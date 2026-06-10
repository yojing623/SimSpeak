import os
import json
import logging
import time
import asyncio
import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from pipeline import SimSpeakAIPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Lightning Async Core API")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./simspeak.db")
Base = declarative_base()

try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as db_err:
    print(f"[DB Error] {db_err}")

class ChatLogModel(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True, nullable=False)
    character_id = Column(String(50), index=True, nullable=False)
    user_text = Column(Text, nullable=False)
    user_audio_url = Column(Text, nullable=True)         
    ai_text_content = Column(Text, nullable=False)
    ai_audio_url = Column(Text, nullable=True)           
    current_affinity = Column(Integer, default=30)       
    summary_context = Column(Text, nullable=True)        
    stage_id = Column(String(50), nullable=True)            
    
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        chat_history_context = Column(JSON, nullable=False)   
        raw_llm_log = Column(JSON, nullable=False)            
    else:
        chat_history_context = Column(JSONB, nullable=False)   
        raw_llm_log = Column(JSONB, nullable=False)            

class CharacterChatLogModel(Base):
    __tablename__ = "character_chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False)
    character_id = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        response_data = Column(JSON, nullable=False)
    else:
        response_data = Column(JSONB, nullable=False)

try:
    Base.metadata.create_all(bind=engine)
except Exception:
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UnifiedChatRequest(BaseModel):
    user_id: str
    character_id: str
    text: str  
    is_video_call: bool
    user_audio_url: Optional[str] = None  
    stage_id: Optional[str] = "stage_1"

pipeline = SimSpeakAIPipeline()

# =================================================================
# 🧠 백그라운드 비동기 워커 (오답노트 및 정밀 발음 리포트 전용 로그)
# =================================================================
async def background_evaluation_worker(user_id: str, char_id: str, stage_id: str, user_audio_url: str, dialogue_result: dict):
    db = SessionLocal()
    try:
        print(f"▶️ [비동기 병렬 피드백 트랙] 스타트 (오답노트 및 정밀 발음 평가 중...)")
        
        user_recognized_text = dialogue_result.get("user_recognized_text", "")
        
        # 1. 오답노트 & 발음 분석 실행
        feedback_payload = await pipeline.run_only_evaluation_track(
            user_id=user_id,
            character_id=char_id,
            user_text=user_recognized_text,
            stage_id=stage_id,
            user_audio_url=user_audio_url
        )
        
        # 로그 일치성을 위해 백그라운드 JSON 객체 내부에도 호감도 데이터 동기화 바인딩
        feedback_json = feedback_payload["system_evaluation"]
        feedback_json["affinity_delta"] = dialogue_result.get("affinity_delta", 0)
        feedback_json["current_total_affinity"] = dialogue_result.get("current_total_affinity", 30)

        print(f"✅ [비동기 병렬 피드백 트랙] 연산 마감 완료!")
        print(f"==================================================================")
        print(f"📊 [TEST MONITORING] 최종 생성된 2차 피드백 JSON 결과 데이터 확인")
        print(f"==================================================================")
        print(json.dumps(feedback_payload, ensure_ascii=False, indent=2))
        print(f"==================================================================")

        # DB 영속화 마감 보관
        new_log = ChatLogModel(
            user_id=user_id,
            character_id=char_id,
            user_text=user_recognized_text,
            user_audio_url=user_audio_url,
            ai_text_content=dialogue_result.get("text_content", ""),
            ai_audio_url=dialogue_result.get("audio_url", ""),
            current_affinity=dialogue_result.get("current_total_affinity", 30), 
            chat_history_context=dialogue_result.get("history_context", []),
            raw_llm_log=dialogue_result.get("raw_llm_log", {}),
            summary_context=dialogue_result.get("summary_context", ""),
            stage_id=stage_id
        )
        db.add(new_log)

        final_monitoring_data = {
            "text_content": dialogue_result.get("text_content", ""),
            "action_description": dialogue_result.get("action_description", ""),
            "audio_url": dialogue_result.get("audio_url", ""),
            "user_recognized_text": user_recognized_text,
            "affinity_delta": dialogue_result.get("affinity_delta", 0),
            "current_total_affinity": dialogue_result.get("current_total_affinity", 30),
            "system_evaluation": feedback_json
        }

        new_monitoring_log = CharacterChatLogModel(
            user_id=user_id,
            character_id=char_id,
            response_data=final_monitoring_data
        )
        db.add(new_monitoring_log)
        db.commit()
        print(f"🎉 [Neon DB] 대사방 로그 + 오답노트 정산본 한 통으로 합치기 최종 성공!")

    except Exception as bg_err:
        db.rollback()
        print(f"❌ [백그라운드 피드백 에러 발생]: {bg_err}")
    finally:
        db.close()


# =================================================================
# 🚀 통합 초고속 엔드포인트 (1차 화면 호감도 반환 버전 원복 완료)
# =================================================================
@app.post("/api/v1/chat/message")
async def process_chat_simultaneously(request: UnifiedChatRequest, db: Session = Depends(get_db)):
    char_id = request.character_id.lower()
    user_id = request.user_id

    last_log = db.query(ChatLogModel).filter(
        ChatLogModel.user_id == user_id, 
        ChatLogModel.character_id == char_id
    ).order_by(ChatLogModel.id.desc()).first()

    history = list(last_log.chat_history_context) if last_log else []
    current_affinity = last_log.current_affinity if last_log else 30
    current_summary = last_log.summary_context or "" if last_log else ""

    temp_session_db = {
        user_id: {
            char_id: {
                "history": history,
                "current_affinity": current_affinity,
                "summary_context": current_summary
            }
        }
    }

    try:
        # 1. 1차 대사방 연산 실행
        dialogue_result = await pipeline.run_only_dialogue_track(
            session_db=temp_session_db,
            user_id=user_id,
            character_id=char_id,
            user_text=request.text,
            is_video_call=request.is_video_call,
            user_audio_url=request.user_audio_url,
            stage_id=request.stage_id
        )

        # 2. 백그라운드로 오답노트 및 정밀 발음 평가 독립 위임 처리
        asyncio.create_task(
            background_evaluation_worker(
                user_id=user_id,
                char_id=char_id,
                stage_id=request.stage_id,
                user_audio_url=request.user_audio_url,
                dialogue_result=dialogue_result
            )
        )

        # 🟢 [원복 완료] 유저가 응답을 즉시 받아볼 때, 호감도 결과 지표를 1차 필드에 포함하여 즉시 패킹 반환합니다!
        return {
          "text_content": dialogue_result.get("text_content"),
          "action_description": dialogue_result.get("action_description"),
          "audio_url": dialogue_result.get("audio_url"),
          "user_recognized_text": dialogue_result.get("user_recognized_text"),
          "affinity_delta": dialogue_result.get("affinity_delta"),           # 👈 다시 전면 배치 완료
          "current_total_affinity": dialogue_result.get("current_total_affinity") # 👈 다시 전면 배치 완료
        }

    except Exception as e:
        print(f"❌ [메인 트랙 치명적 에러]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
