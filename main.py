import os
import json
import requests
import uuid # 파일 덮어쓰기 에러 방지용으로 추가
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# 라이브러리 로드
from openai import AzureOpenAI
# pyrefly: ignore [missing-import]
import azure.cognitiveservices.speech as speechsdk

# 환경변수 로드 (절대 경로 적용으로 터미널 꼬임 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Production Pronunciation Core API")

# 1. 메모리 세션 DB
session_db = {}

# 2. API 요청 데이터 스키마 (팀원 코드 유지)
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  
    text: str
    is_video_call: bool
    user_audio_url: Optional[str] = None  # 💡 다른 팀원이 Blob에 저장 후 넘겨줄 오디오 URL 주소![cite: 3]

# 3. 💡 [정우님 투트랙 코어 엔진] 클라우드 URL 오디오 실시간 채점 + Whisper 한영 추출 
def evaluate_dual_track_from_url(audio_url: str) -> tuple[str, dict]:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    whisper_text = ""
    error_response = {
        "accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []
    }

    if not audio_url:
        return whisper_text, error_response

    try:
        # 다른 팀원이 클라우드(Blob)에 올려둔 진짜 음성 파일 다운로드[cite: 3]
        response = requests.get(audio_url, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ 클라우드 오디오 다운로드 실패: {audio_url}")
            return whisper_text, error_response
            
        audio_buffer = response.content
        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        with open(temp_eval_path, "wb") as f:
            f.write(audio_buffer)

        # --------------------------------------------------
        # ★ 트랙 1: 정우님의 Whisper (정확한 혼용 텍스트 추출)
        # --------------------------------------------------
        try:
            openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_WHISPER_API_KEY"),
                api_version="2024-02-15-preview"
            )
            with open(temp_eval_path, "rb") as audio_file:
                whisper_result = openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="drinkingmool-whisper", 
                    prompt="이 오디오는 영어와 한국어가 섞여 있습니다. Hello 안녕하세요.", 
                    language="ko" 
                )
            whisper_text = whisper_result.text
        except Exception as e:
            whisper_text = f"[Whisper 에러: {e}]"

        # --------------------------------------------------
        # ★ 트랙 2: 팀원의 Azure Speech (기존 채점 엔진 그대로 유지)[cite: 3]
        # --------------------------------------------------
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
        
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text="",
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word
        )
        pronunciation_config.enable_prosody_assessment()
        
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language="en-US", audio_config=audio_config
        )
        pronunciation_config.apply_to(speech_recognizer)
        result = speech_recognizer.recognize_once_async().get()
        
        # 가비지 컬렉터 메모리 해제 및 임시 파일 파기[cite: 3]
        del speech_recognizer
        del audio_config
        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)
        
        detailed_score = error_response
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            assessment_result = speechsdk.PronunciationAssessmentResult(result)
            word_details_list = []
            for word in assessment_result.words:
                error_type = word.error_type if word.error_type != "None" else None
                word_details_list.append({
                    "word": word.word.strip(),
                    "accuracy": int(word.accuracy_score),
                    "error_type": error_type
                })
            detailed_score = {
                "accuracy": int(assessment_result.accuracy_score),
                "fluency": int(assessment_result.fluency_score),
                "completeness": int(assessment_result.completeness_score),
                "prosody": int(assessment_result.prosody_score),
                "word_details": word_details_list
            }
        return whisper_text, detailed_score

    except Exception as e:
        print(f"⚠️ 코어 채점 엔진 내부 연산 중 오류 발생: {e}")
        return whisper_text, error_response


def get_character_prompt(character_id: str) -> str:
    file_path = f"prompts/{character_id.lower()}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 4. 🚀 깔끔하게 정리된 완성형 엔드포인트
@app.post("/chat")
async def chat_with_character(request: ChatRequest):
    char_id = request.character_id.lower()
    print(f"📥 [User: {request.user_id}] -> [{char_id}] 초기 입력: {request.text}")

    # 세션 초기화[cite: 3]
    if request.user_id not in session_db:
        session_db[request.user_id] = {}
    if char_id not in session_db[request.user_id]:
        session_db[request.user_id][char_id] = {"history": [], "current_affinity": 30}
    user_data = session_db[request.user_id][char_id]

    real_pronunciation_score = None
    penalty_message = ""
    
    # 💡 [핵심 연동] 오디오 URL이 들어오면 정우님의 투트랙 + 패널티 가동
    if request.user_audio_url:
        print(f"🎙️ 오디오 URL 감지됨 ➡️ 투트랙 가동: {request.user_audio_url}")
        extracted_text, real_pronunciation_score = evaluate_dual_track_from_url(request.user_audio_url)
        
        # Whisper가 텍스트를 무사히 뽑아왔다면 프론트엔드의 빈 텍스트를 이걸로 덮어씌움
        if extracted_text and not extracted_text.startswith("[Whisper"):
            request.text = extracted_text
            
        # ★ 정우님의 패널티 주입 로직 (종합 발음 점수인 'accuracy'가 50 미만일 때)
        score_val = real_pronunciation_score.get("accuracy", 100)
        if score_val < 50:
            penalty_message = "\n[SYSTEM OVERRIDE MESSAGE: 방금 유저의 발음 점수가 낮거나 한국어가 감지되었습니다. 쌀쌀맞게 대하거나 발음을 지적하고, 무조건 affinity_delta를 -3으로 고정하십시오. 예외는 없습니다.]"
    else:
        print("⌨️ 텍스트 전용 채팅 모드 ➡️ 발음 채점을 진행하지 않습니다.")

    # 프롬프트 조립 및 패널티 결합
    base_prompt = get_character_prompt(char_id)
    system_prompt = base_prompt + f"\n\n[LIVE] Affinity: {user_data['current_affinity']}/100" + penalty_message
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_data["history"][-10:])
    messages.append({"role": "user", "content": request.text})

    try:
        # Azure OpenAI 답변 생성[cite: 3]
        ai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            response_format={"type": "json_object"},
            messages=messages
        )
        ai_response_text = response.choices[0].message.content
        ai_result = json.loads(ai_response_text)
        
        # 친밀도 수치 갱신 및 히스토리 저장[cite: 3]
        affinity_delta = ai_result.get("affinity_delta", 0)
        user_data["history"].append({"role": "user", "content": request.text})
        user_data["history"].append({"role": "assistant", "content": ai_response_text})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + affinity_delta))

        # 결과 주머니 패키징[cite: 3]
        ai_result["audio_url"] = "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        ai_result["user_recognized_text"] = request.text # 프론트에 Whisper 텍스트 반환
        
        if "system_evaluation" not in ai_result:
            ai_result["system_evaluation"] = {}
            
        # 🔥 실시간으로 연산된 리얼 발음 점수 주입[cite: 3]
        ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
        
        return ai_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))