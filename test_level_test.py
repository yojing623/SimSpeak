import asyncio
import httpx

async def run_test():
    url = "http://127.0.0.1:8000/api/v1/chat/level_test"
    user_id = "2"
    char_id = "chloe"

    accumulated_answers = []

    print("🚀 [레벨 테스트 플로우 시뮬레이션 시작]")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 1번부터 7번 문항 답변 시뮬레이션
        for i in range(1, 8):
            print(f"\n▶️ [Q{i}] 전송 중...")
            payload = {
                "user_id": user_id,
                "character_id": char_id,
                "current_question_index": i,
                "user_text": f"This is a dummy answer for question {i}. I am practicing my English.",
            }
            
            response = await client.post(url, json=payload)
            data = response.json()
            
            print(f"✅ [Q{i} 완료] 백엔드가 읽어줄 다음 질문: {data.get('next_question_text')}")
            
            # 클라이언트(프론트)단에서 누적하는 답변 기록 배열
            accumulated_answers.append({
                "question_index": i,
                "text": data.get("user_recognized_text", ""),
                "accuracy": 85,  # 가상의 발음 점수
                "fluency": 80
            })

        # 2. 대망의 8번 문항 (최종 제출)
        print("\n▶️ [Q8] 최종 문항 전송 및 종합 평가 워커 트리거!")
        final_payload = {
            "user_id": user_id,
            "character_id": char_id,
            "current_question_index": 8,
            "user_text": "I strongly agree. First impressions are really crucial in building relationships, although it's not everything.",
            "accumulated_answers": accumulated_answers
        }
        
        response = await client.post(url, json=final_payload)
        data = response.json()
        print("\n✅ [Q8 응답 완료] 8번 응답:", data)
        print("💡 이제 백엔드 터미널 창을 확인해보세요! '▶️ [레벨 테스트 종합 평가] 스타트' 로그가 찍히고 DB에 저장될 것입니다.")

if __name__ == "__main__":
    asyncio.run(run_test())
