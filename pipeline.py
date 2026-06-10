import os
import uuid
import json
import asyncio
import httpx
import io
import aiofiles
import re
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

class SimSpeakAIPipeline:
    def __init__(self):
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.whisper_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        self.whisper_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
        self.whisper_deployment = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT_NAME", "drinkingmool-whisper")
        
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.storage_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    async def call_llm_with_retry(self, ai_client, **kwargs):
        max_retries = 2
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                return await ai_client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(backoff)
                else:
                    raise e

    async def generate_lightning_dialogue(self, messages: list) -> str:
        async_client = AsyncAzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
            api_version="2024-02-15-preview"
        )
        safe_messages = []
        for m in messages:
            if m["role"] == "system":
                safe_messages.append(m)
            else:
                safe_messages.append({
                    "role": m["role"], 
                    "content": [{"type": "text", "text": str(m["content"])}]
                })

        try:
            response = await self.call_llm_with_retry(
                async_client,
                model="gpt-4o-mini", # ⚡ [복구 완료] 대사 초고속 출력을 위해 mini 모델 하드코딩 유지
                messages=safe_messages,
                max_tokens=250
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"🚨 [초고속 대사 엔진] 장애 우회 처리: {e}")
            return '{"text_content": "앗, 미안해! 데이터가 살짝 밀렸나 봐. 다시 말해줄래?", "action_description": "멋쩍게 웃는다."}'
    
    def make_ssml(self, character_id: str, text_content: str) -> str:
        import re
        char_id = character_id.lower()
        voice_name = "en-US-AndrewMultilingualNeural"
        rate, pitch = "0%", "0%"
        if char_id == "liam":
            voice_name = "en-GB-OllieMultilingualNeural"
            rate, pitch = "-10%", "-5%"
        elif char_id == "chloe":
            voice_name = "en-US-AvaMultilingualNeural"
            rate, pitch = "+10%", "+5%"
            
        pattern = re.compile(r'([\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+(?:\s+[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+)*)')
        wrapped_text = pattern.sub(r'<lang xml:lang="ko-KR">\1</lang>', text_content)
        return f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="{voice_name}"><prosody rate="{rate}" pitch="{pitch}">{wrapped_text}</prosody></voice></speak>'

    async def quick_whisper_transcription(self, user_id: str, audio_url: str) -> str:
        if not audio_url: return ""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=15.0)
                if response.status_code != 200: return ""
                audio_bytes = response.content

            whisper_client = AsyncAzureOpenAI(azure_endpoint=self.whisper_endpoint, api_key=self.whisper_key, api_version="2024-02-15-preview")
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "speech.wav"
            
            whisper_result = await whisper_client.audio.transcriptions.create(
                file=audio_file, model=self.whisper_deployment, prompt="Hello! 안녕하세요.", language="en"
            )
            print(f" 🔍 [ASYNC FLOW] User '{user_id}' - Whisper Text Extracted: '{whisper_result.text}'")
            return whisper_result.text
        except Exception as e:
            print(f" ❌ [WHISPER ERROR] User '{user_id}' - {e}")
            return ""

    async def run_azure_pronunciation_assessment(self, user_id: str, audio_url: str, reference_text: str) -> dict:
        error_response = {"accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details_json": []}
        if not audio_url or not reference_text or reference_text.strip() == "": 
            return error_response
            
        temp_audio_file = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=15.0)
                if response.status_code != 200: 
                    print(f" ❌ [SPEECH ACC] 오디오 다운로드 에러. HTTP {response.status_code}")
                    return error_response
                
                with open(temp_audio_file, "wb") as f:
                    f.write(response.content)

            def run_speech_assessment():
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_file)
                
                pure_english_reference = re.sub(r'[^a-zA-Z\s\']', ' ', reference_text)
                pure_english_reference = " ".join(pure_english_reference.split())
                
                if not pure_english_reference:
                    return None

                pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text=pure_english_reference,
                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                    granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
                )
                pronunciation_config.phoneme_alphabet = "IPA"
                pronunciation_config.enable_prosody_assessment()
                
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
                pronunciation_config.apply_to(speech_recognizer)
                return speech_recognizer.recognize_once_async().get()

            result = await asyncio.to_thread(run_speech_assessment)
            
            if result and result.reason == speechsdk.ResultReason.RecognizedSpeech:
                assessment_result = speechsdk.PronunciationAssessmentResult(result)
                word_details_list = []
                for word in assessment_result.words:
                    ipa_guide = f"[{''.join(p.phoneme for p in word.phonemes)}]" if word.phonemes else ""
                    guide = ipa_guide if word.accuracy_score < 75 else ""
                    word_details_list.append({
                        "word": word.word.strip(), 
                        "accuracy": int(word.accuracy_score), 
                        "error_type": word.error_type if word.error_type != "None" else None, 
                        "guide": guide
                    })
                return {
                    "accuracy": int(assessment_result.accuracy_score), 
                    "fluency": int(assessment_result.fluency_score), 
                    "completeness": int(assessment_result.completeness_score), 
                    "prosody": int(assessment_result.prosody_score), 
                    "word_details_json": word_details_list
                }
            else:
                return error_response
        except Exception as e:
            print(f" ❌ [SPEECH CRITICAL ERROR] {e}")
            return error_response
        finally:
            if os.path.exists(temp_audio_file):
                try: os.remove(temp_audio_file)
                except: pass

    async def generate_tts(self, user_id: str, character_id: str, text_content: str) -> str:
        if not text_content or text_content.strip() == "":
            return ""
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        try:
            def run_tts_synthesis():
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                synthesizer.speak_ssml_async(self.make_ssml(character_id, text_content)).get()
            await asyncio.to_thread(run_tts_synthesis)

            def upload_to_blob():
                blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
                blob_client = blob_service_client.get_blob_client(container="audio-files", blob=temp_filename)
                with open(temp_filename, "rb") as data: blob_client.upload_blob(data, overwrite=True)
                return blob_client.url
            blob_url = await asyncio.to_thread(upload_to_blob)
            return blob_url
        except Exception:
            return ""
        finally:
            if os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except: pass

    async def get_character_prompt(self, character_id: str) -> str:
        async with aiofiles.open(f"prompts/{character_id.lower()}.txt", "r", encoding="utf-8") as f: return await f.read()

    # =========================================================================
    # ⚡ 1차 초고속 대사 처리용 최적화 런타임
    # =========================================================================
    async def run_only_dialogue_track(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None, stage_id: str = "stage_1") -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]: session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]
        current_summary = user_data.get("summary_context", "")

        if user_audio_url:
            extracted_text = await self.quick_whisper_transcription(user_id, user_audio_url)
            if extracted_text: user_text = extracted_text

        base_prompt = await self.get_character_prompt(char_id)
        summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
        
        json_injection_rule = """
        [CRITICAL OUTPUT RULE]
        You MUST respond ONLY with a raw, pure JSON object matching this schema. Do not include markdown blocks like ```json.
        {
          "text_content": "Your verbal response in English (keep it under 2 short sentences)",
          "action_description": "Friendly behavioral status in Korean"
        }
        """
        system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}\n\n{json_injection_rule}"
        messages = [{"role": "system", "content": system_prompt}]
        
        for turn in user_data["history"][-6:]:
            try:
                data = json.loads(turn["content"])
                messages.append({"role": turn["role"], "content": data.get("text_content", turn["content"])})
            except Exception:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_text})

        print(f" 🧠 [ASYNC LLM CALL] User '{user_id}' - Requesting Dialogue 가속엔진 가동...")
        raw_response = await self.generate_lightning_dialogue(messages)
        clean_json_str = re.sub(r'```json\s*|```', '', raw_response).strip()
        
        try:
            ai_result = json.loads(clean_json_str)
        except Exception:
            text_match = re.search(r'"text_content"\s*:\s*"([^"]+)"', clean_json_str)
            action_match = re.search(r'"action_description"\s*:\s*"([^"]+)"', clean_json_str)
            ai_result = {
                "text_content": text_match.group(1) if text_match else "Oh, sorry! I got distracted. What were you saying?",
                "action_description": action_match.group(1) if action_match else "여유롭게 웃어 보인다."
            }

        ai_result["affinity_delta"] = 1
        ai_result["system_evaluation"] = {"is_penalty": False}

        words = user_text.split()
        if words:
            korean_word_count = sum(1 for w in words if any(0xAC00 <= ord(c) <= 0xD7A3 for c in w))
            if (korean_word_count / len(words)) >= 0.30:
                ai_result["affinity_delta"] = -1

        user_data["history"].append({"role": "user", "content": user_text})
        user_data["history"].append({"role": "assistant", "content": json.dumps(ai_result, ensure_ascii=False)})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + ai_result["affinity_delta"]))

        main_audio_url = await self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))

        ai_result["audio_url"] = main_audio_url
        ai_result["user_recognized_text"] = user_text
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        ai_result["summary_context"] = current_summary
        ai_result["history_context"] = user_data["history"]
        ai_result["raw_llm_log"] = {"model": "gpt-4o-mini (Lightning)"}

        return ai_result

    # =========================================================================
    # ⚡ 2차 오답노트 백그라운드 (동적 IPA 가이드라인 탑재)
    # =========================================================================
    async def run_only_evaluation_track(self, user_id: str, character_id: str, user_text: str, stage_id: str = "stage_1", user_audio_url: str = None) -> dict:
        char_id = character_id.lower()
        ai_client = AsyncAzureOpenAI(azure_endpoint=self.openai_endpoint, api_key=self.openai_key, api_version="2024-02-15-preview")
        
        if not user_audio_url or user_audio_url.strip() == "":
            return {"system_evaluation": {"is_penalty": False, "grammar_feedback": "음성 파일 링크가 존재하지 않아 평가가 스킵되었습니다.", "corrections_json": [], "pronunciation_evaluations": None, "pronunciation_feedback": None}}

        real_pronunciation_evaluations = await self.run_azure_pronunciation_assessment(user_id, user_audio_url, user_text)

        # 🟢 [수정됨] 프롬프트 내부에 "ipa_guides" 생성을 지시하여, AI가 유저 단어의 발음 기호 사전을 실시간으로 뱉어내게 만듭니다.
        system_feedback_prompt = """
        너는 영어 교육 평가 시스템이야. 유저가 콩글리시나 어색한 한글 섞인 표현을 썼다면 세련된 100% 원어민 영문으로 고쳐줘.
        또한, 유저 원문(original_sentence)에 포함된 주요 영단어들에 대한 정확한 원어민 IPA 발음 기호(예: [kæˈfeɪ])를 'ipa_guides' 딕셔너리에 포함해줘.
        
        응답은 반드시 마크다운 블록 없이 순수 JSON만 반환해라:
        {
            "is_penalty": false,
            "grammar_feedback": "교정 이유를 설명하는 한국어 피드백",
            "corrections_json": [
                {"original_sentence": "유저 원문", "corrected_sentence": "원어민 네이티브 영문"}
            ],
            "ipa_guides": {
                "word1": "[ipa1]",
                "word2": "[ipa2]"
            }
        }
        """
        try:
            response = await self.call_llm_with_retry(
                ai_client, 
                model="gpt-4o-mini", # ⚡ [복구 완료] 초고속 피드백 연산을 위해 mini 모델 적용 유지
                messages=[{"role": "system", "content": system_feedback_prompt}, {"role": "user", "content": [{"type": "text", "text": str(user_text)}]}],
            )
            clean_feedback = re.sub(r'```json\s*|```', '', response.choices[0].message.content).strip()
            feedback_json = json.loads(clean_feedback)
        except Exception:
            feedback_json = {"is_penalty": False, "grammar_feedback": "시스템 분석 지연으로 실시간 문법 교정이 불가능합니다.", "corrections_json": [], "ipa_guides": {}}

        if "corrections" in feedback_json:
            feedback_json["corrections_json"] = feedback_json.pop("corrections")
        
        if "corrections_json" not in feedback_json or not feedback_json["corrections_json"]:
            feedback_json["corrections_json"] = [{
                "original_sentence": user_text,
                "corrected_sentence": user_text
            }]
            if "grammar_feedback" not in feedback_json or not feedback_json["grammar_feedback"]:
                feedback_json["grammar_feedback"] = "추가적인 콩글리시 패턴이나 문법적 오류가 감지되지 않은 완성도 높은 문장입니다."

        for corr in feedback_json["corrections_json"]:
            sentence = corr.get("corrected_sentence", "")
            if sentence and sentence != user_text:
                corr["corrected_audio_url"] = await self.generate_tts(user_id, char_id, sentence)
            else:
                corr["corrected_audio_url"] = None

        # 🟢 [수정됨] 하드코딩되었던 fallback_ipa_map 완전 삭제
        # GPT가 실시간으로 분석해준 ipa_guides 딕셔너리를 소문자 키값으로 가져옵니다.
        gpt_ipa_map = {k.lower(): v for k, v in feedback_json.get("ipa_guides", {}).items()}

        if real_pronunciation_evaluations and len(real_pronunciation_evaluations.get("word_details_json", [])) > 0:
            for word_obj in real_pronunciation_evaluations.get("word_details_json", []):
                acc = word_obj.get("accuracy", 0)
                w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                
                if acc >= 75:
                    word_obj["guide"] = ""
                else:
                    g_val = word_obj.get("guide", "")
                    # 🟢 [수정됨] Azure가 발음 기호를 안 줬을 때, GPT가 동적으로 생성한 사전에서 가져옵니다.
                    if not g_val and w_lower in gpt_ipa_map: 
                        g_val = gpt_ipa_map[w_lower]
                    word_obj["guide"] = g_val if (g_val.startswith("[") or not g_val) else f"[{g_val}]"
            
            feedback_json["pronunciation_evaluations"] = real_pronunciation_evaluations
            feedback_json["pronunciation_feedback"] = "전반적인 문장 억양과 발음 분석이 성공적으로 마감되었습니다."
        else:
            feedback_json["pronunciation_evaluations"] = None
            feedback_json["pronunciation_feedback"] = "오디오 입력 상태 미비 또는 매칭 텍스트 부재로 인해 정밀 발음 평가를 수립할 수 없습니다."

        return {"system_evaluation": feedback_json}
