import argparse
from pathlib import Path

from .config import ROOT, Settings, load_system_prompt
from .conversation import Conversation
from .errors import ConfigurationError, MumuError
from .service import ChatService
from .stt import WhisperTranscriber
from .tts import SupertoneSynthesizer, play_audio
from .watsonx import WatsonxGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="MUMU 보관 코드 정리본 — 로컬 대화 실행")
    parser.add_argument("--voice", action="store_true", help="/음성 5초 녹음과 답변 음성 재생 활성화")
    parser.add_argument("--prompt", type=Path, help="사용할 시스템 프롬프트 파일 경로")
    args = parser.parse_args()
    try:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ConfigurationError("requirements.txt 의존성을 먼저 설치해 주세요.") from exc
        load_dotenv(ROOT / ".env")
        settings = Settings.from_environment()
        prompt = load_system_prompt(args.prompt)
        synthesizer = (SupertoneSynthesizer(settings.supertone_api_key, settings.voice_id,
                                           settings.timeout_seconds) if args.voice else None)
        conversation = Conversation(WatsonxGenerator.from_settings(settings), prompt,
                                    settings.max_history)
        service = ChatService(conversation, synthesizer, play_audio if args.voice else None)
        transcriber = WhisperTranscriber(settings.whisper_model) if args.voice else None
        print("무무 >> 안녕~ 나는 무무야! 같이 얘기하자 :)")
        print("종료: exit / 초기화: /초기화" + (" / 5초 녹음: /음성" if args.voice else ""))
        while True:
            try:
                text = input("너 >> ").strip()
                if text.lower() in ("exit", "quit"):
                    break
                if text == "/초기화":
                    conversation.clear()
                    print("무무 >> 대화 이력을 초기화했어. 처음부터 다시 얘기하자 :)")
                    continue
                if text == "/음성":
                    if transcriber is None:
                        print("안내 >> 음성 입력은 --voice 옵션으로 실행해 주세요.")
                        continue
                    print("안내 >> 5초 동안 말해 주세요.")
                    text = transcriber.record_and_transcribe()
                    print(f"인식 >> {text}")
                reply = service.respond(text)
                print(f"무무 >> {reply.text}")
                if reply.audio_error:
                    print(f"음성 안내 >> {reply.audio_error}")
            except MumuError as exc:
                print(f"안내 >> {exc}")
    except MumuError as exc:
        print(f"설정 안내 >> {exc}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
