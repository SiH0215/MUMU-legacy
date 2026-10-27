"""MUMU 레거시 통합본.

보관된 개별 코드를 바탕으로 2026-10-27 재구성한 파일입니다.
해커톤 당시 최종 통합 소스가 아니며, 종료된 Supertone 연동을 보존합니다.
외부 API 실행을 권장하는 자료가 아닙니다. 코드 열람과 구조 설명을 위한 보관본입니다.
"""


# ============================================================
# errors.py
# ============================================================
class MumuError(Exception):
    """사용자에게 원문 자격 증명 없이 표시할 수 있는 오류."""


class ConfigurationError(MumuError):
    pass


class InputError(MumuError):
    pass


class ServiceError(MumuError):
    pass


# ============================================================
# config.py
# ============================================================
from dataclasses import dataclass, field
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    project_id: str
    cloud_url: str
    supertone_api_key: str = field(default="", repr=False)
    voice_id: str = ""
    model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    max_new_tokens: int = 900
    max_history: int = 5
    timeout_seconds: float = 30.0
    whisper_model: str = "base"

    @classmethod
    def from_environment(cls) -> "Settings":
        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise ConfigurationError(f".env의 {name} 설정이 필요합니다.")
            return value

        return cls(
            api_key=required("API_KEY"),
            project_id=required("PROJECT_ID"),
            cloud_url=required("IBM_CLOUD_URL"),
            supertone_api_key=os.getenv("SUPERTONE_API", "").strip(),
            voice_id=os.getenv("VOICE_ID", "").strip(),
            model_id=os.getenv("WATSONX_MODEL_ID", cls.model_id).strip(),
            whisper_model=os.getenv("WHISPER_MODEL", "base").strip(),
        )


def load_system_prompt(path: Path | None = None) -> str:
    path = path or ROOT / "SYSTEMPROMPT.txt"
    try:
        prompt = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise ConfigurationError(
            "SYSTEMPROMPT.txt를 읽을 수 없습니다. 원본 프롬프트 또는 예시를 준비하세요."
        ) from exc
    if not prompt:
        raise ConfigurationError("시스템 프롬프트가 비어 있습니다.")
    return prompt


# ============================================================
# prompting.py
# ============================================================
"""원본 프롬프트 토큰과 메시지 순서를 유지한다."""

BOS = "<|begin_of_text|>"
EOT = "<|eot|>"
USER_TAG = "<|header_start|>user<|header_end|>"
SYS_TAG = "<|header_start|>system<|header_end|>"
ASSISTANT_TAG = "<|header_start|>assistant<|header_end|>"
RESERVED_TOKENS = (BOS, EOT, USER_TAG, SYS_TAG, ASSISTANT_TAG)


def contains_reserved_tokens(text: str) -> bool:
    return any(token in text for token in RESERVED_TOKENS)


def build_prompt(system: str, history: list[tuple[str, str]], user_input: str) -> str:
    parts = [f"{BOS}\n{SYS_TAG}\n{system.strip()}\n{EOT}\n"]
    for user, assistant in history:
        parts.append(f"{USER_TAG}\n{user.strip()}\n{EOT}\n")
        parts.append(f"{ASSISTANT_TAG}\n{assistant.strip()}\n{EOT}\n")
    parts.append(f"{USER_TAG}\n{user_input.strip()}\n{EOT}\n")
    parts.append(f"{ASSISTANT_TAG}\n")
    return "".join(parts)


def clean_response(text: str) -> str:
    return text.split(EOT, 1)[0].strip()


# ============================================================
# conversation.py
# ============================================================
from typing import Protocol



class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class Conversation:
    """성공한 대화만 보관하고 최근 N개 쌍을 모델에 전달한다."""

    def __init__(self, generator: TextGenerator, system_prompt: str, max_history: int = 5):
        if max_history < 1:
            raise ValueError("max_history는 1 이상이어야 합니다.")
        self.generator = generator
        self.system_prompt = system_prompt
        self.max_history = max_history
        self._history: list[tuple[str, str]] = []

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._history)

    def clear(self) -> None:
        self._history.clear()

    def reply(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            raise InputError("대화 내용을 입력해 주세요.")
        if contains_reserved_tokens(user_input):
            raise InputError("모델 예약 토큰은 입력할 수 없습니다.")
        prompt = build_prompt(self.system_prompt, self._history, user_input)
        answer = clean_response(self.generator.generate(prompt))
        if not answer:
            raise ServiceError("모델이 빈 답변을 반환했습니다. 다시 시도해 주세요.")
        self._history.append((user_input, answer))
        self._history = self._history[-self.max_history:]
        return answer


# ============================================================
# watsonx.py
# ============================================================
from typing import Any



class WatsonxGenerator:
    """보관 코드의 SDK와 generate 응답 형식을 유지한 어댑터."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "WatsonxGenerator":
        try:
            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
        except ImportError as exc:
            raise ConfigurationError("텍스트 실행용 의존성을 먼저 설치해 주세요.") from exc
        try:
            model = Model(
                model_id=settings.model_id,
                params={GenTextParamsMetaNames.MAX_NEW_TOKENS: settings.max_new_tokens,
                        "stop_sequences": ["<|eot|>"]},
                credentials={"apikey": settings.api_key, "url": settings.cloud_url},
                project_id=settings.project_id,
            )
        except Exception as exc:
            raise ServiceError("watsonx 초기화에 실패했습니다. 계정·리전·모델 설정을 확인하세요.") from exc
        return cls(model)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate(prompt=prompt)
            text = response["results"][0]["generated_text"]
            if not isinstance(text, str):
                raise TypeError("generated_text must be a string")
            return text
        except Exception as exc:
            raise ServiceError("답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.") from exc


# ============================================================
# stt.py
# ============================================================
import math
from pathlib import Path
import tempfile
from typing import Any
import wave


SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK = 1024


def record_audio(seconds: float = 5.0) -> Path:
    """로컬 마이크에서 16kHz·모노·16비트 WAV를 녹음한다."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("녹음 시간은 유한한 양수여야 합니다.")
    try:
        import pyaudio
    except ImportError as exc:
        raise ConfigurationError("마이크 녹음에는 PyAudio가 필요합니다.") from exc
    audio = pyaudio.PyAudio()
    stream = None
    path = None
    try:
        sample_width = audio.get_sample_size(pyaudio.paInt16)
        stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS,
                            rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        remaining = int(SAMPLE_RATE * seconds)
        while remaining > 0:
            count = min(CHUNK, remaining)
            frames.append(stream.read(count, exception_on_overflow=False))
            remaining -= count
        # Windows에서도 wave.open이 접근할 수 있도록 임시 핸들을 먼저 닫는다.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            path = Path(temp.name)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(sample_width)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"".join(frames))
        return path
    except Exception as exc:
        if path is not None:
            path.unlink(missing_ok=True)
        raise ServiceError("마이크 녹음에 실패했습니다. 입력 장치와 권한을 확인하세요.") from exc
    finally:
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        finally:
            audio.terminate()


class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model: Any = None

    def transcribe_file(self, path: Path) -> str:
        if not path.is_file():
            raise InputFileError("음성 파일을 찾을 수 없습니다.")
        try:
            import whisper
        except ImportError as exc:
            raise ConfigurationError("음성 인식에는 openai-whisper가 필요합니다.") from exc
        try:
            if self._model is None:
                self._model = whisper.load_model(self.model_name)
            result = self._model.transcribe(str(path), language="ko")
            text = result["text"]
            if not isinstance(text, str):
                raise TypeError("STT text must be a string")
            return text.strip()
        except Exception as exc:
            raise ServiceError("음성을 인식하지 못했습니다. 파일과 Whisper 실행 환경을 확인하세요.") from exc

    def record_and_transcribe(self, seconds: float = 5.0) -> str:
        path = record_audio(seconds)
        try:
            return self.transcribe_file(path)
        finally:
            path.unlink(missing_ok=True)


class InputFileError(ServiceError):
    pass


# ============================================================
# tts.py
# ============================================================
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse



class SupertoneSynthesizer:
    """음성 본문과 JSON audio_url 두 종류의 보관 응답 방식을 처리한다."""

    def __init__(self, api_key: str, voice_id: str, timeout: float = 30, session: Any = None):
        if not api_key or not voice_id:
            raise ConfigurationError("음성 출력에는 SUPERTONE_API와 VOICE_ID가 필요합니다.")
        if timeout <= 0:
            raise ValueError("timeout은 양수여야 합니다.")
        self.api_key = api_key
        self.voice_id = voice_id
        self.timeout = timeout
        self.session = session

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise InputError("빈 답변은 음성으로 변환할 수 없습니다.")
        if self.session is None:
            try:
                import requests
            except ImportError as exc:
                raise ConfigurationError("음성 합성에는 requests가 필요합니다.") from exc
            self.session = requests.Session()
        try:
            response = self.session.post(
                f"https://supertoneapi.com/v1/text-to-speech/{self.voice_id}",
                json={"text": text, "language": "ko", "style": "Sad",
                      "model": "sona_speech_1", "output_format": "wav",
                      "voice_settings": {"pitch_shift": 0, "pitch_variance": 1, "speed": 1.1}},
                headers={"x-sup-api-key": self.api_key, "x-tts-engine": "supercage",
                         "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type.startswith("audio/"):
                audio = response.content
            elif "application/json" in content_type:
                url = response.json().get("audio_url")
                if not isinstance(url, str) or urlparse(url).scheme != "https":
                    raise ValueError("Expected an HTTPS audio_url")
                # 원래 API 키를 외부 다운로드 URL에 전달하지 않는다.
                downloaded = self.session.get(url, timeout=self.timeout)
                downloaded.raise_for_status()
                if not downloaded.headers.get("Content-Type", "").lower().startswith("audio/"):
                    raise ValueError("Expected audio content")
                audio = downloaded.content
            else:
                raise ValueError("Unsupported response type")
            if not audio:
                raise ValueError("Empty audio response")
            return audio
        except Exception as exc:
            raise ServiceError("음성 합성에 실패했습니다. API 설정과 서비스 응답을 확인하세요.") from exc


def play_audio(audio: bytes) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise ConfigurationError("로컬 음성 재생에는 pygame이 필요합니다.") from exc
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(audio)
        path = Path(temp.name)
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as exc:
        raise ServiceError("음성을 재생하지 못했습니다. 출력 장치를 확인하세요.") from exc
    finally:
        try:
            pygame.mixer.quit()
        finally:
            path.unlink(missing_ok=True)


# ============================================================
# service.py
# ============================================================
from dataclasses import dataclass
from typing import Callable, Protocol



class Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...


@dataclass(frozen=True)
class Reply:
    text: str
    audio_error: str | None = None


class ChatService:
    def __init__(self, conversation: Conversation, synthesizer: Synthesizer | None = None,
                 player: Callable[[bytes], None] | None = None):
        self.conversation = conversation
        self.synthesizer = synthesizer
        self.player = player

    def respond(self, text: str) -> Reply:
        answer = self.conversation.reply(text)
        if self.synthesizer is not None:
            try:
                audio = self.synthesizer.synthesize(answer)
                if self.player is not None:
                    self.player(audio)
            except MumuError as exc:
                # 음성 출력 실패가 성공한 텍스트 답변과 대화 이력을 지우지 않는다.
                return Reply(answer, str(exc))
        return Reply(answer)


# ============================================================
# __main__.py
# ============================================================
import argparse
from pathlib import Path



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
