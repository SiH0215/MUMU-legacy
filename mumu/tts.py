from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

from .errors import ConfigurationError, InputError, ServiceError


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
