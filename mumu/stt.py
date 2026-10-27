import math
from pathlib import Path
import tempfile
from typing import Any
import wave

from .errors import ConfigurationError, ServiceError

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
