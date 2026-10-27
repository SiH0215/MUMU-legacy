from dataclasses import dataclass, field
import os
from pathlib import Path

from .errors import ConfigurationError

ROOT = Path(__file__).resolve().parent.parent


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
