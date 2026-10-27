from typing import Any

from .config import Settings
from .errors import ConfigurationError, ServiceError


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
