from typing import Protocol

from .errors import InputError, ServiceError
from .prompting import build_prompt, clean_response, contains_reserved_tokens


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
