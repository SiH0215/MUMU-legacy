from dataclasses import dataclass
from typing import Callable, Protocol

from .conversation import Conversation
from .errors import MumuError


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
