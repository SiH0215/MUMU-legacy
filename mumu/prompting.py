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
