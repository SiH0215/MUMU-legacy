# MUMU Legacy

LLM 대화, 음성 인식, 음성 합성 연동 코드를 보관하는 저장소입니다. 통합본과 역할별 모듈, 코드 동작 설명을 포함합니다.

**레거시 코드 보관용입니다. 사용하던 Supertone 음성 서비스 종료로 전체 음성 대화는 실행할 수 없습니다.** 대체 TTS나 서비스 복구는 구현하지 않았습니다.

## 코드

- [mumu_legacy.py](mumu_legacy.py): 개별 보관 소스를 한 파일로 재구성한 통합본
- [mumu/](mumu/): 설정, 프롬프트, 대화 관리와 외부 API를 분리한 모듈별 정리본
- [코드 설명](docs/CODE_GUIDE.md): 처리 흐름, 함수 역할, 응답 처리와 예외 처리
- [tests/](tests/): 외부 API와 음성 장치 없이 실행하는 테스트

```text
로컬 마이크 → Whisper STT → 사용자 문장
                                ↓
시스템 프롬프트 + 최근 대화 → watsonx LLM → 답변 문장
                                              ↓
                                      Supertone TTS → 음성 출력
```

## 보관 범위

당시 최종 통합 파일과 원본 시스템 프롬프트가 남아 있지 않아, 개별 소스를 바탕으로 통합 구조를 재구성했습니다. 통합본과 모듈별 정리본은 같은 처리 구조를 담습니다. `SYSTEMPROMPT.example.txt`는 새로 작성한 예시이며 원본 프롬프트가 아닙니다.

보관한 동작은 프롬프트 구성, 최근 5개 대화 쌍 관리, watsonx 모델 호출, Whisper 한국어 인식, Supertone 요청·응답 처리입니다. 화면 코드, 감정 일기 저장, 요약과 RAG 구현은 포함하지 않습니다.

모델 ID·프롬프트 토큰·API 요청 형식은 보관 소스의 흔적을 유지합니다. 현재 SDK나 외부 서비스와의 호환성을 보장하지 않습니다. `requirements*.txt`와 `.env.example`은 의존성과 설정 항목을 설명하기 위해 남겼습니다. 실제 API 키와 개인 계정 정보는 포함하지 않습니다.

## 확인 방법

Python 3.10 이상에서 외부 서비스에 연결하지 않고 아래 검사를 실행할 수 있습니다.

```shell
python -m unittest discover -s tests -v
python -m compileall -q mumu mumu_legacy.py tests
```

테스트는 모의 모델·HTTP·음성 장치로 코드 동작을 확인합니다. 전체 음성 대화나 실제 API 연동을 검증한 결과는 아닙니다.
