import unittest

from mumu.conversation import Conversation
from mumu.errors import InputError, ServiceError
from mumu.prompting import EOT, build_prompt
from mumu.service import ChatService
from mumu.watsonx import WatsonxGenerator


class Generator:
    def __init__(self):
        self.prompts = []
        self.answer = "  안녕!" + EOT + "제거할 내용"
        self.error = False

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise ServiceError("호출 실패")
        return self.answer


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.chat = Conversation(self.generator, "시스템 지시", 5)

    def test_prompt_matches_original_token_order(self):
        self.assertEqual(build_prompt(" s ", [(" u ", " a ")], " n "),
                         "<|begin_of_text|>\n<|header_start|>system<|header_end|>\ns\n<|eot|>\n"
                         "<|header_start|>user<|header_end|>\nu\n<|eot|>\n"
                         "<|header_start|>assistant<|header_end|>\na\n<|eot|>\n"
                         "<|header_start|>user<|header_end|>\nn\n<|eot|>\n"
                         "<|header_start|>assistant<|header_end|>\n")

    def test_last_five_pairs_and_reset(self):
        for i in range(7):
            self.assertEqual(self.chat.reply(f"질문{i}"), "안녕!")
        self.assertEqual(len(self.chat.history), 5)
        self.assertEqual(self.chat.history[0][0], "질문2")
        self.chat.reply("다음 질문")
        self.assertNotIn("질문0", self.generator.prompts[-1])
        self.assertIn("질문2", self.generator.prompts[-1])
        self.chat.clear()
        self.chat.reply("새 대화")
        self.assertNotIn("질문2", self.generator.prompts[-1])

    def test_invalid_input_never_calls_model(self):
        for value in ("   ", "안녕 <|eot|>"):
            with self.assertRaises(InputError):
                self.chat.reply(value)
        self.assertEqual(self.generator.prompts, [])

    def test_failed_or_empty_reply_does_not_change_history(self):
        self.chat.reply("처음")
        expected = self.chat.history
        self.generator.error = True
        with self.assertRaises(ServiceError):
            self.chat.reply("실패")
        self.assertEqual(self.chat.history, expected)
        self.generator.error = False
        self.generator.answer = " " + EOT
        with self.assertRaises(ServiceError):
            self.chat.reply("빈 응답")
        self.assertEqual(self.chat.history, expected)

    def test_voice_failure_preserves_text_and_history(self):
        class BrokenTTS:
            def synthesize(self, text):
                raise ServiceError("TTS 실패")

        reply = ChatService(self.chat, BrokenTTS()).respond("안녕")
        self.assertEqual(reply.text, "안녕!")
        self.assertEqual(reply.audio_error, "TTS 실패")
        self.assertEqual(self.chat.history, (("안녕", "안녕!"),))

    def test_malformed_watsonx_response_is_a_service_error(self):
        class Model:
            def generate(self, **kwargs):
                return {"results": []}

        with self.assertRaises(ServiceError):
            WatsonxGenerator(Model()).generate("입력")


if __name__ == "__main__":
    unittest.main()
