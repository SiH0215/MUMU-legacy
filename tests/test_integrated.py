import unittest

import mumu_legacy as legacy
from mumu.conversation import Conversation
from mumu.prompting import build_prompt


class Generator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return " 답변 <|eot|>제거할 내용"


class IntegratedTests(unittest.TestCase):
    def test_prompt_matches_modular_version(self):
        args = ("시스템", [("이전 입력", "이전 답변")], "현재 입력")
        self.assertEqual(legacy.build_prompt(*args), build_prompt(*args))

    def test_history_reply_and_reset_match(self):
        left_generator, right_generator = Generator(), Generator()
        left = legacy.Conversation(left_generator, "시스템")
        right = Conversation(right_generator, "시스템")
        for i in range(7):
            self.assertEqual(left.reply(str(i)), right.reply(str(i)))
        self.assertEqual(left.history, right.history)
        self.assertEqual(left_generator.prompts, right_generator.prompts)
        left.clear()
        right.clear()
        self.assertEqual(left.history, right.history)


if __name__ == "__main__":
    unittest.main()
