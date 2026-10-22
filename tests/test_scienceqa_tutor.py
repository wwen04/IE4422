from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_scienceqa_tutor as runner
from IE4422_Qwen_Tutor.tutor_pipeline import extract_evidence, diagnose, generate_hint


class TutorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data/images").mkdir(parents=True)
        (self.root / "data/images/example.png").write_bytes(b"mock image")
        self.questions = self.root / "data/processed/scienceqa/questions"
        self.questions.mkdir(parents=True)
        self.item = {
            "item_id": "validation_000003", "split": "validation",
            "question": "Which arrow points up?", "context": "Use the arrow labels.",
            "choices": [{"id": "A", "text": "left"}, {"id": "B", "text": "up"}],
            "image_path": "data/images/example.png",
            "answer": "SECRET GOLD", "lecture": "SECRET LECTURE", "solution": "SECRET SOLUTION",
        }
        self.write_question()

    def write_question(self):
        (self.questions / "validation.jsonl").write_text(json.dumps(self.item) + "\n", encoding="utf-8")

    def test_public_prompt_keeps_context_and_order_without_reference_fields(self):
        item = runner.load_question(self.root, "validation", self.item["item_id"])
        prompt = runner.question_prompt(item)
        self.assertIn("Use the arrow labels.", prompt)
        self.assertIn("A. left\nB. up", prompt)
        self.assertNotIn("SECRET", str(item))
        calls = []

        class FakeTutor:
            def ask(self, image, instruction, **kwargs):
                calls.append(instruction)
                return "uncertain"

        tutor = FakeTutor()
        evidence = extract_evidence(tutor, item["image_path"], prompt)
        diagnosis = diagnose(tutor, item["image_path"], prompt, "A", "My reasoning", evidence)
        generate_hint(tutor, item["image_path"], prompt, evidence, diagnosis)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("SECRET", "\n".join(calls))

    def test_missing_data_item_and_image_fail_before_inference(self):
        with self.assertRaisesRegex(FileNotFoundError, "Run prepare_data.py"):
            runner.load_question(self.root, "train", "train_000003")
        with self.assertRaisesRegex(ValueError, "not found"):
            runner.load_question(self.root, "validation", "validation_999999")
        (self.root / self.item["image_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing prepared image"):
            runner.load_question(self.root, "validation", self.item["item_id"])

    def test_split_mismatch_and_image_outside_repository_are_rejected(self):
        self.item["split"] = "test"
        self.write_question()
        with self.assertRaisesRegex(ValueError, "split"):
            runner.load_question(self.root, "validation", self.item["item_id"])
        self.item["split"] = "validation"
        self.item["image_path"] = "../outside.png"
        self.write_question()
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            runner.load_question(self.root, "validation", self.item["item_id"])

    def test_cli_passes_student_response_and_saves_linked_result(self):
        output = self.root / "outputs/result.json"
        with patch.object(runner, "ROOT", self.root), patch.object(
                runner, "run_pipeline", return_value={"evidence": "E", "diagnosis": "D", "hint": "H"}
        ) as pipeline, redirect_stdout(StringIO()):
            runner.main(["--item-id", self.item["item_id"], "--answer", "A",
                         "--explanation", "My reasoning", "--hint-level", "2",
                         "--output", str(output)])
        arguments = pipeline.call_args.args
        self.assertEqual(arguments[2:5], ("A", "My reasoning", 2))
        self.assertNotIn("SECRET", str(arguments))
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["item_id"], self.item["item_id"])
        self.assertEqual(result["hint"], "H")


if __name__ == "__main__":
    unittest.main()
