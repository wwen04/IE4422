from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import check_scienceqa_answers as checker


class AnswerCheckerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=checker.ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.items = []
        for index in range(2):
            key = f"train_{index:06}"
            image = self.root / f"data/final/scienceqa/images/{key}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"synthetic image; fake model does not decode")
            self.items.append({"item_id": key, "split": "train", "question": "Which arrow points up?",
                               "context": "Read the labels.", "image_path": image.relative_to(self.root).as_posix(),
                               "choices": [{"id": "A", "text": "left"}, {"id": "B", "text": "up"}],
                               "answer": "SECRET ANSWER", "lecture": "SECRET LECTURE",
                               "solution": "SECRET SOLUTION"})
        self.write("manifests/scienceqa/train_final_ids.json", [item["item_id"] for item in self.items])
        self.write_rows("data/final/scienceqa/questions/train.jsonl", self.items)
        self.references = [{"item_id": item["item_id"], "gold_option_id": "B",
                            "lecture": "SECRET REFERENCE LECTURE", "solution": "SECRET REFERENCE SOLUTION"}
                           for item in self.items]
        self.write_rows("data/final/scienceqa/references/train.jsonl", self.references)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_rows(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_training_loader_needs_no_other_splits_and_strips_reference_fields(self):
        items = checker.load_train_questions(self.root, limit=1)
        self.assertEqual(len(items), 1)
        self.assertNotIn("answer", items[0])
        prompt = checker.answer_prompt({**items[0], "solution": "SECRET EXTRA FIELD"})
        self.assertIn("Read the labels.", prompt)
        self.assertLess(prompt.index("A. left"), prompt.index("B. up"))
        self.assertNotIn("SECRET", prompt)
        self.assertTrue(Path(items[0]["image_path"]).is_absolute())
        selected = checker.load_train_questions(self.root, item_ids=["train_000001"])
        self.assertEqual(selected[0]["item_id"], "train_000001")

    def test_checker_rejects_nontraining_excluded_and_duplicate_ids(self):
        for keys in (["validation_000000"], ["test_000000"], ["train_999999"],
                     ["train_000000", "train_000000"]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                checker.load_train_questions(self.root, item_ids=keys)
        with self.assertRaises(ValueError):
            checker.load_train_questions(self.root, limit=0)
        tutor = Mock()
        with self.assertRaises(ValueError):
            checker.predict_answers(tutor, [{**self.items[0], "split": "test"}])
        tutor.ask.assert_not_called()

    def test_invalid_outputs_and_failures_remain_in_accuracy_denominator(self):
        items = [{**self.items[0], "item_id": f"train_{index:06}"} for index in range(6)]
        tutor = Mock()
        tutor.ask.side_effect = [" b\n", "A because it points up", "Z", "", RuntimeError("generation failed"), "A"]
        saved = []
        predictions = checker.predict_answers(tutor, items, saved.append)
        report = checker.score_predictions(predictions, {item["item_id"]: "B" for item in items})
        self.assertEqual(tutor.ask.call_count, 6)
        self.assertEqual(saved, predictions)
        self.assertEqual(predictions[0]["raw_response"], " b\n")
        self.assertEqual(predictions[0]["predicted_option_id"], "B")
        self.assertIn("generation failed", predictions[4]["error"])
        self.assertEqual(report["summary"], {
            "total": 6, "correct": 1, "incorrect": 1, "invalid_outputs": 3,
            "generation_errors": 1, "accuracy": 1 / 6, "random_guess_accuracy": 0.5})
        for call in tutor.ask.call_args_list:
            self.assertNotIn("SECRET", call.args[1])
            self.assertEqual(call.kwargs["max_new_tokens"], checker.MAX_NEW_TOKENS)

    def test_missing_duplicate_and_out_of_range_references_fail_scoring(self):
        items = checker.load_train_questions(self.root)
        for rows in (self.references[:1], self.references + self.references[:1],
                     [{**self.references[0], "gold_option_id": "Z"}, self.references[1]]):
            self.write_rows("data/final/scienceqa/references/train.jsonl", rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                checker.load_train_references(self.root, items)

    def test_cli_saves_results_and_loads_references_after_all_model_calls(self):
        tutor = Mock()
        tutor.ask.side_effect = ["B", "A"]
        engine = ModuleType("qwen_engine")
        engine.MODEL_ID = "fixed-fake-model"
        engine.IMAGE_SIZE = {"shortest_edge": 1, "longest_edge": 2}
        engine.QwenTutor = Mock(return_value=tutor)
        load_references = checker.load_train_references

        def after_generation(root, items):
            self.assertEqual(tutor.ask.call_count, 2)
            return load_references(root, items)

        output = self.root / "run"
        with patch.object(checker, "ROOT", self.root), patch.dict(sys.modules, {"qwen_engine": engine}), \
                patch.object(checker, "load_train_references", side_effect=after_generation), \
                redirect_stdout(StringIO()):
            checker.main(["--output-dir", str(output), "--model-path", "fixed-checkpoint"])
        engine.QwenTutor.assert_called_once_with("fixed-checkpoint")
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(predictions), 2)
        self.assertNotIn("gold_option_id", predictions[0])
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["summary"]["accuracy"], 0.5)
        self.assertEqual(result["results"][0]["gold_option_id"], "B")
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["split"], "train")
        self.assertEqual(run["item_ids"], ["train_000000", "train_000001"])


if __name__ == "__main__":
    unittest.main()
