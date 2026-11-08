import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluate_visual_evidence import (
    evaluate,
    parse_choice_id,
    summarize,
    write_csv_report,
    write_html_report,
)


class VisualEvidenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        questions = self.root / "data/final/scienceqa/questions"
        references = self.root / "data/final/scienceqa/references"
        images = self.root / "data/final/scienceqa/images"
        questions.mkdir(parents=True)
        references.mkdir(parents=True)
        images.mkdir(parents=True)
        (images / "validation_000001.png").write_bytes(b"test")
        self.question = {
            "item_id": "validation_000001",
            "split": "validation",
            "question": "Which arrow points up?",
            "context": "",
            "choices": [{"id": "A", "text": "left"}, {"id": "B", "text": "up"}],
            "image_path": "data/final/scienceqa/images/validation_000001.png",
        }
        (questions / "validation.jsonl").write_text(
            json.dumps(self.question) + "\n", encoding="utf-8"
        )
        manifests = self.root / "manifests/scienceqa"
        manifests.mkdir(parents=True)
        (manifests / "validation_final_ids.json").write_text(
            json.dumps([self.question["item_id"]]), encoding="utf-8"
        )
        (references / "validation.jsonl").write_text(
            json.dumps({
                "item_id": self.question["item_id"],
                "gold_option_id": "B",
                "lecture": "Secret reference",
                "solution": "Secret solution",
            }) + "\n",
            encoding="utf-8",
        )

    def test_choice_parser_accepts_common_answer_formats_and_rejects_invalid(self):
        choices = self.question["choices"]
        self.assertEqual(parse_choice_id("B", choices), "B")
        self.assertEqual(parse_choice_id("Answer: **b**.", choices), "B")
        self.assertEqual(parse_choice_id("The answer is: B", choices), "B")
        self.assertEqual(parse_choice_id("Option A - because...", choices), "A")
        self.assertIsNone(parse_choice_id("The answer is probably B", choices))
        self.assertIsNone(parse_choice_id("F", choices))
        self.assertIsNone(parse_choice_id("", choices))

    def test_summary_reports_paired_accuracy_and_percentage_point_change(self):
        report = summarize([
            {"baseline_correct": True, "evidence_guided_correct": True},
            {"baseline_correct": True, "evidence_guided_correct": False},
            {"baseline_correct": False, "evidence_guided_correct": True},
            {"baseline_correct": False, "evidence_guided_correct": False},
        ])
        self.assertEqual(report["baseline"]["accuracy_percent"], 50)
        self.assertEqual(report["evidence_guided"]["accuracy_percent"], 50)
        self.assertEqual(report["delta_percentage_points"], 0)
        self.assertEqual(report["paired"], {
            "both_correct": 1,
            "baseline_only_correct": 1,
            "evidence_guided_only_correct": 1,
            "both_incorrect": 1,
        })

    def test_evaluate_runs_paired_prompts_without_reference_leakage(self):
        tutor = Mock()
        tutor.ask.side_effect = ["A", '{"labels": ["up arrow"]}', "B"]

        report = evaluate(root=self.root, tutor=tutor)

        self.assertEqual(report["n"], 1)
        self.assertEqual(report["available_n"], 1)
        self.assertIsNone(report["limit_per_split"])
        self.assertEqual(report["baseline"]["accuracy_percent"], 0)
        self.assertEqual(report["evidence_guided"]["accuracy_percent"], 100)
        self.assertEqual(report["delta_percentage_points"], 100)
        self.assertEqual(report["items"][0]["baseline_prediction"], "A")
        self.assertEqual(report["items"][0]["evidence_guided_prediction"], "B")
        self.assertNotIn("Secret", str(tutor.ask.call_args_list))
        evidence_instruction = tutor.ask.call_args_list[1].args[1]
        self.assertIn("minimum sufficient visual observations", evidence_instruction)
        self.assertIn("Which arrow points up?", evidence_instruction)
        self.assertIn("B. up", evidence_instruction)
        self.assertEqual(tutor.ask.call_count, 3)

    def test_evaluate_rejects_missing_tutor_and_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "loaded tutor"):
            evaluate(root=self.root)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            evaluate(root=self.root, tutor=Mock(), limit=0)

    def test_all_splits_are_evaluated_and_reported_separately(self):
        for split in ("train", "test"):
            item_id = f"{split}_000001"
            item = {
                **self.question,
                "item_id": item_id,
                "split": split,
                "image_path": f"data/final/scienceqa/images/{item_id}.png",
            }
            (self.root / item["image_path"]).write_bytes(b"test")
            (self.root / "data/final/scienceqa/questions" / f"{split}.jsonl").write_text(
                json.dumps(item) + "\n", encoding="utf-8"
            )
            (self.root / "data/final/scienceqa/references" / f"{split}.jsonl").write_text(
                json.dumps({
                    "item_id": item_id,
                    "gold_option_id": "B",
                    "lecture": "",
                    "solution": "",
                }) + "\n",
                encoding="utf-8",
            )
            (self.root / f"manifests/scienceqa/{split}_final_ids.json").write_text(
                json.dumps([item_id]), encoding="utf-8"
            )
        tutor = Mock()
        tutor.ask.side_effect = [
            "B", "train evidence", "B",
            "B", "validation evidence", "A",
            "A", "test evidence", "A",
        ]

        report = evaluate(root=self.root, split="all", tutor=tutor)

        self.assertEqual(report["n"], 3)
        self.assertEqual(set(report["by_split"]), {"train", "validation", "test"})
        self.assertEqual(report["by_split"]["train"]["n"], 1)
        self.assertEqual(report["by_split"]["validation"]["n"], 1)
        self.assertEqual(report["by_split"]["test"]["n"], 1)
        self.assertEqual([item["split"] for item in report["items"]],
                         ["train", "validation", "test"])

    def test_checkpoint_resumes_completed_questions_and_exports_reports(self):
        checkpoint = self.root / "outputs/accuracy_comparison.checkpoint.json"
        checkpoint.parent.mkdir()
        tutor = Mock()
        tutor.ask.side_effect = ["B", '{"entities": ["arrow"]}', "B"]
        first = evaluate(
            root=self.root,
            tutor=tutor,
            resume_path=checkpoint,
        )
        self.assertTrue(checkpoint.is_file())
        with checkpoint.open("ab") as stream:
            stream.write(b'{"split":"validation"')
        resumed_tutor = Mock(side_effect=AssertionError("Completed item should resume"))
        second = evaluate(
            root=self.root,
            tutor=resumed_tutor,
            resume_path=checkpoint,
        )
        self.assertEqual(first["items"], second["items"])
        resumed_tutor.assert_not_called()
        self.assertTrue(checkpoint.read_bytes().endswith(b"\n"))
        csv_path = self.root / "comparison.csv"
        html_path = self.root / "comparison.html"
        write_csv_report(csv_path, second)
        write_html_report(html_path, second)
        self.assertIn("baseline_correct", csv_path.read_text(encoding="utf-8-sig"))
        self.assertIn("Default model vs visual-evidence prompting",
                      html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
