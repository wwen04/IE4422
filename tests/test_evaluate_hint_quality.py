import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from evaluate_hint_quality import (
    CSV_FIELDS,
    generate_review_set,
    summarize_reviews,
)


class HintQualityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.questions = self.root / "data/final/scienceqa/questions"
        self.images = self.root / "data/final/scienceqa/images"
        self.questions.mkdir(parents=True)
        self.images.mkdir(parents=True)
        self.items = []
        for index in range(2):
            item_id = f"validation_{index:06d}"
            image_path = f"data/final/scienceqa/images/{item_id}.png"
            (self.root / image_path).write_bytes(b"image")
            item = {
                "item_id": item_id,
                "split": "validation",
                "question": "Which diagram label is higher?",
                "context": "Use the diagram.",
                "choices": [{"id": "A", "text": "first"}, {"id": "B", "text": "second"}],
                "image_path": image_path,
            }
            self.items.append(item)
        (self.questions / "validation.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in self.items),
            encoding="utf-8",
        )
        manifests = self.root / "manifests/scienceqa"
        manifests.mkdir(parents=True)
        (manifests / "validation_final_ids.json").write_text(
            json.dumps([item["item_id"] for item in self.items]),
            encoding="utf-8",
        )
        references = self.root / "data/final/scienceqa/references"
        references.mkdir(parents=True)
        (references / "validation.jsonl").write_text(
            json.dumps({
                "item_id": self.items[0]["item_id"],
                "gold_option_id": "B",
                "lecture": "SECRET ANSWER MATERIAL",
                "solution": "SECRET SOLUTION",
            }) + "\n",
            encoding="utf-8",
        )

    def test_generates_blinded_conditions_without_loading_references(self):
        tutor = Mock()
        tutor.ask.side_effect = [
            "evidence one", "direct 1", "direct 2", "direct 3",
            "guided 1", "guided 2", "guided 3",
            "evidence two", "direct 1", "direct 2", "direct 3",
            "guided 1", "guided 2", "guided 3",
        ]
        output = self.root / "review.csv"
        result = generate_review_set(
            root=self.root, limit=2, seed=42, tutor=tutor, output_path=output,
            model="test-model",
        )

        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        key = json.loads(result["key_path"].read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows[0]), set(CSV_FIELDS))
        self.assertEqual({row["condition_code"] for row in rows}, {"A", "B"})
        self.assertEqual(len(key["items"]), 2)
        self.assertFalse(key["contains_answer_key"])
        self.assertNotIn("SECRET", output.read_text(encoding="utf-8-sig"))
        self.assertNotIn("SECRET", str(tutor.ask.call_args_list))
        evidence_calls = tutor.ask.call_args_list[::7]
        hint_calls = [
            call for index, call in enumerate(tutor.ask.call_args_list)
            if index % 7 != 0
        ]
        self.assertTrue(all("A. first" in call.args[1] for call in evidence_calls))
        self.assertTrue(all("A. first" not in call.args[1] for call in hint_calls))
        self.assertEqual(tutor.ask.call_count, 14)

    def test_review_summary_validates_ratings_and_reports_method_metrics(self):
        tutor = Mock()
        tutor.ask.side_effect = [
            "evidence", "direct 1", "direct 2", "direct 3",
            "guided 1", "guided 2", "guided 3",
            "evidence", "direct 1", "direct 2", "direct 3",
            "guided 1", "guided 2", "guided 3",
        ]
        output = self.root / "review.csv"
        result = generate_review_set(
            root=self.root, limit=2, seed=3, tutor=tutor, output_path=output
        )
        key = json.loads(result["key_path"].read_text(encoding="utf-8"))
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            code_map = key["conditions"][row["item_id"]]
            is_guided = code_map[row["condition_code"]] == "evidence_guided"
            for level in (1, 2, 3):
                row.update({
                    f"level_{level}_factual_accuracy_1_to_4":
                        "4" if is_guided else "3",
                    f"level_{level}_image_groundedness_1_to_4": "4",
                    f"level_{level}_pedagogical_usefulness_1_to_4": "3",
                    f"level_{level}_answer_leakage_yes_no_unclear": "no",
                    f"level_{level}_unsupported_claim_yes_no": "no",
                })
            row.update({"reviewer": "reviewer", "comments": ""})
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        summary = summarize_reviews(output, result["key_path"])

        self.assertEqual(summary["methods"]["direct_visual"]["n"], 2)
        self.assertEqual(
            summary["methods"]["evidence_guided"]["levels"]["1"][
                "factual_accuracy_1_to_4"
            ],
            4,
        )
        self.assertEqual(
            summary["methods"]["evidence_guided"]["levels"]["1"][
                "answer_leakage"
            ]["yes"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
