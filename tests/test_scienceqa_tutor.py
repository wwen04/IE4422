import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluate_visual_evidence import answer_prompt
import scienceqa_tutor as runner
from tutor_pipeline import (
    assess_explanation,
    diagnose,
    extract_evidence,
    generate_hint,
    run_answered_pipeline,
    run_initial_hint,
)


class TutorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data/final/scienceqa/images").mkdir(parents=True)
        (self.root / "data/final/scienceqa/images/example.png").write_bytes(b"mock image")
        self.questions = self.root / "data/final/scienceqa/questions"
        self.questions.mkdir(parents=True)
        self.references = self.root / "data/final/scienceqa/references"
        self.references.mkdir(parents=True)
        self.item = {
            "item_id": "validation_000003", "split": "validation",
            "question": "Which arrow points up?", "context": "Use the arrow labels.",
            "choices": [{"id": "A", "text": "left"}, {"id": "B", "text": "up"}],
            "image_path": "data/final/scienceqa/images/example.png",
            "answer": "SECRET GOLD", "lecture": "SECRET LECTURE", "solution": "SECRET SOLUTION",
        }
        self.write_question()
        self.write_reference()
        manifest = self.root / "manifests/scienceqa/validation_final_ids.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps([self.item["item_id"]]), encoding="utf-8")

    def write_question(self):
        (self.questions / "validation.jsonl").write_text(json.dumps(self.item) + "\n", encoding="utf-8")

    def write_reference(self):
        reference = {
            "item_id": self.item["item_id"],
            "gold_option_id": "B",
            "lecture": "Reference lecture",
            "solution": "Reference solution",
        }
        (self.references / "validation.jsonl").write_text(
            json.dumps(reference) + "\n", encoding="utf-8"
        )

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
        evidence_prompt = calls[0]
        self.assertIn("List the visible observations", evidence_prompt)
        self.assertIn("One observation per line", evidence_prompt)
        self.assertIn("Do not infer, explain, interpret, predict", evidence_prompt)
        diagnosis = diagnose(tutor, item["image_path"], prompt, "A", "My reasoning", evidence)
        generate_hint(tutor, item["image_path"], prompt, evidence, diagnosis)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("SECRET", "\n".join(calls))

    def test_evidence_guided_answer_checks_visual_observations_before_solving(self):
        item = {
            "question": "Which solution is more concentrated?",
            "context": "Count the particles and compare the volumes.",
            "choices": [{"id": "A", "text": "Solution A"},
                        {"id": "B", "text": "Solution B"}],
        }

        baseline = answer_prompt(item)
        guided = answer_prompt(item, '{"observations": ["A: 5 particles"]}')

        self.assertNotIn("Proposed visual observations", baseline)
        self.assertIn("Inspect the image yourself", guided)
        self.assertIn("correct or discard", guided)
        self.assertIn("before matching the result to an option", guided)
        self.assertTrue(guided.endswith("Output exactly one option ID and no explanation."))

    def test_missing_data_item_and_image_fail_before_inference(self):
        with self.assertRaisesRegex(FileNotFoundError, "Run prepare_data.py"):
            runner.load_question(self.root, "train", "train_000003")
        with self.assertRaisesRegex(ValueError, "not found"):
            runner.load_question(self.root, "validation", "validation_999999")
        (self.root / self.item["image_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing final image"):
            runner.load_question(self.root, "validation", self.item["item_id"])

    def test_reference_loader_is_separate_from_public_question(self):
        item = runner.load_question(self.root, "validation", self.item["item_id"])
        reference = runner.load_reference(self.root, "validation", self.item["item_id"])
        self.assertNotIn("gold_option_id", item)
        self.assertEqual(reference["gold_option_id"], "B")
        self.assertNotIn("SECRET", str(reference))

    def test_default_rejects_excluded_processed_rows_and_missing_final_exports(self):
        processed = self.root / "data/processed/scienceqa/questions"
        processed.mkdir(parents=True)
        excluded = {**self.item, "item_id": "validation_000999"}
        (processed / "validation.jsonl").write_text(json.dumps(excluded) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not found"):
            runner.load_question(self.root, "validation", excluded["item_id"])
        with self.assertRaisesRegex(ValueError, "audited"):
            runner.load_reference(self.root, "validation", excluded["item_id"])
        (self.questions / "validation.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "finalize_data.py"):
            runner.load_question(self.root, "validation", excluded["item_id"])

    def test_final_manifest_and_final_image_directory_are_enforced(self):
        manifest = self.root / "manifests/scienceqa/validation_final_ids.json"
        manifest.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audited.*manifest"):
            runner.load_question(self.root, "validation", self.item["item_id"])
        manifest.write_text(json.dumps([self.item["item_id"]]), encoding="utf-8")
        image = self.root / "data/processed/scienceqa/images/example.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"old candidate")
        self.item["image_path"] = image.relative_to(self.root).as_posix()
        self.write_question()
        with self.assertRaisesRegex(ValueError, "audited images"):
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

    def test_multiple_questions_reuse_the_supplied_tutor_without_loading_a_model(self):
        item = runner.load_question(self.root, "validation", self.item["item_id"])
        tutor = Mock()
        tutor.ask.side_effect = ["E1", "H1", "E2", "H2"]
        constructor = Mock(side_effect=AssertionError("The pipeline must not load a model"))
        with patch.dict("sys.modules", {
            "qwen_engine": SimpleNamespace(QwenTutor=constructor)
        }):
            first = runner.run_hint(tutor, item, 1)
            second = runner.run_hint(tutor, item, 2)
        constructor.assert_not_called()
        self.assertEqual(tutor.ask.call_count, 4)
        self.assertEqual((first["hint"], second["hint"]), ("H1", "H2"))
        self.assertEqual(first["item_id"], second["item_id"])
        self.assertNotIn("SECRET", str(tutor.ask.call_args_list))

    def test_invalid_hint_level_fails_before_generation(self):
        tutor = Mock()
        with self.assertRaisesRegex(ValueError, "hint_level"):
            run_initial_hint(tutor, "image.png", "Question", 4)
        with self.assertRaisesRegex(ValueError, "hint_level"):
            run_answered_pipeline(tutor, "image.png", "Question", "A", "My reasoning",
                                  "A. left", "B", "Reference", 4)
        tutor.ask.assert_not_called()

    def test_checked_question_links_result_and_limits_reference_to_assessment(self):
        item = runner.load_question(self.root, "validation", self.item["item_id"])
        reference = runner.load_reference(self.root, "validation", self.item["item_id"])
        tutor = Mock()
        tutor.ask.side_effect = ["Evidence", "NEEDS_WORK", "Diagnosis", "Hint"]
        stages = []
        result = runner.run_checked_question(
            tutor, item, reference, "A", "My reasoning",
            on_stage=stages.append,
        )
        self.assertEqual(stages, ["evidence", "assessment", "diagnosis", "hint"])
        self.assertEqual(result["item_id"], self.item["item_id"])
        self.assertEqual(result["hint"], "Hint")
        for index, call in enumerate(tutor.ask.call_args_list):
            self.assertIn("A. left\nB. up", call.args[1])
            self.assertEqual("Reference lecture" in call.args[1], index == 1)
            self.assertEqual("Reference solution" in call.args[1], index == 1)

    def test_initial_hint_needs_no_student_answer_or_explanation(self):
        tutor = Mock()
        evidence = '1. Count: left panel, 3 circles.\n2. Text: "A", above left panel.'
        tutor.ask.side_effect = [evidence, "Start by noticing the labels."]
        stages = []

        result = run_initial_hint(
            tutor, "image.png", "Question", hint_level=1, on_stage=stages.append
        )

        self.assertEqual(stages, ["evidence", "hint"])
        self.assertEqual(result["outcome"], "hint")
        self.assertIsNone(result["diagnosis"])
        self.assertEqual(result["evidence"], evidence)
        self.assertEqual(result["hint"], "Start by noticing the labels.")
        self.assertIn(f"Visual evidence (unverified): {evidence}\n", tutor.ask.call_args_list[1].args[1])
        self.assertEqual([call.kwargs["max_new_tokens"] for call in tutor.ask.call_args_list], [320, 160])
        self.assertNotIn("Student answer", tutor.ask.call_args_list[1].args[1])

    def test_correct_answer_and_supported_explanation_get_feedback(self):
        tutor = Mock()
        tutor.ask.side_effect = ["Evidence", "SUPPORTED"]
        stages = []

        result = run_answered_pipeline(
            tutor, "image.png", "Question", "B", "My reasoning",
            "B. right", "B", "Reference explanation", on_stage=stages.append,
        )

        self.assertEqual(stages, ["evidence", "assessment", "feedback"])
        self.assertEqual(result["outcome"], "correct")
        self.assertTrue(result["answer_correct"])
        self.assertEqual(result["explanation_status"], "correct")
        self.assertIsNone(result["hint"])
        self.assertNotIn("Reference explanation", result["feedback"])
        self.assertEqual(tutor.ask.call_count, 2)

    def test_incorrect_answer_gets_a_follow_up_hint(self):
        tutor = Mock()
        tutor.ask.side_effect = ["Evidence", "NEEDS_WORK", "Diagnosis", "Hint"]
        stages = []

        result = run_answered_pipeline(
            tutor, "image.png", "Question", "A", "My reasoning",
            "A. left", "B", "Reference explanation", on_stage=stages.append,
        )

        self.assertEqual(stages, ["evidence", "assessment", "diagnosis", "hint"])
        self.assertEqual(result["outcome"], "try_again")
        self.assertFalse(result["answer_correct"])
        self.assertEqual(result["hint"], "Hint")
        self.assertEqual(tutor.ask.call_count, 4)

    def test_wrong_choice_can_still_receive_credit_for_supported_reasoning(self):
        tutor = Mock()
        tutor.ask.side_effect = [
            "Evidence", "SUPPORTED", "Diagnosis", "Hint",
        ]
        result = run_answered_pipeline(
            tutor, "image.png", "Question", "A", "Mostly sound idea",
            "A. left", "B", "Reference explanation",
        )
        self.assertFalse(result["answer_correct"])
        self.assertEqual(result["explanation_status"], "correct")
        self.assertIn("useful idea", result["feedback"])
        self.assertEqual(result["hint"], "Hint")

    def test_correct_answer_with_weak_explanation_gets_another_hint(self):
        tutor = Mock()
        tutor.ask.side_effect = ["Evidence", "NEEDS_WORK", "Diagnosis", "Hint"]

        result = run_answered_pipeline(
            tutor, "image.png", "Question", "B", "Unclear reasoning",
            "B. right", "B", "Reference explanation",
        )

        self.assertEqual(result["outcome"], "try_again")
        self.assertTrue(result["answer_correct"])
        self.assertEqual(result["explanation_status"], "needs_work")
        self.assertEqual(result["hint"], "Hint")

    def test_unclear_explanation_is_not_marked_correct(self):
        tutor = Mock()
        tutor.ask.return_value = "I cannot determine whether this is supported."
        status, _ = assess_explanation(
            tutor, "image.png", "Question", "B. right", "Maybe", "Reference"
        )
        self.assertEqual(status, "uncertain")


if __name__ == "__main__":
    unittest.main()
