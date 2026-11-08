from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from experiments import run as runner
from experiments.data import freeze_ai2d_judging, load_questions, load_references
from experiments.prepare_judging import prepare, verify_run
from experiments.summarize import summarize
from experiments.judge_validate import read_judgment
from prepare_ai2d import METHOD, prepare_rows
from tutor_pipeline import extract_evidence, generate_initial_hint


def judgment():
    value = {key: {"reason": "Supported by the attached diagram.", "verdict": "yes"}
             for key in ("tier_compliance", "visual_grounding", "scientific_correctness")}
    value.update(steers_toward_wrong_answer={"reason": "Does not misdirect.", "verdict": "no"},
                 unsupported_visual_claims={"unsupported_claims": [], "count": 0},
                 answer_disclosure={"reason": "Leaves the answer for the learner.", "score": 0},
                 actionability={"reason": "Provides a concrete next move.", "score": 4})
    return value


class BatchExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("prompts/initial_hint.txt", "experiments/judge_rubric.md"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.ROOT / name, target)
        config = {"dataset_id": "fixture/ai2d", "dataset_revision": "fixed", "split": "test",
                  "data_files": ["test.parquet"], "expected_rows": 31,
                  "selection": {"size": 31, "seed": 42, "method": METHOD}}
        (self.root / "configs").mkdir()
        (self.root / "configs/ai2d.json").write_text(json.dumps(config))
        rows = []
        for index in range(31):
            stream = BytesIO()
            Image.new("RGB", (3, 2), (index * 7, index, 30)).save(stream, format="PNG")
            rows.append({"question": f"Which label identifies diagram {index}?", "options": ["c", "D", "b", "a"],
                         "answer": "1", "image": {"path": f"{index}.png", "bytes": stream.getvalue()}})
        prepare_rows(self.root, config, iter(rows), {"test.parquet": "fixture"})
        self.frozen = freeze_ai2d_judging(self.root)
        self.engine = ModuleType("qwen_engine")
        self.engine.MODEL_ID = "fixed-fake-model"
        self.engine.IMAGE_SIZE = {"shortest_edge": 1, "longest_edge": 2}
        self.tutor = Mock()
        self.tutor.ask.return_value = "Look at the diagram label."
        self.engine.QwenTutor = Mock(return_value=self.tutor)

    def run_cli(self, experiment, name=None, extra=()):
        output = self.root / (name or experiment)
        with patch.object(runner, "ROOT", self.root), patch.dict(sys.modules, {"qwen_engine": self.engine}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            runner.main(["--dataset", "ai2d", "--experiment", experiment,
                         "--output-dir", str(output), *extra])
        return output

    def test_preflight_and_b_work_without_reference_exports(self):
        shutil.rmtree(self.root / "data/final/ai2d/references")
        self.run_cli("B", extra=("--preflight",))
        self.engine.QwenTutor.assert_not_called()
        output = self.run_cli("B")
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(run["completed_records"], 93)
        self.assertEqual(run["split"], "test")
        self.assertEqual(self.tutor.ask.call_count, 93)
        rows = [json.loads(line) for line in (output / "hints.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["evidence"] is None for row in rows))
        self.assertTrue(all("gold_option_id" not in str(row) for row in rows))
        self.assertIn("A. c\nB. D\nC. b\nD. a", self.tutor.ask.call_args.args[1])
        self.assertFalse((output / "judge_package").exists())

    def test_a_loads_gold_only_after_generation_and_keeps_invalid_responses(self):
        # Reproduce Colab LF data restored alongside a Windows-generated report.
        reference = self.root / "data/final/ai2d/references/test.jsonl"
        lf_content = reference.read_bytes().replace(b"\r\n", b"\n")
        reference.write_bytes(lf_content)
        audit_path = self.root / "reports/ai2d/integrity_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["artifact_files_sha256"]["data/final/ai2d/references/test.jsonl"] = runner.digest(lf_content.replace(b"\n", b"\r\n"))
        audit_path.write_text(json.dumps(audit))
        self.tutor.ask.side_effect = [RuntimeError("generation failure"), "Not a valid option", *(["B"] * 29)]
        from experiments.data import load_references

        def references(*args):
            self.assertEqual(self.tutor.ask.call_count, 31)
            return load_references(*args)

        with patch.object(runner, "load_references", side_effect=references):
            output = self.run_cli("A")
        report = json.loads((output / "result.json").read_text())
        self.assertEqual(report["summary"]["total"], 31)
        self.assertEqual(report["summary"]["correct"], 29)
        self.assertEqual(report["summary"]["invalid_outputs"], 1)
        self.assertEqual(report["summary"]["generation_errors"], 1)
        self.assertEqual(report["dataset_name"], "ai2d")
        self.assertEqual(self.tutor.ask.call_count, 31)

    def test_b_prompt_matches_ui_without_evidence_and_does_not_load_judge_rubric(self):
        items = load_questions(self.root, "ai2d", limit=1)
        spec = runner.load_prompt_spec(self.root)
        plans = runner.make_plans(self.root, "ai2d", "B", items, spec, "fixed")
        for plan in plans.values():
            tutor = Mock()
            generate_initial_hint(tutor, items[0]["image_path"], runner.question_prompt(items[0]),
                                  "VISIBLE DETAIL", level=plan["tier"])
            expected = tutor.ask.call_args.args[1].replace("Visual evidence (unverified): VISIBLE DETAIL\n", "", 1)
            self.assertEqual(plan["prompt"]["rendered_text"], expected)
            self.assertNotIn("Unedited hint", expected)
            self.assertNotIn("A1. Tier compliance", expected)
        (self.root / "experiments/judge_rubric.md").unlink()
        self.assertEqual(runner.make_plans(self.root, "ai2d", "B", items, runner.load_prompt_spec(self.root), "fixed"), plans)

    def test_a_with_evidence_uses_two_passes_and_scores_only_after_generation(self):
        evidence = '1. Text: "w", upper left.'
        self.tutor.ask.side_effect = [evidence, "b", evidence, "Not an option"]

        def references(*args):
            self.assertEqual(self.tutor.ask.call_count, 4)
            return load_references(*args)

        with patch.object(runner, "load_references", side_effect=references):
            output = self.run_cli("A", extra=("--with-evidence", "--limit", "2"))
        manifest = json.loads((output / "run.json").read_text())
        self.assertEqual(manifest["prompt_version"], runner.EVIDENCE_ANSWER_VERSION)
        self.assertEqual(manifest["generation"]["evidence_max_new_tokens"], 320)
        records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        spec = runner.load_prompt_spec(self.root)
        plans = runner.make_plans(self.root, "ai2d", "A", load_questions(self.root, "ai2d", limit=2),
                                  spec, manifest["run_id"], with_evidence=True)
        for index, row in enumerate(records):
            runner.validate_record(row, plans[row["hint_id"]], spec)
            self.assertEqual(row["evidence"]["text"], evidence)
            self.assertIn(evidence, self.tutor.ask.call_args_list[index * 2 + 1].args[1])
            self.assertEqual(self.tutor.ask.call_args_list[index * 2].args[1], row["evidence_prompt"])
            self.assertNotIn("gold_option_id", str(row))
            self.assertNotIn("lecture", row["effective_prompt"])
        self.assertEqual([call.kwargs["max_new_tokens"] for call in self.tutor.ask.call_args_list], [320, 8] * 2)
        summary = json.loads((output / "result.json").read_text())["summary"]
        self.assertEqual((summary["total"], summary["correct"], summary["invalid_outputs"]), (2, 1, 1))
        self.assertEqual(summary["accuracy"], 0.5)

    def test_a_with_evidence_failures_skip_answering_and_remain_in_denominator(self):
        self.tutor.ask.side_effect = ["  ", RuntimeError("evidence failure"), "observation", RuntimeError("answer failure")]
        output = self.run_cli("A", extra=("--with-evidence", "--limit", "3"))
        self.assertEqual(self.tutor.ask.call_count, 4)
        manifest = json.loads((output / "run.json").read_text())
        spec = runner.load_prompt_spec(self.root)
        plans = runner.make_plans(self.root, "ai2d", "A", load_questions(self.root, "ai2d", limit=3),
                                  spec, manifest["run_id"], with_evidence=True)
        rows = list(runner.read_records(output / "predictions.jsonl", plans, spec).values())
        self.assertEqual([row["evidence"]["status"] for row in rows], ["empty_output", "generation_error", "ok"])
        summary = json.loads((output / "result.json").read_text())["summary"]
        self.assertEqual((summary["total"], summary["generation_errors"], summary["accuracy"]), (3, 3, 0))

    def test_a_with_evidence_resume_retains_mode_and_rejects_tampered_prompt(self):
        self.tutor.ask.side_effect = ["observation", "B", KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli("A", "resume-evidence", extra=("--with-evidence", "--limit", "2"))
        output = self.root / "resume-evidence"
        before = (output / "predictions.jsonl").read_bytes()
        self.tutor.ask.reset_mock(side_effect=True)
        self.tutor.ask.side_effect = ["another observation", "B"]
        self.run_cli("A", "resume-evidence", extra=("--resume",))
        self.assertEqual(self.tutor.ask.call_count, 2)
        self.assertTrue((output / "predictions.jsonl").read_bytes().startswith(before))
        rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        rows[0]["effective_prompt"] += "changed"
        (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.engine.QwenTutor.reset_mock()
        with self.assertRaises(SystemExit):
            self.run_cli("A", "resume-evidence", extra=("--resume",))
        self.engine.QwenTutor.assert_not_called()

    def test_evidence_mode_cannot_be_added_to_baseline_resume_or_hint_conditions(self):
        self.tutor.ask.return_value = "B"
        output = self.run_cli("A", extra=("--limit", "1"))
        before = (output / "predictions.jsonl").read_bytes()
        with self.assertRaises(SystemExit):
            self.run_cli("A", extra=("--resume", "--with-evidence"))
        self.assertEqual((output / "predictions.jsonl").read_bytes(), before)
        for experiment in ("B", "D"):
            with self.assertRaises(SystemExit):
                self.run_cli(experiment, extra=("--with-evidence", "--preflight"))

    def test_a_with_evidence_preflight_is_reference_free_and_reports_two_calls(self):
        shutil.rmtree(self.root / "data/final/ai2d/references")
        with patch.object(runner, "ROOT", self.root), redirect_stdout(StringIO()) as stdout:
            runner.main(["--dataset", "ai2d", "--experiment", "A", "--with-evidence", "--preflight"])
        report = json.loads(stdout.getvalue())
        self.assertEqual((report["planned_outputs"], report["model_calls"]), (31, 62))
        self.assertFalse(report["reference_inputs_loaded"])
        self.engine.QwenTutor.assert_not_called()

    def test_answer_prompt_does_not_include_extra_reference_fields(self):
        item = load_questions(self.root, "ai2d", limit=1)[0]
        prompt = runner.answer_prompt({**item, "gold_option_id": "SECRET GOLD", "lecture": "SECRET LECTURE", "solution": "SECRET SOLUTION"})
        self.assertNotIn("SECRET", prompt)
        self.assertIn("A. c\nB. D\nC. b\nD. a", prompt)

    def test_reference_hash_accepts_lf_crlf_but_rejects_changed_answer(self):
        path = self.root / "data/final/ai2d/references/test.jsonl"
        items = load_questions(self.root, "ai2d")
        original = path.read_bytes().replace(b"\r\n", b"\n")
        for content in (original, original.replace(b"\n", b"\r\n")):
            path.write_bytes(content)
            refs = load_references(self.root, "ai2d", items)
            self.assertEqual(len(refs), 31)
            self.assertTrue(all(ref["gold_option_id"] == "B" for ref in refs.values()))
        path.write_bytes(original.replace(b'"gold_option_id": "B"', b'"gold_option_id": "A"', 1))
        with self.assertRaisesRegex(ValueError, "differs"):
            load_references(self.root, "ai2d", items)

    def test_changed_reference_fails_normal_a_without_writing_result(self):
        self.tutor.ask.return_value = "B"
        reference = self.root / "data/final/ai2d/references/test.jsonl"
        content = reference.read_bytes()
        reference.write_bytes(content.replace(b'"gold_option_id": "B"', b'"gold_option_id": "A"', 1))
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_cli("A", "invalid-reference", extra=("--limit", "2"))
        output = self.root / "invalid-reference"
        self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
        self.assertFalse((output / "result.json").exists())
        self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")

    def test_d_uses_two_calls_per_tier_without_prior_hint_history_or_gold(self):
        evidence = '1. Text: "w", upper left.\n2. Arrow: upper left -> lower right.'
        self.tutor.ask.side_effect = lambda image, prompt, **kwargs: (evidence if kwargs["max_new_tokens"] == 320 else 'Notice label w.')
        with patch.object(runner, "extract_evidence", wraps=extract_evidence) as shared_extractor:
            output = self.run_cli("D", extra=("--limit", "2"))
        self.assertEqual(shared_extractor.call_count, 6)
        self.assertEqual(self.tutor.ask.call_count, 12)
        run, records, _, _, _ = verify_run(self.root, output)
        self.assertEqual(run["generation"]["evidence_reuse"], False)
        self.assertEqual(len(records), 6)
        for index, row in enumerate(records):
            self.assertEqual(row["evidence"]["text"], evidence)
            self.assertIn(f'Visual evidence (unverified): {evidence}', row["effective_prompt"])
            self.assertNotIn("Notice label w.", row["effective_prompt"])
            self.assertEqual(row["generation"]["status"], "ok")
            evidence_call = self.tutor.ask.call_args_list[index * 2]
            self.assertEqual(evidence_call.args[1], row["evidence_prompt"])
            ui_tutor = Mock()
            extract_evidence(ui_tutor, evidence_call.args[0], row["public_inputs"])
            self.assertEqual(evidence_call, ui_tutor.ask.call_args)
        self.assertEqual([call.kwargs["max_new_tokens"] for call in self.tutor.ask.call_args_list], [320, 160] * 6)

    def test_interruption_resume_preserves_saved_errors_and_rejects_changed_checkpoint(self):
        self.tutor.ask.side_effect = [RuntimeError("retained failure"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli("B", "resume", extra=("--limit", "2"))
        output = self.root / "resume"
        before = (output / "hints.jsonl").read_bytes()
        self.tutor.ask.reset_mock(side_effect=True)
        self.tutor.ask.return_value = "Remaining hint"
        self.run_cli("B", "resume", extra=("--resume",))
        self.assertEqual(self.tutor.ask.call_count, 5)
        self.assertTrue((output / "hints.jsonl").read_bytes().startswith(before))
        self.engine.QwenTutor.reset_mock()
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.run_cli("B", "resume", extra=("--resume", "--model-path", "different"))
        self.engine.QwenTutor.assert_not_called()

    def test_same_frozen_30_and_all_tiers_are_packaged_for_b_and_d_without_provenance(self):
        b = self.run_cli("B")
        self.tutor.ask.side_effect = lambda image, prompt, **kwargs: ('1. Text: "w", upper left.' if kwargs["max_new_tokens"] == 320 else 'Notice label w.')
        d = self.run_cli("D")
        expected = {entry["item_id"] for entry in self.frozen["items"]}
        for output in (b, d):
            before = {path.name: path.read_bytes() for path in (output / "hints.jsonl", output / "run.json")}
            report = prepare(self.root, output, output / "judge_package")
            self.assertEqual(report["selected_question_count"], 30)
            self.assertEqual(report["case_count"], 90)
            mapping = [json.loads(line) for line in (output / "judge_package/record_map_PRIVATE.jsonl").read_text().splitlines()]
            self.assertEqual({row["item_id"] for row in mapping}, expected)
            self.assertEqual({(row["item_id"], row["tier"]) for row in mapping}, {(key, t) for key in expected for t in (1, 2, 3)})
            for path in (output / "judge_package/attachments/cases").glob("*/hints.jsonl"):
                case = json.loads(path.read_text())
                self.assertNotIn("experiment", case)
                self.assertNotIn("evidence", case)
                self.assertNotIn("model_path", case)
                self.assertNotIn("item_id", case)
                self.assertEqual(case["gold_reference"]["lecture"], "")
                self.assertEqual(case["gold_reference"]["option_id"], "B")
            for name, content in before.items():
                self.assertEqual((output / name).read_bytes(), content)

    def test_judgments_summary_is_separate_and_reports_missing_invalid_and_failed_outcomes(self):
        output = self.run_cli("B")
        package = output / "judge_package"
        prepare(self.root, output, package)
        mapping = [json.loads(line) for line in (package / "record_map_PRIVATE.jsonl").read_text().splitlines()]
        judgments = output / "judgments"
        judgments.mkdir()
        for entry in mapping[1:]:
            (judgments / entry["expected_response_file"]).write_text(json.dumps(judgment()))
        summary = summarize(self.root, output, package, judgments, output / "summary")
        self.assertEqual(summary["dataset_name"], "ai2d")
        self.assertEqual(summary["planned_hints"], 90)
        self.assertEqual(summary["valid_judgments"], 89)
        self.assertEqual(len(summary["pending_judgments"]), 1)
        self.assertIsNone(summary["good_hint_rate_planned"])
        (judgments / mapping[0]["expected_response_file"]).write_text(json.dumps(judgment()))
        complete = summarize(self.root, output, package, judgments, output / "summary_complete")
        self.assertEqual(complete["good_hint_rate_planned"], 1.0)
        self.assertEqual(complete["overall"]["mean_actionability"], 4)

    def test_modified_public_inputs_and_saved_prompts_are_rejected_before_packaging(self):
        output = self.run_cli("B")
        path = output / "hints.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt"]["rendered_text"] += "CHANGED"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "provenance"):
            prepare(self.root, output, output / "judge_package")
        self.assertFalse((output / "judge_package").exists())
        public = self.root / "data/final/ai2d/questions/test.jsonl"
        original = public.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[0]["choices"].reverse()
        public.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_questions(self.root, "ai2d")

    def test_empty_generation_remains_in_planned_denominator_and_failed_evidence_skips_hint(self):
        self.tutor.ask.return_value = ""
        output = self.run_cli("B")
        report = prepare(self.root, output, output / "judge_package")
        self.assertEqual(report["selected_records"], 90)
        self.assertEqual(report["case_count"], 0)
        result = summarize(self.root, output, output / "judge_package", output / "judgments", output / "summary")
        self.assertEqual(result["planned_hints"], 90)
        self.assertEqual(result["valid_judgments"], 0)
        self.assertEqual(result["good_hint_rate_planned"], 0)
        self.tutor.ask.reset_mock()
        output = self.run_cli("D", extra=("--limit", "2"))
        self.assertEqual(self.tutor.ask.call_count, 6)
        _, records, _, _, _ = verify_run(self.root, output)
        self.assertTrue(all(row["generation"]["status"] == "generation_error" for row in records))
        self.assertTrue(all(row["evidence"]["status"] == "empty_output" for row in records))

    def test_duplicate_judgment_fields_and_bool_scores_are_rejected(self):
        path = self.root / "judgment.json"
        value = judgment()
        value["actionability"]["score"] = True
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            read_judgment(path)
        path.write_text('{"actionability": 1, "actionability": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_judgment(path)


if __name__ == "__main__":
    unittest.main()
