from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image

from experiments import legacy as generator
from experiments.data import load_questions
from experiments.prepare_judging import judge_prompt, prepare, prepare_package, verify_run, write_json, write_jsonl
from experiments import run as batch


class JudgePreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=generator.ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("prompts/initial_hint.txt", "configs/scienceqa.json", "notebooks/colab_config.json",
                     "experiments/judge_rubric.md"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generator.ROOT / name, target)
        questions, references = [], []
        for i in range(2):
            key = f"train_{i:06}"
            image = self.root / f"data/final/scienceqa/images/{key}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (i * 100, 20, 30)).save(image)
            questions.append({"item_id": key, "split": "train", "question": "Which block is warmer?",
                              "context": "Use the labels.", "choices": [{"id": "A", "text": "left"},
                                                                           {"id": "B", "text": "right"}],
                              "image_path": image.relative_to(self.root).as_posix()})
            references.append({"item_id": key, "gold_option_id": "A", "lecture": "Reference science.", "solution": ""})
        for relative, rows in (("data/final/scienceqa/questions/train.jsonl", questions),
                               ("data/final/scienceqa/references/train.jsonl", references)):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(target, rows)
        manifest = self.root / "manifests/scienceqa/train_final_ids.json"
        manifest.parent.mkdir(parents=True)
        write_json(manifest, [row["item_id"] for row in questions])
        items = load_questions(self.root, "scienceqa")
        plans = generator.planned_records(self.root, items, generator.load_prompt_spec(self.root), "private-run")
        # Existing B files are read-only legacy input; no legacy generator is needed.
        self.rows = [{**plan, "hint_text": f"Unedited hint {i}\n", "word_count": 3,
                      "generation": {"status": "ok", "seconds": 0, "error": None,
                                     "input_tokens": None, "output_tokens": None, "finish_reason": None}}
                     for i, plan in enumerate(plans.values())]
        dataset = json.loads((self.root / "configs/scienceqa.json").read_text())
        pinned = json.loads((self.root / "notebooks/colab_config.json").read_text())
        self.run = {"experiment": "B", "run_id": "private-run", "split": "train", "status": "complete",
                    "tiers": [1, 2, 3], "item_ids": [item["item_id"] for item in items],
                    "prompt_version": generator.PROMPT_VERSION, "schema_version": generator.SCHEMA_VERSION,
                    "generation": {"do_sample": False, "max_new_tokens": generator.MAX_NEW_TOKENS},
                    "dataset": dataset, "plan_sha256": generator.json_digest({"plans": list(plans.values()),
                                                                            "dataset": dataset,
                                                                            "max_new_tokens": generator.MAX_NEW_TOKENS}),
                    "expected_records": 6, "completed_records": 6, "remaining_records": 0,
                    "counts": {"ok": 6, "empty_output": 0, "generation_error": 0},
                    "model_identity": {"checkpoint_marker": {"model": pinned}},
                    "private_marker": "SECRET GENERATOR IDENTITY"}
        self.run_folder = self.root / "run"
        self.run_folder.mkdir()
        write_json(self.run_folder / "run.json", self.run)
        write_jsonl(self.run_folder / "hints.jsonl", self.rows)

    def test_package_keeps_originals_and_each_case_has_one_unchanged_hint_and_image(self):
        before = {name: (self.run_folder / name).read_bytes() for name in ("run.json", "hints.jsonl")}
        output = self.root / "prepared"
        report = prepare_package(self.root, self.run_folder, output)
        self.assertEqual((report["valid_records"], report["question_count"]), (6, 2))
        self.assertFalse(report["quality_judged"])
        self.assertEqual({path.name for path in output.iterdir()},
                         {"README.md", "record_map_PRIVATE.jsonl", "attachments"})
        self.assertEqual({path.name for path in (output / "attachments").iterdir()},
                         {"judge_prompt.md", "cases"})
        originals = {row["hint_id"]: row for row in self.rows}
        mapping = [json.loads(line) for line in (output / "record_map_PRIVATE.jsonl").read_text().splitlines()]
        self.assertEqual(len(mapping), 6)
        for entry in mapping:
            folder = output / "attachments/cases" / entry["record_id"]
            case_lines = (folder / "hints.jsonl").read_text().splitlines()
            self.assertEqual(len(case_lines), 1)
            case = json.loads(case_lines[0])
            original = originals[entry["hint_id"]]
            self.assertEqual(case["hint_text"], original["hint_text"])
            self.assertEqual(case["choices"], original["public_inputs"]["choices"])
            self.assertEqual(case["gold_reference"]["option_id"], "A")
            self.assertEqual(generator.digest((folder / case["image_file"]).read_bytes()), case["image_sha256"])
            self.assertFalse({"run_id", "hint_id", "item_id", "prompt", "generation", "word_count"} & set(case))
        for path in (output / "attachments").rglob("*"):
            if path.suffix in (".jsonl", ".md"):
                self.assertNotIn(b"SECRET GENERATOR IDENTITY", path.read_bytes())
        for name, content in before.items():
            self.assertEqual((self.run_folder / name).read_bytes(), content)

    def test_default_scienceqa_packages_reuse_saved_b_question_ids_for_d(self):
        # A small fixture exercises the same manifest mechanism as the real 100.
        public = next(row["public_inputs"] for row in self.rows if row["item_id"] == "train_000001")
        frozen = {"dataset_name": "scienceqa", "size": 1, "seed": 42, "tiers": [1, 2, 3],
                  "source_configuration": self.run["dataset"], "method": "Reuse exact existing B question IDs",
                  "items": [{"item_id": "train_000001", "input_sha256": public["input_sha256"],
                             "image_sha256": public["image_sha256"]}]}
        manifest = self.root / "manifests/scienceqa/train_judge_selection.json"
        write_json(manifest, frozen)
        items = load_questions(self.root, "scienceqa")
        spec = batch.load_prompt_spec(self.root)
        plans = batch.make_plans(self.root, "scienceqa", "D", items, spec, "new-d-run")
        tutor = SimpleNamespace(ask=lambda image, prompt, **kwargs: '{"labels": ["left"]}' if "Return a JSON object" in prompt else "Notice the left block.")
        records = [batch.generate_record(tutor, self.root, plan, spec) for plan in plans.values()]
        d_run = {"schema_version": batch.SCHEMA_VERSION, "dataset_name": "scienceqa", "experiment": "D",
                 "run_id": "new-d-run", "split": "train", "data_source": "final", "status": "complete",
                 "tiers": [1, 2, 3], "dataset": self.run["dataset"], "item_ids": [item["item_id"] for item in items],
                 "prompt_spec": spec, "prompt_version": batch.VERSIONS["D"], "generation": batch.settings("D"),
                 "expected_records": 6, "completed_records": 6, "remaining_records": 0,
                 "counts": {"ok": 6, "empty_output": 0, "invalid_output": 0, "generation_error": 0},
                 "plan_sha256": batch.json_digest({"plans": list(plans.values()), "dataset": self.run["dataset"],
                                                  "prompt_spec": spec, "generation": batch.settings("D")})}
        d_folder = self.root / "d-run"
        d_folder.mkdir()
        write_json(d_folder / "run.json", d_run)
        write_jsonl(d_folder / "hints.jsonl", records)
        for folder in (self.run_folder, d_folder):
            destination = folder / "fixed-package"
            report = prepare(self.root, folder, destination, seed=999)
            mapping = [json.loads(line) for line in (destination / "record_map_PRIVATE.jsonl").read_text().splitlines()]
            self.assertEqual({row["item_id"] for row in mapping}, {"train_000001"})
            self.assertEqual({row["tier"] for row in mapping}, {1, 2, 3})
            self.assertEqual(report["selected_records"], 3)
        frozen["items"][0]["input_sha256"] = "changed"
        write_json(manifest, frozen)
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare(self.root, d_folder, self.root / "bad-fixed-package")
        self.assertFalse((self.root / "bad-fixed-package").exists())

    def test_modified_hint_prompt_fingerprint_is_rejected_without_creating_package(self):
        changed = deepcopy(self.rows)
        changed[-1]["prompt"]["sha256"] = "0" * 64
        write_jsonl(self.run_folder / "hints.jsonl", changed)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            prepare_package(self.root, self.run_folder, self.root / "bad-package")
        self.assertFalse((self.root / "bad-package").exists())

    def test_question_sample_has_all_tiers_preserves_case_ids_and_repeats_exactly(self):
        full = self.root / "full"
        prepare_package(self.root, self.run_folder, full)
        full_map = {row["hint_id"]: row for row in
                    map(json.loads, (full / "record_map_PRIVATE.jsonl").read_text().splitlines())}
        sampled = self.root / "sampled"
        report = prepare_package(self.root, self.run_folder, sampled, sample_questions=1)
        mapping = list(map(json.loads, (sampled / "record_map_PRIVATE.jsonl").read_text().splitlines()))
        self.assertEqual((report["selected_question_count"], report["selected_records"], report["case_count"]), (1, 3, 3))
        self.assertEqual(len({row["item_id"] for row in mapping}), 1)
        self.assertEqual({row["tier"] for row in mapping}, {1, 2, 3})
        for row in mapping:
            self.assertEqual(row["record_id"], full_map[row["hint_id"]]["record_id"])
            relative = Path("attachments/cases") / row["record_id"] / "hints.jsonl"
            self.assertEqual((sampled / relative).read_bytes(), (full / relative).read_bytes())
        repeated = self.root / "repeated"
        prepare_package(self.root, self.run_folder, repeated, sample_questions=1)
        self.assertEqual((sampled / "record_map_PRIVATE.jsonl").read_bytes(),
                         (repeated / "record_map_PRIVATE.jsonl").read_bytes())

    def test_calibration_exclusions_and_saved_selection_are_reused_without_redrawing(self):
        sampled = self.root / "sampled"
        prepare_package(self.root, self.run_folder, sampled, sample_questions=1,
                        exclude_item_ids=["train_000000"])
        sample_map = sampled / "record_map_PRIVATE.jsonl"
        reused = self.root / "reused"
        report = prepare_package(self.root, self.run_folder, reused, seed=999, sample_from=sample_map)
        mapping = list(map(json.loads, (reused / "record_map_PRIVATE.jsonl").read_text().splitlines()))
        self.assertEqual(report["selection"]["selected_item_ids"], ["train_000001"])
        self.assertEqual({(row["item_id"], row["tier"]) for row in mapping},
                         {("train_000001", tier) for tier in (1, 2, 3)})
        changed = list(map(json.loads, sample_map.read_text().splitlines()))
        changed[0]["input_sha256"] = "0" * 64
        write_jsonl(sample_map, changed)
        with self.assertRaisesRegex(ValueError, "different public inputs"):
            prepare_package(self.root, self.run_folder, self.root / "bad-reuse", sample_from=sample_map)
        self.assertFalse((self.root / "bad-reuse").exists())

    def test_failed_generation_stays_in_selected_question_and_planned_denominator(self):
        changed = deepcopy(self.rows)
        changed[0].update(hint_text=None, word_count=None)
        changed[0]["generation"].update(status="generation_error", error={"type": "Example", "message": "Failed"})
        write_jsonl(self.run_folder / "hints.jsonl", changed)
        run = deepcopy(self.run)
        run["counts"] = {"ok": 5, "empty_output": 0, "generation_error": 1}
        write_json(self.run_folder / "run.json", run)
        output = self.root / "sampled"
        report = prepare_package(self.root, self.run_folder, output, sample_questions=1,
                                 exclude_item_ids=["train_000001"])
        mapping = list(map(json.loads, (output / "record_map_PRIVATE.jsonl").read_text().splitlines()))
        self.assertEqual((report["selected_records"], report["case_count"]), (3, 2))
        failed = [row for row in mapping if row["generation_status"] == "generation_error"]
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0]["expected_response_file"])
        self.assertFalse((output / "attachments/cases" / failed[0]["record_id"]).exists())
        self.assertEqual(len(list((output / "attachments/cases").iterdir())), 2)

    def test_invalid_sample_size_or_exclusions_do_not_create_package(self):
        for count in (0, -1, 3):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "Sample size"):
                prepare_package(self.root, self.run_folder, self.root / "invalid", sample_questions=count)
        with self.assertRaisesRegex(ValueError, "excluded question"):
            prepare_package(self.root, self.run_folder, self.root / "invalid",
                            sample_questions=1, exclude_item_ids=["train_999999"])
        self.assertFalse((self.root / "invalid").exists())

    def test_duplicate_rows_and_missing_references_fail_verification(self):
        duplicate = self.rows[:-1] + self.rows[:1]
        write_jsonl(self.run_folder / "hints.jsonl", duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_run(self.root, self.run_folder)
        write_jsonl(self.run_folder / "hints.jsonl", self.rows)
        references = self.root / "data/final/scienceqa/references/train.jsonl"
        references.write_text(references.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Missing answer-key"):
            verify_run(self.root, self.run_folder)

    def test_prompt_preserves_scoring_content_and_excludes_author_notes(self):
        rubric = (self.root / "experiments/judge_rubric.md").read_text(encoding="utf-8")
        prompt = judge_prompt(rubric)
        table_start = rubric.index("| Tier | Definition |")
        table = rubric[table_start:rubric.index("\n\n---", table_start)].rstrip()
        criteria_start = rubric.index("### Criteria", rubric.index("## Part A:"))
        criteria = rubric[criteria_start:rubric.index("### Derived metrics", criteria_start)].rstrip()
        self.assertIn(table, prompt)
        self.assertIn(criteria, prompt)
        self.assertEqual(prompt.count("You are evaluating a tutoring hint"), 1)
        for author_note in ("Inputs given to the judge", "Instructions to the judge", "Derived metrics",
                            "computed in code", "Experiment A", "used by the generator AND the judge",
                            "## Part B:", "## Part C:", "## Part D:"):
            self.assertNotIn(author_note, prompt)
        self.assertIn("exactly one hint", prompt)
        self.assertIn("seven-field Part A JSON object", prompt)
        self.assertIn("unsupported-claim count must equal the list length", prompt)


if __name__ == "__main__":
    unittest.main()
