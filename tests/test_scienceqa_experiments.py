from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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
from experiments.data import freeze_scienceqa_judging, load_questions
from experiments.prepare_judging import prepare
from experiments.common import write_json, write_jsonl


class ScienceQATestExperiments(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("configs/scienceqa.json", "prompts/initial_hint.txt", "experiments/judge_rubric.md"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.ROOT / name, target)
        for split, count in (("train", 2), ("test", 31)):
            questions, references = [], []
            for index in range(count):
                key = f"{split}_{index:06}"
                image = self.root / f"data/final/scienceqa/images/{key}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (3, 2), (index, 20, 30)).save(image)
                questions.append({"item_id": key, "split": split, "question": f"{split} question {index}?",
                                  "context": f"{split} context", "choices": [{"id": "A", "text": "left"},
                                                                            {"id": "B", "text": "right"}],
                                  "image_path": image.relative_to(self.root).as_posix()})
                references.append({"item_id": key, "gold_option_id": "A" if split == "test" else "B",
                                   "lecture": f"PRIVATE {split} lecture", "solution": "PRIVATE solution"})
            for kind, rows in (("questions", questions), ("references", references)):
                path = self.root / f"data/final/scienceqa/{kind}/{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(path, rows)
            manifest = self.root / f"manifests/scienceqa/{split}_final_ids.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            write_json(manifest, [row["item_id"] for row in questions])
        self.tutor = Mock()
        self.tutor.ask.side_effect = lambda image, prompt, **kw: (
            "1. Object: left block." if kw["max_new_tokens"] == 320 else
            "A" if kw["max_new_tokens"] == 8 else "Look at the left block.")
        self.engine = ModuleType("qwen_engine")
        self.engine.MODEL_ID = "fake-model"
        self.engine.IMAGE_SIZE = {"shortest_edge": 1, "longest_edge": 2}
        self.engine.QwenTutor = Mock(return_value=self.tutor)

    def run_cli(self, experiment, name=None, extra=()):
        output = self.root / (name or experiment)
        with patch.object(runner, "ROOT", self.root), patch.dict(sys.modules, {"qwen_engine": self.engine}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            runner.main(["--dataset", "scienceqa", "--experiment", experiment,
                         "--output-dir", str(output), *extra])
        return output

    def test_all_four_modes_use_only_test_questions_and_matching_references(self):
        for experiment, evidence in (("A", False), ("A", True), ("B", False), ("D", False)):
            with self.subTest(experiment=experiment, evidence=evidence):
                self.tutor.ask.reset_mock()
                name = "A_with_evidence" if evidence else experiment
                output = self.run_cli(experiment, name, ("--with-evidence",) if evidence else ())
                manifest = json.loads((output / "run.json").read_text())
                self.assertEqual(manifest["split"], "test")
                self.assertEqual(len(manifest["item_ids"]), 31)
                self.assertTrue(all(key.startswith("test_") for key in manifest["item_ids"]))
                self.assertNotIn("training", manifest["purpose"])
                expected_records = 31 if experiment == "A" else 93
                self.assertEqual(manifest["completed_records"], expected_records)
                self.assertEqual(self.tutor.ask.call_count, expected_records * (2 if evidence or experiment == "D" else 1))
                for call in self.tutor.ask.call_args_list:
                    self.assertIn("test question", call.args[1])
                    self.assertNotIn("train question", call.args[1])
                    self.assertNotIn("PRIVATE", call.args[1])
                if experiment == "A":
                    summary = json.loads((output / "result.json").read_text())["summary"]
                    self.assertEqual((summary["total"], summary["correct"]), (31, 31))

    def test_default_packages_use_same_frozen_30_test_questions_and_90_tiers(self):
        frozen = freeze_scienceqa_judging(self.root)
        snapshot = (self.root / "manifests/scienceqa/test_judge_selection.json").read_bytes()
        selected = {row["item_id"] for row in frozen["items"]}
        self.assertEqual(len(selected), 30)
        self.assertEqual(frozen["split"], "test")
        self.assertTrue(all(key.startswith("test_") for key in selected))
        self.assertEqual(freeze_scienceqa_judging(self.root), frozen)
        for experiment in ("B", "D"):
            output = self.run_cli(experiment)
            report = prepare(self.root, output, output / "judge_package", seed=999)
            mapping = [json.loads(line) for line in (output / "judge_package/record_map_PRIVATE.jsonl").read_text().splitlines()]
            self.assertEqual((report["selected_question_count"], report["selected_records"], report["case_count"]), (30, 90, 90))
            self.assertEqual({(row["item_id"], row["tier"]) for row in mapping},
                             {(key, tier) for key in selected for tier in (1, 2, 3)})
            for path in (output / "judge_package/attachments/cases").glob("*/hints.jsonl"):
                case = json.loads(path.read_text())
                self.assertEqual(case["gold_reference"]["lecture"], "PRIVATE test lecture")
        self.assertEqual((self.root / "manifests/scienceqa/test_judge_selection.json").read_bytes(), snapshot)
        value = json.loads(snapshot)
        value["items"][0]["input_sha256"] = "changed"
        write_json(self.root / "manifests/scienceqa/test_judge_selection.json", value)
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare(self.root, output, output / "bad-package")
        self.assertFalse((output / "bad-package").exists())

    def test_train_ids_and_training_resume_are_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            load_questions(self.root, "scienceqa", item_ids=["train_000000"])
        output = self.root / "historical"
        output.mkdir()
        write_json(output / "run.json", {"schema_version": runner.SCHEMA_VERSION,
                   "dataset_name": "scienceqa", "experiment": "A", "split": "train"})
        with self.assertRaises(SystemExit):
            self.run_cli("A", "historical", ("--resume",))
        self.engine.QwenTutor.assert_not_called()

    def test_test_preflight_does_not_need_training_data_or_reference_exports(self):
        for kind in ("questions", "references"):
            (self.root / f"data/final/scienceqa/{kind}/train.jsonl").unlink()
        (self.root / "data/final/scienceqa/references/test.jsonl").unlink()
        with patch.object(runner, "ROOT", self.root), redirect_stdout(StringIO()) as stdout:
            runner.main(["--dataset", "scienceqa", "--experiment", "D", "--preflight"])
        self.assertEqual(json.loads(stdout.getvalue())["questions"], 31)
        self.engine.QwenTutor.assert_not_called()
        self.assertEqual(len(freeze_scienceqa_judging(self.root)["items"]), 30)


if __name__ == "__main__":
    unittest.main()
