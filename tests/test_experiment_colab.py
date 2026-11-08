import json
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from notebooks import experiments_colab as support
from notebooks.build_experiment_bundle import ROOT, build_bundle
from notebooks.build_experiment_notebook import build


class ExperimentColabTests(unittest.TestCase):
    def test_source_upload_uses_bytes_returned_with_colab_numbered_filenames(self):
        notebook = json.loads((ROOT / "notebooks/IE4422_Experiments.ipynb").read_text(encoding="utf-8"))
        bootstrap = next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code" and "uploaded = files.upload()" in cell["source"])
        fragment = "uploaded = files.upload()" + bootstrap.split("uploaded = files.upload()", 1)[1].split("SOURCE_ROOT =", 1)[0]
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("IE4422/bundle_info.json", json.dumps({"kind": "batch_experiments"}))
        payload = buffer.getvalue()
        for name in ("IE4422_Experiments_Source.zip", "IE4422_Experiments_Source (1).zip", "IE4422_Experiments_Source (2).zip"):
            files = Mock()
            files.upload.return_value = {name: payload}
            scope = {"files": files}
            exec(fragment, scope)
            self.assertEqual(scope["archive_content"], payload)
            with zipfile.ZipFile(BytesIO(scope["archive_content"])) as archive:
                self.assertEqual(json.loads(archive.read("IE4422/bundle_info.json"))["kind"], "batch_experiments")
        self.assertIn("zipfile.ZipFile(BytesIO(archive_content))", bootstrap)

    def test_notebook_cells_compile_and_expensive_controls_default_off(self):
        notebook = json.loads((ROOT / "notebooks/IE4422_Experiments.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        code = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
        for flag in ("RUN_A", "RUN_B", "RUN_D", "PREPARE_B_PACKAGE", "PREPARE_D_PACKAGE", "UPLOAD_JUDGMENTS", "COMPUTE_SUMMARY"):
            self.assertIn(flag + " = False", code)
        self.assertNotIn("ui_server", code)
        self.assertIn("DATASET = 'scienceqa'", code)
        self.assertIn("RUN_D = False", code)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = build(Path(directory) / "rebuilt.ipynb")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), notebook)

    def test_generated_notebook_uses_lf_and_preserves_content_with_windows_line_endings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = build(Path(directory) / "rebuilt.ipynb")
            generated = path.read_bytes()
            self.assertNotIn(b"\r\n", generated)
            windows_copy = Path(directory) / "windows.ipynb"
            windows_copy.write_bytes(generated.replace(b"\n", b"\r\n"))
            self.assertNotEqual(generated, windows_copy.read_bytes())
            self.assertEqual(json.loads(generated), json.loads(windows_copy.read_bytes()))

    def test_dataset_and_condition_namespaces_and_drive_persistence(self):
        folder = support.run_folder(ROOT, "ai2d", "B", "fixed", Path("drive"))
        self.assertEqual(folder, Path("drive/runs/experiments/ai2d/B/fixed"))
        for label in ("../outside", "..", "a/b", "a\\b", ""):
            with self.assertRaises(ValueError):
                support.run_folder(ROOT, "ai2d", "B", label)
        with patch.object(support, "run_visible") as call:
            result = support.run_experiment(ROOT, "python", "checkpoint", "ai2d", "D", "fixed", resume=True)
            self.assertIn("--resume", call.call_args.args[0])
            self.assertEqual(result, ROOT / "outputs/experiments/ai2d/D/fixed")

    def test_judgment_upload_rejects_traversal_and_does_not_replace_valid_conflicts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("../judgment_r_000001.json", "{}")
            with self.assertRaises(ValueError):
                support.import_judgments(archive, root / "judgments")
            self.assertFalse((root / "judgments").exists())
            archive = root / "empty.zip"
            with zipfile.ZipFile(archive, "w"):
                pass
            with self.assertRaises(ValueError):
                support.import_judgments(archive, root / "judgments")

    def test_bundle_contains_research_sources_and_selections_but_no_data_or_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            archive = build_bundle(ROOT, Path(directory) / "experiments.zip")
            with zipfile.ZipFile(archive) as source:
                names = source.namelist()
                for name in ("experiments/run.py", "experiments/common.py", "experiments/prepare_judging.py",
                             "manifests/ai2d/test_judge_selection.json", "notebooks/IE4422_Experiments.ipynb"):
                    self.assertIn("IE4422/" + name, names)
                self.assertFalse(any(name.startswith(("IE4422/data/", "IE4422/outputs/", "IE4422/models/")) for name in names))
                self.assertEqual(json.loads(source.read("IE4422/bundle_info.json"))["kind"], "batch_experiments")

    def test_prepared_final_data_restores_only_matching_fingerprinted_archives(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            original, restored, persistent = base / "original", base / "restored", base / "drive"
            names = ("configs/ai2d.json", "manifests/ai2d/test_final_ids.json", "manifests/ai2d/test_selection.json", "prepare_ai2d.py")
            for root in (original, restored):
                for name in names:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
            final = original / "data/final/ai2d"
            final.mkdir(parents=True)
            (final / "sample.txt").write_text("Original prepared bytes")
            with patch.object(support, "run_visible"):
                support.prepare_dataset(original, "python", "ai2d", persistent)
                support.prepare_dataset(restored, "python", "ai2d", persistent)
            self.assertEqual((restored / "data/final/ai2d/sample.txt").read_text(), "Original prepared bytes")
            config = restored / "configs/ai2d.json"
            config.write_text(config.read_text() + "\n")
            third = base / "third"
            for name in names:
                target = third / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(restored / name, target)
            with patch.object(support, "run_visible"), self.assertRaisesRegex(ValueError, "differs"):
                support.prepare_dataset(third, "python", "ai2d", persistent)
            self.assertFalse((third / "data/final/ai2d").exists())


if __name__ == "__main__":
    unittest.main()
