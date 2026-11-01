import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import zipfile

from notebooks import colab_support as support
from notebooks.build_bundle import ROOT, build_bundle
from notebooks.run_demo import run_demo


class ColabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self, root):
        (root / "configs").mkdir(parents=True, exist_ok=True)
        support.write_json(root / "configs/scienceqa.json", {"dataset_revision": "fixed"})
        (root / "prepare_data.py").write_text("# preparation\n", encoding="utf-8")
        for name in support.AUDIT_SOURCES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
        (root / "requirements-lock.txt").write_text("datasets==5.1.0\n", encoding="utf-8")
        for name in support.prepared_files():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]\n", encoding="utf-8")
        for stage in ("processed", "final"):
            image = root / f"data/{stage}/scienceqa/images/example.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"original image bytes")

    def test_prepared_cache_roundtrip_and_stale_source_rejection(self):
        self.fixture(self.root)
        archive = self.root / "prepared.zip"
        support.archive_prepared(self.root, archive)
        target = self.root / "restored"
        (target / "configs").mkdir(parents=True)
        for name in ("configs/scienceqa.json", "prepare_data.py", "requirements-lock.txt", *support.AUDIT_SOURCES):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes((self.root / name).read_bytes())
        self.assertTrue(support.restore_prepared(target, archive))
        self.assertEqual((target / "data/processed/scienceqa/images/example.png").read_bytes(),
                         b"original image bytes")
        self.assertTrue((target / "data/processed/scienceqa/references/validation.jsonl").is_file())
        self.assertEqual((target / "data/final/scienceqa/images/example.png").read_bytes(), b"original image bytes")
        self.assertTrue((target / "manifests/scienceqa/validation_final_ids.json").is_file())
        (target / "prepare_data.py").write_text("# changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "REBUILD_DATA"):
            support.restore_prepared(target, archive)

    def test_prepared_cache_rejects_changed_audit_decisions(self):
        self.fixture(self.root)
        archive = self.root / "prepared.zip"
        support.archive_prepared(self.root, archive)
        (self.root / "reviews/scienceqa/annotations.jsonl").write_text('{"changed": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "REBUILD_DATA"):
            support.restore_prepared(self.root, archive)

    def test_notebook_finalizes_before_caching_and_revalidates_reused_data(self):
        self.fixture(self.root)
        support.write_json(self.root / "reports/scienceqa/visual_audit.json", {"final_counts": {}})
        notebook = support.read_json(ROOT / "notebooks/IE4422_Colab.ipynb")
        code = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "prepare-data")
        events = []
        namespace = {
            "WORK": self.root, "PYTHON": Path(sys.executable), "PERSISTENT": None,
            "REBUILD_DATA": False, "prepared_signature": support.prepared_signature,
            "prepared_files": support.prepared_files, "read_json": support.read_json,
            "write_json": support.write_json,
            "run_visible": lambda command, **kwargs: events.append(command[1]),
            "archive_prepared": lambda *args: events.append("archive"),
        }
        with redirect_stdout(StringIO()):
            exec(compile(code, "prepare-data", "exec"), namespace)
            self.assertEqual(events, ["prepare_data.py", "finalize_data.py", "archive"])
            events.clear()
            exec(compile(code, "prepare-data", "exec"), namespace)
        self.assertEqual(events, ["finalize_data.py"])

    def test_notebook_selects_final_ids_and_rejects_excluded_candidates(self):
        self.fixture(self.root)
        retained = "validation_000008"
        excluded = "validation_000003"
        row = {"item_id": retained, "split": "validation", "question": "Read labels.",
               "context": "Student passage", "choices": [{"id": "A", "text": "Choice"}],
               "image_path": "data/final/scienceqa/images/example.png"}
        (self.root / "data/final/scienceqa/questions/validation.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8")
        support.write_json(self.root / "manifests/scienceqa/validation_final_ids.json", [retained])
        support.write_json(self.root / "manifests/scienceqa/validation_valid_ids.json", [excluded, retained])
        (self.root / "data/processed/scienceqa/questions/validation.jsonl").write_text(
            json.dumps({**row, "item_id": excluded}) + "\n", encoding="utf-8")
        notebook = support.read_json(ROOT / "notebooks/IE4422_Colab.ipynb")
        code = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "select-question")
        display_module = ModuleType("IPython.display")
        display_module.Image = Mock()
        display_module.display = Mock()
        namespace = {"WORK": self.root, "read_json": support.read_json}
        with patch.dict(sys.modules, {"IPython.display": display_module}), redirect_stdout(StringIO()):
            exec(compile(code, "select-question", "exec"), namespace)
            self.assertEqual(namespace["item"]["item_id"], retained)
            self.assertIn("data/final/", namespace["item"]["image_path"].replace("\\", "/"))
            # Simulate entering an excluded ID through the notebook's parameter form.
            excluded_code = code.replace("ITEM_ID = ''", f"ITEM_ID = '{excluded}'")
            with self.assertRaisesRegex(ValueError, "not found.*final"):
                exec(compile(excluded_code, "select-question", "exec"), namespace)

    def test_missing_preparation_and_unsafe_archives_fail(self):
        with self.assertRaises(FileNotFoundError):
            support.archive_prepared(self.root, self.root / "prepared.zip")
        for name in ("../outside.txt", "/outside.txt", "C:/outside.txt", "folder\\..\\outside.txt"):
            with self.subTest(name=name):
                archive = self.root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(name, "unsafe")
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    support.safe_extract(archive, self.root / "destination")

    def test_checkpoint_rejects_missing_changed_and_wrong_revision_files(self):
        folder = self.root / "model"
        folder.mkdir()
        config = {"model_id": "example", "model_revision": "fixed"}
        for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
            (folder / name).write_text("{}", encoding="utf-8")
        support.write_json(folder / "model.safetensors.index.json", {"weight_map": {"w": "shard.safetensors"}})
        self.assertFalse(support.complete_checkpoint(folder, config))
        (folder / "shard.safetensors").write_bytes(b"weights")
        support.write_checkpoint_marker(folder, config)
        self.assertTrue(support.complete_checkpoint(folder, config))
        self.assertFalse(support.complete_checkpoint(folder, {**config, "model_revision": "other"}))
        # A different length cannot be mistaken for the recorded complete file.
        (folder / "shard.safetensors").write_bytes(b"part")
        self.assertFalse(support.complete_checkpoint(folder, config))

    def test_dependency_profile_preserves_runtime_cuda_and_exact_dependencies(self):
        (self.root / "requirements.txt").write_text(
            "-c requirements-lock.txt\ndatasets==5.1.0\ntorch==2.14.1\ntorchvision==0.29.1\n")
        (self.root / "requirements-lock.txt").write_text(
            "datasets==5.1.0\ntorch==2.14.1\ntorchvision==0.29.1\nsympy==1.14.0\n")
        versions = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "sympy": "1.13.1"}
        dependencies = {"torch": ["sympy==1.13.1"], "torchvision": ["torch==2.6.0"]}
        with patch.object(support.metadata, "version", side_effect=versions.__getitem__), patch.object(
                support.metadata, "requires", side_effect=dependencies.__getitem__):
            target = support.dependency_profile(self.root, self.root / "profile")
        text = target.read_text(encoding="utf-8")
        lock = (target.parent / "constraints.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.6.0+cu124", text)
        self.assertIn("torchvision==0.21.0+cu124", text)
        self.assertIn("sympy==1.13.1", lock)
        self.assertNotIn("torch==2.14.1", lock)
        self.assertIn("datasets==5.1.0", text)

    def test_setup_skips_ensurepip_and_targets_project_python(self):
        commands = []

        def simulate_colab(command, **kwargs):
            command = list(map(str, command))
            if "venv" in command and "--without-pip" not in command:
                raise subprocess.CalledProcessError(1, command)  # Reported ensurepip failure.
            commands.append(command)

        requirements = self.root / "requirements.txt"
        environment = self.root / "project_env"
        with patch.object(support, "dependency_profile", return_value=requirements), patch.object(
                support, "run_visible", side_effect=simulate_colab), patch.object(
                support.sys, "version_info", (3, 13, 0)), redirect_stdout(StringIO()):
            support.setup_environment(self.root, environment)
        self.assertIn("--without-pip", commands[0])
        self.assertIn("--system-site-packages", commands[0])
        self.assertNotIn("--clear", commands[0])
        self.assertEqual(commands[1][1:4], ["-m", "pip", "--python"])
        self.assertEqual(commands[1][-1], "--version")
        self.assertEqual(commands[2][4], commands[1][4])
        self.assertEqual(commands[2][5:7], ["install", "-r"])
        self.assertEqual(commands[3][2], "check")
        self.assertIn("torch.cuda.is_available()", commands[4][-1])

    def test_torch_211_bound_preserves_compatible_colab_setuptools(self):
        (self.root / "requirements.txt").write_text(
            "-c requirements-lock.txt\ndatasets==5.1.0\ntorch==2.14.1\ntorchvision==0.29.1\n")
        original = "datasets==5.1.0\ntorch==2.14.1\ntorchvision==0.29.1\nsetuptools==84.2.1\n"
        (self.root / "requirements-lock.txt").write_text(original)
        versions = {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130", "setuptools": "80.9.0"}
        dependencies = {"torch": ["setuptools<82"], "torchvision": ["torch==2.11.0"]}
        with patch.object(support.metadata, "version", side_effect=versions.__getitem__), patch.object(
                support.metadata, "requires", side_effect=dependencies.__getitem__):
            target = support.dependency_profile(self.root, self.root / "profile")
        lock = (target.parent / "constraints.txt").read_text(encoding="utf-8")
        self.assertIn("setuptools==80.9.0", lock)
        self.assertNotIn("setuptools==84.2.1", lock)
        self.assertIn("torch==2.11.0+cu130", target.read_text(encoding="utf-8"))
        self.assertEqual((self.root / "requirements-lock.txt").read_text(), original)

    def test_torch_bounds_are_resolved_when_base_package_is_incompatible_or_missing(self):
        (self.root / "requirements.txt").write_text("torch==2.14.1\ntorchvision==0.29.1\n")
        (self.root / "requirements-lock.txt").write_text(
            "torch==2.14.1\ntorchvision==0.29.1\nsetuptools==84.2.1\n")
        dependencies = {"torch": ["setuptools<82", "setuptools>=77"], "torchvision": ["torch==2.11.0"]}
        for installed in ("84.2.1", None):
            with self.subTest(installed=installed):
                def version(name):
                    if name == "setuptools":
                        if installed is None:
                            raise support.metadata.PackageNotFoundError(name)
                        return installed
                    return {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}[name]

                with patch.object(support.metadata, "version", side_effect=version), patch.object(
                        support.metadata, "requires", side_effect=dependencies.__getitem__):
                    target = support.dependency_profile(self.root, self.root / "profile")
                lock = (target.parent / "constraints.txt").read_text(encoding="utf-8")
                self.assertIn("setuptools<82,>=77", lock)
                self.assertNotIn("setuptools==84.2.1", lock)

    def test_command_logging_exposes_stdout_stderr_and_preserves_failure(self):
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as failure:
            support.run_visible([sys.executable, "-c", "import sys; print('stdout detail'); "
                                 "print('stderr detail', file=sys.stderr); sys.exit(7)"])
        self.assertEqual(failure.exception.returncode, 7)
        self.assertIn("stdout detail", output.getvalue())
        self.assertIn("stderr detail", output.getvalue())

    def test_demo_reuses_model_and_intermediates_without_reference_fields(self):
        item = {"item_id": "validation_000003", "split": "validation", "image_path": "image.png",
                "question": "Which arrow points up?", "context": "Read labels",
                "choices": [{"id": "A", "text": "up"}, {"id": "B", "text": "down"}],
                "answer": "SECRET GOLD", "solution": "SECRET SOLUTION"}
        tutor = Mock()
        tutor.ask.side_effect = ["Evidence", "Diagnosis", "Visual hint", "Concept hint", "Reasoning hint"]
        saved = []
        result = run_demo(tutor, item, "B", "Student reasoning", [1, 2, 3],
                          lambda value: saved.append(copy.deepcopy(value)))
        self.assertEqual(tutor.ask.call_count, 5)
        self.assertEqual(result["hints"]["3"], "Reasoning hint")
        self.assertEqual(len(saved), 5)
        self.assertEqual(saved[0]["hints"], {})
        self.assertNotIn("SECRET", str(tutor.ask.call_args_list))
        for call in tutor.ask.call_args_list[2:]:
            self.assertIn("Evidence", call.args[1])
            self.assertIn("Diagnosis", call.args[1])
        tutor.reset_mock()
        with self.assertRaises(ValueError):
            run_demo(tutor, item, "B", "Student reasoning", [4], saved.append)
        tutor.ask.assert_not_called()

    def test_notebook_code_compiles_and_bundle_contains_only_selected_source(self):
        notebook = support.read_json(ROOT / "notebooks/IE4422_Colab.ipynb")
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
        archive = build_bundle(ROOT, self.root / "source.zip")
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            self.assertIn("IE4422/prepare_data.py", names)
            for required in (*support.AUDIT_SOURCES, "inspect_question.py", "ui_server.py", "ui/app.js",
                             "notebooks/colab_ui.py"):
                self.assertIn("IE4422/" + required, names)
            self.assertIn("IE4422/notebooks/run_demo.py", names)
            self.assertEqual(source.read("IE4422/notebooks/IE4422_Colab.ipynb"),
                             (ROOT / "notebooks/IE4422_Colab.ipynb").read_bytes())
            self.assertTrue(all(name.startswith("IE4422/") for name in names))
            for prefix in ("IE4422/data/", "IE4422/models/", "IE4422/.venv/", "IE4422/.git/"):
                self.assertFalse(any(name.startswith(prefix) for name in names))
            self.assertNotIn("IE4422/AGENTS.md", names)
            self.assertNotIn("IE4422/IMPLEMENTATION_PLAN.md", names)
            self.assertNotIn("IE4422/notebooks/IE4422_Source.zip", names)
            info = json.loads(source.read("IE4422/bundle_info.json"))
            self.assertEqual(set(info["files"]), {name.removeprefix("IE4422/") for name in names
                                                 if name != "IE4422/bundle_info.json"})

    def test_notebook_ui_launch_uses_proxy_embed_and_drive_without_model_import(self):
        notebook = support.read_json(ROOT / "notebooks/IE4422_Colab.ipynb")
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        ids = list(cells)
        self.assertLess(ids.index("prepare-data"), ids.index("prepare-model"))
        self.assertLess(ids.index("prepare-model"), ids.index("start-ui"))
        output = Mock()
        output.eval_js.side_effect = ["https://session.example.com/", "https://output.example.com"]
        google = ModuleType("google")
        colab = ModuleType("google.colab")
        colab.output = output
        google.colab = colab
        server = Mock()
        server.start.return_value = {"model_loaded": False}
        namespace = {"INTERFACE": "browser UI", "WORK": self.root, "PYTHON": Path(sys.executable),
                     "PERSISTENT": self.root / "drive", "UI_SERVER": server}
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}), redirect_stdout(StringIO()):
            exec(compile(cells["start-ui"]["source"], "start-ui", "exec"), namespace)
        self.assertEqual(server.start.call_args.kwargs["proxy_origin"], "https://session.example.com")
        self.assertEqual(server.start.call_args.kwargs["persistent_output"], self.root / "drive/runs/ui")
        self.assertEqual(server.start.call_args.kwargs["frame_ancestors"],
                         ("https://colab.research.google.com", "https://output.example.com"))
        output.serve_kernel_port_as_iframe.assert_called_once_with(8765, height=1100, cache_in_notebook=False)
        server.stop.assert_not_called()

    def test_browser_mode_skips_demo_inputs_generation_and_default_stop(self):
        notebook = support.read_json(ROOT / "notebooks/IE4422_Colab.ipynb")
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        server = Mock()
        namespace = {"INTERFACE": "browser UI", "UI_SERVER": server}
        with patch("builtins.input", side_effect=AssertionError("UI mode must not prompt")), \
                redirect_stdout(StringIO()):
            for id in ("select-question", "student-attempt", "generate-hints", "stop-ui", "download-results"):
                exec(compile(cells[id]["source"], id, "exec"), namespace)
            server.stop.assert_not_called()
            exec(cells["stop-ui"]["source"].replace("STOP_UI = False", "STOP_UI = True"), namespace)
        server.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
