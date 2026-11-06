import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from notebooks.colab_support import write_checkpoint_marker, write_json
from notebooks.colab_ui import NotebookUIServer


class NotebookUIServerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config = {"model_id": "fake", "model_revision": "fixed"}
        write_json(self.root / "notebooks/colab_config.json", config)
        checkpoint = self.root / "models/fixed"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                     "tokenizer.json", "model.safetensors"):
            (checkpoint / name).write_text("fake", encoding="utf-8")
        write_checkpoint_marker(checkpoint, config)
        # Exercise real process/HTTP lifecycle without importing torch or loading weights.
        (self.root / "ui_server.py").write_text(
            "import argparse, json, os\n"
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--port', type=int)\n"
            "p.add_argument('--instance-id')\n"
            "args, rest=p.parse_known_args()\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  body=json.dumps({'pid':os.getpid(), 'model_loaded':False, 'instance_id':args.instance_id}).encode()\n"
            "  self.send_response(200)\n"
            "  self.send_header('Content-Length', str(len(body)))\n"
            "  self.end_headers()\n"
            "  self.wfile.write(body)\n"
            "ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()\n",
            encoding="utf-8",
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.server = NotebookUIServer()
        self.addCleanup(self.server.stop)
        self.options = {"port": port, "proxy_origin": "https://session.example.com",
                        "frame_ancestors": ("https://colab.research.google.com",), "timeout": 10}

    def start(self, **overrides):
        # Windows venv python.exe is a redirector process; the CPU-only fixture
        # can use the interpreter directly, as Colab's Linux venv does.
        return self.server.start(sys._base_executable, self.root, **{**self.options, **overrides})

    def test_reuse_restart_and_stop_own_process(self):
        first = self.start()
        process = self.server.process
        self.assertEqual(self.start()["pid"], first["pid"])
        self.assertIs(self.server.process, process)
        self.assertIn("GET /api/status", self.server.logs())
        with self.assertRaisesRegex(RuntimeError, "settings changed"):
            self.start(persistent_output=self.root / "drive")
        self.assertTrue(self.server.running)
        second = self.start(restart=True)
        self.assertIsNotNone(process.poll())
        self.assertNotEqual(second["pid"], first["pid"])
        replacement = self.server.process
        self.server.stop()
        self.server.stop()
        self.assertIsNotNone(replacement.poll())
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.server.status()

    def test_launch_preserves_virtualenv_python_symlink(self):
        python = self.root / "project_env/bin/python"
        base_python = Path(sys._base_executable).resolve()
        resolve = Path.resolve

        def resolve_path(path, *args, **kwargs):
            # Model Colab's venv symlink on Windows without requiring symlink privileges.
            return base_python if path == python else resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve_path), \
                patch("notebooks.colab_ui.subprocess.Popen") as spawn, \
                patch("notebooks.colab_ui.socket.socket") as socket_factory, \
                patch.object(self.server, "status", return_value={"model_loaded": False}):
            spawn.return_value.poll.return_value = None
            socket_factory.return_value.__enter__.return_value.connect_ex.return_value = 1
            self.server.start(python, self.root, **self.options)

        self.assertEqual(spawn.call_args.args[0][0], str(python.absolute()))

    def test_incomplete_checkpoint_never_spawns(self):
        (self.root / "models/fixed/model.safetensors").unlink()
        with patch("notebooks.colab_ui.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "checkpoint first"):
                self.start()
        spawn.assert_not_called()

    def test_occupied_port_is_not_reused_or_killed(self):
        with socket.socket() as other:
            other.bind(("127.0.0.1", self.options["port"]))
            other.listen()
            with patch("notebooks.colab_ui.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    self.start()
            spawn.assert_not_called()
            self.assertFalse(self.server.running)

    def test_failed_start_surfaces_logs_and_cleans_up(self):
        (self.root / "ui_server.py").write_text("raise RuntimeError('startup detail')\n")
        with self.assertRaisesRegex(RuntimeError, "startup detail"):
            self.start()
        self.assertFalse(self.server.running)

    def test_timeout_stops_child_and_wrong_process_status_is_rejected(self):
        (self.root / "ui_server.py").write_text("import time\ntime.sleep(60)\n")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.start(timeout=.1)
        self.assertFalse(self.server.running)
        # A different HTTP service must never be mistaken for this worker.
        from io import BytesIO
        from unittest.mock import Mock
        self.server.process = Mock(pid=123, poll=Mock(return_value=None))
        with patch("notebooks.colab_ui.urlopen", return_value=BytesIO(json.dumps({"instance_id": "other"}).encode())):
            with self.assertRaisesRegex(RuntimeError, "does not belong"):
                self.server.status()
        self.server.process = None


if __name__ == "__main__":
    unittest.main()
