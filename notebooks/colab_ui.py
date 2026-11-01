"""Manage the notebook's UI child process without importing GPU packages."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
from urllib.request import urlopen

from notebooks.colab_support import complete_checkpoint, read_json


class NotebookUIServer:
    """Keep this object in the notebook namespace so reruns reuse the same worker."""

    def __init__(self):
        self.process = None
        self.settings = None
        self.log_path = None
        self.port = None
        self.instance_id = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.settings = None

    def logs(self, limit=12000):
        if not self.log_path or not self.log_path.is_file():
            return "No UI log yet."
        with self.log_path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            source.seek(max(0, source.tell() - limit))
            return source.read().decode("utf-8", errors="replace")

    def status(self):
        if not self.running:
            raise RuntimeError("The UI server is stopped. Rerun Start browser UI.")
        with urlopen(f"http://127.0.0.1:{self.port}/api/status", timeout=1) as response:
            status = json.load(response)
        if not self.running or status.get("instance_id") != self.instance_id:
            raise RuntimeError("The port does not belong to this notebook's UI process.")
        return status

    def start(self, python, root, *, port=8765, proxy_origin, frame_ancestors,
              persistent_output=None, restart=False, timeout=120):
        from ui_server import validate_origin

        root = Path(root).resolve()
        python = Path(python).resolve()
        if not 1 <= port <= 65535:
            raise ValueError("Choose a port between 1 and 65535.")
        proxy_origin = validate_origin(proxy_origin)
        frame_ancestors = tuple(dict.fromkeys(validate_origin(value) for value in frame_ancestors))
        config = read_json(root / "notebooks/colab_config.json")
        checkpoint = root / "models" / config["model_revision"]
        if not complete_checkpoint(checkpoint, config):
            raise RuntimeError("Run Download/restore the fixed Qwen checkpoint first.")
        persistent_output = Path(persistent_output).resolve() if persistent_output else None
        settings = (str(python), str(root), port, proxy_origin, frame_ancestors,
                    str(persistent_output), config["model_revision"])
        if self.running and not restart:
            if settings != self.settings:
                raise RuntimeError("UI settings changed. Set RESTART_UI=True to replace the server.")
            return self.status()
        self.stop()
        # Fail before spawning if another process is using this port. Never kill it.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already in use. Choose another UI_PORT.")

        self.port = port
        self.instance_id = uuid.uuid4().hex
        self.log_path = root / "outputs/colab_ui/server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(python), "-u", str(root / "ui_server.py"), "--no-browser",
                   "--port", str(port), "--model-path", str(checkpoint),
                   "--allowed-origin", proxy_origin, "--instance-id", self.instance_id,
                   "--splits", "train", "validation"]
        for origin in frame_ancestors:
            command.extend(["--frame-ancestor", origin])
        if persistent_output:
            command.extend(["--persistent-output", str(persistent_output)])
        try:
            with self.log_path.open("ab") as log:
                log.write(b"\n--- Starting notebook UI server ---\n")
                log.flush()
                self.process = subprocess.Popen(
                    command, cwd=root, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                )
            self.settings = settings
            deadline = time.monotonic() + timeout
            while self.running and time.monotonic() < deadline:
                try:
                    return self.status()
                except (OSError, ValueError):
                    time.sleep(.25)
            raise RuntimeError("UI startup failed or timed out.\n" + self.logs())
        except BaseException:
            self.stop()
            raise
