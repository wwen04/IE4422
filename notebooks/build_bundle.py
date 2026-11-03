"""Rebuild the small source ZIP after editing the project or its Colab notebook."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "README.md", ".gitignore", "prepare_data.py", "qwen_engine.py", "tutor_pipeline.py",
    "run_scienceqa_tutor.py", "check_scienceqa_answers.py", "run_tutor.py", "inspect_diagram.py", "finalize_data.py", "inspect_question.py",
    "requirements.txt", "requirements-lock.txt", "configs/scienceqa.json",
    "notebooks/README.md", "notebooks/__init__.py",
    "notebooks/IE4422_Colab.ipynb", "notebooks/colab_config.json",
    "notebooks/colab_support.py", "notebooks/colab_ui.py", "notebooks/run_demo.py", "notebooks/build_bundle.py",
    "reviews/scienceqa/README.md", "reviews/scienceqa/annotations.jsonl",
    "reviews/scienceqa/audit_manifest.json", "reports/scienceqa/visual_audit.md",
    "ui_server.py", "ui/index.html", "ui/app.js", "ui/app.css",
)


def build_bundle(root=ROOT, destination=None):
    root = Path(root).resolve()
    destination = Path(destination) if destination else root / "notebooks/IE4422_Source.zip"
    files = sorted(set(SOURCE_FILES) | {
        path.relative_to(root).as_posix()
        for pattern in ("tests/test_*.py", "manifests/scienceqa/*.json", "reports/scienceqa/*.json")
        for path in root.glob(pattern)
    })
    for name in files:
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Missing or linked bundle source: {name}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    info = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source": "Current selected working files, including uncommitted edits",
            "files": files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("IE4422/bundle_info.json", json.dumps(info, indent=2) + "\n")
        for name in files:
            archive.write(root / name, "IE4422/" + name)
    return destination


if __name__ == "__main__":
    path = build_bundle()
    print(f"Created {path} ({path.stat().st_size:,} bytes); no data, weights or environments included.")
