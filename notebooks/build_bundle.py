"""Package the browser tutor and its Colab launcher without research workflows."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "README.md", ".gitignore", "prepare_data.py", "qwen_engine.py", "tutor_pipeline.py", "hint_prompts.py",
    "scienceqa_tutor.py", "finalize_data.py", "inspect_question.py", "prompts/initial_hint.txt",
    "requirements.txt", "requirements-lock.txt", "configs/scienceqa.json",
    "notebooks/README.md", "notebooks/__init__.py",
    "notebooks/IE4422_Colab.ipynb", "notebooks/colab_config.json",
    "notebooks/colab_support.py", "notebooks/colab_ui.py", "notebooks/build_bundle.py",
    "reviews/scienceqa/README.md", "reviews/scienceqa/annotations.jsonl",
    "reviews/scienceqa/audit_manifest.json", "reports/scienceqa/visual_audit.md",
    "ui_server.py", "ui/index.html", "ui/app.js", "ui/app.css",
    "tests/test_prepare_data.py", "tests/test_finalize_data.py", "tests/test_inspect_question.py",
    "tests/test_scienceqa_tutor.py", "tests/test_ui_server.py",
    "tests/test_colab_workflow.py", "tests/test_colab_ui.py",
)


def build_bundle(root=ROOT, destination=None):
    root = Path(root).resolve()
    destination = Path(destination) if destination else root / "notebooks/IE4422_Source.zip"
    files = sorted(set(SOURCE_FILES) | {
        path.relative_to(root).as_posix()
        for pattern in ("manifests/scienceqa/*.json", "reports/scienceqa/*.json")
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
