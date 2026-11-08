"""Package the separate research notebook, frozen selections and batch code."""

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    "README.md", "requirements.txt", "requirements-lock.txt", "prepare_data.py", "prepare_ai2d.py",
    "finalize_data.py", "scienceqa_tutor.py", "tutor_pipeline.py", "qwen_engine.py", "hint_prompts.py",
    "prompts/initial_hint.txt", "notebooks/__init__.py", "notebooks/colab_support.py",
    "notebooks/colab_config.json", "notebooks/experiments_colab.py", "notebooks/build_experiment_bundle.py",
    "notebooks/IE4422_Experiments.ipynb", "notebooks/build_experiment_notebook.py", "notebooks/README.md",
    "tests/test_judge_preparation.py",
    "tests/test_batch_experiments.py", "tests/test_scienceqa_experiments.py",
    "tests/test_experiment_colab.py", "tests/test_prepare_ai2d.py",
)


def build_bundle(root=ROOT, destination=None):
    root = Path(root).resolve()
    destination = Path(destination) if destination else root / "notebooks/IE4422_Experiments_Source.zip"
    names = set(BASE)
    for pattern in ("experiments/**/*.py", "experiments/**/*.md", "configs/*.json",
                    "manifests/scienceqa/*.json", "manifests/ai2d/*.json", "reports/scienceqa/*",
                    "reports/ai2d/*", "reviews/scienceqa/*", "reviews/ai2d/*"):
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
    for name in names:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError(f"Missing or linked experiment source: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("IE4422/bundle_info.json", json.dumps({"kind": "batch_experiments", "files": sorted(names),
                         "created_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
        for name in sorted(names):
            archive.write(root / name, "IE4422/" + name)
    return destination


if __name__ == "__main__":
    print(build_bundle())
