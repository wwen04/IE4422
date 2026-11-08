"""Storage and execution controls for the separate batch-experiment notebook."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from notebooks.colab_support import run_visible, safe_extract


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def prepared_signature(root, dataset):
    root = Path(root)
    names = [f"configs/{dataset}.json"]
    names += sorted(path.relative_to(root).as_posix() for path in (root / f"manifests/{dataset}").glob("*final_ids.json"))
    names += (["prepare_ai2d.py", "manifests/ai2d/test_selection.json"] if dataset == "ai2d" else
              ["prepare_data.py", "finalize_data.py", "reviews/scienceqa/annotations.jsonl", "reviews/scienceqa/audit_manifest.json"])
    return hashlib.sha256(b"".join(name.encode() + b"\0" + (root / name).read_bytes() for name in names)).hexdigest()


def prepare_dataset(root, python, dataset, persistent=None, rebuild=False):
    root = Path(root).resolve()
    if dataset not in ("ai2d", "scienceqa"):
        raise ValueError("Choose ai2d or scienceqa")
    signature = prepared_signature(root, dataset)
    archive = Path(persistent) / "prepared" / f"{dataset}_experiments.zip" if persistent else None
    final = root / f"data/final/{dataset}"
    if not rebuild and not final.exists() and archive and archive.is_file():
        with zipfile.ZipFile(archive) as source:
            manifest = json.loads(source.read("experiment_prepared.json"))
            if manifest["dataset"] != dataset or manifest["signature"] != signature:
                raise ValueError("Drive prepared data differs from the bundled source; enable REBUILD_DATA")
            names = manifest["files"]
            if not names or any(not name.startswith(f"data/final/{dataset}/") for name in names):
                raise ValueError("Invalid prepared-data archive")
            for name, expected in names.items():
                if hashlib.sha256(source.read(name)).hexdigest() != expected:
                    raise ValueError("Prepared-data archive fingerprint mismatch")
        safe_extract(archive, root, members=list(names))
    if rebuild or not final.exists():
        scripts = ["prepare_ai2d.py"] if dataset == "ai2d" else ["prepare_data.py", "finalize_data.py"]
        for script in scripts:
            run_visible([python, root / script], cwd=root)
    run_visible([python, "-m", "experiments.run", "--dataset", dataset, "--experiment", "B", "--preflight"], cwd=root)
    run_visible([python, "-m", "experiments.prepare_judging", f"--freeze-{dataset}-selection"], cwd=root)
    if archive and (rebuild or not archive.exists()):
        archive.parent.mkdir(parents=True, exist_ok=True)
        paths = sorted(path for path in final.rglob("*") if path.is_file())
        files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        temporary = archive.with_suffix(".partial.zip")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
            target.writestr("experiment_prepared.json", json.dumps({"dataset": dataset, "signature": signature, "files": files}))
            for path in paths:
                target.write(path, path.relative_to(root).as_posix())
        temporary.replace(archive)
    print(f"{dataset}: prepared final data validated; no model was loaded.")


def run_folder(root, dataset, experiment, label, persistent=None, *, with_evidence=False):
    if dataset not in ("ai2d", "scienceqa") or experiment not in ("A", "B", "D"):
        raise ValueError("Invalid dataset or experiment")
    if with_evidence and experiment != "A":
        raise ValueError("with_evidence is only available for Experiment A")
    if not label or label in (".", "..") or any(char in label for char in '/\\:'):
        raise ValueError("Use a simple run label without path separators")
    base = Path(persistent) / "runs/experiments" if persistent else Path(root) / "outputs/experiments"
    dataset_folder = "scienceqa_test" if dataset == "scienceqa" else dataset
    return base / dataset_folder / ("A_with_evidence" if with_evidence else experiment) / label


def run_experiment(root, python, model, dataset, experiment, label, persistent=None, resume=False, *, with_evidence=False):
    folder = run_folder(root, dataset, experiment, label, persistent, with_evidence=with_evidence)
    command = [python, "-m", "experiments.run", "--dataset", dataset, "--experiment", experiment,
               "--model-path", model, "--output-dir", folder]
    if with_evidence:
        command.append("--with-evidence")
    if resume:
        command.append("--resume")
    run_visible(command, cwd=root)
    return folder


def package_run(root, python, folder, package_name="judge_package"):
    if not package_name or package_name in (".", "..") or any(c in package_name for c in '/\\:'):
        raise ValueError("Use a simple package name")
    target = Path(folder) / package_name
    run_visible([python, "-m", "experiments.prepare_judging", "--run-dir", folder, "--output-dir", target], cwd=root)
    return target


def import_judgments(archive, destination):
    """Import flat judgment files; preserve any conflicting valid prior judgment."""
    from experiments.judge_validate import validate

    destination = Path(destination)
    files = {}
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe judgment ZIP path")
            if not path.name.startswith("judgment_r_") or path.suffix != ".json":
                raise ValueError("Upload a ZIP containing only judgment JSON files")
            if path.name in files:
                raise ValueError("Duplicate judgment filename in ZIP")
            files[path.name] = source.read(info)
    if not files:
        raise ValueError("No judgments in the uploaded ZIP")
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content and validate(path) is None:
            raise ValueError(f"Conflicting valid prior judgment: {name}; retain both versions separately")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            previous = destination / "archive" / stamp() / name
            previous.parent.mkdir(parents=True, exist_ok=True)
            path.replace(previous)
        path.write_bytes(content)
    return len(files)


def summarize_run(root, python, folder, package, answers=None):
    target = Path(folder) / "summaries" / stamp()
    command = [python, "-m", "experiments.summarize", "--run-dir", folder, "--package-dir", package,
               "--judgments-dir", Path(folder) / "judgments", "--output-dir", target]
    if answers:
        command.extend(["--answers-run-dir", answers])
    run_visible(command, cwd=root)
    return target
