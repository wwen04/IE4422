"""Small setup and storage helpers for the Colab notebook; no tutoring prompts."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile


def run_visible(command, *, cwd=None):
    """Forward child stdout/stderr through notebook output and preserve failures."""
    command = [str(argument) for argument in command]
    print("Running:", " ".join(command), flush=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1,
                          env=environment) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_extract(archive, destination, members=None):
    """Check all member paths before writing anything, including on Windows."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist() if members is None else [source.getinfo(name) for name in members]
        names = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            target = (destination / info.filename).resolve()
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or ":" in info.filename or info.filename in names
                    or not target.is_relative_to(destination)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe or duplicate ZIP member: {info.filename}")
            names.add(info.filename)
        source.extractall(destination, members=[info.filename for info in infos])


def dependency_profile(root, destination):
    """Retain Colab's CUDA wheels while using the shared project dependency pins."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    root, destination = Path(root), Path(destination)
    if Version(metadata.version("torch").split("+")[0]) < Version("2.6"):
        raise RuntimeError("Use a current Colab runtime with PyTorch 2.6 or newer.")
    protected = {name: metadata.version(name) for name in ("torch", "torchvision")}
    shared_lines = (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    shared_pins = {canonicalize_name(line.split("==")[0]): line.split("==", 1)[1]
                   for line in shared_lines if line.strip() and not line.startswith("#")}
    runtime_requirements = {}
    for name in ("torch", "torchvision"):
        for text in metadata.requires(name) or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            runtime_requirements.setdefault(canonicalize_name(req.name), []).append(req)

    overrides = {name: f"{name}=={version}" for name, version in protected.items()}
    for name, requirements in runtime_requirements.items():
        bounds = SpecifierSet(",".join(str(req.specifier) for req in requirements if req.specifier))
        if name in protected:
            if protected[name] not in bounds:
                raise RuntimeError(f"Colab runtime mismatch: {name} {protected[name]} does not satisfy {bounds}")
            continue
        exact = any(spec.operator == "==" for spec in bounds)
        if name in shared_pins and shared_pins[name] in bounds and not exact:
            continue  # Retain a shared pin when it is compatible with this CUDA build.
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        # Cover ranges as well as exact pins (e.g. torch 2.11 needs setuptools<82).
        # An incompatible/missing base package is resolved inside the project env.
        overrides[name] = f"{name}=={installed}" if installed and installed in bounds else f"{name}{bounds}"

    constraints = []
    for line in shared_lines:
        name = canonicalize_name(line.split("==")[0])
        if name not in overrides:
            constraints.append(line)
    constraints.extend(value for name, value in sorted(overrides.items()))
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / "constraints.txt"
    lock.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    requirements = [f"-c {lock.resolve().as_posix()}"]
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("-c ") or line.split("==")[0].lower() in ("torch", "torchvision"):
            continue
        requirements.append(line)
    requirements.extend(f"{name}=={protected[name]}" for name in ("torch", "torchvision"))
    target = destination / "requirements.txt"
    target.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    write_json(destination / "runtime_pins.json", {"preserved_runtime": protected, "dependency_overrides": overrides})
    return target


def check_dependencies(requirements):
    """Check the project's dependency graph, excluding unused Colab applications."""
    from packaging.requirements import Requirement

    pending = [Requirement(line) for line in Path(requirements).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith(("#", "-"))]
    visited = set()
    while pending:
        req = pending.pop()
        actual = metadata.version(req.name)
        if actual not in req.specifier:
            raise RuntimeError(f"Project dependency conflict: {req}; installed {actual}")
        key = (req.name.lower().replace("_", "-"), tuple(sorted(req.extras)))
        if key in visited:
            continue
        visited.add(key)
        for text in metadata.requires(req.name) or []:
            child = Requirement(text)
            if child.marker is None or any(child.marker.evaluate({"extra": extra})
                                           for extra in (req.extras or {""})):
                pending.append(child)
    print(f"Project dependency graph is consistent ({len(visited)} package/extra entries).")


def setup_environment(root, environment):
    if sys.version_info < (3, 12):
        raise RuntimeError("Select a current Colab Python 3.12+ runtime for these dependency pins.")
    root, environment = Path(root), Path(environment)
    requirements = dependency_profile(root, root / "tmp/colab-dependencies")
    print("Creating the project environment without ensurepip; reusing Colab's pip.", flush=True)
    # Colab's system Python can fail in ensurepip. Use the existing pip with an
    # explicit target interpreter instead of bootstrapping a new pip installation.
    # Recreating a partial environment updates its configuration without clearing files.
    run_visible([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(environment)])
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    pip = [sys.executable, "-m", "pip", "--python", str(python)]
    run_visible([*pip, "--version"])
    print("Installing shared dependencies into the project environment.", flush=True)
    run_visible([*pip, "install", "-r", requirements])
    print("Checking project dependencies and CUDA.", flush=True)
    run_visible([python, Path(__file__).resolve(), "check", requirements])
    run_visible([python, "-c", "import torch, torchvision; "
                 "assert torch.cuda.is_available(), 'CUDA unavailable in the project environment'; "
                 "print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"])


CACHE_PREFIXES = ("data/processed/scienceqa/", "manifests/scienceqa/", "reports/scienceqa/")


def prepared_signature(root):
    root = Path(root)
    return {"format": 1, "dataset": read_json(root / "configs/scienceqa.json"),
            "preparation_source": (root / "prepare_data.py").read_text(encoding="utf-8"),
            "dependencies": (root / "requirements-lock.txt").read_text(encoding="utf-8")}


def prepared_files():
    files = [f"data/processed/scienceqa/{folder}/{split}.jsonl"
             for split in ("train", "validation", "test") for folder in ("questions", "references")]
    files.extend(f"manifests/scienceqa/{split}_{kind}_ids.json"
                 for split in ("train", "validation", "test") for kind in ("candidate", "valid"))
    return files + ["reports/scienceqa/metadata_audit.json", "reports/scienceqa/integrity_audit.json"]


def archive_prepared(root, archive):
    root, archive = Path(root), Path(archive)
    for name in prepared_files():
        if not (root / name).is_file():
            raise FileNotFoundError(f"Run full prepare_data.py before caching; missing {name}")
    files = sorted(path for prefix in CACHE_PREFIXES for path in (root / prefix).rglob("*")
                   if path.is_file())
    record = {"signature": prepared_signature(root),
              "files": {path.relative_to(root).as_posix(): path.stat().st_size for path in files}}
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("prepared_cache.json", json.dumps(record))
        for path in files:
            output.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)


def restore_prepared(root, archive):
    root, archive = Path(root), Path(archive)
    if not archive.is_file():
        return False
    with zipfile.ZipFile(archive) as source:
        record = json.loads(source.read("prepared_cache.json"))
        if record["signature"] != prepared_signature(root):
            raise ValueError("Prepared cache differs from source/configuration. Set REBUILD_DATA=True.")
        if not set(prepared_files()).issubset(record["files"]):
            raise ValueError("Prepared cache is missing required exports/manifests/reports. Rebuild it.")
        for name, size in record["files"].items():
            if not name.startswith(CACHE_PREFIXES) or source.getinfo(name).file_size != size:
                raise ValueError(f"Invalid prepared-cache member: {name}")
    safe_extract(archive, root, members=record["files"])
    return True


def checkpoint_files(folder):
    folder = Path(folder)
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"]
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        required.extend(set(read_json(index)["weight_map"].values()))
    else:
        required.append("model.safetensors")
    return required


def write_checkpoint_marker(folder, config):
    folder = Path(folder)
    files = {}
    for name in checkpoint_files(folder):
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint is missing a required file: {name}")
        files[name] = path.stat().st_size
    write_json(folder / "colab_checkpoint.json", {"model": config, "files": files})


def complete_checkpoint(folder, config):
    folder = Path(folder)
    marker = folder / "colab_checkpoint.json"
    if not marker.is_file():
        return False
    record = read_json(marker)
    required = checkpoint_files(folder)
    if record.get("model") != config or set(record.get("files", {})) != set(required):
        return False
    for name in required:
        path = (folder / name).resolve()
        if (not path.is_relative_to(folder.resolve()) or not path.is_file()
                or path.stat().st_size == 0 or path.stat().st_size != record["files"][name]):
            return False
    return True


def prepare_model(root, persistent=None):
    from huggingface_hub import snapshot_download

    root = Path(root)
    config = read_json(root / "notebooks/colab_config.json")
    folder = root / "models" / config["model_revision"]
    cached = Path(persistent) / config["model_revision"] if persistent else None
    if not complete_checkpoint(folder, config):
        if cached and complete_checkpoint(cached, config):
            print("Copying the saved checkpoint from Drive into local storage.")
            shutil.copytree(cached, folder, dirs_exist_ok=True)
        else:
            print("Downloading the pinned model checkpoint (approximately 7.5 GB of weights).")
            snapshot_download(config["model_id"], revision=config["model_revision"], local_dir=folder,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model",
                                              "LICENSE*", "README.md"])
            write_checkpoint_marker(folder, config)
        if not complete_checkpoint(folder, config):
            raise RuntimeError("Checkpoint is incomplete; rerun the model-download cell before loading.")
    if cached and not complete_checkpoint(cached, config):
        print("Saving the complete checkpoint to Drive. This may take a few minutes.")
        (cached / "colab_checkpoint.json").unlink(missing_ok=True)
        # Copy regular checkpoint files; exclude Hub's local download bookkeeping.
        shutil.copytree(folder, cached, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".cache", "colab_checkpoint.json"))
        write_checkpoint_marker(cached, config)  # Completion marker is written last.
    print("Local model folder:", folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup")
    setup.add_argument("root", type=Path)
    setup.add_argument("environment", type=Path)
    check = commands.add_parser("check")
    check.add_argument("requirements", type=Path)
    model = commands.add_parser("model")
    model.add_argument("root", type=Path)
    model.add_argument("--persistent", type=Path)
    args = parser.parse_args()
    if args.command == "setup":
        setup_environment(args.root, args.environment)
    elif args.command == "check":
        check_dependencies(args.requirements)
    else:
        prepare_model(args.root, args.persistent)


if __name__ == "__main__":
    main()
