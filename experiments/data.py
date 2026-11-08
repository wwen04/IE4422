"""Dataset boundaries and public input fingerprints shared by batch experiments."""

import json
from pathlib import Path
import random

from prepare_ai2d import digest as preparation_digest
from scienceqa_tutor import load_public_questions
from experiments.common import digest, json_digest, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"Blank JSONL record: {path}")
    return [json.loads(line) for line in lines]


def text_fingerprint_matches(content, expected_hash):
    """Accept LF/CRLF equivalents of legacy byte hashes, preserving all other bytes."""
    normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return expected_hash in {digest(content), digest(normalized), digest(normalized.replace(b"\n", b"\r\n"))}


def source_config(root, dataset):
    if dataset not in ("ai2d", "scienceqa"):
        raise ValueError("Choose ai2d or scienceqa")
    return read_json(Path(root) / f"configs/{dataset}.json")


def public_snapshot(root, item):
    image = Path(item["image_path"]).resolve()
    snapshot = {"question": item["question"], "context": item["context"],
                "choices": [{"id": c["id"], "text": c["text"]} for c in item["choices"]],
                "image_sha256": digest(image.read_bytes())}
    snapshot["input_sha256"] = json_digest(snapshot)
    snapshot["image_path"] = image.relative_to(Path(root).resolve()).as_posix()
    return snapshot


def load_questions(root, dataset, *, item_ids=None, limit=None):
    """Read public test exports only for all current experiment modes."""
    return _load_questions(root, dataset, split="test", item_ids=item_ids, limit=limit)


def _load_questions(root, dataset, *, split, item_ids=None, limit=None):
    """Shared validation; the legacy read-only verifier can still inspect train runs."""
    root = Path(root).resolve()
    config = source_config(root, dataset)
    if split not in ("train", "test") or dataset == "ai2d" and split != "test":
        raise ValueError("Invalid experiment dataset/split")
    if dataset == "scienceqa":
        rows = load_public_questions(root, split)
    else:
        rows = read_jsonl(root / "data/final/ai2d/questions/test.jsonl")
        ids = read_json(root / "manifests/ai2d/test_final_ids.json")
        frozen = read_json(root / "manifests/ai2d/test_selection.json")
        if (frozen["configuration"] != config or ids != [r["item_id"] for r in rows]
                or ids != [r["item_id"] for r in frozen["items"]]
                or len(ids) != len(set(ids))):
            raise ValueError("AI2D public exports differ from the frozen selection")
        for row, expected in zip(rows, frozen["items"]):
            prepared_row = {**row, "image_path": "data/processed/ai2d/images/" + Path(row["image_path"]).name}
            if preparation_digest(prepared_row) != expected["public_row_sha256"]:
                raise ValueError(f"Frozen AI2D question changed: {row['item_id']}")
    if not rows or len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate public questions")
    if item_ids and limit is not None:
        raise ValueError("Choose item IDs or a limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("The question limit must be positive")
    if item_ids:
        by_id = {row["item_id"]: row for row in rows}
        if len(item_ids) != len(set(item_ids)) or not set(item_ids) <= set(by_id):
            raise ValueError("IDs must be distinct members of this dataset's frozen selection")
        rows = [by_id[key] for key in item_ids]
    elif limit is not None:
        rows = rows[:limit]
    result = []
    image_root = (root / f"data/final/{dataset}/images").resolve()
    for row in rows:
        prefix = "ai2d_test_" if dataset == "ai2d" else f"{split}_"
        if row["split"] != split or not row["item_id"].startswith(prefix):
            raise ValueError("Dataset ID/split mismatch")
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError("Use images within the dataset's final image folder")
        choices = row["choices"]
        if (not 2 <= len(choices) <= 26 or [c["id"] for c in choices] != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)])
                or any(not isinstance(c["text"], str) or not c["text"].strip() for c in choices)
                or not isinstance(row["question"], str) or not row["question"].strip()
                or not isinstance(row["context"], str)):
            raise ValueError("Invalid public question or ordered options")
        if dataset == "ai2d" and digest(image.read_bytes()) != row["metadata"]["image_sha256"]:
            raise ValueError("Frozen AI2D image changed")
        result.append({"item_id": row["item_id"], "split": split, "question": row["question"],
                       "context": row["context"], "choices": [{"id": c["id"], "text": c["text"]} for c in choices],
                       "image_path": str(image)})
    return result


def load_references(root, dataset, items):
    """Evaluator-only join; never called by hint generation."""
    root = Path(root).resolve()
    split = "test"
    path = root / f"data/final/{dataset}/references/{split}.jsonl"
    if dataset == "ai2d":
        report = read_json(root / "reports/ai2d/integrity_audit.json")
        expected_hash = report["artifact_files_sha256"][path.relative_to(root).as_posix()]
        if not text_fingerprint_matches(path.read_bytes(), expected_hash):
            raise ValueError("AI2D reference export differs from the preparation report, including after LF/CRLF normalization")
    rows = read_jsonl(path)
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate reference ID")
    by_id = {row["item_id"]: row for row in rows}
    result = {}
    for item in items:
        ref = by_id.get(item["item_id"])
        if not ref or ref["gold_option_id"] not in {c["id"] for c in item["choices"]}:
            raise ValueError("Missing or invalid reference answer")
        if not isinstance(ref.get("lecture"), str) or not isinstance(ref.get("solution"), str):
            raise ValueError("Reference explanation fields must be strings; empty means unavailable")
        result[item["item_id"]] = {key: ref[key] for key in ("gold_option_id", "lecture", "solution")}
    return result


def freeze_ai2d_judging(root):
    """Choose 30 of the full 100 questions without opening model outcomes."""
    root = Path(root).resolve()
    items = load_questions(root, "ai2d")
    if len(items) < 30:
        raise ValueError("AI2D judging requires at least 30 frozen evaluation questions")
    by_id = {item["item_id"]: item for item in items}
    ids = sorted(random.Random(42).sample(sorted(by_id), 30))
    frozen = {"dataset_name": "ai2d", "size": 30, "seed": 42, "tiers": [1, 2, 3],
              "method": "Random questions from the frozen evaluation sample; independent of generation/accuracy/quality",
              "parent_selection_sha256": json_digest(read_json(root / "manifests/ai2d/test_selection.json")),
              "items": [{"item_id": key, "input_sha256": public_snapshot(root, by_id[key])["input_sha256"]} for key in ids]}
    path = root / "manifests/ai2d/test_judge_selection.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Frozen AI2D judging selection changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, frozen)
    return frozen


def freeze_scienceqa_judging(root):
    """Freeze 30 audited test questions for paired B/D judging, independent of outcomes."""
    root = Path(root).resolve()
    items = load_questions(root, "scienceqa")
    if len(items) < 30:
        raise ValueError("ScienceQA judging requires at least 30 audited test questions")
    by_id = {item["item_id"]: item for item in items}
    ids = sorted(random.Random(42).sample(sorted(by_id), 30))
    snapshots = {key: public_snapshot(root, by_id[key]) for key in ids}
    frozen = {"dataset_name": "scienceqa", "split": "test", "size": 30, "seed": 42, "tiers": [1, 2, 3],
              "method": "Random questions from the audited test set; independent of generation/accuracy/quality",
              "source_configuration": source_config(root, "scienceqa"),
              "parent_selection_sha256": json_digest(read_json(root / "manifests/scienceqa/test_final_ids.json")),
              "items": [{"item_id": key, **{field: snapshots[key][field]
                                           for field in ("input_sha256", "image_sha256")}} for key in ids]}
    path = root / "manifests/scienceqa/test_judge_selection.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Frozen ScienceQA test judging selection changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, frozen)
    return frozen
