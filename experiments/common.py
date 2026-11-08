"""Shared prompts, scoring, fingerprints and run metadata; no dataset-specific runners."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

from hint_prompts import INITIAL_HINT_TIERS, load_initial_hint_template
from scienceqa_tutor import question_prompt

ROOT = Path(__file__).resolve().parents[1]

def digest(value):
    return hashlib.sha256(value).hexdigest()

def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8"))

def write_json(path, value):
    """Replace the manifest only after the new JSON has been fully written."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load_prompt_spec(root):
    """Use the UI's wording and tier descriptions; never read the judge rubric."""
    template = load_initial_hint_template(root)
    return {"template": template, "tier_instructions": dict(INITIAL_HINT_TIERS),
            "template_sha256": digest(template.encode("utf-8")),
            "shared_tiers_sha256": json_digest(INITIAL_HINT_TIERS)}

def runtime_versions():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "datasets", "huggingface-hub", "bitsandbytes"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}

def model_identity(model_path):
    folder = Path(model_path)
    if not folder.is_dir():
        return {"model_id": model_path}
    marker = folder / "colab_checkpoint.json"
    return {"checkpoint_marker": json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None,
            "config_sha256": digest((folder / "config.json").read_bytes()) if (folder / "config.json").is_file() else None,
            "files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in sorted(folder.glob("*.safetensors"))} if not marker.is_file() else None}

def answer_prompt(item, evidence=None):
    """Select public fields explicitly; references never enter this prompt."""
    ids = [choice["id"] for choice in item["choices"]]
    if (len(ids) < 2 or len(set(ids)) != len(ids)
            or any(not isinstance(key, str) or len(key) != 1 or key not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   for key in ids)):
        raise ValueError(f"Invalid option letters for {item['item_id']}")
    return (
        "Answer the multiple-choice science question using the image and supplied text.\n"
        f"Return exactly one option letter ({', '.join(ids)}). "
        "Do not include an explanation, punctuation, or any other text.\n\n"
        + (f"Visual evidence (unverified): {evidence}\n"
           "Check these model-produced observations against the image; they may contain errors.\n\n"
           if evidence is not None else "")
        + question_prompt(item)
    )

def score_predictions(predictions, references):
    """Keep invalid responses and generation failures in the accuracy denominator."""
    if not predictions:
        raise ValueError("No predictions to score.")
    results = []
    for prediction in predictions:
        gold = references[prediction["item_id"]]
        correct = prediction["status"] == "ok" and prediction["predicted_option_id"] == gold
        results.append({**prediction, "gold_option_id": gold, "correct": correct})
    correct = sum(row["correct"] for row in results)
    invalid = sum(row["status"] == "invalid_output" for row in results)
    failed = sum(row["status"] == "generation_error" for row in results)
    total = len(results)
    summary = {"total": total, "correct": correct, "incorrect": total - correct - invalid - failed,
               "invalid_outputs": invalid, "generation_errors": failed,
               "accuracy": correct / total,
               "random_guess_accuracy": sum(1 / len(row["option_ids"]) for row in results) / total}
    return {"summary": summary, "results": results}
