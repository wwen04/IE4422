"""Read-only verification of the completed ScienceQA B format; generation uses run.py."""

from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image

from experiments.common import ROOT, digest, json_digest, load_prompt_spec, read_json, require
from experiments.data import _load_questions
from hint_prompts import render_initial_hint
from scienceqa_tutor import question_prompt

TIERS = (1, 2, 3)
MAX_NEW_TOKENS = 160
PROMPT_VERSION = "b-ui-prompt-without-evidence-v2"
SCHEMA_VERSION = "b-hint-record-v1"


def load_questions(root, dataset, *, item_ids=None, limit=None):
    """Read historical training inputs only for verification; current runs use test."""
    require(dataset == "scienceqa", "Legacy runs must use ScienceQA")
    return _load_questions(root, dataset, split="train", item_ids=item_ids, limit=limit)


def planned_records(root, items, spec, run_id):
    """Snapshot public fields explicitly and construct the full planned ID set."""
    records = {}
    for item in items:
        if item["split"] != "train" or not re.fullmatch(r"train_\d{6}", item["item_id"]):
            raise ValueError("Experiment B accepts audited training questions only.")
        image = Path(item["image_path"]).resolve()
        if not image.is_relative_to((root / "data/final/scienceqa/images").resolve()):
            raise ValueError("Use images from the audited final image folder.")
        public = {"question": item["question"], "context": item["context"],
                  "choices": [{"id": c["id"], "text": c["text"]} for c in item["choices"]],
                  "image_sha256": digest(image.read_bytes())}
        public["input_sha256"] = json_digest(public)
        public["image_path"] = image.relative_to(root).as_posix()
        for tier in TIERS:
            prompt = render_initial_hint(question_prompt(public), tier, template=spec["template"])
            hint_id = f'{run_id}:{item["item_id"]}:T{tier}'
            if hint_id in records:
                raise ValueError("Duplicate planned question/tier.")
            records[hint_id] = {"schema_version": SCHEMA_VERSION, "hint_id": hint_id,
                                "run_id": run_id, "item_id": item["item_id"], "split": "train",
                                "tier": tier, "public_inputs": public,
                                "prompt": {"version": PROMPT_VERSION, "rendered_text": prompt,
                                           "sha256": digest(prompt.encode("utf-8"))}}
    return records


def verify_run(root, folder):
    """Read and verify all source records without repairing or changing them."""
    root, folder = Path(root).resolve(), Path(folder).resolve()
    run_bytes = (folder / "run.json").read_bytes()
    hint_bytes = (folder / "hints.jsonl").read_bytes()
    run = json.loads(run_bytes)
    lines = hint_bytes.decode("utf-8").splitlines()
    require(lines and all(line.strip() for line in lines), "Empty/blank hint records found.")
    records = [json.loads(line) for line in lines]
    require(run["experiment"] == "B" and run["split"] == "train", "Expected a B training run.")
    require(run["status"] == "complete", "The run manifest is not complete.")
    require(run["tiers"] == [1, 2, 3], "Unexpected requested tiers.")
    require(run["prompt_version"] == PROMPT_VERSION and run["schema_version"] == SCHEMA_VERSION,
            "The run uses another prompt or record version.")
    require(run["generation"] == {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "Generation settings differ from the current frozen B settings.")
    dataset = read_json(root / "configs/scienceqa.json")
    require(run["dataset"] == dataset, "Dataset configuration differs from this run.")
    items = load_questions(root, "scienceqa", item_ids=run["item_ids"])
    spec = load_prompt_spec(root)
    planned = planned_records(root, items, spec, run["run_id"])
    signature = json_digest({"plans": list(planned.values()), "dataset": dataset,
                             "max_new_tokens": MAX_NEW_TOKENS})
    require(run["plan_sha256"] == signature, "Input/prompt plan fingerprint mismatch.")
    expected = len(planned)
    require(expected == len(records) == run["expected_records"] == run["completed_records"],
            "Planned, saved and manifest record counts disagree.")
    require(run["remaining_records"] == 0, "The manifest still reports remaining records.")
    image_sources, seen, statuses, tier_counts = {}, set(), Counter(), Counter()
    for row in records:
        key = row["hint_id"]
        require(key in planned and key not in seen, "Unexpected or duplicate hint ID.")
        seen.add(key)
        plan = planned[key]
        require(set(row) == set(plan) | {"hint_text", "word_count", "generation"},
                f"Unexpected record fields: {key}")
        require(all(row[field] == value for field, value in plan.items()),
                f"Public inputs, prompt or provenance mismatch: {key}")
        generation = row["generation"]
        require(set(generation) == {"status", "seconds", "error", "input_tokens", "output_tokens", "finish_reason"},
                f"Invalid generation fields: {key}")
        require(type(generation["seconds"]) in (int, float) and math.isfinite(generation["seconds"])
                and generation["seconds"] >= 0, f"Invalid timing: {key}")
        require(all(v is None or type(v) is int and v >= 0
                    for v in (generation["input_tokens"], generation["output_tokens"])),
                f"Invalid token counts: {key}")
        require(generation["finish_reason"] is None or isinstance(generation["finish_reason"], str),
                f"Invalid finish reason: {key}")
        status, text = generation["status"], row["hint_text"]
        if status in ("ok", "empty_output"):
            require(isinstance(text, str) and type(row["word_count"]) is int
                    and row["word_count"] == len(text.split())
                    and bool(text.strip()) == (status == "ok") and generation["error"] is None,
                    f"Inconsistent text/word count/status: {key}")
        else:
            error = generation["error"]
            require(status == "generation_error" and text is None and row["word_count"] is None
                    and isinstance(error, dict) and set(error) == {"type", "message"}
                    and isinstance(error["type"], str) and isinstance(error["message"], str),
                    f"Inconsistent error record: {key}")
        statuses[status] += 1
        tier_counts[row["tier"]] += 1
        public = row["public_inputs"]
        if public["image_sha256"] not in image_sources:
            path = root / public["image_path"]
            image_bytes = path.read_bytes()
            require(digest(image_bytes) == public["image_sha256"], f"Image hash mismatch: {key}")
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            image_sources[public["image_sha256"]] = path
    require(seen == set(planned), "Some question-tier pairs are missing.")
    counts = {name: statuses[name] for name in ("ok", "empty_output", "generation_error")}
    require(run["counts"] == counts, "Manifest status counts differ from saved records.")
    references = {}
    reference_path = root / "data/final/scienceqa/references/train.jsonl"
    selected = set(run["item_ids"])
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["item_id"] in selected:
            require(row["item_id"] not in references, "Duplicate reference ID.")
            references[row["item_id"]] = {field: row[field] for field in ("gold_option_id", "lecture", "solution")}
    require(set(references) == selected, "Missing answer-key references.")
    for item in items:
        ref = references[item["item_id"]]
        require(ref["gold_option_id"] in {choice["id"] for choice in item["choices"]}, "Invalid gold option.")
        require(isinstance(ref["lecture"], str) and isinstance(ref["solution"], str), "Invalid reference text.")
    pinned = read_json(root / "notebooks/colab_config.json")
    checkpoint = (run.get("model_identity", {}).get("checkpoint_marker") or {}).get("model")
    word_counts = [r["word_count"] for r in records if r["word_count"] is not None]
    report = {
        "integrity_passed": True, "quality_judged": False,
        "run_id": run["run_id"], "run_status": run["status"], "question_count": len(items),
        "expected_records": expected, "valid_records": len(records), "missing_records": 0,
        "duplicate_records": 0, "records_per_tier": dict(tier_counts), "generation_counts": counts,
        "prompt_version": run["prompt_version"], "input_and_prompt_fingerprints_match": True,
        "unique_image_contents": len(image_sources), "all_images_readable": True,
        "reference_question_count": len(references),
        "missing_lecture_count": sum(not r["lecture"].strip() for r in references.values()),
        "missing_solution_count": sum(not r["solution"].strip() for r in references.values()),
        "missing_both_explanations_count": sum(not r["lecture"].strip() and not r["solution"].strip()
                                               for r in references.values()),
        "matches_project_pinned_checkpoint": checkpoint == pinned if checkpoint is not None else None,
        "word_count_range": [min(word_counts), max(word_counts)] if word_counts else None,
        "sha256": {"source_hints": digest(hint_bytes), "source_run": digest(run_bytes),
                   "references": digest(reference_path.read_bytes()), "verified_plan": signature},
    }
    return run, records, references, image_sources, report
