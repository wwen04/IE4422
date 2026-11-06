"""Generate blinded, answer-key-free hint candidates for human review."""
import argparse
import csv
from datetime import datetime, timezone
import json
import random
from pathlib import Path

import run_scienceqa_tutor as runner
from evaluate_visual_evidence import read_question_ids
from tutor_pipeline import extract_evidence, generate_hint_candidate


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "hint_quality_validation.csv"
RATING_DIMENSIONS = (
    "factual_accuracy_1_to_4",
    "image_groundedness_1_to_4",
    "pedagogical_usefulness_1_to_4",
)
RATING_FIELDS = tuple(
    f"level_{level}_{dimension}"
    for level in (1, 2, 3)
    for dimension in (
        *RATING_DIMENSIONS,
        "answer_leakage_yes_no_unclear",
        "unsupported_claim_yes_no",
    )
) + ("reviewer", "comments")
CSV_FIELDS = (
    "item_id",
    "image_path",
    "question",
    "context",
    "choices",
    "condition_code",
    "hint_level_1",
    "hint_level_2",
    "hint_level_3",
    *RATING_FIELDS,
)


def hint_question(item):
    parts = [item["question"]]
    if item["context"].strip():
        parts.append("Supplied problem context: " + item["context"])
    return "\n\n".join(parts)


def generate_review_set(root=ROOT, limit=10, seed=4422, tutor=None,
                        output_path=DEFAULT_OUTPUT, model=None):
    if limit < 1:
        raise ValueError("Question limit must be at least 1.")
    if tutor is None:
        raise ValueError("Pass a loaded tutor to generate_review_set().")

    root = Path(root).resolve()
    ids = read_question_ids(root, "validation", "final")
    selected_ids = random.Random(seed).sample(ids, min(limit, len(ids)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_path = output_path.with_name(output_path.stem + "_key.json")
    mapping_rng = random.Random(seed ^ 0x5A17)
    rows = []
    key = {
        "warning": "Keep this file hidden from reviewers until scoring is complete.",
        "contains_answer_key": False,
        "conditions": {},
        "items": {},
    }

    for item_id in selected_ids:
        item = runner.load_question(root, "validation", item_id, "final")
        question = hint_question(item)
        evidence = extract_evidence(tutor, item["image_path"], question)
        direct_hints = [
            generate_hint_candidate(tutor, item["image_path"], question, level)
            for level in (1, 2, 3)
        ]
        guided_hints = [
            generate_hint_candidate(
                tutor, item["image_path"], question, level, evidence=evidence
            )
            for level in (1, 2, 3)
        ]

        direct_code, guided_code = (
            ("A", "B") if mapping_rng.randrange(2) == 0 else ("B", "A")
        )
        base_row = {
            "item_id": item_id,
            "image_path": str((root / item["image_path"]).resolve()),
            "question": item["question"],
            "context": item["context"],
            "choices": "\n".join(
                f'{choice["id"]}. {choice["text"]}' for choice in item["choices"]
            ),
        }
        for code, hints in ((direct_code, direct_hints), (guided_code, guided_hints)):
            rows.append({
                **base_row,
                "condition_code": code,
                **{f"hint_level_{level}": hint for level, hint in enumerate(hints, 1)},
                **{field: "" for field in RATING_FIELDS},
            })
        key["conditions"][item_id] = {
            direct_code: "direct_visual",
            guided_code: "evidence_guided",
        }
        key["items"][item_id] = {"visual_evidence": evidence}

    random.Random(seed ^ 0xBEEF).shuffle(rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    key.update({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "split": "validation",
        "seed": seed,
        "n_questions": len(selected_ids),
        "n_blinded_rows": len(rows),
    })
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "n_questions": len(selected_ids),
        "n_blinded_rows": len(rows),
        "csv_path": output_path,
        "key_path": key_path,
    }


def _rating(value, field, line_number):
    if value == "":
        return None
    try:
        rating = int(value)
    except ValueError as error:
        raise ValueError(f"CSV row {line_number}: {field} must be an integer.") from error
    if rating not in (1, 2, 3, 4):
        raise ValueError(f"CSV row {line_number}: {field} must be from 1 to 4.")
    return rating


def summarize_reviews(csv_path, key_path):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if key.get("contains_answer_key") is not False:
        raise ValueError("Review key metadata is invalid.")
    grouped = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(CSV_FIELDS)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Hint review CSV is missing required columns.")
        for line_number, row in enumerate(reader, start=2):
            item_id = row["item_id"]
            code = row["condition_code"]
            method = key.get("conditions", {}).get(item_id, {}).get(code)
            if method not in ("direct_visual", "evidence_guided"):
                raise ValueError(
                    f"CSV row {line_number}: item/condition is absent from the review key."
                )
            ratings = {
                field: _rating(row[field], field, line_number)
                for field in RATING_FIELDS
                if field.endswith(RATING_DIMENSIONS)
            }
            level_ratings = {}
            for level in (1, 2, 3):
                leakage = row[f"level_{level}_answer_leakage_yes_no_unclear"].strip().casefold()
                unsupported = row[f"level_{level}_unsupported_claim_yes_no"].strip().casefold()
                if leakage not in ("", "yes", "no", "unclear"):
                    raise ValueError(
                        f"CSV row {line_number}: invalid level-{level} answer-leakage rating."
                    )
                if unsupported not in ("", "yes", "no"):
                    raise ValueError(
                        f"CSV row {line_number}: invalid level-{level} unsupported-claim rating."
                    )
                level_ratings[level] = {
                    "leakage": leakage or None,
                    "unsupported": unsupported or None,
                }
            grouped.setdefault(method, []).append({
                **ratings,
                "levels": level_ratings,
            })

    methods = {}
    for method in ("direct_visual", "evidence_guided"):
        entries = grouped.get(method, [])
        method_summary = {"n": len(entries), "levels": {}}
        for level in (1, 2, 3):
            prefix = f"level_{level}_"
            level_summary = {}
            for dimension in RATING_DIMENSIONS:
                field = prefix + dimension
                values = [
                    entry[field] for entry in entries if entry[field] is not None
                ]
                level_summary[dimension] = (
                    round(sum(values) / len(values), 3) if values else None
                )
            leakage_values = [
                entry["levels"][level]["leakage"]
                for entry in entries
                if entry["levels"][level]["leakage"]
            ]
            rated_leakage = [value for value in leakage_values if value != "unclear"]
            level_summary["answer_leakage"] = {
                "rated_n": len(rated_leakage),
                "yes": sum(value == "yes" for value in rated_leakage),
                "unclear": sum(value == "unclear" for value in leakage_values),
            }
            unsupported_values = [
                entry["levels"][level]["unsupported"]
                for entry in entries
                if entry["levels"][level]["unsupported"]
            ]
            level_summary["unsupported_claims"] = {
                "rated_n": len(unsupported_values),
                "yes": sum(value == "yes" for value in unsupported_values),
            }
            level_summary["fully_rated_rows"] = sum(
                all(
                    entry[prefix + dimension] is not None
                    for dimension in RATING_DIMENSIONS
                )
                and entry["levels"][level]["leakage"] is not None
                and entry["levels"][level]["unsupported"] is not None
                for entry in entries
            )
            method_summary["levels"][str(level)] = level_summary
        methods[method] = method_summary

    return {
        "n_questions": key.get("n_questions"),
        "methods": methods,
        "interpretation": (
            "Human-rated hint quality only; this does not measure student answer "
            "accuracy or learning gains."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4422)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", help="Existing local model checkpoint directory.")
    parser.add_argument(
        "--summarize",
        nargs=2,
        metavar=("REVIEW_CSV", "KEY_JSON"),
        help="Summarize completed human ratings without loading a model.",
    )
    args = parser.parse_args(argv)
    if args.summarize:
        summary = summarize_reviews(*args.summarize)
        summary_path = Path(args.summarize[0]).with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Summary: {summary_path.resolve()}")
        return
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    from qwen_engine import MODEL_ID, QwenTutor

    model = args.model_path or MODEL_ID
    tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()
    result = generate_review_set(
        limit=args.limit,
        seed=args.seed,
        tutor=tutor,
        output_path=args.output,
        model=model,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    print("Review the CSV without opening its paired _key.json file.")


if __name__ == "__main__":
    main()
