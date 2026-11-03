"""Check answer-only Qwen predictions on audited ScienceQA training questions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from run_scienceqa_tutor import load_public_questions, question_prompt


ROOT = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 8
PROMPT_VERSION = "answer-only-v1"


def answer_prompt(item):
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
        + question_prompt(item)
    )


def load_train_questions(root, *, item_ids=None, limit=None):
    """Use only the current audited final training selection, in saved order."""
    root = Path(root).resolve()
    if item_ids and (len(set(item_ids)) != len(item_ids)
                     or any(not key.startswith("train_") for key in item_ids)):
        raise ValueError("Supply distinct train_ item IDs only.")
    if limit is not None and limit < 1:
        raise ValueError("The question limit must be positive.")
    if item_ids and limit is not None:
        raise ValueError("Choose item IDs or a question limit, not both.")
    rows = load_public_questions(root, "train", data_source="final")
    if item_ids:
        by_id = {row["item_id"]: row for row in rows}
        missing = set(item_ids) - by_id.keys()
        if missing:
            raise ValueError(f"Not in the audited training set: {', '.join(sorted(missing))}")
        rows = [by_id[key] for key in item_ids]
    elif limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No audited training questions were selected.")
    items = []
    image_root = (root / "data/final/scienceqa/images").resolve()
    for row in rows:
        if row["split"] != "train" or not row["item_id"].startswith("train_"):
            raise ValueError("The answer checker only accepts training questions.")
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(image_root) or not image.is_relative_to(root):
            raise ValueError("Training images must be inside data/final/scienceqa/images.")
        if not image.is_file():
            raise FileNotFoundError(f"Missing training image: {image}")
        item = {
            "item_id": row["item_id"], "split": "train", "image_path": str(image),
            "question": row["question"], "context": row["context"],
            "choices": [{"id": choice["id"], "text": choice["text"]}
                        for choice in row["choices"]],
        }
        answer_prompt(item)  # Validate before loading GPU weights.
        items.append(item)
    return items


def predict_answers(tutor, items, on_prediction=None):
    """One model call per item; no evidence, hints, reference access or retries."""
    predictions = []
    for item in items:
        if item["split"] != "train" or not item["item_id"].startswith("train_"):
            raise ValueError("The answer checker only accepts training questions.")
        prompt = answer_prompt(item)
        valid_ids = [choice["id"] for choice in item["choices"]]
        record = {"item_id": item["item_id"], "split": "train", "option_ids": valid_ids,
                  "raw_response": None, "predicted_option_id": None, "error": None}
        start = time.perf_counter()
        try:
            response = tutor.ask(item["image_path"], prompt, max_new_tokens=MAX_NEW_TOKENS)
            if not isinstance(response, str):
                raise TypeError("The model response must be text.")
            record["raw_response"] = response
            letter = response.strip().upper()
            record["predicted_option_id"] = letter if letter in valid_ids else None
            record["status"] = "ok" if record["predicted_option_id"] else "invalid_output"
        except Exception as error:
            record.update(status="generation_error", error=f"{type(error).__name__}: {error}")
        record["seconds"] = time.perf_counter() - start
        predictions.append(record)
        if on_prediction:
            on_prediction(record)
    return predictions


def load_train_references(root, items):
    """Read only training answer letters for offline scoring, after generation."""
    expected = {item["item_id"]: {choice["id"] for choice in item["choices"]} for item in items}
    if not expected or any(not key.startswith("train_") for key in expected):
        raise ValueError("Only training references may be scored.")
    path = Path(root) / "data/final/scienceqa/references/train.jsonl"
    references = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["item_id"]
            if key not in expected:
                continue
            gold = row["gold_option_id"]
            if key in references or gold not in expected[key]:
                raise ValueError(f"Duplicate or invalid training reference: {key}")
            references[key] = gold
    missing = expected.keys() - references.keys()
    if missing:
        raise ValueError(f"Missing training references: {', '.join(sorted(missing))}")
    return references


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


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Check the first N audited training questions (default: all).")
    selection.add_argument("--item-id", action="append", help="A train_ ID; repeat to select several questions.")
    parser.add_argument("--model-path", help="Existing checkpoint folder or model ID; defaults to the shared Qwen model.")
    parser.add_argument("--output-dir", type=Path, help="New run folder; defaults to a dated folder in outputs/answer_checks.")
    args = parser.parse_args(argv)
    try:
        items = load_train_questions(ROOT, item_ids=args.item_id, limit=args.limit)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or ROOT / "outputs/answer_checks" / run_id
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Use a new output folder: {error}")
    run = {"run_id": run_id, "split": "train", "data_source": "final",
           "purpose": "Answer-only development check; no model training or final-test evaluation",
           "prompt_version": PROMPT_VERSION, "item_ids": [item["item_id"] for item in items],
           "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
           "status": "loading_model"}
    write_json(output / "run.json", run)
    print(f"Checking {len(items)} audited training questions. Results: {output.resolve()}", flush=True)
    try:
        # Keep --help, data validation and software tests independent of GPU imports.
        from qwen_engine import IMAGE_SIZE, MODEL_ID, QwenTutor

        run.update(model_path=args.model_path or MODEL_ID, image_size=IMAGE_SIZE,
                   quantization="NF4 4-bit; double quantization; float16 compute")
        write_json(output / "run.json", run)
        tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()
        run["status"] = "generating"
        write_json(output / "run.json", run)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            def save_prediction(record):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{record['item_id']}: {record['predicted_option_id'] or record['status']}", flush=True)

            predictions = predict_answers(tutor, items, save_prediction)
        # Reference answers are first loaded here, after all model calls.
        run["status"] = "scoring"
        write_json(output / "run.json", run)
        report = score_predictions(predictions, load_train_references(ROOT, items))
        write_json(output / "result.json", report)
        run["status"] = "complete"
        write_json(output / "run.json", run)
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    summary = report["summary"]
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}")
    print(f"Invalid outputs: {summary['invalid_outputs']}; generation errors: {summary['generation_errors']}")
    print(f"Random-guess baseline for these option counts: {summary['random_guess_accuracy']:.1%}")


if __name__ == "__main__":
    main()
