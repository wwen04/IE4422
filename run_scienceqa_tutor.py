"""Run the local Qwen tutor on a public question exported by prepare_data.py."""

import argparse
import json
from pathlib import Path

from tutor_pipeline import (
    run_answered_pipeline,
    run_initial_hint,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def dataset_root(root, data_source):
    if data_source not in ("processed", "final"):
        raise ValueError(f"Unknown data source: {data_source}")
    return Path(root) / "data" / data_source / "scienceqa"


def load_question(root, split, question_id, data_source="processed"):
    """Read only public questions; never open the reference-answer exports."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(root).resolve()
    source = dataset_root(root, data_source) / "questions" / f"{split}.jsonl"
    if not source.is_file():
        command = "prepare_data.py" if data_source == "processed" else "ui_server.py"
        raise FileNotFoundError(
            f"Missing {data_source} questions: {source}. Run {command} first."
        )
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["item_id"] == question_id:
                if row["split"] != split:
                    raise ValueError("Question split does not match the requested split")
                image = (root / row["image_path"]).resolve()
                if not image.is_relative_to(root.resolve()):
                    raise ValueError("Prepared image path must stay inside the repository")
                if not image.is_file():
                    raise FileNotFoundError(f"Missing prepared image: {image}")
                # Explicitly select public fields even if extra fields are present.
                return {
                    "item_id": row["item_id"], "split": row["split"],
                    "question": row["question"], "context": row["context"],
                    "choices": [{"id": choice["id"], "text": choice["text"]}
                                for choice in row["choices"]],
                    "image_path": str(image),
                }
    raise ValueError(f"Question {question_id!r} was not found in the prepared {split} split")


def load_reference(root, split, question_id, data_source="processed"):
    """Load answer-key material for checking only after a student submits."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(root).resolve()
    source = dataset_root(root, data_source) / "references" / f"{split}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {data_source} answer references: {source}. "
            f"Run {'prepare_data.py' if data_source == 'processed' else 'ui_server.py'} first."
        )
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["item_id"] == question_id:
                return {
                    "item_id": row["item_id"],
                    "gold_option_id": row["gold_option_id"],
                    "lecture": row["lecture"],
                    "solution": row["solution"],
                }
    raise ValueError(
        f"Reference for {question_id!r} was not found in the {split} split"
    )


def question_prompt(item):
    parts = [item["question"]]
    if item["context"].strip():
        parts.append("Supplied problem context: " + item["context"])
    parts.append("Options:\n" + "\n".join(
        f'{choice["id"]}. {choice["text"]}' for choice in item["choices"]))
    return "\n\n".join(parts)


def run_question(tutor, item, student_answer, explanation, hint_level=1,
                 on_stage=None):
    """Generate a linked result using a model that can serve multiple questions."""
    result = run_pipeline(tutor, item["image_path"], question_prompt(item),
                          student_answer, explanation, hint_level, on_stage)
    return {"item_id": item["item_id"], "split": item["split"], **result}


def run_hint(tutor, item, hint_level=1, on_stage=None):
    result = run_initial_hint(
        tutor, item["image_path"], question_prompt(item), hint_level, on_stage
    )
    return {"item_id": item["item_id"], "split": item["split"], **result}


def run_checked_question(tutor, item, reference, student_answer_id,
                         explanation, hint_level=1, on_stage=None):
    choice = next(
        (choice for choice in item["choices"] if choice["id"] == student_answer_id),
        None,
    )
    if choice is None:
        raise ValueError(f"Unknown answer choice: {student_answer_id}")
    reference_text = "\n\n".join(filter(None, (
        reference["lecture"], reference["solution"]
    )))
    result = run_answered_pipeline(
        tutor=tutor,
        image=item["image_path"],
        question=question_prompt(item),
        student_answer=student_answer_id,
        student_explanation=explanation,
        answer_text=f'{choice["id"]}. {choice["text"]}',
        correct_answer_id=reference["gold_option_id"],
        reference=reference_text,
        hint_level=hint_level,
        on_stage=on_stage,
    )
    return {"item_id": item["item_id"], "split": item["split"], **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--item-id", required=True, help="Prepared ID, e.g. validation_000003")
    parser.add_argument("--answer", required=True, help="Student response, not a reference answer")
    parser.add_argument("--explanation", required=True, help="Student's own reasoning")
    parser.add_argument("--hint-level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scienceqa_tutor.json")
    args = parser.parse_args(argv)

    try:
        item = load_question(ROOT, args.split, args.item_id)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))
    # Keep --help, data loading and unit tests independent of GPU imports.
    from qwen_engine import QwenTutor

    tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()
    result = run_question(tutor, item, args.answer, args.explanation, args.hint_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
