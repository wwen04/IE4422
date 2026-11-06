"""Batch-compare direct Qwen answers with answers guided by visual evidence."""

import argparse
from datetime import datetime, timezone
import csv
import html
import json
import os
from pathlib import Path
import re

import scienceqa_tutor as runner
from tutor_pipeline import extract_evidence


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
EXPERIMENT = "paired_direct_vs_question_focused_visual_evidence_prompt_v2"
ALL_SPLITS = (*runner.SPLITS, "all")


def parse_choice_id(response, choices):
    """Return a valid leading option ID, or None for an unparseable response."""
    normalized = response.strip().lstrip("*").strip()
    match = re.match(
        r"^(?:(?:the\s+)?(?:correct\s+)?(?:answer|choice)(?:\s+is)?\s*:?\s*|"
        r"(?:option)\s*)?\*{0,2}([A-Z0-9]+)\*{0,2}(?=$|[\s.):,-])",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    valid_ids = {choice["id"].casefold(): choice["id"] for choice in choices}
    return valid_ids.get(match.group(1).casefold())


def read_question_ids(root, split, data_source):
    questions_path = (
        runner.dataset_root(root, data_source) / "questions" / f"{split}.jsonl"
    )
    if not questions_path.is_file():
        raise FileNotFoundError(f"Missing evaluation questions: {questions_path}")
    item_ids = []
    with questions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item_ids.append(json.loads(line)["item_id"])
    return item_ids


def answer_prompt(item, evidence=None):
    prompt = runner.question_prompt(item)
    if evidence is None:
        return f"""{prompt}

Select the single correct answer. Output exactly one option ID and no explanation."""
    return f"""{prompt}

Proposed visual observations (model-produced; may be incomplete or wrong):
{evidence}
Inspect the image yourself. Check each proposed observation against the image and question;
correct or discard anything that does not match. Do not treat the observations as ground truth.
Then solve the question using the verified image details, supplied context, and relevant science.
For comparisons or calculations, identify the needed quantities and compare or calculate them
before matching the result to an option. Do not guess unreadable image details.
Select the single correct answer. Output exactly one option ID and no explanation."""


def summarize(items):
    total = len(items)
    if total == 0:
        raise ValueError("No questions were evaluated.")
    baseline_correct = sum(item["baseline_correct"] for item in items)
    guided_correct = sum(item["evidence_guided_correct"] for item in items)
    both_correct = sum(
        item["baseline_correct"] and item["evidence_guided_correct"]
        for item in items
    )
    baseline_only = sum(
        item["baseline_correct"] and not item["evidence_guided_correct"]
        for item in items
    )
    guided_only = sum(
        item["evidence_guided_correct"] and not item["baseline_correct"]
        for item in items
    )
    both_incorrect = total - both_correct - baseline_only - guided_only
    baseline_accuracy = baseline_correct / total * 100
    guided_accuracy = guided_correct / total * 100
    return {
        "n": total,
        "baseline": {
            "label": "Default direct-answer prompt",
            "correct": baseline_correct,
            "accuracy_percent": round(baseline_accuracy, 2),
        },
        "evidence_guided": {
            "label": "Project visual-evidence prompt",
            "correct": guided_correct,
            "accuracy_percent": round(guided_accuracy, 2),
        },
        "delta_percentage_points": round(guided_accuracy - baseline_accuracy, 2),
        "paired": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "evidence_guided_only_correct": guided_only,
            "both_incorrect": both_incorrect,
        },
    }


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_checkpoint(path, metadata, expected_items):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return {}
    content = checkpoint_path.read_bytes()
    lines = content.splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Evaluation checkpoint is empty: {checkpoint_path}")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation checkpoint header is invalid: {checkpoint_path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Evaluation checkpoint header is invalid: {checkpoint_path}")
    if any(header.get(key) != value for key, value in metadata.items()):
        raise ValueError(
            f"Checkpoint settings or question list differ from this run: {checkpoint_path}. "
            "Use --no-resume to start a fresh evaluation."
        )

    expected = set(expected_items)
    results = {}
    valid_content = lines[0] if lines[0].endswith(b"\n") else lines[0] + b"\n"
    for index, line in enumerate(lines[1:], start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            is_incomplete_tail = index == len(lines) - 1 and not line.endswith(b"\n")
            if is_incomplete_tail:
                checkpoint_path.write_bytes(valid_content)
                break
            raise ValueError(
                f"Evaluation checkpoint row {index + 1} is invalid: {checkpoint_path}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation checkpoint row {index + 1} is invalid: {checkpoint_path}"
            )
        key = (item.get("split"), item.get("item_id"))
        if key not in expected or key in results:
            raise ValueError(
                f"Evaluation checkpoint contains an unexpected or duplicate item: {key}"
            )
        results[key] = item
        valid_content += line if line.endswith(b"\n") else line + b"\n"
    if content != valid_content:
        checkpoint_path.write_bytes(valid_content)
    return results


def initialize_checkpoint(path, metadata):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, checkpoint_path)


def append_checkpoint_item(path, item):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()


def evaluate(root=ROOT, split="validation", data_source="final", limit=None,
             tutor=None, model=DEFAULT_MODEL_ID, resume_path=None,
             on_progress=None):
    if split not in ALL_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if data_source not in ("processed", "final"):
        raise ValueError(f"Unknown data source: {data_source}")
    if limit is not None and limit < 1:
        raise ValueError("Question limit must be at least 1.")
    if tutor is None:
        raise ValueError("Pass a loaded tutor to evaluate().")

    root = Path(root).resolve()
    splits = runner.SPLITS if split == "all" else (split,)
    split_ids = {name: read_question_ids(root, name, data_source) for name in splits}
    available_counts = {name: len(ids) for name, ids in split_ids.items()}
    if limit is not None:
        split_ids = {name: ids[:limit] for name, ids in split_ids.items()}
    expected_items = [
        (name, item_id)
        for name in splits
        for item_id in split_ids[name]
    ]
    if not expected_items:
        raise ValueError(f"No {split} questions are available for evaluation.")

    resume_metadata = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "model": model,
        "split": split,
        "data_source": data_source,
        "limit": limit,
        "expected_items": [[name, item_id] for name, item_id in expected_items],
    }
    results_by_key = (
        load_checkpoint(resume_path, resume_metadata, expected_items)
        if resume_path is not None else {}
    )
    if resume_path is not None and not Path(resume_path).exists():
        initialize_checkpoint(resume_path, resume_metadata)

    total = len(expected_items)
    for index, (item_split, item_id) in enumerate(expected_items, start=1):
        key = (item_split, item_id)
        if key not in results_by_key:
            item = runner.load_question(root, item_split, item_id, data_source)
            reference = runner.load_reference(root, item_split, item_id, data_source)
            question = runner.question_prompt(item)
            baseline_response = tutor.ask(
                item["image_path"], answer_prompt(item), max_new_tokens=32
            )
            evidence = extract_evidence(tutor, item["image_path"], question)
            guided_response = tutor.ask(
                item["image_path"],
                answer_prompt(item, evidence),
                max_new_tokens=32,
            )
            baseline_prediction = parse_choice_id(baseline_response, item["choices"])
            guided_prediction = parse_choice_id(guided_response, item["choices"])
            results_by_key[key] = {
                "split": item_split,
                "item_id": item_id,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": baseline_prediction == reference["gold_option_id"],
                "baseline_response": baseline_response,
                "evidence_guided_prediction": guided_prediction,
                "evidence_guided_correct": guided_prediction == reference["gold_option_id"],
                "evidence_guided_response": guided_response,
                "visual_evidence": evidence,
            }
            if resume_path is not None:
                append_checkpoint_item(resume_path, results_by_key[key])
        if on_progress is not None:
            item_result = results_by_key[key]
            on_progress({
                "completed": index,
                "total": total,
                "split": item_split,
                "item_id": item_id,
                "baseline_correct": item_result["baseline_correct"],
                "evidence_guided_correct": item_result["evidence_guided_correct"],
            })

    results = [results_by_key[key] for key in expected_items]
    by_split = {
        name: {
            "available_n": available_counts[name],
            **summarize([item for item in results if item["split"] == name]),
        }
        for name in splits
        if any(item["split"] == name for item in results)
    }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "split": split,
        "data_source": data_source,
        "available_n": sum(available_counts.values()),
        "limit_per_split": limit,
        "protocol": (
            "Same frozen model, question, choices, and image. Baseline answers "
            "directly; project condition extracts compact, question-focused "
            "visual observations first and asks the model to verify them against "
            "the image before solving. No fine-tuning."
        ),
        **summarize(results),
        "by_split": by_split,
        "items": results,
    }


def write_csv_report(path, report):
    fields = (
        "split", "item_id", "baseline_prediction", "baseline_correct",
        "baseline_response", "evidence_guided_prediction",
        "evidence_guided_correct", "evidence_guided_response", "visual_evidence",
    )
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields}
                         for item in report["items"])


def write_html_report(path, report):
    def esc(value):
        return html.escape(str(value if value is not None else ""), quote=True)

    def accuracy_rows():
        splits = [("Overall", report), *report["by_split"].items()]
        return "\n".join(
            "<tr><th>{}</th><td>{} / {} ({:.2f}%)</td>"
            "<td>{} / {} ({:.2f}%)</td><td>{:+.2f} pp</td></tr>".format(
                esc(label),
                values["baseline"]["correct"], values["n"],
                values["baseline"]["accuracy_percent"],
                values["evidence_guided"]["correct"], values["n"],
                values["evidence_guided"]["accuracy_percent"],
                values["delta_percentage_points"],
            )
            for label, values in splits
        )

    rows = []
    for item in report["items"]:
        rows.append(
            "<tr>"
            f"<th>{esc(item['split'])}<br>{esc(item['item_id'])}</th>"
            f"<td><b class=\"{'correct' if item['baseline_correct'] else 'incorrect'}\">"
            f"{'Correct' if item['baseline_correct'] else 'Incorrect'}"
            f"{' · ' + esc(item['baseline_prediction']) if item['baseline_prediction'] else ''}</b>"
            f"<pre>{esc(item['baseline_response'])}</pre></td>"
            f"<td><b class=\"{'correct' if item['evidence_guided_correct'] else 'incorrect'}\">"
            f"{'Correct' if item['evidence_guided_correct'] else 'Incorrect'}"
            f"{' · ' + esc(item['evidence_guided_prediction']) if item['evidence_guided_prediction'] else ''}</b>"
            f"<pre>{esc(item['evidence_guided_response'])}</pre></td>"
            f"<td><pre>{esc(item['visual_evidence'])}</pre></td>"
            "</tr>"
        )
    paired = report["paired"]
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Visual evidence accuracy comparison</title><style>
body{{font:15px/1.5 Segoe UI,Arial,sans-serif;color:#202a25;background:#f6f7f3;margin:0;padding:32px}}
main{{max-width:1200px;margin:auto}}h1{{font-size:30px}}.muted{{color:#68746d}}
section{{background:white;border:1px solid #e4e9e1;border-radius:12px;padding:20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e8ece6}}
thead{{background:#f4f7f2}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-width:380px;font-size:12px;color:#46534a}}
.correct{{color:#245c43}}.incorrect{{color:#a14c37}}.bar{{height:10px;border-radius:8px;background:#e9eee8;overflow:hidden}}
.bar i{{display:block;height:100%;background:#245c43}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:700px){{body{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Default model vs visual-evidence prompting</h1>
<p class="muted">{esc(report['model'])} · {esc(report['split'])} · {report['n']} questions · {esc(report['created_at'])}</p>
<section><h2>Accuracy summary</h2><table><thead><tr><th>Split</th><th>Default direct answer</th>
<th>Project visual evidence</th><th>Change</th></tr></thead><tbody>{accuracy_rows()}</tbody></table>
<div class="grid">
<div><p>Default model: {report['baseline']['accuracy_percent']:.2f}%</p>
<div class="bar"><i style="width:{report['baseline']['accuracy_percent']}%"></i></div></div>
<div><p>Project: {report['evidence_guided']['accuracy_percent']:.2f}%</p>
<div class="bar"><i style="width:{report['evidence_guided']['accuracy_percent']}%"></i></div></div></div>
<p>Paired outcomes: {paired['evidence_guided_only_correct']} improved ·
{paired['baseline_only_correct']} regressed · {paired['both_correct']} both correct ·
{paired['both_incorrect']} both incorrect.</p>
<p class="muted">Answer-choice accuracy on the evaluated questions only; this does not establish
hint quality, learning gains, or statistical significance.</p></section>
<section><h2>Per-question outputs</h2><div style="overflow:auto"><table><thead><tr>
<th>Question</th><th>Default output</th><th>Project output</th><th>Extracted visual evidence</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
</main></body></html>"""
    Path(path).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=ALL_SPLITS, default="all",
        help="Evaluate one split or all finalized splits (default: all).",
    )
    parser.add_argument("--data-source", choices=("processed", "final"), default="final")
    parser.add_argument(
        "--limit", type=int,
        help="Evaluate only the first N IDs in each selected split.",
    )
    parser.add_argument("--model-path", help="Existing local model checkpoint directory.")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/accuracy_comparison.json",
        help="JSON report path; matching .html and .csv reports are also written.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore an existing checkpoint and start the batch from the beginning.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    from qwen_engine import MODEL_ID, QwenTutor

    model = args.model_path or MODEL_ID
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.jsonl")
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()

    def report_progress(progress):
        baseline = "correct" if progress["baseline_correct"] else "incorrect"
        guided = "correct" if progress["evidence_guided_correct"] else "incorrect"
        print(
            f"[{progress['completed']}/{progress['total']}] "
            f"{progress['split']}/{progress['item_id']}: "
            f"default {baseline}; visual evidence {guided}",
            flush=True,
        )

    report = evaluate(
        root=ROOT,
        split=args.split,
        data_source=args.data_source,
        limit=args.limit,
        tutor=tutor,
        model=model,
        resume_path=checkpoint_path,
        on_progress=report_progress,
    )
    atomic_write_json(args.output, report)
    write_csv_report(args.output.with_suffix(".csv"), report)
    write_html_report(args.output.with_suffix(".html"), report)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in report.items() if key != "items"},
                     ensure_ascii=False, indent=2))
    print(f"JSON report: {args.output.resolve()}")
    print(f"CSV report:  {args.output.with_suffix('.csv').resolve()}")
    print(f"HTML report: {args.output.with_suffix('.html').resolve()}")


if __name__ == "__main__":
    main()
