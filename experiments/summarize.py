"""Join validated Part A judgments and report one dataset/condition at a time."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from experiments.data import ROOT, freeze_ai2d_judging, read_json, read_jsonl
from experiments.judge_validate import read_judgment
from experiments.prepare_judging import verify_run
from experiments.common import digest, write_json


def good_hint(judgment):
    return (judgment["tier_compliance"]["verdict"] == "yes"
            and judgment["visual_grounding"]["verdict"] == "yes"
            and judgment["unsupported_visual_claims"]["count"] == 0
            and judgment["scientific_correctness"]["verdict"] == "yes"
            and judgment["steers_toward_wrong_answer"]["verdict"] == "no"
            and judgment["answer_disclosure"]["score"] == 0)


def metrics(rows):
    judgments = [row["judgment"] for row in rows]
    size = len(judgments)
    if not size:
        return {"judged_hints": 0}
    return {"judged_hints": size, "good_hint_rate": sum(map(good_hint, judgments)) / size,
            **{key + "_yes_rate": sum(j[key]["verdict"] == "yes" for j in judgments) / size
               for key in ("tier_compliance", "visual_grounding", "scientific_correctness", "steers_toward_wrong_answer")},
            "mean_unsupported_visual_claims": sum(j["unsupported_visual_claims"]["count"] for j in judgments) / size,
            "disclosure_distribution": {str(score): sum(j["answer_disclosure"]["score"] == score for j in judgments) / size for score in (0, 1, 2)},
            "mean_actionability": sum(j["actionability"]["score"] for j in judgments) / size}


def summarize(root, run_folder, package, judgments_folder, output, *, answers_run=None):
    root, run_folder, package, judgments_folder, output = map(Path, (root, run_folder, package, judgments_folder, output))
    run, records, references, _, _ = verify_run(root, run_folder)
    by_hint = {row["hint_id"]: row for row in records}
    mapping = read_jsonl(package / "record_map_PRIVATE.jsonl")
    ids, pairs = set(), set()
    joined, pending, generated = [], [], Counter()
    expected_files = set()
    for entry in mapping:
        pair = (entry["item_id"], entry["tier"])
        if not re.fullmatch(r"r_\d{6}", entry["record_id"]) or entry["record_id"] in ids or pair in pairs:
            raise ValueError("Duplicate case/question-tier in private map")
        ids.add(entry["record_id"])
        pairs.add(pair)
        source = by_hint.get(entry["hint_id"])
        if (source is None or source["item_id"] != entry["item_id"] or source["tier"] != entry["tier"]
                or source["generation"]["status"] != entry["generation_status"]
                or source["public_inputs"]["image_sha256"] != entry["image_sha256"]
                or source["public_inputs"]["input_sha256"] != entry.get("input_sha256")
                or entry["hint_sha256"] != (digest(source["hint_text"].encode("utf-8")) if source["hint_text"] is not None else None)):
            raise ValueError("Private map differs from the verified generation run")
        generated[entry["generation_status"]] += 1
        if entry["generation_status"] != "ok":
            continue
        filename = f"judgment_{entry['record_id']}.json"
        if entry["expected_response_file"] != filename:
            raise ValueError("Unexpected judgment filename in private map")
        case_folder = package / "attachments/cases" / entry["record_id"]
        cases = read_jsonl(case_folder / "hints.jsonl")
        if len(cases) != 1:
            raise ValueError("A judging case must contain exactly one hint")
        case = cases[0]
        ref = references[entry["item_id"]]
        gold_text = next(c["text"] for c in source["public_inputs"]["choices"] if c["id"] == ref["gold_option_id"])
        if (case["record_id"] != entry["record_id"] or case["hint_text"] != source["hint_text"]
                or case["requested_tier"] != entry["tier"]
                or case["image_sha256"] != entry["image_sha256"]
                or any(case[key] != source["public_inputs"][key] for key in ("question", "context", "choices"))
                or case["gold_reference"] != {"option_id": ref["gold_option_id"], "option_text": gold_text,
                                               "lecture": ref["lecture"], "solution": ref["solution"]}
                or Path(case["image_file"]).name != case["image_file"]
                or digest((case_folder / case["image_file"]).read_bytes()) != entry["image_sha256"]):
            raise ValueError("Judge attachments differ from the verified source inputs/hint/references")
        expected_files.add(filename)
        path = judgments_folder / filename
        try:
            judgment = read_judgment(path)
        except (ValueError, OSError, KeyError) as error:
            pending.append({"record_id": entry["record_id"], "status": "missing" if not path.exists() else "invalid", "reason": str(error)})
            continue
        joined.append({"record_id": entry["record_id"], "item_id": entry["item_id"], "tier": entry["tier"],
                       "hint_id": entry["hint_id"], "judgment": judgment})
    question_ids = {item_id for item_id, _ in pairs}
    if not mapping or pairs != {(key, tier) for key in question_ids for tier in (1, 2, 3)}:
        raise ValueError("The private map must preserve all three tiers for every selected question")
    unexpected = sorted(path.name for path in judgments_folder.glob("judgment_*.json") if path.name not in expected_files)
    if unexpected:
        raise ValueError(f"Judgment files do not belong to this package: {unexpected[:5]}")
    dataset = run.get("dataset_name", "scienceqa")
    if dataset == "ai2d" and question_ids != {row["item_id"] for row in freeze_ai2d_judging(root)["items"]}:
        raise ValueError("AI2D package differs from the frozen 30-question judging selection")
    summary = {"dataset_name": dataset, "experiment": run["experiment"], "run_id": run["run_id"],
               "planned_questions": len(question_ids), "planned_hints": len(mapping),
               "generation_counts": dict(generated), "valid_judgments": len(joined), "pending_judgments": pending,
               "metric_denominator": "Valid judgments only; coverage and generation failures are reported separately",
               "overall": metrics(joined), "by_tier": {str(t): metrics([row for row in joined if row["tier"] == t]) for t in (1, 2, 3)},
               "good_hint_rate_planned": sum(good_hint(row["judgment"]) for row in joined) / len(mapping) if not pending else None,
               "good_hint_rate_planned_definition": "Good judged hints / all planned hints, including failed/empty generation; unavailable while judgments are pending",
               "judging_selection_sha256": digest((package / "record_map_PRIVATE.jsonl").read_bytes()),
               "source_hints_sha256": digest((run_folder / "hints.jsonl").read_bytes()),
               "notes": ["Metrics describe this dataset and condition only; no pooled ScienceQA/AI2D score is produced.",
                         "No confidence intervals or claims of independent hints: tiers share questions and may share diagrams."]}
    if answers_run:
        answers_run = Path(answers_run)
        answer_manifest = read_json(answers_run / "run.json")
        answer_report = read_json(answers_run / "result.json")
        if (answer_manifest.get("dataset_name") != dataset or answer_manifest["experiment"] != "A"
                or answer_manifest["status"] != "complete"
                or answer_manifest["dataset"] != run["dataset"]
                or answer_manifest.get("model_identity") != run.get("model_identity")
                or answer_manifest.get("resolved_revision") != run.get("resolved_revision")
                or answer_manifest.get("image_size") != run.get("image_size")):
            raise ValueError("Experiment A must match the dataset, checkpoint and image settings")
        answers = answer_report["results"]
        if len({row["item_id"] for row in answers}) != len(answers):
            raise ValueError("Duplicate Experiment A result")
        by_id = {row["item_id"]: row for row in answers}
        if not question_ids <= set(by_id):
            raise ValueError("Experiment A is missing judging questions")
        current = {row["item_id"]: row["public_inputs"] for row in records}
        for key in question_ids:
            row = by_id[key]
            expected_correct = row["generation"]["status"] == "ok" and row["predicted_option_id"] == references[key]["gold_option_id"]
            if (row["public_inputs"]["input_sha256"] != current[key]["input_sha256"]
                    or row["gold_option_id"] != references[key]["gold_option_id"] or row["correct"] != expected_correct):
                raise ValueError("Experiment A inputs, gold answer or correctness differ")
        summary["by_experiment_a_correctness"] = {
            "correct": metrics([row for row in joined if by_id[row["item_id"]]["correct"]]),
            "incorrect_or_failed": metrics([row for row in joined if not by_id[row["item_id"]]["correct"]])}
        summary["experiment_a_result_sha256"] = digest((answers_run / "result.json").read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", summary)
    (output / "judgments.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in joined), encoding="utf-8")
    lines = [f"# {dataset} Experiment {run['experiment']} judging summary", "",
             f"Questions: {len(question_ids)}. Planned hints: {len(mapping)}. Valid judgments: {len(joined)}. Pending: {len(pending)}.", "",
             "Rates below use valid judgments only; missing judgments are not silently scored as failures or successes.", "",
             "| Tier | Judged hints | Good hint rate | Mean actionability |", "|---|---:|---:|---:|"]
    for tier in (1, 2, 3):
        values = summary["by_tier"][str(tier)]
        good = f"{values['good_hint_rate']:.1%}" if values["judged_hints"] else "unavailable"
        action = f"{values['mean_actionability']:.2f}" if values["judged_hints"] else "unavailable"
        lines.append(f"| {tier} | {values['judged_hints']} | {good} | {action} |")
    lines.extend(["", "See metrics.json for component rates, disclosure distribution, planned-denominator good-hint rate and coverage. Datasets and conditions are kept separate. No new judgments were generated.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--judgments-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--answers-run-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(ROOT, args.run_dir, args.package_dir or args.run_dir / "judge_package",
                           args.judgments_dir or args.run_dir / "judgments",
                           args.output_dir or args.run_dir / "summaries" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
                           answers_run=args.answers_run_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("dataset_name", "experiment", "planned_hints", "valid_judgments")}, indent=2))


if __name__ == "__main__":
    main()
