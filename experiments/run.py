"""Shared no-attempt batch runner for A (answers), B (hints) and D (evidence + hints)."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from experiments.data import ROOT, load_questions, load_references, public_snapshot, read_json, source_config
from experiments.common import answer_prompt, score_predictions, digest, json_digest, load_prompt_spec, model_identity, runtime_versions, write_json
from hint_prompts import render_initial_hint
from scienceqa_tutor import question_prompt
from tutor_pipeline import EVIDENCE_MAX_NEW_TOKENS, evidence_prompt, extract_evidence


SCHEMA_VERSION = "no-attempt-batch-v1"
VERSIONS = {"A": "answer-only-v1", "B": "b-ui-prompt-without-evidence-v2", "D": "d-ui-prompt-with-evidence-v2"}
OUTPUT_FIELDS = {"hint_text", "word_count", "raw_response", "predicted_option_id", "generation", "evidence", "effective_prompt"}


def settings(experiment):
    return {"do_sample": False, "max_new_tokens": 8 if experiment == "A" else 160,
            "evidence_max_new_tokens": EVIDENCE_MAX_NEW_TOKENS if experiment == "D" else None,
            "evidence_reuse": False}


def make_plans(root, dataset, experiment, items, spec, run_id):
    plans = {}
    for item in items:
        public = public_snapshot(root, item)
        question = question_prompt(public)
        for tier in ((None,) if experiment == "A" else (1, 2, 3)):
            prompt = (answer_prompt(item) if experiment == "A" else
                      render_initial_hint(question, tier, None if experiment == "B" else
                                          "<evidence generated in the first pass>", template=spec["template"]))
            key = f"{run_id}:{item['item_id']}:{'A' if tier is None else 'T' + str(tier)}"
            plans[key] = {"schema_version": SCHEMA_VERSION, "hint_id": key, "run_id": run_id,
                          "dataset_name": dataset, "experiment": experiment, "item_id": item["item_id"],
                          "split": item["split"], "tier": tier, "public_inputs": public,
                          "prompt": {"version": VERSIONS[experiment], "rendered_text": prompt,
                                     "sha256": digest(prompt.encode("utf-8"))},
                          "evidence_prompt": evidence_prompt(public) if experiment == "D" else None}
    return plans


def generate_record(tutor, root, plan, spec):
    """No gold references, learner attempts, quality retries or cross-tier history."""
    row = {**plan, "hint_text": None, "word_count": None, "raw_response": None,
           "predicted_option_id": None, "evidence": None, "effective_prompt": None,
           "generation": {"status": "generation_error", "seconds": 0, "error": None,
                          "input_tokens": None, "output_tokens": None, "finish_reason": None}}
    started = time.perf_counter()
    try:
        image = str(Path(root) / plan["public_inputs"]["image_path"])
        if plan["experiment"] == "D":
            evidence = {"text": None, "seconds": 0, "status": "generation_error", "error": None}
            row["evidence"] = evidence
            evidence_started = time.perf_counter()
            try:
                text = extract_evidence(tutor, image, plan["public_inputs"])
                if not isinstance(text, str):
                    raise TypeError("Evidence response must be text")
                evidence.update(text=text, status="ok" if text.strip() else "empty_output")
            except Exception as error:
                evidence["error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                evidence["seconds"] = time.perf_counter() - evidence_started
            if evidence["status"] != "ok":
                raise ValueError("Empty evidence output; the dependent hint was not generated")
            prompt = render_initial_hint(question_prompt(plan["public_inputs"]), plan["tier"],
                                         evidence["text"], template=spec["template"])
        else:
            prompt = plan["prompt"]["rendered_text"]
        row["effective_prompt"] = prompt
        response = tutor.ask(image, prompt, max_new_tokens=settings(plan["experiment"])["max_new_tokens"])
        if not isinstance(response, str):
            raise TypeError("Model response must be text")
        if plan["experiment"] == "A":
            option = response.strip().upper()
            row["raw_response"] = response
            row["predicted_option_id"] = option if option in {c["id"] for c in plan["public_inputs"]["choices"]} else None
            row["generation"]["status"] = "ok" if row["predicted_option_id"] else "invalid_output"
        else:
            row.update(hint_text=response, word_count=len(response.split()))
            row["generation"]["status"] = "ok" if response.strip() else "empty_output"
    except Exception as error:
        row["generation"]["error"] = {"type": type(error).__name__, "message": str(error)}
    row["generation"]["seconds"] = time.perf_counter() - started
    return row


def validate_record(row, plan, spec):
    if (not isinstance(row, dict) or set(row) != set(plan) | OUTPUT_FIELDS
            or any(row[key] != value for key, value in plan.items())):
        raise ValueError("Saved record inputs, prompts or provenance changed")
    generation = row["generation"]
    keys = {"status", "seconds", "error", "input_tokens", "output_tokens", "finish_reason"}
    if (not isinstance(generation, dict) or set(generation) != keys
            or type(generation["seconds"]) not in (int, float)
            or not math.isfinite(generation["seconds"]) or generation["seconds"] < 0
            or any(value is not None for value in (generation["input_tokens"], generation["output_tokens"], generation["finish_reason"]))):
        raise ValueError("Invalid generation metadata")
    status = generation["status"]
    if status == "generation_error":
        error = generation["error"]
        if (not isinstance(error, dict) or set(error) != {"type", "message"}
                or any(not isinstance(value, str) for value in error.values())
                or any(row[key] is not None for key in ("hint_text", "word_count", "raw_response", "predicted_option_id"))):
            raise ValueError("Invalid error outcome")
    elif generation["error"] is not None:
        raise ValueError("Successful outcome contains an error")
    elif plan["experiment"] == "A":
        response = row["raw_response"]
        if not isinstance(response, str):
            raise ValueError("Missing answer response")
        answer = response.strip().upper()
        predicted = answer if answer in {c["id"] for c in plan["public_inputs"]["choices"]} else None
        if (row["predicted_option_id"] != predicted or status != ("ok" if predicted else "invalid_output")
                or row["hint_text"] is not None or row["word_count"] is not None):
            raise ValueError("Answer parsing/status mismatch")
    else:
        text = row["hint_text"]
        if (not isinstance(text, str) or type(row["word_count"]) is not int or row["word_count"] != len(text.split())
                or status != ("ok" if text.strip() else "empty_output")
                or row["raw_response"] is not None or row["predicted_option_id"] is not None):
            raise ValueError("Hint text/status mismatch")
    if plan["experiment"] == "D":
        evidence = row["evidence"]
        if (not isinstance(evidence, dict) or set(evidence) != {"text", "seconds", "status", "error"}
                or type(evidence["seconds"]) not in (int, float) or not math.isfinite(evidence["seconds"])
                or not 0 <= evidence["seconds"] <= generation["seconds"]):
            raise ValueError("Invalid evidence metadata")
        if evidence["status"] == "ok":
            if not isinstance(evidence["text"], str) or not evidence["text"].strip() or evidence["error"] is not None:
                raise ValueError("Invalid evidence output")
            expected_prompt = render_initial_hint(question_prompt(plan["public_inputs"]), plan["tier"],
                                                  evidence["text"], template=spec["template"])
            if row["effective_prompt"] != expected_prompt:
                raise ValueError("Evidence-conditioned prompt changed")
        else:
            if status != "generation_error" or row["effective_prompt"] is not None:
                raise ValueError("Hint generated from failed evidence")
            if evidence["status"] == "empty_output":
                if not isinstance(evidence["text"], str) or evidence["text"].strip() or evidence["error"] is not None:
                    raise ValueError("Invalid empty evidence outcome")
            elif evidence["status"] == "generation_error":
                if (evidence["text"] is not None or not isinstance(evidence["error"], dict)
                        or set(evidence["error"]) != {"type", "message"}
                        or any(not isinstance(value, str) for value in evidence["error"].values())):
                    raise ValueError("Invalid failed evidence outcome")
            else:
                raise ValueError("Invalid evidence status")
    elif row["evidence"] is not None or row["effective_prompt"] != plan["prompt"]["rendered_text"]:
        raise ValueError("Unexpected evidence or effective prompt")


def read_records(path, plans, spec, *, repair_tail=False):
    if not path.exists():
        return {}
    saved, offset = {}, 0
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                path.with_name(f"{path.stem}.interrupted.{time.time_ns()}.bin").write_bytes(line)
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError("Malformed saved record; no outcomes were replaced")
        key = row.get("hint_id") if isinstance(row, dict) else None
        if key not in plans or key in saved:
            raise ValueError("Unexpected or duplicate saved record")
        validate_record(row, plans[key], spec)
        saved[key] = row
        offset += len(line)
    if repair_tail and saved and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ai2d", "scienceqa"), required=True)
    parser.add_argument("--experiment", choices=("A", "B", "D"), required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--limit", type=int, help="First N frozen questions for a pilot; default uses the full selection")
    choice.add_argument("--item-id", action="append")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate public inputs and plan without importing/loading the GPU model")
    args = parser.parse_args(argv)
    if args.resume and (not args.output_dir or args.limit is not None or args.item_id or args.preflight):
        parser.error("Resume requires an existing output folder and its saved selection")
    try:
        old = read_json(args.output_dir / "run.json") if args.resume else None
        if old and (old.get("dataset_name") != args.dataset or old.get("experiment") != args.experiment
                    or old.get("schema_version") != SCHEMA_VERSION):
            raise ValueError("The saved run uses a different dataset, experiment or schema")
        items = load_questions(ROOT, args.dataset, item_ids=old["item_ids"] if old else args.item_id,
                               limit=None if old else args.limit)
        spec = load_prompt_spec(ROOT)
        run_id = old["run_id"] if old else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        plans = make_plans(ROOT, args.dataset, args.experiment, items, spec, run_id)
        dataset = source_config(ROOT, args.dataset)
        signature = json_digest({"plans": list(plans.values()), "dataset": dataset,
                                 "prompt_spec": spec, "generation": settings(args.experiment)})
        if old and old["plan_sha256"] != signature:
            raise ValueError("Public inputs, selection, source or prompts changed")
        if args.preflight:
            print(json.dumps({"dataset": args.dataset, "experiment": args.experiment, "questions": len(items),
                              "planned_outputs": len(plans), "model_calls": len(plans) * (2 if args.experiment == "D" else 1),
                              "reference_inputs_loaded": False, "model_loaded": False}, indent=2))
            return
        output = args.output_dir or ROOT / "outputs/experiments" / args.dataset / args.experiment / run_id
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        path = output / ("predictions.jsonl" if args.experiment == "A" else "hints.jsonl")
        saved = read_records(path, plans, spec, repair_tail=True) if args.resume else {}
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    run = old or {"schema_version": SCHEMA_VERSION, "dataset_name": args.dataset, "dataset": dataset,
                  "experiment": args.experiment, "run_id": run_id, "split": items[0]["split"],
                  "data_source": "final", "item_ids": [item["item_id"] for item in items],
                  "tiers": [] if args.experiment == "A" else [1, 2, 3], "expected_records": len(plans),
                  "prompt_version": VERSIONS[args.experiment], "prompt_spec": spec, "plan_sha256": signature,
                  "generation": settings(args.experiment), "runtime": runtime_versions(), "resume_count": 0,
                  "purpose": "AI2D held-out no-attempt evaluation" if args.dataset == "ai2d" else "ScienceQA training development"}
    if args.resume:
        run["resume_count"] += 1

    def save_manifest(status, error=None):
        counts = Counter(row["generation"]["status"] for row in saved.values())
        run.update(status=status, completed_records=len(saved), remaining_records=len(plans) - len(saved),
                   counts={key: counts[key] for key in ("ok", "empty_output", "invalid_output", "generation_error")},
                   error=error, updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "run.json", run)

    save_manifest("validating_model")
    try:
        from qwen_engine import IMAGE_SIZE, MODEL_ID, QwenTutor

        model_path = args.model_path or run.get("model_path") or MODEL_ID
        identity = model_identity(model_path)
        if old and (run.get("model_identity") not in (None, identity) or run.get("image_size") not in (None, IMAGE_SIZE)):
            raise ValueError("Checkpoint identity or image processing settings changed")
        run.update(model_path=model_path, model_identity=identity, image_size=IMAGE_SIZE,
                   quantization="NF4 4-bit; double quantization; float16 compute")
        pending = [plan for key, plan in plans.items() if key not in saved]
        if pending:
            save_manifest("loading_model")
            started = time.perf_counter()
            tutor = QwenTutor(model_path)
            run["last_model_load_seconds"] = time.perf_counter() - started
            revision = getattr(getattr(getattr(tutor, "model", None), "config", None), "_commit_hash", None)
            if isinstance(revision, str):
                if run.get("resolved_revision") not in (None, revision):
                    raise ValueError("Resolved model revision changed")
                run["resolved_revision"] = revision
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                run["hardware"] = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                for plan in pending:
                    row = generate_record(tutor, ROOT, plan, spec)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    saved[row["hint_id"]] = row
                    save_manifest("generating")
                    print(f"{len(saved)}/{len(plans)} {row['item_id']} T{row['tier']}: {row['generation']['status']}", flush=True)
        if args.experiment == "A":
            save_manifest("scoring")
            # First reference access occurs after all answer-generation calls.
            refs = load_references(ROOT, args.dataset, items)
            predictions = [{**row, "status": row["generation"]["status"],
                            "option_ids": [choice["id"] for choice in row["public_inputs"]["choices"]]}
                           for row in saved.values()]
            report = score_predictions(predictions, {key: value["gold_option_id"] for key, value in refs.items()})
            write_json(output / "result.json", {"dataset_name": args.dataset, "experiment": "A", **report})
            print(json.dumps(report["summary"], indent=2))
        save_manifest("complete")
    except BaseException as error:
        save_manifest("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      {"type": type(error).__name__, "message": str(error)})
        raise
    print(f"Saved {len(saved)} outcomes to {path}. No judge was called.", flush=True)


if __name__ == "__main__":
    main()
