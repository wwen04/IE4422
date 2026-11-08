"""Serve a local browser interface for the ScienceQA tutor."""

import argparse
import csv
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile
from urllib.parse import parse_qs, urlencode, urlparse
import uuid
import webbrowser

import scienceqa_tutor as runner


ROOT = Path(__file__).resolve().parent
UI_ROOT = ROOT / "ui"
STAGES = {
    "queued": ("Waiting to start", 2),
    "loading_model": ("Loading the Qwen model", 10),
    "evidence": ("Reading relevant diagram evidence", 35),
    "assessment": ("Checking the submitted response", 60),
    "diagnosis": ("Considering the student's reasoning", 72),
    "hint": ("Writing a supportive hint", 88),
    "feedback": ("Preparing positive feedback", 88),
    "complete": ("Finished", 100),
}


def validate_origin(value):
    """Accept an explicit HTTP(S) origin, never a wildcard or a URL path."""
    parsed = urlparse(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.path
            or parsed.params or parsed.query or parsed.fragment
            or any(character.isspace() for character in value)
            or "*" in value or ";" in value):
        raise ValueError(f"Expected an exact HTTP(S) origin: {value!r}")
    # Accessing port also rejects malformed port values.
    parsed.port
    return value


def ensure_data_prepared(root=ROOT):
    root = Path(root).resolve()
    output = root / "data/processed/scienceqa"
    required_exports = [
        output / directory / f"{split}.jsonl"
        for directory in ("questions", "references")
        for split in runner.SPLITS
    ] + [
        root / f"manifests/scienceqa/{split}_valid_ids.json"
        for split in runner.SPLITS
    ] + [root / "reports/scienceqa/integrity_audit.json"]
    if all(path.is_file() for path in required_exports):
        print("Prepared ScienceQA data found; skipping preparation.", flush=True)
        return

    script = root / "prepare_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"Data is missing and preparation script was not found: {script}")
    print("ScienceQA data is missing or incomplete; running prepare_data.py.", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)


def ensure_data_ready(root=ROOT):
    root = Path(root).resolve()
    ensure_data_prepared(root)

    script = root / "finalize_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"Data finalization script was not found: {script}")
    print("Validating and building the finalized ScienceQA selection.", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)

    output = root / "data/final/scienceqa"
    required_exports = [
        output / directory / f"{split}.jsonl"
        for directory in ("questions", "references")
        for split in runner.SPLITS
    ] + [
        root / f"manifests/scienceqa/{split}_final_ids.json"
        for split in runner.SPLITS
    ] + [output / "selection.json"]
    missing = [path for path in required_exports if not path.is_file()]
    if missing:
        raise RuntimeError(
            "finalize_data.py completed without creating required outputs: "
            + ", ".join(str(path) for path in missing)
        )


class TutorWebApp:
    def __init__(self, root=ROOT, model_path=None, persistent_output=None,
                 allowed_origins=(), frame_ancestors=(), splits=runner.SPLITS, instance_id=None,
                 d_judge_package=None):
        self.root = Path(root).resolve()
        self.model_path = model_path
        self.instance_id = instance_id or uuid.uuid4().hex
        self.persistent_output = Path(persistent_output) if persistent_output else None
        self.d_judge_package = (
            Path(d_judge_package).resolve() if d_judge_package else None
        )
        self.allowed_origins = {validate_origin(value) for value in allowed_origins}
        self.frame_ancestors = tuple(validate_origin(value) for value in frame_ancestors)
        self.splits = tuple(splits)
        if not self.splits or any(split not in runner.SPLITS for split in self.splits):
            raise ValueError("Choose supported dataset splits.")
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.tutor = None
        self.inspection_lock = threading.Lock()
        self.review_lock = threading.Lock()
        self.evaluation_lock = threading.Lock()
        self.evaluations = {}
        self.evaluations_lock = threading.Lock()
        self.evaluation_running = False
        self.d_judge_package_lock = threading.Lock()
        self._d_judge_package_cache = None

    def generate_inspection(self, split, item_id):
        self.check_split(split)
        runner.load_question(self.root, split, item_id, data_source="final")
        script = self.root / "inspect_question.py"
        if not script.is_file():
            raise FileNotFoundError(f"Question inspection script was not found: {script}")

        output = self.root / "outputs/inspection" / f"{item_id}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.inspection_lock:
                subprocess.run(
                    [sys.executable, str(script), item_id, "--output", str(output)],
                    cwd=self.root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip()
            message = f"Question inspection failed with exit code {error.returncode}."
            if detail:
                message += f" {detail}"
            raise RuntimeError(message) from error

        inspection_page = output.with_suffix(".html")
        if not inspection_page.is_file():
            raise FileNotFoundError(
                f"Question inspection completed without creating {inspection_page}"
            )
        return inspection_page

    def update_job(self, job_id, **updates):
        with self.jobs_lock:
            self.jobs[job_id].update(updates)

    def get_job(self, job_id):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list_questions(self, split):
        self.check_split(split)
        return [{"item_id": row["item_id"], "question": row["question"]}
                for row in runner.load_public_questions(self.root, split)]

    def check_split(self, split):
        if split not in self.splits:
            raise ValueError(f"Dataset split is unavailable in this session: {split}")

    def get_tutor(self):
        if self.tutor is None:
            with self.model_lock:
                if self.tutor is None:
                    from qwen_engine import QwenTutor

                    self.tutor = (
                        QwenTutor(self.model_path)
                        if self.model_path
                        else QwenTutor()
                    )
        return self.tutor

    def start_job(self, payload):
        mode = payload.get("mode")
        if mode not in ("hint", "check"):
            raise ValueError("Choose whether to get a hint or check an answer.")
        split = payload.get("split", "validation")
        self.check_split(split)
        item_id = payload.get("item_id")
        answer = payload.get("student_answer")
        explanation = payload.get("explanation")
        try:
            hint_level = int(payload.get("hint_level", 1))
        except (TypeError, ValueError) as error:
            raise ValueError("Hint level must be 1, 2, or 3.") from error
        if hint_level not in (1, 2, 3):
            raise ValueError("Hint level must be 1, 2, or 3.")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Choose a prepared question first.")

        item = runner.load_question(self.root, split, item_id, data_source="final")
        reference = None
        if mode == "check":
            if answer is not None and not isinstance(answer, str):
                raise ValueError("The selected answer must be text.")
            if explanation is not None and not isinstance(explanation, str):
                raise ValueError("The reasoning must be text.")
            answer = answer.strip() or None if answer is not None else None
            if answer is None:
                raise ValueError("Select an answer before submitting.")
            explanation = (explanation or "").strip()
            if answer is not None and answer not in {choice["id"] for choice in item["choices"]}:
                raise ValueError("The selected answer is not one of this question's choices.")
            reference = runner.load_reference(
                self.root, split, item_id, data_source="final"
            )
        else:
            answer = None
            explanation = None

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "mode": mode,
            "status": "running",
            "stage": "queued",
            "stage_label": STAGES["queued"][0],
            "progress": STAGES["queued"][1],
            "result": None,
            "error": None,
            "output_path": None,
        }
        with self.jobs_lock:
            self.jobs[job_id] = job
        worker = threading.Thread(
            target=self.run_job,
            args=(job_id, mode, item, reference, answer,
                  explanation.strip() if explanation else None, hint_level),
            daemon=True,
        )
        worker.start()
        return job_id

    def run_job(self, job_id, mode, item, reference, answer, explanation, hint_level):
        try:
            with self.inference_lock:
                self.update_job(
                    job_id,
                    stage="loading_model",
                    stage_label=STAGES["loading_model"][0],
                    progress=STAGES["loading_model"][1],
                )
                tutor = self.get_tutor()

                def report_stage(stage):
                    label, progress = STAGES[stage]
                    self.update_job(
                        job_id, stage=stage, stage_label=label, progress=progress
                    )

                if mode == "hint":
                    result = runner.run_hint(tutor, item, hint_level, report_stage)
                else:
                    result = runner.run_checked_question(
                        tutor, item, reference, answer, explanation,
                        hint_level, report_stage,
                    )
            result["hint_level"] = hint_level
            if mode == "check":
                result.update({
                    "student_answer": answer,
                    "student_explanation": explanation,
                })
            output_dir = self.root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"ui_{job_id}.json"
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if self.persistent_output:
                self.persistent_output.mkdir(parents=True, exist_ok=True)
                destination = self.persistent_output / output_path.name
                temporary = destination.with_suffix(".json.partial")
                shutil.copy2(output_path, temporary)
                temporary.replace(destination)
            label, progress = STAGES["complete"]
            self.update_job(
                job_id,
                status="complete",
                stage="complete",
                stage_label=label,
                progress=progress,
                result=result,
                output_file=output_path.name,
            )
        except Exception as error:
            traceback.print_exc()
            self.update_job(
                job_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )

    def start_evaluation(self, payload):
        raw_limit = payload.get("limit", 10)
        if isinstance(raw_limit, bool):
            raise ValueError("Question count must be a whole number from 1 to 25.")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as error:
            raise ValueError("Question count must be a whole number from 1 to 25.") from error
        if not 1 <= limit <= 25:
            raise ValueError("Question count must be from 1 to 25.")
        self.check_split("validation")

        with self.evaluations_lock:
            if self.evaluation_running:
                raise ValueError("A comparison evaluation is already running.")
            self.evaluation_running = True
            job_id = uuid.uuid4().hex
            self.evaluations[job_id] = {
                "job_id": job_id,
                "status": "running",
                "stage": "queued",
                "stage_label": "Waiting to start",
                "progress": 0,
                "completed": 0,
                "total": limit,
                "current_item": None,
                "error": None,
            }

        worker = threading.Thread(
            target=self.run_evaluation,
            args=(job_id, limit),
            daemon=True,
        )
        worker.start()
        return job_id

    def get_evaluation(self, job_id):
        with self.evaluations_lock:
            job = self.evaluations.get(job_id)
            return dict(job) if job else None

    def update_evaluation(self, job_id, **updates):
        with self.evaluations_lock:
            self.evaluations[job_id].update(updates)

    def run_evaluation(self, job_id, limit):
        try:
            with self.inference_lock:
                self.update_evaluation(
                    job_id,
                    stage="loading_model",
                    stage_label="Loading the vision model",
                    progress=2,
                )
                tutor = self.get_tutor()
                from evaluate_hint_quality import generate_review_set
                from evaluate_visual_evidence import (
                    atomic_write_json,
                    evaluate,
                    write_csv_report,
                    write_html_report,
                )
                from qwen_engine import MODEL_ID

                output_dir = self.root / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / "accuracy_comparison.json"
                checkpoint = (
                    output_dir / f"accuracy_comparison.validation-{limit}.checkpoint.jsonl"
                )

                def report_progress(progress):
                    completed = progress["completed"]
                    total = progress["total"]
                    self.update_evaluation(
                        job_id,
                        stage="answer_accuracy",
                        stage_label="Comparing direct and evidence-guided answers",
                        progress=5 + round(completed / total * 70),
                        completed=completed,
                        total=total,
                        current_item=progress["item_id"],
                    )

                report = evaluate(
                    root=self.root,
                    split="validation",
                    data_source="final",
                    limit=limit,
                    tutor=tutor,
                    model=MODEL_ID,
                    resume_path=checkpoint,
                    on_progress=report_progress,
                )
                atomic_write_json(report_path, report)
                write_csv_report(report_path.with_suffix(".csv"), report)
                write_html_report(report_path.with_suffix(".html"), report)
                self.update_evaluation(
                    job_id,
                    stage="hints",
                    stage_label="Preparing paired hint examples",
                    progress=78,
                    current_item=None,
                )

                hint_csv = self.hint_review_path()
                hint_key = hint_csv.with_name(hint_csv.stem + "_key.json")
                if hint_csv.exists() != hint_key.exists():
                    raise RuntimeError(
                        "The hint review files are incomplete. Keep both "
                        "hint_quality_validation.csv and its _key.json together."
                    )
                hints_generated = not hint_csv.exists()
                if hints_generated:
                    def report_hint_progress(progress):
                        completed = progress["completed"]
                        total = progress["total"]
                        self.update_evaluation(
                            job_id,
                            stage="hints",
                            stage_label="Generating paired hint examples",
                            progress=78 + round(completed / total * 18),
                            completed=completed,
                            total=total,
                            current_item=progress["item_id"],
                        )

                    generate_review_set(
                        root=self.root,
                        limit=limit,
                        tutor=tutor,
                        output_path=hint_csv,
                        model=MODEL_ID,
                        on_progress=report_hint_progress,
                    )
                result = {
                    "accuracy_questions": report["n"],
                    "hint_questions": (
                        json.loads(hint_key.read_text(encoding="utf-8"))["n_questions"]
                        if hint_key.exists() else 0
                    ),
                    "hints_generated": hints_generated,
                    "accuracy_report": str(report_path),
                    "hint_review": str(hint_csv),
                }
            self.update_evaluation(
                job_id,
                status="complete",
                stage="complete",
                stage_label="Results are ready",
                progress=100,
                result=result,
            )
        except Exception as error:
            traceback.print_exc()
            self.update_evaluation(
                job_id,
                status="failed",
                stage="failed",
                stage_label="Evaluation stopped",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self.evaluations_lock:
                self.evaluation_running = False

    def status(self):
        counts = {}
        for split in self.splits:
            try:
                counts[split] = len(self.list_questions(split))
            except FileNotFoundError:
                counts[split] = 0
        return {"prepared": any(counts.values()), "counts": counts,
                "pid": os.getpid(), "instance_id": self.instance_id,
                "model_loaded": self.tutor is not None}

    def comparison(self):
        report_path = self.root / "outputs/accuracy_comparison.json"
        if not report_path.is_file():
            return {"available": False}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (not isinstance(report, dict)
                or report.get("schema_version") != 1
                or not isinstance(report.get("items"), list)):
            raise ValueError(f"Invalid comparison report: {report_path}")
        return {**report, "available": True}

    def hint_comparison(self):
        import evaluate_hint_quality as review_tools

        review_path = self.hint_review_path()
        key_path = review_path.with_name(review_path.stem + "_key.json")
        if not review_path.is_file() or not key_path.is_file():
            return {"available": False}

        key = json.loads(key_path.read_text(encoding="utf-8"))
        if (not isinstance(key, dict) or key.get("contains_answer_key") is not False
                or not isinstance(key.get("conditions"), dict)
                or not isinstance(key.get("items"), dict)):
            raise ValueError(f"Invalid hint comparison key: {key_path}")

        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not set(review_tools.CSV_FIELDS).issubset(
                    reader.fieldnames):
                raise ValueError(f"Hint review sheet has an invalid header: {review_path}")
            rows = list(reader)

        grouped = {}
        for line_number, row in enumerate(rows, start=2):
            item_id = row["item_id"]
            code = row["condition_code"]
            code_map = key["conditions"].get(item_id, {})
            method = code_map.get(code) if isinstance(code_map, dict) else None
            if (not isinstance(item_id, str)
                    or not re.fullmatch(r"validation_\d{6}", item_id)
                    or method not in ("direct_visual", "evidence_guided")):
                raise ValueError(
                    f"Hint review row {line_number} is not present in the comparison key."
                )
            item = grouped.setdefault(item_id, {
                "item_id": item_id,
                "question": row["question"],
                "context": row["context"],
                "choices": row["choices"],
                "image_url": (
                    "/api/image?" + urlencode({
                        "split": "validation",
                        "item_id": item_id,
                    })
                ),
                "conditions": {},
            })
            if method in item["conditions"]:
                raise ValueError(
                    f"Duplicate {method} hint row for validation question {item_id}."
                )
            item["conditions"][method] = {
                "hints": [row[f"hint_level_{level}"] for level in (1, 2, 3)],
                "ratings": {
                    field: row.get(field, "")
                    for field in self.hint_review_rating_fields()
                },
            }

        items = []
        for item_id, item in grouped.items():
            if set(item["conditions"]) != {"direct_visual", "evidence_guided"}:
                raise ValueError(
                    f"Expected both hint conditions for validation question {item_id}."
                )
            runner.load_question(self.root, "validation", item_id, data_source="final")
            item_key = key["items"].get(item_id, {})
            evidence = (
                item_key.get("visual_evidence")
                if isinstance(item_key, dict) else None
            )
            items.append({
                **item,
                "visual_evidence": evidence if isinstance(evidence, str) else "",
            })

        summary = review_tools.summarize_reviews(review_path, key_path)
        rating_counts = {
            method: {
                str(level): {
                    dimension: sum(
                        item["conditions"][method]["ratings"][
                            f"level_{level}_{dimension}"
                        ] in ("1", "2", "3", "4")
                        for item in items
                    )
                    for dimension in (
                        "factual_accuracy_1_to_4",
                        "image_groundedness_1_to_4",
                        "pedagogical_usefulness_1_to_4",
                    )
                }
                for level in (1, 2, 3)
            }
            for method in ("direct_visual", "evidence_guided")
        }
        return {
            "available": True,
            "created_at": key.get("created_at"),
            "model": key.get("model"),
            "n_questions": len(items),
            "summary": summary,
            "rating_counts": rating_counts,
            "items": items,
        }

    def hint_review_path(self):
        return self.root / "outputs/hint_quality_validation.csv"

    def experiment_d_judge_package(self):
        if self.d_judge_package is None:
            return {"available": False}
        cases = self._load_experiment_d_judge_package()
        return {
            "available": True,
            "name": self.d_judge_package.stem,
            "n_cases": len(cases),
            "items": [
                {key: value for key, value in case.items() if key != "image_entry"}
                for case in cases
            ],
        }

    def experiment_d_judge_image(self, record_id):
        if not isinstance(record_id, str) or not re.fullmatch(r"r_\d{6}", record_id):
            raise ValueError("Invalid Experiment D case ID.")
        cases = self._load_experiment_d_judge_package()
        case = next((item for item in cases if item["record_id"] == record_id), None)
        if case is None:
            raise FileNotFoundError(f"Experiment D image not found for {record_id}.")
        with zipfile.ZipFile(self.d_judge_package) as archive:
            return archive.read(case["image_entry"])

    def _load_experiment_d_judge_package(self):
        with self.d_judge_package_lock:
            if self._d_judge_package_cache is not None:
                return self._d_judge_package_cache
            if self.d_judge_package is None:
                raise FileNotFoundError("No Experiment D judging package is connected.")
            if not self.d_judge_package.is_file():
                raise FileNotFoundError(
                    f"Experiment D judging package not found: {self.d_judge_package}"
                )
            try:
                with zipfile.ZipFile(self.d_judge_package) as archive:
                    entries = archive.infolist()
                    if len(entries) > 2000 or sum(item.file_size for item in entries) > 128 * 1024 * 1024:
                        raise ValueError("Experiment D judging package is larger than supported.")
                    files = {item.filename: item for item in entries if not item.is_dir()}
                    case_entries = {}
                    for filename, info in files.items():
                        match = re.fullmatch(
                            r"(?:[^/]+/)?(?:judge_package/)?attachments/cases/"
                            r"(r_\d{6})/hints\.jsonl",
                            filename,
                        )
                        if match:
                            if match.group(1) in case_entries:
                                raise ValueError("Duplicate case in Experiment D judging package.")
                            case_entries[match.group(1)] = (filename, info)
                    if not case_entries:
                        raise ValueError(
                            "No judging cases found in the Experiment D package."
                        )

                    cases = []
                    for record_id, (hints_entry, info) in sorted(case_entries.items()):
                        if info.file_size > 128 * 1024:
                            raise ValueError(f"Experiment D case is too large: {record_id}.")
                        raw = archive.read(hints_entry).decode("utf-8")
                        lines = [line for line in raw.splitlines() if line.strip()]
                        if len(lines) != 1:
                            raise ValueError(
                                f"Expected exactly one hint in Experiment D case {record_id}."
                            )
                        row = json.loads(lines[0])
                        image_name = f"image_{record_id}.png"
                        image_entry = hints_entry.rsplit("/", 1)[0] + "/" + image_name
                        image_info = files.get(image_entry)
                        if (not isinstance(row, dict)
                                or row.get("record_id") != record_id
                                or row.get("image_file") != image_name
                                or type(row.get("requested_tier")) is not int
                                or row["requested_tier"] not in (1, 2, 3)
                                or not isinstance(row.get("question"), str)
                                or not isinstance(row.get("context"), str)
                                or not isinstance(row.get("hint_text"), str)
                                or not isinstance(row.get("choices"), list)
                                or image_info is None
                                or image_info.file_size > 8 * 1024 * 1024):
                            raise ValueError(
                                f"Invalid or incomplete Experiment D case {record_id}."
                            )
                        choices = []
                        for choice in row["choices"]:
                            if (not isinstance(choice, dict)
                                    or not isinstance(choice.get("id"), str)
                                    or not isinstance(choice.get("text"), str)):
                                raise ValueError(
                                    f"Invalid choices in Experiment D case {record_id}."
                                )
                            choices.append({"id": choice["id"], "text": choice["text"]})
                        image = archive.read(image_entry)
                        if (not image.startswith(b"\x89PNG\r\n\x1a\n")
                                or hashlib.sha256(image).hexdigest() != row.get("image_sha256")):
                            raise ValueError(
                                f"Image fingerprint mismatch in Experiment D case {record_id}."
                            )
                        cases.append({
                            "record_id": record_id,
                            "question": row["question"],
                            "context": row["context"],
                            "choices": choices,
                            "tier": row["requested_tier"],
                            "hint_text": row["hint_text"],
                            "image_url": (
                                "/api/experiment-d-image?"
                                + urlencode({"record_id": record_id})
                            ),
                            "image_entry": image_entry,
                        })
            except zipfile.BadZipFile as error:
                raise ValueError("Experiment D judging package is not a valid ZIP file.") from error
            self._d_judge_package_cache = cases
            return cases

    @staticmethod
    def hint_review_rating_fields():
        return tuple(
            f"level_{level}_{dimension}"
            for level in (1, 2, 3)
            for dimension in (
                "factual_accuracy_1_to_4",
                "image_groundedness_1_to_4",
                "pedagogical_usefulness_1_to_4",
                "answer_leakage_yes_no_unclear",
                "unsupported_claim_yes_no",
            )
        )

    def read_hint_reviews(self):
        import evaluate_hint_quality as review_tools

        review_path = self.hint_review_path()
        if not review_path.is_file():
            raise FileNotFoundError(
                f"Hint review sheet not found: {review_path}. "
                "Generate it with evaluate_hint_quality.py first."
            )
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required_fields = set(review_tools.CSV_FIELDS)
            if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
                raise ValueError(f"Hint review sheet has an invalid header: {review_path}")
            rows = list(reader)

        grouped = {}
        for row in rows:
            item_id = row["item_id"]
            if not re.fullmatch(r"validation_\d{6}", item_id):
                raise ValueError(f"Invalid validation item ID in hint review sheet: {item_id}")
            runner.load_question(self.root, "validation", item_id, data_source="final")
            item = grouped.setdefault(item_id, {
                "item_id": item_id,
                "image_url": (
                    "/api/image?" + urlencode({
                        "split": "validation",
                        "item_id": item_id,
                    })
                ),
                "question": row["question"],
                "context": row["context"],
                "choices": row["choices"],
                "conditions": [],
            })
            item["conditions"].append({
                "condition_code": row["condition_code"],
                "hints": [row[f"hint_level_{level}"] for level in (1, 2, 3)],
                "ratings": {
                    field: row.get(field, "")
                    for field in self.hint_review_rating_fields()
                },
                "reviewer": row.get("reviewer", ""),
                "comments": row.get("comments", ""),
            })
        return {"items": list(grouped.values())}

    def save_hint_review(self, payload):
        review_path = self.hint_review_path()
        item_id = payload.get("item_id")
        code = payload.get("condition_code")
        ratings = payload.get("ratings")
        reviewer = payload.get("reviewer", "")
        comments = payload.get("comments", "")
        if not isinstance(item_id, str) or not re.fullmatch(
                r"validation_\d{6}", item_id):
            raise ValueError("Choose a valid validation review item.")
        if code not in ("A", "B"):
            raise ValueError("Review condition must be A or B.")
        if not isinstance(ratings, dict):
            raise ValueError("Ratings must be provided as an object.")
        if not isinstance(reviewer, str) or len(reviewer) > 100:
            raise ValueError("Reviewer name must be 100 characters or fewer.")
        if not isinstance(comments, str) or len(comments) > 2000:
            raise ValueError("Comments must be 2000 characters or fewer.")

        valid_fields = set(self.hint_review_rating_fields())
        if set(ratings) != valid_fields:
            raise ValueError("Rating fields do not match the hint review form.")
        validated = {}
        for field, value in ratings.items():
            if not isinstance(value, str):
                raise ValueError(f"{field} must be submitted as text.")
            if field.endswith("_1_to_4") and value not in ("", "1", "2", "3", "4"):
                raise ValueError(f"{field} must be blank or a rating from 1 to 4.")
            if field.endswith("_answer_leakage_yes_no_unclear") and value not in (
                    "", "yes", "no", "unclear"):
                raise ValueError(f"{field} must be blank, yes, no, or unclear.")
            if field.endswith("_unsupported_claim_yes_no") and value not in (
                    "", "yes", "no"):
                raise ValueError(f"{field} must be blank, yes, or no.")
            validated[field] = value

        if not review_path.is_file():
            raise FileNotFoundError(
                f"Hint review sheet not found: {review_path}. "
                "Generate it with evaluate_hint_quality.py first."
            )
        with self.review_lock:
            with review_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                if fieldnames is None or not set(self.hint_review_rating_fields()).issubset(
                        fieldnames):
                    raise ValueError(f"Hint review sheet has an invalid header: {review_path}")
                rows = list(reader)
            matched = [
                row for row in rows
                if row.get("item_id") == item_id
                and row.get("condition_code") == code
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"Expected exactly one review row for {item_id} condition {code}."
                )
            matched[0].update(validated)
            matched[0]["reviewer"] = reviewer
            matched[0]["comments"] = comments
            temporary = review_path.with_name(review_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, review_path)

        complete = all(value != "" for value in validated.values())
        return {"saved": True, "complete": complete}

    def handler(self):
        app = self
        ancestors = " ".join(("'self'", *app.frame_ancestors)) if app.frame_ancestors else "'none'"

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "ScienceTutorUI/1.0"

            def log_message(self, format_string, *args):
                print(f"[UI] {self.address_string()} - {format_string % args}")

            def send_bytes(
                self, status, content, content_type, download_name=None,
                content_security_policy=None,
            ):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    content_security_policy or (
                        "default-src 'self'; img-src 'self'; style-src 'self'; "
                        "script-src 'self'; object-src 'none'; base-uri 'none'; "
                        f"frame-ancestors {ancestors}"
                    ),
                )
                self.send_header("Referrer-Policy", "no-referrer")
                if download_name:
                    self.send_header(
                        "Content-Disposition", f'attachment; filename="{download_name}"'
                    )
                self.end_headers()
                self.wfile.write(content)

            def send_json(self, status, value):
                content = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_bytes(status, content, "application/json; charset=utf-8")

            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                try:
                    if parsed.path == "/":
                        self.send_static("index.html")
                    elif parsed.path == "/review":
                        self.send_static("review.html")
                    elif parsed.path == "/compare":
                        self.send_static("compare.html")
                    elif parsed.path == "/assets/app.css":
                        self.send_static("app.css")
                    elif parsed.path == "/assets/app.js":
                        self.send_static("app.js")
                    elif parsed.path == "/assets/compare.css":
                        self.send_static("compare.css")
                    elif parsed.path == "/assets/compare.js":
                        self.send_static("compare.js")
                    elif parsed.path == "/assets/review.css":
                        self.send_static("review.css")
                    elif parsed.path == "/assets/review.js":
                        self.send_static("review.js")
                    elif parsed.path == "/api/status":
                        self.send_json(200, app.status())
                    elif parsed.path == "/api/comparison":
                        self.send_json(200, app.comparison())
                    elif parsed.path == "/api/hint-comparison":
                        self.send_json(200, app.hint_comparison())
                    elif parsed.path == "/api/experiment-d-judge-package":
                        self.send_json(200, app.experiment_d_judge_package())
                    elif parsed.path == "/api/experiment-d-image":
                        record_id = params.get("record_id", [""])[0]
                        self.send_bytes(
                            200,
                            app.experiment_d_judge_image(record_id),
                            "image/png",
                        )
                    elif parsed.path.startswith("/api/evaluations/"):
                        job_id = parsed.path.rsplit("/", 1)[-1]
                        evaluation = app.get_evaluation(job_id)
                        if evaluation is None:
                            self.send_json(404, {"error": "Evaluation job not found."})
                        else:
                            self.send_json(200, evaluation)
                    elif parsed.path == "/api/hint-reviews":
                        self.send_json(200, app.read_hint_reviews())
                    elif parsed.path == "/api/comparison/html":
                        report_path = app.root / "outputs/accuracy_comparison.html"
                        if not report_path.is_file():
                            self.send_json(404, {"error": "HTML comparison report not found."})
                        else:
                            self.send_bytes(
                                200,
                                report_path.read_bytes(),
                                "text/html; charset=utf-8",
                                content_security_policy=(
                                    "default-src 'none'; style-src 'unsafe-inline'; "
                                    "object-src 'none'; base-uri 'none'; frame-ancestors 'self'"
                                ),
                            )
                    elif parsed.path == "/api/questions":
                        split = params.get("split", ["validation"])[0]
                        self.send_json(200, {
                            "split": split,
                            "questions": app.list_questions(split),
                        })
                    elif parsed.path == "/api/question":
                        split = params.get("split", [""])[0]
                        app.check_split(split)
                        item_id = params.get("item_id", [""])[0]
                        item = runner.load_question(
                            app.root, split, item_id, data_source="final"
                        )
                        app.generate_inspection(split, item_id)
                        item["inspection_url"] = (
                            f"/api/inspection/{split}/{item_id}"
                        )
                        image_query = parse_qs(parsed.query)
                        item["image_url"] = (
                            "/api/image?"
                            + urlencode({
                                "split": image_query["split"][0],
                                "item_id": image_query["item_id"][0],
                            })
                        )
                        self.send_json(200, item)
                    elif parsed.path == "/api/image":
                        split = params.get("split", [""])[0]
                        app.check_split(split)
                        item_id = params.get("item_id", [""])[0]
                        item = runner.load_question(
                            app.root, split, item_id, data_source="final"
                        )
                        content = Path(item["image_path"]).read_bytes()
                        content_type = mimetypes.guess_type(item["image_path"])[0]
                        self.send_bytes(
                            200, content, content_type or "application/octet-stream"
                        )
                    elif parsed.path.startswith("/api/inspection/"):
                        parts = parsed.path.strip("/").split("/")
                        if len(parts) != 4:
                            self.send_json(404, {"error": "Inspection page not found."})
                            return
                        _, _, split, item_id = parts
                        if (split not in app.splits
                                or not re.fullmatch(rf"{re.escape(split)}_\d{{6}}", item_id)):
                            self.send_json(404, {"error": "Inspection page not found."})
                            return
                        item = runner.load_question(
                            app.root, split, item_id, data_source="final"
                        )
                        inspection_page = (
                            app.root / "outputs/inspection" / f"{item_id}.html"
                        ).resolve()
                        if not inspection_page.is_relative_to(
                                (app.root / "outputs/inspection").resolve()):
                            raise ValueError("Invalid inspection page path.")
                        content = inspection_page.read_text(encoding="utf-8")
                        image_path = Path(item["image_path"])
                        relative_image = Path(
                            os.path.relpath(image_path, inspection_page.parent)
                        ).as_posix().replace(" ", "%20")
                        image_url = (
                            "/api/image?"
                            + urlencode({"split": split, "item_id": item_id})
                        )
                        content = content.replace(
                            html.escape(relative_image, quote=True),
                            html.escape(image_url, quote=True),
                        )
                        self.send_bytes(
                            200,
                            content.encode("utf-8"),
                            "text/html; charset=utf-8",
                            content_security_policy=(
                                "default-src 'none'; img-src 'self'; "
                                "style-src 'unsafe-inline'; object-src 'none'; "
                                "base-uri 'none'; frame-ancestors "
                                + " ".join(("'self'", *app.frame_ancestors))
                            ),
                        )
                    elif parsed.path.startswith("/api/jobs/"):
                        job_id = parsed.path.rsplit("/", 1)[-1]
                        job = app.get_job(job_id)
                        if job is None:
                            self.send_json(404, {"error": "Job not found."})
                        else:
                            self.send_json(200, job)
                    elif parsed.path.startswith("/api/results/"):
                        job_id = parsed.path.rsplit("/", 1)[-1]
                        job = app.get_job(job_id)
                        if job is None or job["status"] != "complete":
                            self.send_json(404, {"error": "Completed result not found."})
                        else:
                            path = (app.root / "outputs" / job["output_file"]).resolve()
                            if path.parent != (app.root / "outputs").resolve():
                                raise ValueError("Invalid result path.")
                            self.send_bytes(
                                200,
                                path.read_bytes(),
                                "application/json; charset=utf-8",
                                path.name,
                            )
                    else:
                        self.send_json(404, {"error": "Not found."})
                except FileNotFoundError as error:
                    self.send_json(404, {"error": str(error)})
                except (ValueError, KeyError, json.JSONDecodeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    traceback.print_exc()
                    self.send_json(500, {"error": f"{type(error).__name__}: {error}"})

            def send_static(self, filename):
                path = UI_ROOT / filename
                self.send_bytes(
                    200,
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )

            def do_POST(self):
                request_path = urlparse(self.path).path
                if request_path not in (
                        "/api/jobs", "/api/hint-reviews", "/api/evaluations"):
                    self.send_json(404, {"error": "Not found."})
                    return
                try:
                    origin = self.headers.get("Origin")
                    if (origin and urlparse(origin).netloc != self.headers.get("Host")
                            and origin not in app.allowed_origins):
                        self.send_json(403, {"error": "Cross-origin requests are not allowed."})
                        return
                    if not self.headers.get("Content-Type", "").startswith(
                            "application/json"):
                        self.send_json(415, {"error": "Send the request as JSON."})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 64 * 1024:
                        raise ValueError("Request body must be between 1 byte and 64 KB.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    if request_path == "/api/jobs":
                        job_id = app.start_job(payload)
                        self.send_json(202, {"job_id": job_id})
                    elif request_path == "/api/evaluations":
                        job_id = app.start_evaluation(payload)
                        self.send_json(202, {"job_id": job_id})
                    else:
                        self.send_json(200, app.save_hint_review(payload))
                except (ValueError, KeyError, json.JSONDecodeError) as error:
                    self.send_json(400, {"error": str(error)})
                except FileNotFoundError as error:
                    self.send_json(404, {"error": str(error)})
                except Exception as error:
                    traceback.print_exc()
                    self.send_json(500, {"error": f"{type(error).__name__}: {error}"})

        return RequestHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--persistent-output", type=Path,
                        help="Also save each completed UI result to this folder (e.g. mounted Drive).")
    parser.add_argument("--d-judge-package", type=Path,
                        help="Read-only Experiment D judge-package ZIP to show on /compare.")
    parser.add_argument("--allowed-origin", action="append", default=[], type=validate_origin,
                        help="Exact trusted proxy origin for POST requests; repeat if needed.")
    parser.add_argument("--frame-ancestor", action="append", default=[], type=validate_origin,
                        help="Exact trusted notebook origin allowed to embed the UI; repeat if needed.")
    parser.add_argument("--splits", nargs="+", choices=runner.SPLITS, default=runner.SPLITS)
    parser.add_argument("--instance-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("The tutor UI only supports localhost for safety.")

    ensure_data_ready(ROOT)
    app = TutorWebApp(model_path=args.model_path, persistent_output=args.persistent_output,
                      allowed_origins=args.allowed_origin, frame_ancestors=args.frame_ancestor,
                      splits=args.splits, instance_id=args.instance_id,
                      d_judge_package=args.d_judge_package)
    server = ThreadingHTTPServer((args.host, args.port), app.handler())
    url = f"http://{args.host}:{server.server_port}"
    print(f"Science tutor UI is running at {url}")
    print("Press Ctrl+C to stop the server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the tutor UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
