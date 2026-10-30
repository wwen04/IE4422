"""Serve a local browser interface for the ScienceQA tutor."""

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import traceback
from urllib.parse import parse_qs, urlencode, urlparse
import uuid
import webbrowser

import run_scienceqa_tutor as runner


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
    def __init__(self, root=ROOT, model_path=None):
        self.root = Path(root).resolve()
        self.model_path = model_path
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.tutor = None
        self.inspection_lock = threading.Lock()

    def generate_inspection(self, split, item_id):
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
        if split not in runner.SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return [{"item_id": row["item_id"], "question": row["question"]}
                for row in runner.load_public_questions(self.root, split)]

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
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Select the student's answer before checking it.")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("Enter the student's reasoning before checking it.")
            if answer not in {choice["id"] for choice in item["choices"]}:
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

    def status(self):
        counts = {}
        for split in runner.SPLITS:
            try:
                counts[split] = len(self.list_questions(split))
            except FileNotFoundError:
                counts[split] = 0
        return {"prepared": any(counts.values()), "counts": counts}

    def handler(self):
        app = self

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
                        "frame-ancestors 'none'"
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
                    elif parsed.path == "/assets/app.css":
                        self.send_static("app.css")
                    elif parsed.path == "/assets/app.js":
                        self.send_static("app.js")
                    elif parsed.path == "/api/status":
                        self.send_json(200, app.status())
                    elif parsed.path == "/api/questions":
                        split = params.get("split", ["validation"])[0]
                        self.send_json(200, {
                            "split": split,
                            "questions": app.list_questions(split),
                        })
                    elif parsed.path == "/api/question":
                        split = params.get("split", [""])[0]
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
                        if (split not in runner.SPLITS
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
                                "base-uri 'none'; frame-ancestors 'self'"
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
                if request_path != "/api/jobs":
                    self.send_json(404, {"error": "Not found."})
                    return
                try:
                    origin = self.headers.get("Origin")
                    if origin and urlparse(origin).netloc != self.headers.get("Host"):
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
                    job_id = app.start_job(payload)
                    self.send_json(202, {"job_id": job_id})
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
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("The tutor UI only supports localhost for safety.")

    ensure_data_ready(ROOT)
    app = TutorWebApp(model_path=args.model_path)
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
