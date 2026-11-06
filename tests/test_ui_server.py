import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer

from evaluate_hint_quality import CSV_FIELDS, RATING_FIELDS
import scienceqa_tutor as runner
from ui_server import TutorWebApp, ensure_data_prepared, ensure_data_ready, main
from ui_server import validate_origin

RATING_VALUE_FIELDS = RATING_FIELDS[:-2]


class TutorWebAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.questions_dir = self.root / "data/final/scienceqa/questions"
        self.questions_dir.mkdir(parents=True)
        reference_dir = self.root / "data/final/scienceqa/references"
        reference_dir.mkdir(parents=True)
        (reference_dir / "validation.jsonl").write_text(
            json.dumps({
                "item_id": "validation_000001",
                "gold_option_id": "A",
                "lecture": "Reference lecture",
                "solution": "Reference solution",
            }) + "\n",
            encoding="utf-8",
        )
        image_dir = self.root / "data/final/scienceqa/images"
        image_dir.mkdir(parents=True)
        (image_dir / "example.png").write_bytes(b"test-image")
        self.item = {
            "item_id": "validation_000001",
            "split": "validation",
            "question": "What does this diagram show?",
            "context": "Use the labels.",
            "choices": [{"id": "A", "text": "A sample choice"}],
            "image_path": "data/final/scienceqa/images/example.png",
            "gold_option_id": "SECRET ANSWER",
            "solution": "SECRET SOLUTION",
        }
        (self.questions_dir / "validation.jsonl").write_text(
            json.dumps(self.item) + "\n", encoding="utf-8"
        )
        manifest = self.root / "manifests/scienceqa/validation_final_ids.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps([self.item["item_id"]]), encoding="utf-8")
        (self.root / "inspect_question.py").write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "output.parent.mkdir(parents=True, exist_ok=True)\n"
            "output.write_text('# inspection\\n', encoding='utf-8')\n"
            "output.with_suffix('.html').write_text("
            "\"<!doctype html><html><body>"
            "<img src='../../data/final/scienceqa/images/"
            "example.png'>"
            "<p>Reference answer: A</p></body></html>\", encoding='utf-8')\n",
            encoding="utf-8",
        )
        self.app = TutorWebApp(self.root)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.app.handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, method="GET", payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
        )
        with urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read()

    def test_prepare_data_runs_only_when_exports_are_missing(self):
        script = self.root / "prepare_data.py"
        script.write_text("# test script\n", encoding="utf-8")
        with patch("ui_server.subprocess.run") as run:
            ensure_data_prepared(self.root)
        run.assert_called_once_with(
            [sys.executable, str(script)],
            cwd=self.root,
            check=True,
        )

        for directory in ("questions", "references"):
            target = self.root / "data/processed/scienceqa" / directory
            target.mkdir(parents=True, exist_ok=True)
            for split in runner.SPLITS:
                (target / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
        for split in runner.SPLITS:
            target = self.root / f"manifests/scienceqa/{split}_valid_ids.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("[]\n", encoding="utf-8")
        report = self.root / "reports/scienceqa/integrity_audit.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
        with patch("ui_server.subprocess.run") as run:
            ensure_data_prepared(self.root)
        run.assert_not_called()

    def test_startup_runs_preparation_then_finalization(self):
        (self.root / "prepare_data.py").write_text("# preparation\n", encoding="utf-8")
        (self.root / "finalize_data.py").write_text("# finalization\n", encoding="utf-8")
        calls = []

        def run_script(command, cwd, check):
            script_name = Path(command[1]).name
            calls.append(script_name)
            if script_name == "prepare_data.py":
                for directory in ("questions", "references"):
                    target = self.root / "data/processed/scienceqa" / directory
                    target.mkdir(parents=True, exist_ok=True)
                    for split in runner.SPLITS:
                        (target / f"{split}.jsonl").write_text("", encoding="utf-8")
                for split in runner.SPLITS:
                    target = self.root / f"manifests/scienceqa/{split}_valid_ids.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("[]\n", encoding="utf-8")
                report = self.root / "reports/scienceqa/integrity_audit.json"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text("{}\n", encoding="utf-8")
            elif script_name == "finalize_data.py":
                output = self.root / "data/final/scienceqa"
                for directory in ("questions", "references"):
                    target = output / directory
                    target.mkdir(parents=True, exist_ok=True)
                    for split in runner.SPLITS:
                        (target / f"{split}.jsonl").write_text("", encoding="utf-8")
                for split in runner.SPLITS:
                    target = self.root / f"manifests/scienceqa/{split}_final_ids.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("[]\n", encoding="utf-8")
                (output / "selection.json").write_text("{}\n", encoding="utf-8")
            else:
                self.fail(f"Unexpected startup script: {script_name}")

        with patch("ui_server.subprocess.run", side_effect=run_script) as run:
            ensure_data_ready(self.root)
        self.assertEqual(calls, ["prepare_data.py", "finalize_data.py"])
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_main_prepares_data_before_starting_ui_server(self):
        events = []

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                events.append("server")

            def serve_forever(self):
                events.append("serve")
                raise KeyboardInterrupt

            def server_close(self):
                events.append("close")

        with patch("ui_server.ensure_data_ready",
                   side_effect=lambda root: events.append("prepare")), \
                patch("ui_server.ThreadingHTTPServer", FakeServer), \
                patch("ui_server.webbrowser.open"):
            main(["--no-browser"])
        self.assertEqual(events, ["prepare", "server", "serve", "close"])

    def test_ui_and_question_routes_only_expose_public_question_fields(self):
        status, _, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"science hint studio", page.lower())
        self.assertIn(b"Show inspection report", page)
        self.assertIn(b"Does visual evidence improve answer accuracy?", page)
        self.assertIn(b"/review", page)

        _, _, response = self.request("/api/questions?split=validation")
        self.assertNotIn(b"SECRET", response)
        self.assertIn(b"validation_000001", response)

        _, _, response = self.request(
            "/api/question?split=validation&item_id=validation_000001"
        )
        self.assertNotIn(b"SECRET", response)
        question = json.loads(response)
        self.assertEqual(question["context"], "Use the labels.")
        self.assertEqual(
            question["image_url"],
            "/api/image?split=validation&item_id=validation_000001",
        )

        _, headers, image = self.request(question["image_url"])
        self.assertEqual(headers.get_content_type(), "image/png")
        self.assertEqual(image, b"test-image")

    def test_blinded_hint_review_ui_loads_and_saves_ratings_without_key(self):
        output = self.root / "outputs/hint_quality_validation.csv"
        output.parent.mkdir(parents=True)
        row = {
            "item_id": "validation_000001",
            "image_path": str(self.root / "data/final/scienceqa/images/example.png"),
            "question": "What does this diagram show?",
            "context": "Use the labels.",
            "choices": "A. A sample choice",
            "condition_code": "A",
            "hint_level_1": "Look at the diagram.",
            "hint_level_2": "Notice a relevant label.",
            "hint_level_3": "Compare the evidence.",
            **{field: "" for field in RATING_FIELDS},
        }
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

        status, _, page = self.request("/review")
        self.assertEqual(status, 200)
        self.assertIn(b"Review the hint", page)

        status, _, body = self.request("/api/hint-reviews")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertNotIn("SECRET", body.decode("utf-8"))
        self.assertNotIn("evidence_guided", body.decode("utf-8"))
        self.assertEqual(result["items"][0]["conditions"][0]["condition_code"], "A")

        ratings = {field: "" for field in RATING_VALUE_FIELDS}
        ratings["level_1_factual_accuracy_1_to_4"] = "4"
        payload = {
            "item_id": "validation_000001",
            "condition_code": "A",
            "ratings": ratings,
            "reviewer": "QA reviewer",
            "comments": "Looks grounded.",
        }
        status, _, body = self.request(
            "/api/hint-reviews", method="POST", payload=payload
        )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["saved"])
        saved = output.read_text(encoding="utf-8-sig")
        self.assertIn("QA reviewer", saved)
        self.assertIn("Looks grounded.", saved)
        self.assertIn(",4,", saved)

    def test_hint_review_save_rejects_invalid_ratings(self):
        output = self.root / "outputs/hint_quality_validation.csv"
        output.parent.mkdir(parents=True)
        row = {
            "item_id": "validation_000001",
            "image_path": "data/final/scienceqa/images/example.png",
            "question": "Question",
            "context": "",
            "choices": "A. Choice",
            "condition_code": "A",
            "hint_level_1": "Hint 1",
            "hint_level_2": "Hint 2",
            "hint_level_3": "Hint 3",
            **{field: "" for field in RATING_FIELDS},
        }
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        payload = {
            "item_id": "validation_000001",
            "condition_code": "A",
            "ratings": {field: "" for field in RATING_VALUE_FIELDS},
            "reviewer": "",
            "comments": "",
        }
        payload["ratings"]["level_1_factual_accuracy_1_to_4"] = "5"
        request = Request(
            self.base_url + "/api/hint-reviews",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 400)
        self.assertIn(b"rating from 1 to 4", raised.exception.read())

    def test_comparison_route_reports_missing_file_then_serves_generated_metrics(self):
        _, _, body = self.request("/api/comparison")
        self.assertEqual(json.loads(body), {"available": False})

        report_path = self.root / "outputs/accuracy_comparison.json"
        report_path.parent.mkdir(parents=True)
        report = {
            "schema_version": 1,
            "n": 2,
            "baseline": {"correct": 1, "accuracy_percent": 50},
            "evidence_guided": {"correct": 2, "accuracy_percent": 100},
            "delta_percentage_points": 50,
            "paired": {
                "both_correct": 1,
                "baseline_only_correct": 0,
                "evidence_guided_only_correct": 1,
                "both_incorrect": 0,
            },
            "items": [],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        _, _, body = self.request("/api/comparison")
        result = json.loads(body)
        self.assertTrue(result["available"])
        self.assertEqual(result["delta_percentage_points"], 50)

        html_path = self.root / "outputs/accuracy_comparison.html"
        html_path.write_text("<!doctype html><title>Report</title>", encoding="utf-8")
        status, headers, body = self.request("/api/comparison/html")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/html")
        self.assertIn("style-src 'unsafe-inline'", headers.get("Content-Security-Policy"))
        self.assertIn(b"Report", body)

    def test_inspection_runs_when_question_is_selected_and_display_is_optional(self):
        with patch("ui_server.subprocess.run", wraps=subprocess.run) as run:
            status, _, response = self.request(
                "/api/question?split=validation&item_id=validation_000001"
            )
            self.assertEqual(status, 200)
            question = json.loads(response)
            self.assertNotIn("SECRET", response.decode("utf-8"))
            page_url = question["inspection_url"]
            self.assertEqual(
                run.call_args.args[0][1],
                str(self.root / "inspect_question.py"),
            )
            status, headers, page = self.request(page_url)

        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/html")
        self.assertEqual(
            headers.get("Content-Security-Policy"),
            "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'self'",
        )
        self.assertIn(b"/api/image?split=validation&amp;item_id=validation_000001", page)
        self.assertIn(b"Reference answer: A", page)
        self.assertEqual(run.call_count, 1)

    def test_run_validation_rejects_missing_student_response(self):
        request = Request(
            self.base_url + "/api/jobs",
            data=json.dumps({
                "split": "validation",
                "item_id": self.item["item_id"],
                "mode": "check",
                "student_answer": "",
                "explanation": "Reasoning",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 400)
        self.assertIn(b"Select the student's answer", raised.exception.read())

    def test_run_rejects_cross_origin_and_non_json_requests(self):
        body = json.dumps({
            "mode": "check",
            "split": "validation",
            "item_id": self.item["item_id"],
            "student_answer": "A",
            "explanation": "Reasoning",
        }).encode("utf-8")
        for headers, expected_status in (
                ({"Content-Type": "application/json", "Origin": "https://example.com"}, 403),
                ({"Content-Type": "text/plain"}, 415)):
            request = Request(
                self.base_url + "/api/jobs", data=body, headers=headers, method="POST"
            )
            with self.subTest(expected_status=expected_status):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, expected_status)

    def test_run_creates_visible_job_and_downloadable_local_result(self):
        self.app.persistent_output = self.root / "drive/runs/ui"
        generated = {
            "mode": "check", "evidence": "Evidence", "diagnosis": "Diagnosis",
            "hint": "Hint", "outcome": "try_again", "answer_correct": False,
            "explanation_status": "not_assessed",
        }
        with patch.object(self.app, "get_tutor", return_value=object()), patch(
                "ui_server.runner.run_checked_question", return_value=generated):
            _, _, body = self.request(
                "/api/jobs",
                method="POST",
                payload={
                    "split": "validation",
                    "mode": "check",
                    "item_id": self.item["item_id"],
                    "student_answer": "A",
                    "explanation": "Example reasoning",
                    "hint_level": 1,
                },
            )
            job_id = json.loads(body)["job_id"]
            for _ in range(50):
                _, _, body = self.request(f"/api/jobs/{job_id}")
                job = json.loads(body)
                if job["status"] in ("complete", "failed"):
                    break
                threading.Event().wait(.02)

        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"]["hint"], "Hint")
        self.assertTrue((self.root / "outputs" / job["output_file"]).is_file())
        self.assertEqual(
            (self.app.persistent_output / job["output_file"]).read_bytes(),
            (self.root / "outputs" / job["output_file"]).read_bytes(),
        )
        self.assertEqual(list(self.app.persistent_output.glob("*.partial")), [])
        _, headers, result_file = self.request(f"/api/results/{job_id}")
        self.assertEqual(headers.get_content_type(), "application/json")
        self.assertEqual(json.loads(result_file)["student_explanation"], "Example reasoning")

    def test_hint_job_does_not_require_or_load_answer_reference(self):
        generated = {
            "mode": "hint", "evidence": "Evidence",
            "diagnosis": None, "hint": "Look at the diagram.",
            "outcome": "hint", "feedback": None,
        }
        with patch.object(self.app, "get_tutor", return_value=object()), patch(
                "ui_server.runner.run_hint", return_value=generated) as run_hint, patch(
                "ui_server.runner.load_reference",
                side_effect=AssertionError("Pre-answer hints must not load the answer key")):
            _, _, body = self.request(
                "/api/jobs",
                method="POST",
                payload={
                    "mode": "hint",
                    "split": "validation",
                    "item_id": self.item["item_id"],
                    "hint_level": 1,
                },
            )
            job_id = json.loads(body)["job_id"]
            for _ in range(50):
                _, _, body = self.request(f"/api/jobs/{job_id}")
                job = json.loads(body)
                if job["status"] in ("complete", "failed"):
                    break
                threading.Event().wait(.02)

        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["mode"], "hint")
        self.assertEqual(job["result"]["hint"], "Look at the diagram.")
        run_hint.assert_called_once()

    def test_check_job_loads_reference_only_after_submission(self):
        with patch.object(self.app, "get_tutor", return_value=object()), patch(
                "ui_server.runner.run_checked_question",
                return_value={
                    "mode": "check", "evidence": "Evidence", "diagnosis": None,
                    "hint": None, "outcome": "correct", "answer_correct": True,
                    "explanation_status": "correct", "feedback": "Well done.",
                }) as checked, patch(
                "ui_server.runner.load_reference",
                wraps=runner.load_reference) as load_reference:
            _, _, body = self.request(
                "/api/jobs",
                method="POST",
                payload={
                    "mode": "check",
                    "split": "validation",
                    "item_id": self.item["item_id"],
                    "student_answer": "A",
                    "explanation": "Example reasoning",
                    "hint_level": 2,
                },
            )
            job_id = json.loads(body)["job_id"]
            for _ in range(50):
                _, _, body = self.request(f"/api/jobs/{job_id}")
                job = json.loads(body)
                if job["status"] in ("complete", "failed"):
                    break
                threading.Event().wait(.02)

        self.assertEqual(job["status"], "complete")
        load_reference.assert_called_once_with(
            self.root, "validation", self.item["item_id"], data_source="final"
        )
        self.assertEqual(checked.call_args.args[2]["gold_option_id"], "A")

    def test_model_is_loaded_once_from_supplied_checkpoint(self):
        engine = ModuleType("qwen_engine")
        engine.QwenTutor = Mock()
        self.app.model_path = "cached/checkpoint"
        with patch.dict(sys.modules, {"qwen_engine": engine}):
            first = self.app.get_tutor()
            self.assertIs(self.app.get_tutor(), first)
        engine.QwenTutor.assert_called_once_with("cached/checkpoint")

    def test_proxy_origin_is_explicit_and_other_origins_remain_blocked(self):
        origin = "https://session.example.com"
        self.app.allowed_origins = {origin}
        for candidate, expected in (
                (origin, 202),
                (origin + ".attacker.invalid", 403),
                ("null", 403)):
            with self.subTest(origin=candidate), patch.object(self.app, "start_job", return_value="job") as start:
                request = Request(self.base_url + "/api/jobs", data=b'{}',
                                  headers={"Content-Type": "application/json", "Origin": candidate})
                if expected == 202:
                    with urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 202)
                    start.assert_called_once()
                else:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request, timeout=5)
                    self.assertEqual(error.exception.code, expected)
                    start.assert_not_called()

    def test_embedding_is_opt_in_and_applies_to_nested_inspection(self):
        _, headers, _ = self.request("/")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        origins = ("https://colab.research.google.com", "https://output.example.com")
        self.app.frame_ancestors = origins
        # handler() captures CSP when the server is created.
        self.server.RequestHandlerClass = self.app.handler()
        _, headers, _ = self.request("/")
        expected = "frame-ancestors 'self' " + " ".join(origins)
        self.assertIn(expected, headers["Content-Security-Policy"])
        _, _, response = self.request("/api/question?split=validation&item_id=validation_000001")
        _, headers, _ = self.request(json.loads(response)["inspection_url"])
        self.assertIn(expected, headers["Content-Security-Policy"])
        for invalid in ("https://*.example.com", "https://example.com/path", "null",
                        "https://example.com;", "https://user@example.com", "https://example.com\n"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_origin(invalid)

    def test_development_session_rejects_held_out_data_in_all_routes(self):
        self.app.splits = ("train", "validation")
        self.assertNotIn("test", self.app.status()["counts"])
        for path, code in (("/api/questions?split=test", 400),
                           ("/api/question?split=test&item_id=test_000001", 400),
                           ("/api/image?split=test&item_id=test_000001", 400),
                           ("/api/inspection/test/test_000001", 404)):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, code)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.app.start_job({"mode": "hint", "split": "test", "item_id": "test_000001"})

    def test_failed_drive_copy_is_visible_and_keeps_local_result(self):
        self.app.persistent_output = self.root / "drive"
        self.app.jobs["copy-failure"] = {"status": "running"}
        with patch.object(self.app, "get_tutor", return_value=object()), patch(
                "ui_server.runner.run_hint", return_value={"hint": "Saved locally"}), patch(
                "ui_server.shutil.copy2", side_effect=OSError("Drive quota exceeded")), patch(
                "ui_server.traceback.print_exc"):
            self.app.run_job("copy-failure", "hint", self.item, None, None, None, 1)
        job = self.app.get_job("copy-failure")
        self.assertEqual(job["status"], "failed")
        self.assertIn("Drive quota exceeded", job["error"])
        local = self.root / "outputs/ui_copy-failure.json"
        self.assertEqual(json.loads(local.read_text())["hint"], "Saved locally")


if __name__ == "__main__":
    unittest.main()
