import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer

import run_scienceqa_tutor as runner
from ui_server import TutorWebApp


class TutorWebAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.questions_dir = self.root / "data/processed/scienceqa/questions"
        self.questions_dir.mkdir(parents=True)
        reference_dir = self.root / "data/processed/scienceqa/references"
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
        image_dir = self.root / "data/images"
        image_dir.mkdir(parents=True)
        (image_dir / "example.png").write_bytes(b"test-image")
        self.item = {
            "item_id": "validation_000001",
            "split": "validation",
            "question": "What does this diagram show?",
            "context": "Use the labels.",
            "choices": [{"id": "A", "text": "A sample choice"}],
            "image_path": "data/images/example.png",
            "gold_option_id": "SECRET ANSWER",
            "solution": "SECRET SOLUTION",
        }
        (self.questions_dir / "validation.jsonl").write_text(
            json.dumps(self.item) + "\n", encoding="utf-8"
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

    def test_ui_and_question_routes_only_expose_public_question_fields(self):
        status, _, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"science hint studio", page.lower())

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
            self.root, "validation", self.item["item_id"]
        )
        self.assertEqual(checked.call_args.args[2]["gold_option_id"], "A")


if __name__ == "__main__":
    unittest.main()
