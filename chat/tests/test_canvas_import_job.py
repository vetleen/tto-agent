"""Canvas import on the worker: view queues, worker fills, status reports."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils import timezone

from chat.canvas_import import EMPTY_IMPORT_ERROR, expire_stale_imports, run_canvas_import
from chat.models import CanvasImportJob, ChatCanvas, ChatThread

User = get_user_model()


def _txt(name="brief.txt", body=b"# Brief\n\nBody text."):
    return SimpleUploadedFile(name, body, content_type="text/plain")


class _Base(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cij@test.com", password="pass")
        self.client.login(email="cij@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user, title="T")

    def url(self, canvas_id=None):
        mid = f"{canvas_id}/" if canvas_id else ""
        return f"/chat/threads/{self.thread.id}/canvas/{mid}import/"

    def post(self, f, canvas_id=None):
        with patch("chat.tasks.process_canvas_import.delay") as delay:
            resp = self.client.post(self.url(canvas_id), {"file": f}, format="multipart")
        return resp, delay


class ImportViewTests(_Base):
    def test_returns_202_and_queues_without_extracting(self):
        with patch("chat.services.import_file_to_canvas") as extract:
            resp, delay = self.post(_txt())
        self.assertEqual(resp.status_code, 202)
        data = resp.json()
        job = CanvasImportJob.objects.get(pk=data["job_id"])
        self.assertEqual(job.state, "pending")
        self.assertTrue(job.created_canvas)
        self.assertTrue(job.file)
        self.assertEqual(job.original_filename, "brief.txt")
        delay.assert_called_once_with(str(job.pk))
        extract.assert_not_called()  # no extraction inside the request
        canvas = ChatCanvas.objects.get(pk=data["canvas_id"])
        self.assertEqual(canvas.title, "brief")
        self.assertEqual(data["title"], "brief")
        self.assertIn(str(job.pk), data["status_url"])
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas_id, canvas.pk)  # tab focused

    def test_validation_still_happens_in_the_request(self):
        exe = SimpleUploadedFile("evil.exe", b"MZ", content_type="application/x-msdownload")
        resp, delay = self.post(exe)
        self.assertEqual(resp.status_code, 400)
        delay.assert_not_called()
        self.assertFalse(CanvasImportJob.objects.exists())
        self.assertFalse(ChatCanvas.objects.exists())

    def test_second_import_into_same_canvas_is_refused(self):
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Doc", content="x")
        first, _ = self.post(_txt(), canvas_id=canvas.pk)
        self.assertEqual(first.status_code, 202)
        second, delay = self.post(_txt(), canvas_id=canvas.pk)
        self.assertEqual(second.status_code, 409)
        delay.assert_not_called()

    def test_canvas_cap_enforced_for_a_new_canvas(self):
        from chat.services import MAX_CANVASES_PER_THREAD

        for i in range(MAX_CANVASES_PER_THREAD):
            ChatCanvas.objects.create(thread=self.thread, title=f"C{i}", content="x")
        resp, delay = self.post(_txt("new.txt"))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("maximum", resp.json()["error"])
        delay.assert_not_called()

    def test_broker_failure_returns_503_and_drops_created_canvas(self):
        with patch("chat.tasks.process_canvas_import.delay", side_effect=RuntimeError("broker down")):
            resp = self.client.post(self.url(), {"file": _txt()}, format="multipart")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())
        job = CanvasImportJob.objects.get()
        self.assertEqual(job.state, "failed")
        self.assertFalse(job.file)


class ImportWorkerTests(_Base):
    def _queued(self, f=None, canvas_id=None):
        resp, _ = self.post(f or _txt(), canvas_id=canvas_id)
        return CanvasImportJob.objects.get(pk=resp.json()["job_id"])

    @patch("chat.canvas_import._push")
    def test_fills_canvas_with_checkpoint_and_baseline(self, push):
        job = self._queued()
        self.assertEqual(run_canvas_import(job.pk), "ready")
        job.refresh_from_db()
        self.assertEqual(job.state, "ready")
        self.assertIsNotNone(job.finished_at)
        self.assertFalse(job.file)  # stored upload removed
        canvas = job.canvas
        canvas.refresh_from_db()
        self.assertIn("Body text.", canvas.content)
        self.assertIsNone(canvas.truncated_from_chars)
        self.assertEqual(canvas.accepted_checkpoint.source, "import")
        event = push.call_args_list[0].args[1]
        self.assertEqual(event["event_type"], "canvas.updated")
        self.assertEqual(event["canvas_id"], str(canvas.pk))  # str: the JS compares with ===
        self.assertEqual(event["import_job_id"], str(job.pk))
        self.assertFalse(event["pending_ai_review"])

    @patch("chat.canvas_import._push")
    def test_records_truncation(self, _push):
        from chat.services import CANVAS_MAX_CHARS

        job = self._queued(_txt("long.txt", b"w" * (CANVAS_MAX_CHARS + 7)))
        run_canvas_import(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.original_chars, CANVAS_MAX_CHARS + 7)
        job.canvas.refresh_from_db()
        self.assertEqual(job.canvas.truncated_from_chars, CANVAS_MAX_CHARS + 7)
        self.assertEqual(len(job.canvas.content), CANVAS_MAX_CHARS)

    @patch("chat.canvas_import._push")
    @patch("chat.services.import_file_to_canvas", return_value=("scan", "  ", None))
    def test_empty_text_fails_and_drops_created_canvas(self, _extract, _push):
        job = self._queued(SimpleUploadedFile("scan.pdf", b"%PDF", content_type="application/pdf"))
        canvas_pk = job.canvas_id
        self.assertEqual(run_canvas_import(job.pk), "failed")
        job.refresh_from_db()
        self.assertEqual(job.error, EMPTY_IMPORT_ERROR)
        self.assertFalse(ChatCanvas.objects.filter(pk=canvas_pk).exists())

    @patch("chat.canvas_import._push")
    @patch("chat.services.import_file_to_canvas", return_value=("scan", "", None))
    def test_empty_text_keeps_an_existing_canvas(self, _extract, _push):
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Keep", content="keep me")
        job = self._queued(SimpleUploadedFile("scan.pdf", b"%PDF", content_type="application/pdf"), canvas_id=canvas.pk)
        run_canvas_import(job.pk)
        canvas.refresh_from_db()
        self.assertEqual(canvas.content, "keep me")

    @patch("chat.canvas_import._push")
    def test_user_edits_made_while_importing_are_kept_in_history(self, _push):
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Doc", content="old")
        job = self._queued(canvas_id=canvas.pk)
        ChatCanvas.objects.filter(pk=canvas.pk).update(content="typed by the user meanwhile")
        run_canvas_import(job.pk)
        contents = list(canvas.checkpoints.order_by("order").values_list("content", flat=True))
        self.assertIn("typed by the user meanwhile", contents)
        canvas.refresh_from_db()
        self.assertIn("Body text.", canvas.content)

    @patch("chat.canvas_import._push")
    @patch("chat.services.generate_canvas_title", return_value="Generated")
    def test_titles_an_untitled_thread(self, gen, push):
        ChatThread.objects.filter(pk=self.thread.pk).update(title="")
        job = self._queued()
        run_canvas_import(job.pk)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.title, "Generated")
        self.assertEqual(gen.call_args.kwargs["conversation_id"], str(self.thread.id))
        events = [c.args[1]["event_type"] for c in push.call_args_list]
        self.assertEqual(events, ["canvas.updated", "thread.title_updated"])

    @patch("chat.canvas_import._push")
    @patch("chat.services.generate_canvas_title")
    def test_keeps_an_existing_title(self, gen, _push):
        job = self._queued()
        run_canvas_import(job.pk)
        gen.assert_not_called()

    def test_push_failure_does_not_fail_the_job(self):
        job = self._queued()
        with patch("channels.layers.get_channel_layer", side_effect=RuntimeError("redis down")):
            self.assertEqual(run_canvas_import(job.pk), "ready")

    @patch("chat.canvas_import._push")
    def test_duplicate_delivery_is_a_no_op(self, push):
        job = self._queued()
        run_canvas_import(job.pk)
        self.assertEqual(run_canvas_import(job.pk), "ready")
        self.assertEqual(len(push.call_args_list), 1)

    @patch("chat.canvas_import._push")
    def test_deleted_canvas_fails_the_job(self, _push):
        job = self._queued()
        ChatCanvas.objects.filter(pk=job.canvas_id).update(deleted_at=timezone.now())
        self.assertEqual(run_canvas_import(job.pk), "failed")

    def test_task_failure_settles_the_job(self):
        from chat.tasks import _CanvasImportTask

        job = self._queued()
        _CanvasImportTask().on_failure(RuntimeError("boom"), "tid", (str(job.pk),), {}, None)
        job.refresh_from_db()
        self.assertEqual(job.state, "failed")
        self.assertFalse(ChatCanvas.objects.filter(pk=job.canvas_id).exists())

    def test_stale_jobs_expire(self):
        job = self._queued()
        CanvasImportJob.objects.filter(pk=job.pk).update(created_at=timezone.now() - timedelta(minutes=20))
        self.assertEqual(expire_stale_imports(), 1)
        job.refresh_from_db()
        self.assertEqual(job.state, "failed")

    @patch("chat.canvas_import._push")
    def test_reports_image_progress(self, _push):
        from documents.services.progress import canvas_imports

        stages = []
        real = canvas_imports.set_stage

        def spy(owner_id, stage, **kw):
            stages.append(stage)
            return real(owner_id, stage, **kw)

        def fake_extract(uploaded, user, *, canvas=None, progress=None):
            progress("describing_images", 1, 0)
            return "brief", "Body", None

        job = self._queued()
        with patch.object(canvas_imports, "set_stage", side_effect=spy), \
                patch("chat.services.import_file_to_canvas", side_effect=fake_extract):
            run_canvas_import(job.pk)
        self.assertEqual(stages, ["extracting", "describing_images"])


class ImportStatusTests(_Base):
    def test_pending_then_ready_payload(self):
        resp, _ = self.post(_txt())
        status_url = resp.json()["status_url"]
        pending = self.client.get(status_url).json()
        self.assertEqual(pending["state"], "pending")
        self.assertIn("progress", pending)
        self.assertNotIn("canvas", pending)
        with patch("chat.canvas_import._push"):
            run_canvas_import(resp.json()["job_id"])
        ready = self.client.get(status_url).json()
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["canvas"]["event_type"], "canvas.updated")
        self.assertEqual(ready["canvas"]["canvas_id"], resp.json()["canvas_id"])
        self.assertIn("Body text.", ready["canvas"]["content"])

    def test_failed_reports_error(self):
        resp, _ = self.post(SimpleUploadedFile("scan.pdf", b"%PDF", content_type="application/pdf"))
        with patch("chat.services.import_file_to_canvas", return_value=("scan", "", None)), \
                patch("chat.canvas_import._push"):
            run_canvas_import(resp.json()["job_id"])
        failed = self.client.get(resp.json()["status_url"]).json()
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error"], EMPTY_IMPORT_ERROR)

    def test_other_users_cannot_read_status(self):
        resp, _ = self.post(_txt())
        User.objects.create_user(email="other-cij@test.com", password="pass")
        self.client.logout()
        self.client.login(email="other-cij@test.com", password="pass")
        self.assertEqual(self.client.get(resp.json()["status_url"]).status_code, 404)


class CanvasAssetSinkProgressTests(TestCase):
    def test_sink_reports_each_picture(self):
        from chat.assets import _describe_and_store_sink

        calls = []
        image = MagicMock(content_type="image/png", alt_text="")
        image.open.return_value.__enter__.return_value.read.return_value = b"png"
        store = MagicMock(return_value=MagicMock(id="00000000-0000-0000-0000-000000000001"))
        sink = _describe_and_store_sink(store, None, max_described=0, progress=lambda *a: calls.append(a))
        sink(image, 1)
        sink(image, 2)
        self.assertEqual(calls, [("describing_images", 1, 0), ("describing_images", 2, 0)])
