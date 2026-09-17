"""Tests for processing-progress plumbing and the two-phase embedded-image
describer's concurrency + org-scoped description cache.

These use a locmem cache (the real cache backend is Redis, which is fail-open and
would no-op under test) so ``documents.services.progress`` and the describer's
``imgdesc:`` cache actually round-trip.
"""

from __future__ import annotations

import io
import tempfile
import threading
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.services.image_assets import EmbeddedImageDescriber

User = get_user_model()

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}


def _png(seed: int) -> bytes:
    """A small, valid, content-distinct PNG (distinct color -> distinct sha256)."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeImg:
    def __init__(self, data, content_type="image/png", alt_text=""):
        self._data = data
        self.content_type = content_type
        self.alt_text = alt_text

    def open(self):
        return io.BytesIO(self._data)


@override_settings(CACHES=LOCMEM)
class ProgressStoreTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_set_and_read_round_trip(self):
        from documents.services import progress

        progress.set_stage(42, "describing_images", current=3, total=10)
        out = progress.read_many([42, 99])
        self.assertEqual(out[42], {"stage": "describing_images", "current": 3, "total": 10})
        self.assertNotIn(99, out)  # no dict stored -> absent, not an error

    def test_bump_overwrites_counter(self):
        from documents.services import progress

        progress.set_stage(7, "extracting")
        progress.bump_progress(7, "describing_images", 5, 8)
        self.assertEqual(
            progress.read_many([7])[7],
            {"stage": "describing_images", "current": 5, "total": 8},
        )

    def test_clear(self):
        from documents.services import progress

        progress.set_stage(7, "chunking")
        progress.clear(7)
        self.assertEqual(progress.read_many([7]), {})

    def test_read_many_empty_input(self):
        from documents.services import progress

        self.assertEqual(progress.read_many([]), {})
        self.assertEqual(progress.read_many([None]), {})


@override_settings(CACHES=LOCMEM, ALLOWED_HOSTS=["testserver"])
class DocumentStatusProgressTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(email="p@example.com", password="x")
        self.data_room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.client.force_login(self.user)

    def _doc(self, status):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.pdf", status=status,
        )
        v = DataRoomDocumentVersion.objects.create(document=doc, version_index=0)
        doc.current_version = v
        doc.save(update_fields=["current_version"])
        return doc, v

    def test_progress_returned_for_processing_doc(self):
        doc, v = self._doc(DataRoomDocument.Status.PROCESSING)
        from documents.services import progress

        progress.set_stage(v.id, "describing_images", current=2, total=5)
        resp = self.client.get(reverse("document_status", args=[self.data_room.uuid]))
        data = resp.json()
        self.assertEqual(data["statuses"][str(doc.id)], "processing")
        self.assertEqual(
            data["progress"][str(doc.id)],
            {"stage": "describing_images", "current": 2, "total": 5},
        )

    def test_no_progress_for_terminal_doc(self):
        doc, v = self._doc(DataRoomDocument.Status.READY)
        from documents.services import progress

        # A stale dict left over from processing must not leak for a ready doc.
        progress.set_stage(v.id, "describing_images", current=2, total=5)
        resp = self.client.get(reverse("document_status", args=[self.data_room.uuid]))
        data = resp.json()
        self.assertEqual(data["statuses"][str(doc.id)], "ready")
        self.assertNotIn(str(doc.id), data.get("progress", {}))


@override_settings(CACHES=LOCMEM)
class ImageDescribeCacheTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(email="c@example.com", password="x")
        self.data_room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.pdf", status=DataRoomDocument.Status.UPLOADED,
        )
        self.v1 = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=0)
        self.v2 = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=1)

    def _describer(self, version, org_id):
        d = EmbeddedImageDescriber(version, self.doc)
        d.org_id = org_id  # force a stable org id (no Membership in this test)
        d.model = "anthropic/claude-opus-4-8"
        return d

    def test_org_cache_short_circuits_second_identical_image(self):
        png = _png(3)
        with tempfile.TemporaryDirectory() as tmpdir, self.settings(MEDIA_ROOT=tmpdir):
            with patch("chat.services.describe_image", return_value="A shared logo") as mock_describe:
                d1 = self._describer(self.v1, org_id=7)
                d1.sink(_FakeImg(png), 1)
                self.assertEqual(d1.total, 1)  # queued for a vision call
                described = d1.run_descriptions()
                self.assertEqual(len(described), 1)

                # A different document in the SAME org, same image bytes.
                d2 = self._describer(self.v2, org_id=7)
                token = d2.sink(_FakeImg(png), 1)
                self.assertEqual(d2.total, 0)  # cache hit -> not queued
                self.assertIn("A shared logo", token)  # label already final
                self.assertEqual(d2.run_descriptions(), {})
        self.assertEqual(mock_describe.call_count, 1)  # described exactly once

    def test_no_cache_when_org_is_none(self):
        png = _png(4)
        with tempfile.TemporaryDirectory() as tmpdir, self.settings(MEDIA_ROOT=tmpdir):
            with patch("chat.services.describe_image", return_value="desc") as mock_describe:
                d1 = self._describer(self.v1, org_id=None)
                d1.sink(_FakeImg(png), 1)
                d1.run_descriptions()
                d2 = self._describer(self.v2, org_id=None)
                d2.sink(_FakeImg(png), 1)
                d2.run_descriptions()
        self.assertEqual(mock_describe.call_count, 2)  # no cross-doc reuse


@override_settings(CACHES=LOCMEM, DOCUMENT_IMAGE_DESCRIBE_CONCURRENCY=4)
class ImageDescribeConcurrencyTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(email="k@example.com", password="x")
        self.data_room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.pdf", status=DataRoomDocument.Status.UPLOADED,
        )
        self.version = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=0)

    def test_run_descriptions_runs_concurrently_and_counts(self):
        n = 4
        # A barrier of n parties only releases if all n calls are in flight at
        # once — i.e. it deadlocks (and times out) if descriptions run serially.
        barrier = threading.Barrier(n, timeout=8)
        threads_seen = set()

        def fake_describe(image_bytes, content_type, user, alt_text=None, model=None):
            threads_seen.add(threading.get_ident())
            barrier.wait()
            return "desc"

        describer = EmbeddedImageDescriber(self.version, self.doc)
        describer.org_id = None
        describer.model = "anthropic/claude-opus-4-8"

        progress_calls = []
        with tempfile.TemporaryDirectory() as tmpdir, self.settings(MEDIA_ROOT=tmpdir):
            for i in range(n):
                describer.sink(_FakeImg(_png(i)), i + 1)
            self.assertEqual(describer.total, n)
            with patch("chat.services.describe_image", side_effect=fake_describe):
                described = describer.run_descriptions(
                    progress_cb=lambda c, t: progress_calls.append((c, t)),
                )

        self.assertEqual(len(described), n)
        self.assertEqual(len(threads_seen), n)  # truly ran on n threads
        self.assertEqual(progress_calls[-1], (n, n))  # counter reached N/N


class DescribeImagePromptTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="q@example.com", password="x")

    def test_prompt_keeps_verbatim_transcription(self):
        """The shortened prompt must still instruct verbatim text transcription."""
        from chat.services import describe_image

        resp = MagicMock()
        resp.message.content = "desc"
        fake_service = MagicMock()
        fake_service.run.return_value = resp
        # describe_image does `from llm import get_llm_service` at call time.
        with patch("llm.get_llm_service", return_value=fake_service):
            describe_image(_png(1), "image/png", self.user, model="anthropic/claude-opus-4-8")

        request = fake_service.run.call_args.args[1]
        prompt_text = request.messages[0].content[0]["text"].lower()
        self.assertIn("verbatim", prompt_text)
