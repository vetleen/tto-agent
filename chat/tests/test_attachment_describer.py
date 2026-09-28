"""``chat.assets.AttachmentImageDescriber`` — the chat/meeting-attachment arm of
the shared two-phase describer: concurrent vision calls, in-document dedupe,
the org cache shared with data rooms, the unique-picture cap, and the progress
hook ``extract_attachment_text`` exposes to the worker task.

Locmem cache (the real backend is fail-open Redis) and in-memory file storage,
as in ``documents/tests/test_progress.py`` / ``test_attachment_processing.py``.
"""
from __future__ import annotations

import threading
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings

from chat.assets import AttachmentImageDescriber, store_attachment_image
from chat.models import Asset, ChatThread
from chat.tests.test_attachment_processing import _DOCX_MIME, _PDF_MIME, _make_attachment
from chat.tests.test_attachments import _docx_with_image
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.services.image_assets import EmbeddedImageDescriber
from documents.tests.test_progress import LOCMEM, _FakeImg, _png

User = get_user_model()
_MODEL = "anthropic/claude-opus-4-8"
_STORAGE = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)


@_STORAGE
@override_settings(CACHES=LOCMEM, DOCUMENT_IMAGE_DESCRIBE_CONCURRENCY=4)
class AttachmentImageDescriberTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(email="desc@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.att = _make_attachment(self.thread, self.user, "d.pdf", _PDF_MIME, b"%PDF")

    def _describer(self, **kw):
        kw.setdefault("model", _MODEL)
        org_id = kw.pop("org_id", None)
        d = AttachmentImageDescriber(self.att, self.user, **kw)
        d.org_id = org_id  # stable org id without a Membership fixture
        return d

    def _embedded(self):
        return Asset.objects.filter(attachment=self.att, role=Asset.ROLE_EMBEDDED)

    def test_runs_concurrently_and_reports_progress(self):
        n = 4
        barrier = threading.Barrier(n, timeout=8)  # deadlocks if calls run serially
        threads_seen = set()

        def fake_describe(image_bytes, content_type, user, alt_text=None, model=None):
            threads_seen.add(threading.get_ident())
            barrier.wait()
            return f"desc {len(image_bytes)}"

        d = self._describer()
        tokens = [d.sink(_FakeImg(_png(i)), i + 1) for i in range(n)]
        self.assertEqual(d.total, n)
        progress_calls = []
        with patch("chat.services.describe_image", side_effect=fake_describe):
            described = d.run_descriptions(progress_cb=lambda c, t: progress_calls.append((c, t)))

        self.assertEqual(len(described), n)
        self.assertEqual(len(threads_seen), n)
        self.assertEqual(progress_calls[-1], (n, n))
        self.assertEqual(self._embedded().count(), n)
        self.assertTrue(all(a.description.startswith("desc ") for a in self._embedded()))
        text = d.substitute("\n".join(tokens), described)
        self.assertNotIn("PNG image", text)

    def test_dedupes_identical_pictures_and_numbers_unique_ones(self):
        d = self._describer()
        with patch("chat.services.describe_image", return_value="A logo") as describe:
            t1 = d.sink(_FakeImg(_png(1)), 1)
            t2 = d.sink(_FakeImg(_png(1)), 2)  # same bytes again (docx/pptx repeat)
            t3 = d.sink(_FakeImg(_png(2)), 3)
            self.assertEqual(d.total, 2)
            d.run_descriptions()
        self.assertEqual(t1, t2)
        self.assertIn("Image 1:", t1)
        self.assertIn("Image 2:", t3)
        self.assertEqual(self._embedded().count(), 2)
        self.assertEqual(describe.call_count, 2)

    @override_settings(CHAT_ATTACHMENT_MAX_DESCRIBED_IMAGES=2)
    def test_cap_comes_from_the_setting_and_counts_unique_pictures(self):
        d = self._describer()
        self.assertEqual(d.max_described, 2)
        tokens = [d.sink(_FakeImg(_png(i)), i + 1) for i in (1, 1, 2, 3)]
        self.assertEqual(d.total, 2)  # the repeat is not a new picture
        self.assertIn("PNG image", tokens[3])  # third unique picture: fallback label

    def test_org_cache_is_shared_with_data_rooms(self):
        room = DataRoom.objects.create(name="R", slug="r-desc", created_by=self.user)
        doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="d.pdf",
            status=DataRoomDocument.Status.UPLOADED,
        )
        version = DataRoomDocumentVersion.objects.create(document=doc, version_index=0)
        png = _png(9)
        with patch("chat.services.describe_image", return_value="A shared chart") as describe:
            data_room = EmbeddedImageDescriber(version, doc)
            data_room.org_id = 7
            data_room.model = _MODEL
            data_room.sink(_FakeImg(png), 1)
            data_room.run_descriptions()

            chat = self._describer(org_id=7)
            token = chat.sink(_FakeImg(png), 1)
            self.assertEqual(chat.total, 0)  # cache hit → no call
            self.assertIn("A shared chart", token)
            self.assertEqual(chat.run_descriptions(), {})
        self.assertEqual(describe.call_count, 1)
        self.assertEqual(self._embedded().get().description, "A shared chart")

    def test_reprocessing_reuses_rows_and_keeps_real_descriptions(self):
        png = _png(5)
        existing = store_attachment_image(
            self.att, img_bytes=png, content_type="image/png", description="Old description",
            created_by=self.user,
        )
        d = self._describer()
        with patch("chat.services.describe_image", return_value="new") as describe:
            token = d.sink(_FakeImg(png), 1)
            d.run_descriptions()
        self.assertIn(str(existing.id), token)
        self.assertIn("Old description", token)
        self.assertEqual(d.total, 0)
        describe.assert_not_called()
        self.assertEqual(self._embedded().count(), 1)

    def test_reprocessing_redescribes_a_fallback_labelled_row(self):
        png = _png(6)
        existing = store_attachment_image(
            self.att, img_bytes=png, content_type="image/png", description="PNG image",
            created_by=self.user,
        )
        d = self._describer()
        with patch("chat.services.describe_image", return_value="Finally described"):
            d.sink(_FakeImg(png), 1)
            self.assertEqual(d.total, 1)
            d.run_descriptions()
        existing.refresh_from_db()
        self.assertEqual(existing.description, "Finally described")
        self.assertEqual(self._embedded().count(), 1)

    def test_no_vision_model_stores_pictures_with_fallback_labels(self):
        with patch("chat.services.resolve_vision_model", return_value=None) as resolver, \
             patch("chat.services.describe_image") as describe:
            d = AttachmentImageDescriber(self.att, self.user)
            token = d.sink(_FakeImg(_png(3)), 1)
            self.assertEqual(d.run_descriptions(), {})
        resolver.assert_called_once_with(self.user)
        describe.assert_not_called()
        self.assertIn("PNG image", token)
        self.assertEqual(self._embedded().count(), 1)

    def test_model_is_resolved_once_per_attachment(self):
        with patch("chat.services.resolve_vision_model", return_value=_MODEL) as resolver, \
             patch("chat.services.describe_image", return_value="d") as describe:
            d = AttachmentImageDescriber(self.att, self.user)
            for i in range(3):
                d.sink(_FakeImg(_png(10 + i)), i + 1)
            d.run_descriptions()
        self.assertEqual(resolver.call_count, 1)
        self.assertEqual(describe.call_count, 3)
        self.assertTrue(all(c.kwargs["model"] == _MODEL for c in describe.call_args_list))


@_STORAGE
@override_settings(CACHES=LOCMEM)
class ExtractAttachmentTextProgressTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(email="stages@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_stages_are_reported_and_final_labels_are_cached(self):
        from chat.services import get_or_extract_attachment_text

        att = _make_attachment(self.thread, self.user, "n.docx", _DOCX_MIME, _docx_with_image())
        seen: list = []
        with patch("chat.services.resolve_vision_model", return_value=_MODEL), \
             patch("chat.services.describe_image", return_value="A red rectangle"):
            text = get_or_extract_attachment_text(
                att, _docx_with_image(), user=self.user, progress=lambda s, c, t: seen.append((s, c, t)),
            )
        self.assertEqual(seen, [("extracting", 0, 0), ("describing_images", 0, 1), ("describing_images", 1, 1)])
        att.refresh_from_db()
        self.assertIn("Image 1: A red rectangle", text)
        self.assertEqual(att.extracted_content, text)  # cached with the final label

    def test_progress_hook_errors_never_fail_extraction(self):
        from chat.services import get_or_extract_attachment_text

        att = _make_attachment(self.thread, self.user, "n.docx", _DOCX_MIME, _docx_with_image())

        def boom(*_a):
            raise RuntimeError("redis down")

        with patch("chat.services.resolve_vision_model", return_value=_MODEL), \
             patch("chat.services.describe_image", return_value="ok"):
            text = get_or_extract_attachment_text(att, _docx_with_image(), user=self.user, progress=boom)
        self.assertIn("Image 1: ok", text)
