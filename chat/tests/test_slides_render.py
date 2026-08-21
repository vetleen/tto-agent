"""Tests for chat.slides.render_service (LibreOffice step mocked)."""

from __future__ import annotations

import base64
import io
import uuid
import zipfile
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import Asset, ChatThread, SlideRender, SlideRenderRun, SlideSet
from chat.slides import render_service
from chat.slides.render import RenderUnavailable

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _fake_render(pptx_bytes, **kwargs):
    """Return one fake PNG per slide actually built into the pptx."""
    z = zipfile.ZipFile(io.BytesIO(pptx_bytes))
    n = len([
        x for x in z.namelist()
        if x.startswith("ppt/slides/slide") and x.endswith(".xml") and "rels" not in x
    ])
    return b"%PDF-1.4 fake", [(_PNG, 100, 56) for _ in range(n)]


class RenderServiceTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"r+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.deck = SlideSet.objects.create(
            thread=self.thread, title="Deck",
            content={"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"id": "s1", "name": "A", "elements": []},
                {"id": "s2", "name": "B", "elements": []},
            ]},
        )

    def _run(self, purpose, slide_ids=None):
        return SlideRenderRun.objects.create(
            slide_set=self.deck, purpose=purpose, slide_ids=slide_ids or []
        )

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_user_preview_renders_all_and_notifies(self, m_render, m_notify):
        run = self._run(SlideRenderRun.Purpose.USER_PREVIEW)
        render_service.execute_render_run(str(run.id))
        run.refresh_from_db()
        self.assertEqual(run.status, SlideRenderRun.Status.COMPLETED)
        self.assertEqual(len(run.result["slides"]), 2)
        self.assertEqual(SlideRender.objects.filter(slide_set=self.deck).count(), 2)
        # Both fake PNGs are identical bytes, so store_slide_set_image dedupes to
        # one asset (real renders differ per slide). Every SlideRender has an asset.
        self.assertEqual(SlideRender.objects.filter(slide_set=self.deck, asset__isnull=False).count(), 2)
        self.assertGreaterEqual(Asset.objects.filter(slide_set=self.deck, kind=Asset.KIND_IMAGE).count(), 1)
        m_notify.assert_called_once()
        self.assertEqual(m_notify.call_args.args[2], "slidedeck.rendered")

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_cache_skips_unchanged_slides(self, m_render, m_notify):
        r1 = self._run(SlideRenderRun.Purpose.USER_PREVIEW)
        render_service.execute_render_run(str(r1.id))
        self.assertEqual(m_render.call_count, 1)
        # Second render of the unchanged deck: nothing dirty -> no LibreOffice call.
        r2 = self._run(SlideRenderRun.Purpose.USER_PREVIEW)
        render_service.execute_render_run(str(r2.id))
        self.assertEqual(m_render.call_count, 1)
        r2.refresh_from_db()
        self.assertEqual(len(r2.result["slides"]), 2)

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_changed_slide_re_renders(self, m_render, m_notify):
        render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))
        self.assertEqual(m_render.call_count, 1)
        # Change slide s2 only.
        self.deck.content["slides"][1]["name"] = "B changed"
        self.deck.save(update_fields=["content"])
        render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))
        self.assertEqual(m_render.call_count, 2)

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_pdf_export_stores_pdf_asset(self, m_render, m_notify):
        run = self._run(SlideRenderRun.Purpose.PDF_EXPORT)
        render_service.execute_render_run(str(run.id))
        run.refresh_from_db()
        self.assertIn("pdf_asset_id", run.result)
        self.assertTrue(Asset.objects.filter(pk=run.result["pdf_asset_id"], kind=Asset.KIND_FILE).exists())
        self.assertEqual(m_notify.call_args.args[2], "slidedeck.pdf_ready")

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=RenderUnavailable("no soffice"))
    def test_failed_render_marks_failed_and_notifies(self, m_render, m_notify):
        run = self._run(SlideRenderRun.Purpose.USER_PREVIEW)
        render_service.execute_render_run(str(run.id))
        run.refresh_from_db()
        self.assertEqual(run.status, SlideRenderRun.Status.FAILED)
        self.assertIn("no soffice", run.error)
        self.assertEqual(m_notify.call_args.args[2], "slidedeck.render_failed")

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_agent_preview_does_not_notify(self, m_render, m_notify):
        run = self._run(SlideRenderRun.Purpose.AGENT_PREVIEW, slide_ids=["s1"])
        render_service.execute_render_run(str(run.id))
        run.refresh_from_db()
        self.assertEqual(run.status, SlideRenderRun.Status.COMPLETED)
        self.assertEqual(len(run.result["slides"]), 1)
        m_notify.assert_not_called()

    @mock.patch("chat.slides.render_service.notify_render_event")
    def test_re_render_deletes_superseded_asset(self, m_notify):
        counter = {"n": 0}

        def uniq_render(pptx_bytes, **kwargs):
            counter["n"] += 1
            n = len([
                x for x in zipfile.ZipFile(io.BytesIO(pptx_bytes)).namelist()
                if x.startswith("ppt/slides/slide") and x.endswith(".xml") and "rels" not in x
            ])
            return b"%PDF", [(_PNG + f"-{counter['n']}-{i}".encode(), 100, 56) for i in range(n)]

        with mock.patch("chat.slides.render_service.render_pptx", side_effect=uniq_render):
            render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))
            s1_asset = SlideRender.objects.get(slide_set=self.deck, slide_id="s1").asset_id
            s2_asset = SlideRender.objects.get(slide_set=self.deck, slide_id="s2").asset_id
            # change only s2
            self.deck.content["slides"][1]["name"] = "B changed"
            self.deck.save(update_fields=["content"])
            render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))

        self.assertFalse(Asset.objects.filter(pk=s2_asset).exists(), "old s2 asset should be deleted")
        self.assertTrue(Asset.objects.filter(pk=s1_asset).exists(), "s1 asset should be untouched")
        self.assertNotEqual(SlideRender.objects.get(slide_set=self.deck, slide_id="s2").asset_id, s2_asset)

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_removed_slide_render_is_pruned(self, m_render, m_notify):
        render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))
        self.assertEqual(SlideRender.objects.filter(slide_set=self.deck).count(), 2)
        # remove s2 from the deck
        self.deck.content["slides"] = [self.deck.content["slides"][0]]
        self.deck.save(update_fields=["content"])
        render_service.execute_render_run(str(self._run(SlideRenderRun.Purpose.USER_PREVIEW).id))
        self.assertFalse(SlideRender.objects.filter(slide_set=self.deck, slide_id="s2").exists())
        self.assertEqual(SlideRender.objects.filter(slide_set=self.deck).count(), 1)

    @mock.patch("chat.slides.render_service.notify_render_event")
    @mock.patch("chat.slides.render_service.render_pptx", side_effect=_fake_render)
    def test_completed_run_is_idempotent(self, m_render, m_notify):
        run = self._run(SlideRenderRun.Purpose.USER_PREVIEW)
        render_service.execute_render_run(str(run.id))
        render_service.execute_render_run(str(run.id))  # second call is a no-op
        self.assertEqual(m_render.call_count, 1)
