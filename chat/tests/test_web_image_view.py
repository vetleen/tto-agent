"""Tests for the web_image_view tool: handle resolution, SSRF-safe fetch,
sanitization gate, thread-asset provenance, and per-call caps."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from PIL import Image

from chat.models import Asset, ChatThread
from chat.web_image_tools import WebImageViewTool
from llm.tools.web_fetch import _SSRFBlocked
from llm.types.context import RunContext

User = get_user_model()

_FETCH = "llm.tools.web_fetch._pinned_get_following_redirects"
_IMG_URL = "https://cdn.example.com/diagram.png"
_PAGE_URL = "https://example.com/article/"


def _png(size=(4, 4)):
    img = Image.new("RGB", size, (0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resp(content, content_type="image/png"):
    r = MagicMock()
    r.headers = {"Content-Type": content_type} if content_type is not None else {}
    r.content = content
    r.raise_for_status = MagicMock()
    return r


class WebImageViewToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="v@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        self.handle = self.ctx.allocate_web_image_handle({
            "url": _IMG_URL, "page_url": _PAGE_URL, "filename": "diagram.png", "alt": "A diagram",
        })  # -> "img-1"

    def _invoke(self, args, *, return_value=None, side_effect=None):
        tool = WebImageViewTool()
        tool.set_context(self.ctx)
        kwargs = {}
        if side_effect is not None:
            kwargs["side_effect"] = side_effect
        else:
            kwargs["return_value"] = (return_value, _IMG_URL)
        with patch(_FETCH, **kwargs):
            return tool.invoke(args)

    def test_view_creates_asset_with_provenance(self):
        result = self._invoke({"handles": ["img-1"]}, return_value=_resp(_png()))

        self.assertIn("img-1: viewed", result)
        self.assertIn("[[image:", result)
        self.assertIn("paste this token", result)  # directs the model to embed
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 1)
        asset = Asset.objects.get(thread=self.thread)
        self.assertEqual(asset.source_url, _IMG_URL)
        self.assertEqual(asset.source_page_url, _PAGE_URL)
        self.assertEqual(asset.content_type, "image/png")
        self.assertTrue(asset.blob)
        # Surfaced to the model this turn.
        self.assertEqual(len(self.ctx.pending_image_assets), 1)
        self.assertEqual(self.ctx.pending_image_assets[0]["media_type"], "image/png")

    def test_unknown_handle_errors(self):
        result = self._invoke({"handles": ["img-99"]}, return_value=_resp(_png()))
        self.assertIn("unknown handle", result)
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 0)

    def test_mixed_known_and_unknown(self):
        result = self._invoke({"handles": ["img-99", "img-1"]}, return_value=_resp(_png()))
        self.assertIn("img-99: unknown handle", result)
        self.assertIn("img-1: viewed", result)
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 1)

    def test_non_image_content_type_rejected(self):
        result = self._invoke({"handles": ["img-1"]}, return_value=_resp(b"<html>", content_type="text/html"))
        self.assertIn("not an image", result)
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 0)

    def test_spoofed_mime_rejected_by_sanitizer(self):
        # Declares image/png but the body is not a real image.
        result = self._invoke({"handles": ["img-1"]}, return_value=_resp(b"not an image", content_type="image/png"))
        self.assertIn("img-1", result)
        self.assertIn("sanitiz", result.lower())
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 0)

    def test_ssrf_blocked_reports_error(self):
        result = self._invoke({"handles": ["img-1"]}, side_effect=_SSRFBlocked("private or reserved IP"))
        self.assertIn("could not fetch", result)
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 0)

    def test_same_image_reuses_thread_asset(self):
        self._invoke({"handles": ["img-1"]}, return_value=_resp(_png()))
        self._invoke({"handles": ["img-1"]}, return_value=_resp(_png()))
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 1)

    def test_duplicate_handles_are_fetched_once(self):
        tool = WebImageViewTool()
        tool.set_context(self.ctx)
        with patch(_FETCH, return_value=(_resp(_png()), _IMG_URL)) as fetch:
            result = tool.invoke({"handles": ["img-1"] * 50})

        fetch.assert_called_once()
        self.assertEqual(result.count("img-1: viewed"), 1)
        self.assertEqual(len(self.ctx.pending_image_assets), 1)

    def test_failed_fetches_are_capped_at_first_four_unique_handles(self):
        urls = [_IMG_URL]
        for i in range(5):
            url = f"https://cdn.example.com/i{i}.png"
            urls.append(url)
            self.ctx.allocate_web_image_handle({
                "url": url, "page_url": _PAGE_URL,
                "filename": f"i{i}.png", "alt": f"img {i}",
            })
        tool = WebImageViewTool()
        tool.set_context(self.ctx)
        with patch(_FETCH, side_effect=_SSRFBlocked("private or reserved IP")) as fetch:
            result = tool.invoke({"handles": [f"img-{n}" for n in range(1, 7)]})

        self.assertEqual(fetch.call_count, 4)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], urls[:4])
        self.assertIn("max 4 images per call", result)
        self.assertEqual(len(self.ctx.pending_image_assets), 0)

    def test_per_call_attachment_cap(self):
        for i in range(6):
            self.ctx.allocate_web_image_handle({
                "url": f"https://cdn.example.com/i{i}.png", "page_url": _PAGE_URL,
                "filename": f"i{i}.png", "alt": f"img {i}",
            })
        handles = [f"img-{n}" for n in range(2, 8)]  # 6 fresh valid handles
        result = self._invoke({"handles": handles}, return_value=_resp(_png()))
        self.assertIn("max 4 images per call", result)
        self.assertEqual(len(self.ctx.pending_image_assets), 4)

    def test_subagent_context_can_view(self):
        sub_ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        sub_ctx.agent_kind = "subagent"
        handle = sub_ctx.allocate_web_image_handle({
            "url": _IMG_URL, "page_url": _PAGE_URL, "filename": "diagram.png", "alt": "A diagram",
        })
        tool = WebImageViewTool()
        tool.set_context(sub_ctx)
        with patch(_FETCH, return_value=(_resp(_png()), _IMG_URL)):
            result = tool.invoke({"handles": [handle]})
        self.assertIn("viewed", result)
        self.assertEqual(Asset.objects.filter(thread=self.thread).count(), 1)
