"""Tests for the Group A security fixes from the chat-app review.

Covers:
- A1: document_open_to_canvas is main-audience (not exposed to sub-agents).
- A2: canvas-export SSRF guards (WeasyPrint url_fetcher + remote-<img> stripping).
- A3: document_open_to_canvas gates a still-scanning upload, keeps agent remediation.
- A4: document_view_image gates quarantined / still-scanning documents.
- A5: the Loops page embeds JSON via json_script (no </script> breakout).
"""
from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from chat.models import ChatThread
from chat.pdf_export import _safe_url_fetcher
from chat.tools import DocumentViewImageTool, OpenDocumentToCanvasTool
from chat.views import _strip_remote_images
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.tests._helpers import make_document, make_version
from llm.types.context import RunContext

User = get_user_model()

SCANNING = DataRoomDocument.Status.SCANNING
READY = DataRoomDocument.Status.READY


# ---------------------------------------------------------------------------
# A1 — audience
# ---------------------------------------------------------------------------
class OpenToCanvasAudienceTests(TestCase):
    def test_tool_is_main_audience(self):
        tool = OpenDocumentToCanvasTool()
        # Root cause: both the preferences filter and the pipeline's defensive
        # _resolve_tools filter key on .audience; "main" excludes sub-agents.
        self.assertEqual(tool.audience, "main")
        # The dead subagent_section="chat" override was removed.
        self.assertIsNone(getattr(tool, "subagent_section", None))

    def test_excluded_from_computed_subagent_tools(self):
        from core.preferences import get_preferences

        user = User.objects.create_user(email="aud@x.io", password="p")
        prefs = get_preferences(user)
        self.assertNotIn("document_open_to_canvas", prefs.allowed_subagent_tools)


# ---------------------------------------------------------------------------
# A2 — SSRF guards
# ---------------------------------------------------------------------------
class ExportSsrfGuardTests(TestCase):
    def test_url_fetcher_blocks_remote_and_file(self):
        for url in (
            "http://169.254.169.254/latest/meta-data/",
            "https://evil.example/x.png",
            "file:///app/.env",
            "ftp://host/x",
        ):
            with self.assertRaises(ValueError):
                _safe_url_fetcher(url)

    def test_strip_remote_images_drops_non_data_src(self):
        html = (
            '<p>a</p>'
            '<img src="http://169.254.169.254/x.png">'
            '<img src="file:///etc/passwd">'
            '<img src="data:image/png;base64,AAAA" alt="ok">'
        )
        out = _strip_remote_images(html)
        self.assertNotIn("169.254.169.254", out)
        self.assertNotIn("file:///etc/passwd", out)
        # The legitimate embedded (data:) image survives.
        self.assertIn("data:image/png;base64,AAAA", out)


# ---------------------------------------------------------------------------
# A3 — open-to-canvas scan gate
# ---------------------------------------------------------------------------
class OpenToCanvasGateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="otc@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(self.thread.id),
            data_room_ids=[self.room.pk],
        )

    def _open(self, doc):
        tool = OpenDocumentToCanvasTool()
        tool.set_context(self.ctx)
        return json.loads(tool.invoke({"doc_index": doc.doc_index}))

    def test_scanning_upload_is_blocked(self):
        doc = make_document(self.room, self.user, status=SCANNING, chunks=["x"])
        res = self._open(doc)
        self.assertIn("error", res)
        self.assertIn("still processing", res["error"])

    def test_ready_document_opens(self):
        doc = make_document(self.room, self.user, status=READY, chunks=["clean text"])
        res = self._open(doc)
        self.assertEqual(res.get("status"), "ok")

    def test_quarantined_agent_draft_still_opens_for_remediation(self):
        # A quarantined agent-authored draft is status=READY (release sets READY,
        # only withholds the searchable pointer) → must stay openable to remediate.
        doc = make_document(self.room, self.user, status=READY, chunks=["clean"])
        make_version(
            doc, version_index=1, status=READY, is_quarantined=True,
            origin=DataRoomDocumentVersion.Origin.AGENT_CREATED, chunks=["flagged"],
            make_active=False,
        )
        res = self._open(doc)
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("warning", res)


# ---------------------------------------------------------------------------
# A4 — view-image scan gate
# ---------------------------------------------------------------------------
class ViewImageGateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="vi@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])

    def _view(self, doc):
        tool = DocumentViewImageTool()
        tool.set_context(self.ctx)
        return tool.invoke({"doc_indices": [doc.doc_index]})

    def test_scanning_document_attaches_nothing(self):
        doc = make_document(self.room, self.user, status=SCANNING, chunks=["x"])
        msg = self._view(doc)
        self.assertIn("No document with index", msg)
        self.assertEqual(list(self.ctx.pending_image_assets), [])

    def test_quarantined_document_attaches_nothing(self):
        doc = make_document(self.room, self.user, status=READY, is_quarantined=True, chunks=["x"])
        msg = self._view(doc)
        self.assertIn("quarantined", msg)
        self.assertEqual(list(self.ctx.pending_image_assets), [])


# ---------------------------------------------------------------------------
# A5 — json_script escaping on the Loops page
# ---------------------------------------------------------------------------
class LoopsPageEscapingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="loopsec@x.io", password="p")
        self.client.force_login(self.user)

    def test_malicious_loop_prompt_is_escaped(self):
        payload = {
            "prompt": "</script><img src=x onerror=alert(1)>",
            "history_mode": "fresh",
            "cadence_kind": "interval",
            "interval_value": 6, "interval_unit": "hours",
            "first_run_mode": "now",
        }
        create = self.client.post(
            reverse("loop_create"), data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(create.status_code, 200)

        resp = self.client.get(reverse("loops_list"))
        self.assertEqual(resp.status_code, 200)
        # The raw breakout sequence must never appear; json_script escapes '<'.
        self.assertNotIn(b"</script><img", resp.content)
        self.assertIn(b"\\u003C", resp.content)
