"""Tests for canvas PDF export (``?format=pdf``).

The WeasyPrint render is mocked for wiring tests so they run anywhere; one
end-to-end render test runs only where the native Pango libs are available.
"""
import io
import json
import logging
from unittest import mock, skipUnless

from django.test import TestCase

from accounts.models import Membership, Organization, User
from chat.models import ChatCanvas, ChatThread
from chat.pdf_export import (
    _DropNotdefGlyphWarnings,
    _install_weasyprint_log_filter,
    weasyprint_available,
)

FAKE_PDF = b"%PDF-1.4\n% fake\n"


class CanvasPdfExportTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="pdf@test.com", password="pass")
        self.client.login(email="pdf@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(
            thread=self.thread,
            title="My NDA",
            content="# NDA\n\nThis is ==highlighted== text with a [^1] cite.\n\n[^1]: A source.",
        )
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])
        self.url = f"/chat/threads/{self.thread.id}/canvas/export/"

    def _post(self, query=""):
        return self.client.post(
            self.url + query, json.dumps({}), content_type="application/json"
        )

    def test_pdf_export_wiring(self):
        with mock.patch("chat.pdf_export.render_canvas_pdf", return_value=FAKE_PDF) as render:
            resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/pdf")
        self.assertIn("My NDA.pdf", resp.get("Content-Disposition", ""))
        self.assertEqual(b"".join(resp.streaming_content), FAKE_PDF)
        render.assert_called_once()
        # Default org (none) => all Calibri => metric substitute, no warning header.
        self.assertNotIn("X-Font-Fallbacks", resp)

    def test_docx_still_default(self):
        resp = self._post()  # no format => docx
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp["Content-Type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertIn("My NDA.docx", resp.get("Content-Disposition", ""))

    def test_font_fallback_header(self):
        org = Organization.objects.create(name="Acme", slug="acme")
        org.preferences = {"styles": {"body_font": "Zzqx Nonexistent Font"}}
        org.save(update_fields=["preferences"])
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)

        with mock.patch("chat.pdf_export.render_canvas_pdf", return_value=FAKE_PDF):
            resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("X-Font-Fallbacks", resp)
        notes = json.loads(resp["X-Font-Fallbacks"])
        self.assertTrue(any(n["fidelity"] == "fallback" for n in notes))
        self.assertTrue(any("Zzqx Nonexistent Font" in n["note"] for n in notes))

    def test_render_failure_returns_500(self):
        with mock.patch("chat.pdf_export.render_canvas_pdf", side_effect=RuntimeError("boom")):
            resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("error", resp.json())

    def test_other_users_canvas_forbidden(self):
        other = User.objects.create_user(email="other@test.com", password="pass")
        self.client.force_login(other)
        with mock.patch("chat.pdf_export.render_canvas_pdf", return_value=FAKE_PDF):
            resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 404)

    @skipUnless(weasyprint_available(), "WeasyPrint native libs not installed (expected on Windows)")
    def test_real_render_end_to_end(self):
        from pypdf import PdfReader

        resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 200)
        data = b"".join(resp.streaming_content)
        self.assertTrue(data.startswith(b"%PDF"))
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() for page in reader.pages)
        self.assertIn("NDA", text)


class WeasyPrintNotdefFilterTests(TestCase):
    """WeasyPrint's '.notdef glyph rendered ...' warnings fire once per character
    with no glyph in the export fonts (e.g. emoji ✅/❌ pasted into a canvas) and
    stormed Sentry one event per char (WILFRED-66/67). The export still works, so
    they must be filtered off the weasyprint logger while other warnings pass.
    """

    def _record(self, msg):
        return logging.LogRecord(
            name="weasyprint", level=logging.WARNING, pathname=__file__,
            lineno=1, msg=msg, args=(), exc_info=None,
        )

    def test_filter_drops_notdef_but_keeps_other_warnings(self):
        f = _DropNotdefGlyphWarnings()
        notdef = self._record(
            '.notdef glyph rendered for Unicode string unsupported by fonts: "✅" (U+2705)'
        )
        other = self._record("Failed to load image at 'https://x/y.png'")
        self.assertFalse(f.filter(notdef))
        self.assertTrue(f.filter(other))

    def test_install_is_idempotent_and_suppresses_on_logger(self):
        wp_logger = logging.getLogger("weasyprint")
        # Clean slate, then install twice — must not stack duplicate filters.
        for flt in [f for f in wp_logger.filters if isinstance(f, _DropNotdefGlyphWarnings)]:
            wp_logger.removeFilter(flt)
        try:
            _install_weasyprint_log_filter()
            _install_weasyprint_log_filter()
            installed = [f for f in wp_logger.filters if isinstance(f, _DropNotdefGlyphWarnings)]
            self.assertEqual(len(installed), 1)
            # Attached filter drops a .notdef record but keeps a real warning.
            self.assertFalse(wp_logger.filter(self._record(
                '.notdef glyph rendered for Unicode string unsupported by fonts: "❌" (U+274C)'
            )))
            self.assertTrue(bool(wp_logger.filter(self._record("some other weasyprint warning"))))
        finally:
            for flt in [f for f in wp_logger.filters if isinstance(f, _DropNotdefGlyphWarnings)]:
                wp_logger.removeFilter(flt)
