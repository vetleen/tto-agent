"""Tests for core.files.safe_filename (shared by documents + meetings)."""
from __future__ import annotations

from django.test import SimpleTestCase

from core.files import safe_filename


class SafeFilenameTests(SimpleTestCase):
    def test_empty_returns_fallback(self):
        self.assertEqual(safe_filename(""), "file")
        self.assertEqual(safe_filename("   "), "file")
        self.assertEqual(safe_filename(None), "file")

    def test_custom_fallback(self):
        self.assertEqual(safe_filename("", fallback="document"), "document")

    def test_strips_unix_and_windows_paths(self):
        self.assertEqual(safe_filename("/etc/passwd"), "passwd")
        self.assertEqual(safe_filename(r"C:\Users\me\report.pdf"), "report.pdf")
        self.assertEqual(safe_filename("../../secret.txt"), "secret.txt")

    def test_passthrough_normal_name(self):
        self.assertEqual(safe_filename("notes.md"), "notes.md")

    def test_truncation_preserves_extension(self):
        name = "a" * 300 + ".pdf"
        out = safe_filename(name, max_length=20)
        self.assertEqual(len(out), 20)
        self.assertTrue(out.endswith(".pdf"))

    def test_truncation_without_extension(self):
        out = safe_filename("a" * 300, max_length=10)
        self.assertEqual(out, "a" * 10)
