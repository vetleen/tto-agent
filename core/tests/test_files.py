"""Tests for core.files (shared by documents + meetings)."""
from __future__ import annotations

import hashlib

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from core.files import safe_filename, sha256_of_bytes, sha256_of_upload


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


class Sha256Tests(SimpleTestCase):
    def test_bytes_matches_hashlib(self):
        self.assertEqual(sha256_of_bytes(b"hello"), hashlib.sha256(b"hello").hexdigest())

    def test_empty_bytes(self):
        self.assertEqual(sha256_of_bytes(b""), hashlib.sha256(b"").hexdigest())

    def test_upload_matches_hashlib(self):
        upload = SimpleUploadedFile("a.txt", b"hello world", content_type="text/plain")
        self.assertEqual(sha256_of_upload(upload), hashlib.sha256(b"hello world").hexdigest())

    def test_upload_spans_multiple_chunks(self):
        payload = b"x" * (3 * 1024 * 1024 + 17)
        upload = SimpleUploadedFile("big.txt", payload, content_type="text/plain")
        self.assertEqual(sha256_of_upload(upload), hashlib.sha256(payload).hexdigest())

    def test_upload_is_rewound_so_caller_can_still_save_it(self):
        upload = SimpleUploadedFile("a.txt", b"hello", content_type="text/plain")
        sha256_of_upload(upload)
        self.assertEqual(upload.read(), b"hello")

    def test_unreadable_upload_returns_empty_string(self):
        class Broken:
            def chunks(self, size=None):
                raise OSError("storage is gone")

        with self.assertLogs("core.files", level="WARNING"):
            self.assertEqual(sha256_of_upload(Broken()), "")
