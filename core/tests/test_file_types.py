"""Tests for the unified file-type capability table (core.file_types)."""

from django.test import SimpleTestCase

from core import file_types as ft


class FileTypeTableTests(SimpleTestCase):
    def test_data_room_includes_images(self):
        exts = ft.allowed_extensions(ft.DATA_ROOM_KINDS)
        for e in ("png", "jpg", "jpeg", "gif", "webp"):
            self.assertIn(e, exts)

    def test_chat_excludes_audio_and_email(self):
        mimes = ft.accepted_mimes_for_kinds(ft.CHAT_KINDS)
        self.assertIn("image/png", mimes)
        self.assertNotIn("audio/mpeg", mimes)
        self.assertNotIn("message/rfc822", mimes)
        self.assertNotIn("application/vnd.ms-outlook", mimes)

    def test_kind_lookups(self):
        self.assertEqual(ft.kind_for_extension("PNG"), ft.KIND_IMAGE)
        self.assertEqual(ft.kind_for_extension(".pdf"), ft.KIND_PDF)
        self.assertEqual(ft.kind_for_extension("mp3"), ft.KIND_AUDIO)
        self.assertIsNone(ft.kind_for_extension("zip"))
        self.assertEqual(ft.kind_for_mime("image/jpeg"), ft.KIND_IMAGE)
        self.assertEqual(ft.kind_for_mime("audio/mpeg"), ft.KIND_AUDIO)

    def test_canonical_mime_for_extension(self):
        self.assertEqual(ft.canonical_mime_for_extension("png"), "image/png")
        self.assertEqual(ft.canonical_mime_for_extension("jpg"), "image/jpeg")
        self.assertIsNone(ft.canonical_mime_for_extension("zip"))

    def test_is_image_extension(self):
        self.assertTrue(ft.is_image_extension("webp"))
        self.assertFalse(ft.is_image_extension("pdf"))

    def test_global_mimes_include_octet_stream_and_images(self):
        mimes = ft.global_allowed_mimes(ft.DATA_ROOM_KINDS)
        self.assertIn("application/octet-stream", mimes)
        self.assertIn("image/png", mimes)

    def test_extension_mime_map_reproduces_csv_variants(self):
        m = ft.extension_mime_map(ft.DATA_ROOM_KINDS)
        self.assertEqual(
            m["csv"],
            {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
        )
