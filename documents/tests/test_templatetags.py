from django.test import SimpleTestCase

from documents.templatetags.document_filters import truncate_project_name


class TruncateProjectNameTests(SimpleTestCase):
    def test_short_name_unchanged(self):
        result = truncate_project_name("Short name", 60)
        self.assertEqual(result, "Short name")

    def test_long_name_truncated(self):
        long_name = "A" * 100
        result = truncate_project_name(long_name, 60)
        self.assertLessEqual(len(result), 60)
        self.assertTrue(result.endswith("\u2026"), f"Expected ellipsis, got: {result!r}")

    def test_none_returns_empty_string(self):
        self.assertEqual(truncate_project_name(None), "")

    def test_invalid_max_chars_falls_back_to_60(self):
        long_name = "B" * 100
        result = truncate_project_name(long_name, "bad")
        # Falls back to 60; result should be truncated
        self.assertLessEqual(len(result), 60)

    def test_custom_max_chars(self):
        name = "Hello World Extra Text"
        result = truncate_project_name(name, 10)
        self.assertLessEqual(len(result), 10)
        self.assertTrue(result.endswith("\u2026"))
