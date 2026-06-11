import io

from django.test import SimpleTestCase
from PIL import Image

from feedback.validation import (
    MAX_CONSOLE_ERRORS,
    MAX_CONSOLE_ERRORS_RAW_CHARS,
    _CONSOLE_STR_MAXLEN,
    clean_feedback_url,
    reencode_screenshot,
    sanitize_console_errors,
)


def _image_bytes(fmt, size=(2, 2), color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    buf.seek(0)
    return buf


class ReencodeScreenshotTests(SimpleTestCase):
    def test_png_accepted_and_named(self):
        result = reencode_screenshot(_image_bytes("PNG"))
        self.assertIsNotNone(result)
        name, content = result
        self.assertEqual(name, "screenshot.png")
        # Re-encoded bytes are still a valid PNG.
        with Image.open(io.BytesIO(content.read())) as img:
            self.assertEqual(img.format, "PNG")

    def test_jpeg_accepted_and_named(self):
        result = reencode_screenshot(_image_bytes("JPEG"))
        self.assertIsNotNone(result)
        name, content = result
        self.assertEqual(name, "screenshot.jpg")
        with Image.open(io.BytesIO(content.read())) as img:
            self.assertEqual(img.format, "JPEG")

    def test_webp_accepted_and_named(self):
        result = reencode_screenshot(_image_bytes("WEBP"))
        self.assertIsNotNone(result)
        name, content = result
        self.assertEqual(name, "screenshot.webp")
        with Image.open(io.BytesIO(content.read())) as img:
            self.assertEqual(img.format, "WEBP")

    def test_disallowed_format_rejected(self):
        # A real image, but not one of the allowed formats.
        self.assertIsNone(reencode_screenshot(_image_bytes("GIF")))

    def test_garbage_bytes_rejected(self):
        self.assertIsNone(reencode_screenshot(io.BytesIO(b"not an image at all")))

    def test_truncated_image_rejected(self):
        full = _image_bytes("PNG", size=(64, 64)).getvalue()
        truncated = io.BytesIO(full[: len(full) // 2])
        self.assertIsNone(reencode_screenshot(truncated))

    def test_seeks_before_reading(self):
        # A file already advanced past the header must still be re-encoded.
        buf = _image_bytes("PNG")
        buf.read()  # exhaust it
        result = reencode_screenshot(buf)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "screenshot.png")


class SanitizeConsoleErrorsTests(SimpleTestCase):
    def test_empty_and_none_return_empty_list(self):
        self.assertEqual(sanitize_console_errors(""), [])
        self.assertEqual(sanitize_console_errors(None), [])
        self.assertEqual(sanitize_console_errors("[]"), [])

    def test_invalid_json_returns_empty_list(self):
        self.assertEqual(sanitize_console_errors("not-json"), [])

    def test_non_list_returns_empty_list(self):
        self.assertEqual(sanitize_console_errors('{"message": "oops"}'), [])

    def test_deeply_nested_json_returns_empty_list(self):
        # Regression: deeply nested JSON used to raise RecursionError uncaught.
        deeply_nested = "[" * 100_000
        self.assertEqual(sanitize_console_errors(deeply_nested), [])

    def test_oversize_raw_returns_empty_list(self):
        oversize = "a" * (MAX_CONSOLE_ERRORS_RAW_CHARS + 1)
        self.assertEqual(sanitize_console_errors(oversize), [])

    def test_whitelists_keys_and_truncates_strings(self):
        raw = (
            '[{"message": "' + "x" * 5000 + '", "lineno": 12, "colno": 3, '
            '"source": "app.js", "stack": "trace", "timestamp": "2026-01-01", '
            '"evil": "drop me", "nested": {"a": 1}}]'
        )
        result = sanitize_console_errors(raw)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(set(entry), {"message", "lineno", "colno", "source", "stack", "timestamp"})
        self.assertEqual(len(entry["message"]), _CONSOLE_STR_MAXLEN)
        self.assertEqual(entry["lineno"], 12)
        self.assertNotIn("evil", entry)
        self.assertNotIn("nested", entry)

    def test_bool_not_kept_as_int(self):
        result = sanitize_console_errors('[{"message": "m", "lineno": true}]')
        self.assertEqual(result, [{"message": "m"}])

    def test_non_dict_entries_dropped(self):
        raw = '[1, "string", null, [1, 2], {"message": "kept"}]'
        result = sanitize_console_errors(raw)
        self.assertEqual(result, [{"message": "kept"}])

    def test_entries_with_no_whitelisted_keys_dropped(self):
        self.assertEqual(sanitize_console_errors('[{"foo": "bar"}, {}]'), [])

    def test_capped_at_max_entries(self):
        raw = "[" + ",".join('{"message": "m"}' for _ in range(MAX_CONSOLE_ERRORS + 25)) + "]"
        result = sanitize_console_errors(raw)
        self.assertEqual(len(result), MAX_CONSOLE_ERRORS)


class CleanFeedbackUrlTests(SimpleTestCase):
    def test_valid_http_and_https_preserved(self):
        self.assertEqual(clean_feedback_url("https://example.com/path"), "https://example.com/path")
        self.assertEqual(clean_feedback_url("http://localhost/chat/"), "http://localhost/chat/")

    def test_dangerous_schemes_rejected(self):
        self.assertEqual(clean_feedback_url("javascript:alert(1)"), "")
        self.assertEqual(clean_feedback_url("data:text/html,<script>alert(1)</script>"), "")

    def test_malformed_and_empty_rejected(self):
        self.assertEqual(clean_feedback_url("not a url"), "")
        self.assertEqual(clean_feedback_url(""), "")
        self.assertEqual(clean_feedback_url(None), "")

    def test_whitespace_stripped(self):
        self.assertEqual(clean_feedback_url("  https://example.com  "), "https://example.com")

    def test_truncated_to_max_length(self):
        long_url = "https://example.com/" + "a" * 3000
        result = clean_feedback_url(long_url)
        self.assertEqual(len(result), 2000)
