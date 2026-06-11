import io
import json
import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core import mail
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from feedback.models import Feedback


def _image_upload(fmt, name, content_type, size=(2, 2)):
    buf = io.BytesIO()
    Image.new("RGB", size, (0, 128, 255)).save(buf, format=fmt)
    return SimpleUploadedFile(name, buf.getvalue(), content_type=content_type)

_RATE_LIMIT_SETTINGS = {
    "ALLOWED_HOSTS": ["testserver"],
    "RATELIMIT_ENABLE": True,
    "CACHES": {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
}


class SubmitFeedbackTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="tester@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("feedback:feedback_submit")
        # Screenshots are written to MEDIA_ROOT and the filesystem isn't rolled
        # back between tests, so isolate each test in its own temp dir to avoid
        # name collisions (which storage resolves by appending a random suffix).
        media_root = tempfile.mkdtemp()
        override = override_settings(MEDIA_ROOT=media_root)
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(shutil.rmtree, media_root, ignore_errors=True)

    def test_unauthenticated_redirects(self):
        response = self.client.post(self.url, {"text": "hello"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/logged-out/", response.url)

    def test_get_not_allowed(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)

    def test_submit_text_only(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"text": "Great app!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        fb = Feedback.objects.get()
        self.assertEqual(fb.text, "Great app!")
        self.assertEqual(fb.user, self.user)
        self.assertEqual(fb.url, "")
        self.assertFalse(fb.screenshot)

    def test_submit_with_metadata(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Bug on this page",
            "url": "http://localhost/chat/",
            "user_agent": "TestBrowser/1.0",
            "viewport": "1920x1080",
            "console_errors": json.dumps([{"message": "TypeError"}]),
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.url, "http://localhost/chat/")
        self.assertEqual(fb.user_agent, "TestBrowser/1.0")
        self.assertEqual(fb.viewport, "1920x1080")
        self.assertEqual(len(fb.console_errors), 1)
        self.assertEqual(fb.console_errors[0]["message"], "TypeError")

    def test_submit_missing_text_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"text": ""})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(Feedback.objects.count(), 0)

    def test_submit_whitespace_only_text_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"text": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Feedback.objects.count(), 0)

    def test_text_too_long_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"text": "x" * 5001})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Feedback.objects.count(), 0)

    def test_submit_with_screenshot(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "See screenshot",
            "screenshot": _image_upload("JPEG", "screenshot.jpg", "image/jpeg"),
        }, format="multipart")
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertTrue(fb.screenshot)
        self.assertIn("feedback/", fb.screenshot.name)
        self.assertTrue(fb.screenshot.name.endswith("screenshot.jpg"))

    def test_screenshot_wrong_content_type_dropped_but_feedback_saved(self):
        self.client.login(email=self.user.email, password=self.password)
        fake = io.BytesIO(b"not an image")
        fake.name = "file.txt"
        response = self.client.post(self.url, {
            "text": "Bad file",
            "screenshot": fake,
        })
        # The screenshot is captured client-side, so an invalid one isn't the
        # user's fault: it's dropped quietly and the feedback is still saved.
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.text, "Bad file")
        self.assertFalse(fb.screenshot)

    def test_invalid_console_errors_json_handled(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Bad JSON",
            "console_errors": "not-json",
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.console_errors, [])

    def test_console_errors_non_list_handled(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Object instead of list",
            "console_errors": json.dumps({"error": "oops"}),
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.console_errors, [])

    def test_deeply_nested_console_errors_handled(self):
        # Regression: deeply nested JSON used to raise RecursionError → 500.
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Nested",
            "console_errors": "[" * 100_000,
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.console_errors, [])

    def test_console_errors_keys_whitelisted_and_truncated(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Whitelist",
            "console_errors": json.dumps([
                {"message": "x" * 5000, "lineno": 7, "evil": "drop me"},
                {"foo": "no whitelisted keys"},
                "not a dict",
            ]),
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(len(fb.console_errors), 1)
        entry = fb.console_errors[0]
        self.assertEqual(len(entry["message"]), 2000)
        self.assertEqual(entry["lineno"], 7)
        self.assertNotIn("evil", entry)

    def test_png_screenshot_accepted_and_renamed(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "PNG shot",
            "screenshot": _image_upload("PNG", "ignored.png", "image/png"),
        }, format="multipart")
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertTrue(fb.screenshot)
        self.assertTrue(fb.screenshot.name.endswith("screenshot.png"))

    def test_webp_screenshot_accepted_and_renamed(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "WEBP shot",
            "screenshot": _image_upload("WEBP", "ignored.webp", "image/webp"),
        }, format="multipart")
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertTrue(fb.screenshot.name.endswith("screenshot.webp"))

    def test_spoofed_content_type_non_image_dropped(self):
        self.client.login(email=self.user.email, password=self.password)
        bogus = SimpleUploadedFile("evil.png", b"not really an image", content_type="image/png")
        response = self.client.post(self.url, {
            "text": "Spoofed",
            "screenshot": bogus,
        }, format="multipart")
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertFalse(fb.screenshot)

    def test_hostile_filename_discarded(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Hostile name",
            "screenshot": _image_upload("PNG", "../../evil.html", "image/png"),
        }, format="multipart")
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertTrue(fb.screenshot.name.endswith("screenshot.png"))
        self.assertNotIn("evil", fb.screenshot.name)

    def test_dangerous_url_scheme_dropped(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "JS url",
            "url": "javascript:alert(1)",
        })
        self.assertEqual(response.status_code, 200)
        fb = Feedback.objects.get()
        self.assertEqual(fb.url, "")

    def test_request_too_large_returns_413(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url, {"text": "hi"}, CONTENT_LENGTH="8000000"
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(Feedback.objects.count(), 0)


@override_settings(
    EMAIL_SENDING_ENABLED=True,
    ADMINS=[("Admin", "admin@example.com")],
    DEFAULT_FROM_EMAIL="noreply@example.com",
)
class FeedbackAdminEmailTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="emailer@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("feedback:feedback_submit")
        mail.outbox = []

    def test_admin_email_marks_fields_as_user_supplied(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {
            "text": "Click here please",
            "url": "https://example.com/page",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        body = mail.outbox[0].body
        self.assertIn("user-supplied", body)
        self.assertIn("Click here please", body)
        self.assertIn("https://example.com/page", body)


@override_settings(**_RATE_LIMIT_SETTINGS)
class FeedbackRateLimitTests(TestCase):
    """Submissions are throttled per user (10/h) — storage/email spam guard."""

    def setUp(self):
        cache.clear()
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="limited@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("feedback:feedback_submit")
        self.client.login(email=self.user.email, password=self.password)

    def test_eleventh_post_within_an_hour_is_throttled(self):
        for _ in range(10):
            response = self.client.post(self.url, {"text": "hello"})
            self.assertEqual(response.status_code, 200)
        response = self.client.post(self.url, {"text": "hello"})
        self.assertEqual(response.status_code, 429)
        # The widget posts multipart with Accept */*, so it must get JSON it can
        # surface via data.error — not the branded HTML page.
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("error", response.json())

    def test_throttle_is_per_user(self):
        for _ in range(11):
            self.client.post(self.url, {"text": "hello"})
        other = get_user_model().objects.create_user(
            email="other@example.com",
            password=self.password,
        )
        other.email_verified = True
        other.save(update_fields=["email_verified"])
        self.client.logout()
        self.client.login(email=other.email, password=self.password)
        response = self.client.post(self.url, {"text": "hello"})
        self.assertEqual(response.status_code, 200)
