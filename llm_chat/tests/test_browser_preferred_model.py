"""
Tests for preferred chat model: API (always runs) and optional browser test (Playwright).

- PreferredModelAPITest: uses Django test client to set model several times and GET to verify.
  Run: python manage.py test llm_chat.tests.test_browser_preferred_model.PreferredModelAPITest -v 2

- BrowserPreferredModelTest: uses Playwright to change the dropdown in the UI and reload to verify.
  Requires Playwright; skipped on environments where Playwright fails (e.g. Python 3.14 on Windows).
  Run: python manage.py test llm_chat.tests.test_browser_preferred_model.BrowserPreferredModelTest -v 2
"""
import unittest

from django.contrib.auth import get_user_model
from django.test import LiveServerTestCase, override_settings, TestCase
from django.urls import reverse

from llm_service.conf import get_allowed_models

User = get_user_model()

# Need at least 3 models to switch between in the test (used when LLM_ALLOWED_MODELS is overridden)
TEST_ALLOWED_MODELS = [
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2-thinking",
    "openai/gpt-5-nano",
]


@override_settings(
    LLM_ALLOWED_MODELS=TEST_ALLOWED_MODELS,
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
)
class PreferredModelAPITest(TestCase):
    """Test that POST preferred-model updates the setting and GET returns it (no browser)."""

    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(email="api-test@example.com", password="testpass123")
        # Use project's allowed list so view and test agree (override_settings applies to this test class)
        self._allowed = get_allowed_models()
        if len(self._allowed) < 3:
            self._allowed = TEST_ALLOWED_MODELS  # fallback if setting not applied

    def _get_preferred_model(self, client):
        r = client.get(reverse("chat_preferred_model_update"))
        self.assertEqual(r.status_code, 200)
        return r.json().get("model", "")

    def _set_preferred_model(self, client, model):
        r = client.post(reverse("chat_preferred_model_update"), data={"model": model})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json().get("model"), model)

    def test_set_model_several_times_persists_immediately(self):
        """Change preferred model several times via API; each GET reflects the last POST."""
        if len(self._allowed) < 3:
            self.skipTest("Need at least 3 models in LLM_ALLOWED_MODELS for this test")
        first, second, third = self._allowed[0], self._allowed[1], self._allowed[2]
        self.client.force_login(self.user)
        self._set_preferred_model(self.client, first)
        self.assertEqual(self._get_preferred_model(self.client), first)
        self._set_preferred_model(self.client, second)
        self.assertEqual(self._get_preferred_model(self.client), second)
        self._set_preferred_model(self.client, third)
        self.assertEqual(self._get_preferred_model(self.client), third)
        self._set_preferred_model(self.client, first)
        self.assertEqual(self._get_preferred_model(self.client), first)


@override_settings(
    LLM_ALLOWED_MODELS=TEST_ALLOWED_MODELS,
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
)
class BrowserPreferredModelTest(LiveServerTestCase):
    """Test that changing the model dropdown in the UI updates the stored user setting."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._playwright = None
        cls._skip_reason = None
        try:
            from playwright.sync_api import sync_playwright
            cls._playwright = sync_playwright().start()
        except ImportError:
            cls._skip_reason = "Playwright not installed. pip install playwright && playwright install chromium"
        except Exception as e:
            cls._skip_reason = f"Playwright could not start (e.g. Python 3.14/Windows): {e!r}"

    @classmethod
    def tearDownClass(cls):
        if cls._playwright:
            try:
                cls._playwright.stop()
            except Exception:
                pass
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        if self._skip_reason:
            raise unittest.SkipTest(self._skip_reason)
        self.user = User.objects.create_user(email="browser-test@example.com", password="testpass123")
        self.browser = self._playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def tearDown(self):
        if hasattr(self, "page") and self.page:
            self.page.close()
        if hasattr(self, "context") and self.context:
            self.context.close()
        if hasattr(self, "browser") and self.browser:
            self.browser.close()
        super().tearDown()

    def _login(self):
        """Log in via the login page."""
        self.page.goto(self.live_server_url + reverse("accounts:login"))
        self.page.get_by_label("Email").fill("browser-test@example.com")
        self.page.get_by_label("Password").fill("testpass123")
        self.page.get_by_role("button", name="Log in").click()
        self.page.wait_for_load_state("networkidle")

    def _open_chat(self):
        """Navigate to chat page."""
        self.page.goto(self.live_server_url + reverse("chat"))
        self.page.wait_for_load_state("networkidle")

    def _get_model_select_value(self):
        """Return the current value of the model dropdown."""
        sel = self.page.locator("#model-select")
        sel.wait_for(state="visible", timeout=5000)
        return sel.input_value()

    def _select_model_by_value(self, value: str):
        """Select an option in the model dropdown by value."""
        self.page.locator("#model-select").select_option(value=value)
        # Wait for the fetch that saves the preference to complete
        self.page.wait_for_timeout(500)

    def test_dropdown_changes_update_setting_immediately(self):
        """Change the model dropdown several times; after each change, reload and verify the setting persisted."""
        self._login()
        self._open_chat()

        select = self.page.locator("#model-select")
        select.wait_for(state="visible", timeout=5000)
        options = select.locator("option")
        count = options.count()
        self.assertGreaterEqual(count, 3, "Need at least 3 models in dropdown to run this test")

        values = [options.nth(i).get_attribute("value") for i in range(count)]
        values = [v for v in values if v]

        # First selection: switch to second model
        first, second, third = values[0], values[1], values[2]
        self._select_model_by_value(second)
        self.page.reload()
        self.page.wait_for_load_state("networkidle")
        self.assertEqual(self._get_model_select_value(), second, "After selecting second model and reload, dropdown should show second model")

        # Second selection: switch to third model
        self._select_model_by_value(third)
        self.page.reload()
        self.page.wait_for_load_state("networkidle")
        self.assertEqual(self._get_model_select_value(), third, "After selecting third model and reload, dropdown should show third model")

        # Third selection: switch back to first
        self._select_model_by_value(first)
        self.page.reload()
        self.page.wait_for_load_state("networkidle")
        self.assertEqual(self._get_model_select_value(), first, "After selecting first model and reload, dropdown should show first model")
