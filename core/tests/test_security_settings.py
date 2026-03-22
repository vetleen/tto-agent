"""Tests for production security settings in config/settings.py."""

from django.test import SimpleTestCase


class SecuritySettingsTest(SimpleTestCase):
    """Verify security settings are configured."""

    def test_secure_proxy_ssl_header_set(self):
        """SECURE_PROXY_SSL_HEADER must be set for Heroku SSL termination."""
        from django.conf import settings

        self.assertEqual(
            settings.SECURE_PROXY_SSL_HEADER,
            ("HTTP_X_FORWARDED_PROTO", "https"),
        )

    def test_production_security_block_exists(self):
        """Verify the production security settings block is present in the config module."""
        import config.settings as config_mod

        # When DEBUG is False (production), these should be set.
        # We can't re-import the module, but we can verify the settings
        # ARE defined when DEBUG is False by checking the source.
        import inspect

        source = inspect.getsource(config_mod)
        self.assertIn("SESSION_COOKIE_SECURE = True", source)
        self.assertIn("CSRF_COOKIE_SECURE = True", source)
        self.assertIn("SECURE_HSTS_SECONDS", source)
        self.assertIn("SECURE_SSL_REDIRECT = True", source)

    def test_security_middleware_present(self):
        """SecurityMiddleware must be in the middleware stack."""
        from django.conf import settings

        self.assertIn("django.middleware.security.SecurityMiddleware", settings.MIDDLEWARE)
