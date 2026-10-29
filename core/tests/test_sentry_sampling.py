"""Tests for the allowlist-aware Sentry traces sampler (``core.sentry_sampling``)."""
from __future__ import annotations

from django.test import SimpleTestCase, override_settings

from core.ip_access import parse_ip_allowlist
from core.sentry_sampling import traces_sampler


NTNU_NETWORKS = parse_ip_allowlist("129.241.0.0/16")


def _scope(client_ip: str, scope_type: str = "http") -> dict:
    """A minimal ASGI scope as Sentry hands it to the sampler (``asgi_scope``)."""
    return {
        "type": scope_type,
        "path": "/.env",
        "headers": [(b"x-forwarded-for", client_ip.encode())],
        "client": ("10.0.0.1", 1234),
    }


@override_settings(CLIENT_IP_ALLOWLIST=NTNU_NETWORKS, SENTRY_TRACES_SAMPLE_RATE=0.25)
class TracesSamplerTests(SimpleTestCase):
    def test_request_outside_allowlist_is_not_traced(self):
        self.assertEqual(traces_sampler({"asgi_scope": _scope("203.0.113.7")}), 0.0)

    def test_websocket_outside_allowlist_is_not_traced(self):
        self.assertEqual(
            traces_sampler({"asgi_scope": _scope("203.0.113.7", "websocket")}), 0.0
        )

    def test_denied_request_overrides_parent_decision(self):
        self.assertEqual(
            traces_sampler(
                {"asgi_scope": _scope("203.0.113.7"), "parent_sampled": True}
            ),
            0.0,
        )

    def test_rightmost_forwarded_address_controls_decision(self):
        scope = _scope("203.0.113.7")
        scope["headers"] = [(b"x-forwarded-for", b"129.241.236.80, 203.0.113.7")]
        self.assertEqual(traces_sampler({"asgi_scope": scope}), 0.0)

    def test_allowed_request_uses_configured_rate(self):
        self.assertEqual(traces_sampler({"asgi_scope": _scope("129.241.236.80")}), 0.25)

    def test_allowed_request_honours_parent_decision(self):
        self.assertIs(
            traces_sampler(
                {"asgi_scope": _scope("129.241.236.80"), "parent_sampled": False}
            ),
            False,
        )

    def test_non_asgi_transactions_use_configured_rate(self):
        self.assertEqual(
            traces_sampler({"celery_job": {"task": "chat.tasks.tick_and_scan_loops"}}),
            0.25,
        )

    @override_settings(CLIENT_IP_ALLOWLIST=())
    def test_empty_allowlist_traces_everyone(self):
        self.assertEqual(traces_sampler({"asgi_scope": _scope("203.0.113.7")}), 0.25)
