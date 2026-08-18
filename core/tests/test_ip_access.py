"""Tests for the shared HTTP and WebSocket client-IP allowlist."""
from __future__ import annotations

from pathlib import Path

from channels.testing import WebsocketCommunicator
from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase, override_settings

from core.ip_access import (
    IP_ACCESS_DENIED_CLOSE_CODE,
    ClientIPAllowlistASGIMiddleware,
    client_ip_from_scope,
    client_ip_is_allowed,
    parse_ip_allowlist,
)
from core.middleware import ClientIPAllowlistMiddleware


NTNU_NETWORKS = parse_ip_allowlist("129.241.0.0/16")


class IPAllowlistPolicyTests(SimpleTestCase):
    def test_parses_cidrs_and_individual_addresses(self):
        networks = parse_ip_allowlist(
            "129.241.0.0/16, 203.0.113.7, 2001:db8::/48"
        )
        self.assertEqual(
            [str(network) for network in networks],
            ["129.241.0.0/16", "203.0.113.7/32", "2001:db8::/48"],
        )

    def test_empty_value_disables_policy(self):
        self.assertEqual(parse_ip_allowlist("  , "), ())
        self.assertTrue(client_ip_is_allowed("not-an-ip", ()))

    def test_invalid_value_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_ip_allowlist("129.241.0.0/16,not-a-network")

    def test_verified_ntnu_address_is_allowed(self):
        self.assertTrue(client_ip_is_allowed("129.241.236.80", NTNU_NETWORKS))

    def test_outside_and_malformed_addresses_are_denied(self):
        self.assertFalse(client_ip_is_allowed("203.0.113.7", NTNU_NETWORKS))
        self.assertFalse(client_ip_is_allowed("not-an-ip", NTNU_NETWORKS))
        self.assertFalse(client_ip_is_allowed("", NTNU_NETWORKS))

    def test_asgi_scope_uses_rightmost_forwarded_address(self):
        scope = {
            "headers": [
                (b"x-forwarded-for", b"spoofed, 10.0.0.1"),
                (b"X-Forwarded-For", b"129.241.236.80"),
            ],
            "client": ("127.0.0.1", 1234),
        }
        self.assertEqual(client_ip_from_scope(scope), "129.241.236.80")

    def test_asgi_scope_falls_back_to_peer_address(self):
        scope = {"headers": [], "client": ("129.241.236.80", 1234)}
        self.assertEqual(client_ip_from_scope(scope), "129.241.236.80")


@override_settings(CLIENT_IP_ALLOWLIST=NTNU_NETWORKS, STATIC_URL="/static/")
class ClientIPAllowlistMiddlewareTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.reached_view = False

        def get_response(request):
            self.reached_view = True
            return HttpResponse("ok")

        self.middleware = ClientIPAllowlistMiddleware(get_response)

    def test_allowed_request_reaches_view(self):
        response = self.middleware(
            self.factory.get(
                "/chat/",
                HTTP_X_FORWARDED_FOR="spoofed, 129.241.236.80",
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.reached_view)

    def test_rightmost_forwarded_address_controls_access(self):
        response = self.middleware(
            self.factory.get(
                "/chat/",
                HTTP_X_FORWARDED_FOR="129.241.236.80, 203.0.113.7",
            )
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.reached_view)

    def test_denied_request_gets_branded_non_cacheable_page(self):
        response = self.middleware(
            self.factory.get(
                "/chat/",
                HTTP_X_FORWARDED_FOR="203.0.113.7",
            )
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response["Cache-Control"], "no-store")
        self.assertContains(response, "Access denied", status_code=403)
        self.assertContains(response, "Connect to the NTNU VPN", status_code=403)
        self.assertNotContains(response, "203.0.113.7", status_code=403)
        self.assertFalse(self.reached_view)

    def test_static_assets_bypass_gate(self):
        response = self.middleware(
            self.factory.get(
                "/static/src/output.css",
                HTTP_X_FORWARDED_FOR="203.0.113.7",
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.reached_view)

    @override_settings(CLIENT_IP_ALLOWLIST=())
    def test_empty_setting_disables_gate(self):
        response = self.middleware(
            self.factory.get(
                "/chat/",
                HTTP_X_FORWARDED_FOR="203.0.113.7",
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.reached_view)


async def _accepting_websocket_app(scope, receive, send):
    event = await receive()
    if event["type"] != "websocket.connect":
        return
    await send({"type": "websocket.accept"})
    await receive()


@override_settings(CLIENT_IP_ALLOWLIST=NTNU_NETWORKS)
class ClientIPAllowlistASGITests(SimpleTestCase):
    async def test_allowed_websocket_reaches_application(self):
        communicator = WebsocketCommunicator(
            ClientIPAllowlistASGIMiddleware(_accepting_websocket_app),
            "/ws/chat/",
        )
        communicator.scope["headers"] = [
            (b"x-forwarded-for", b"129.241.236.80")
        ]
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        await communicator.disconnect()

    async def test_denied_websocket_gets_terminal_close_code(self):
        communicator = WebsocketCommunicator(
            ClientIPAllowlistASGIMiddleware(_accepting_websocket_app),
            "/ws/chat/",
        )
        communicator.scope["headers"] = [
            (b"x-forwarded-for", b"203.0.113.7")
        ]
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        event = await communicator.receive_output()
        self.assertEqual(event["type"], "websocket.close")
        self.assertEqual(event["code"], IP_ACCESS_DENIED_CLOSE_CODE)


class IPAllowlistIntegrationWiringTests(SimpleTestCase):
    def test_http_gate_runs_before_session_and_authentication(self):
        from django.conf import settings

        gate = settings.MIDDLEWARE.index(
            "core.middleware.ClientIPAllowlistMiddleware"
        )
        self.assertGreater(
            gate, settings.MIDDLEWARE.index("whitenoise.middleware.WhiteNoiseMiddleware")
        )
        self.assertLess(
            gate,
            settings.MIDDLEWARE.index(
                "django.contrib.sessions.middleware.SessionMiddleware"
            ),
        )

    def test_browser_clients_handle_terminal_close_code(self):
        from django.conf import settings

        base_dir = Path(settings.BASE_DIR)
        chat_source = (base_dir / "templates/chat/chat.html").read_text(
            encoding="utf-8"
        )
        meeting_source = (
            base_dir / "static/meetings/transcribe.js"
        ).read_text(encoding="utf-8")
        self.assertIn("event.code === 4413", chat_source)
        self.assertIn("4413: 'Access denied from this network.", meeting_source)
