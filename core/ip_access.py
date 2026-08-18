"""Client-IP access policy shared by Django HTTP and Channels WebSockets."""
from __future__ import annotations

from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network
from typing import Iterable


IP_ACCESS_DENIED_CLOSE_CODE = 4413

IPAddressNetwork = IPv4Network | IPv6Network


def parse_ip_allowlist(value: str) -> tuple[IPAddressNetwork, ...]:
    """Parse a comma-separated list of CIDRs or individual IP addresses."""
    return tuple(
        ip_network(item.strip(), strict=False)
        for item in value.split(",")
        if item.strip()
    )


def client_ip_from_values(forwarded_for: str, remote_addr: str) -> str:
    """Return the Heroku-observed client IP, falling back to the peer address.

    Heroku appends its observed address to the right of any client-supplied
    ``X-Forwarded-For`` values, so only the final non-empty entry is trusted.
    """
    if forwarded_for:
        forwarded_ip = forwarded_for.rsplit(",", 1)[-1].strip()
        if forwarded_ip:
            return forwarded_ip
    return remote_addr


def client_ip_from_request(request) -> str:
    """Extract the client IP from a Django request."""
    return client_ip_from_values(
        request.META.get("HTTP_X_FORWARDED_FOR", ""),
        request.META.get("REMOTE_ADDR", ""),
    )


def client_ip_from_scope(scope: dict) -> str:
    """Extract the client IP from an ASGI connection scope."""
    forwarded_values = [
        value.decode("latin-1")
        for name, value in scope.get("headers", [])
        if name.lower() == b"x-forwarded-for"
    ]
    client = scope.get("client") or ("", 0)
    remote_addr = client[0] if client else ""
    return client_ip_from_values(",".join(forwarded_values), remote_addr)


def client_ip_is_allowed(
    client_ip: str,
    networks: Iterable[IPAddressNetwork],
) -> bool:
    """Return whether an address is permitted by the configured networks.

    An empty network collection disables the gate. When the gate is enabled,
    malformed or missing client addresses fail closed.
    """
    networks = tuple(networks)
    if not networks:
        return True
    try:
        address = ip_address(client_ip)
    except ValueError:
        return False
    return any(address in network for network in networks)


class ClientIPAllowlistASGIMiddleware:
    """Reject WebSockets whose client address is outside the allowlist."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        from django.conf import settings

        networks = settings.CLIENT_IP_ALLOWLIST
        if networks and not client_ip_is_allowed(
            client_ip_from_scope(scope), networks
        ):
            # Accept before closing so browsers receive the application close
            # code. A close during the HTTP upgrade is exposed as generic 1006,
            # which would make the clients retry a deterministic rejection.
            event = await receive()
            if event["type"] != "websocket.connect":
                return
            await send({"type": "websocket.accept"})
            await send(
                {
                    "type": "websocket.close",
                    "code": IP_ACCESS_DENIED_CLOSE_CODE,
                    "reason": "Client IP is outside the allowed networks.",
                }
            )
            return
        await self.application(scope, receive, send)
