"""Rate-limit helpers for django-ratelimit (client IP behind Heroku, key callables)."""
from __future__ import annotations


def client_ip(request) -> str:
    """Return the real client IP behind the Heroku router.

    Heroku appends the true client IP as the LAST X-Forwarded-For entry; any
    earlier entries are client-controlled and must be ignored. (django-ratelimit
    would also choke outright on a multi-entry header: it feeds the value to
    ``ipaddress.ip_network``.) Falls back to REMOTE_ADDR when the header is
    absent or empty (local dev, tests), so it is safe to wire in unconditionally
    via RATELIMIT_IP_META_KEY. django-ratelimit calls this as ``fn(request)``
    and applies IPv4 /32 or IPv6 /64 masking to the returned string.
    """
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if xff:
        ip = xff.rsplit(",", 1)[-1].strip()
        if ip:
            return ip
    return request.META.get("REMOTE_ADDR", "")


def login_username_or_ip(group, request) -> str:
    """Rate-limit key for the login view: the normalized submitted username.

    Bounds distributed (multi-IP) brute force against a single account. Falls
    back to the client IP when the field is missing/empty so malformed posts
    from many sources don't all share one "" bucket. The prefixes keep the
    username and IP namespaces from aliasing; django-ratelimit hashes key
    values (sha256) before caching, so no email ever lands in Redis.
    django-ratelimit calls key callables as ``fn(group, request)``.
    """
    username = (request.POST.get("username") or "").strip().lower()
    if username:
        return f"username:{username}"
    return f"ip:{client_ip(request)}"
