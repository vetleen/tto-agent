"""Web Fetch tool — fetch and extract clean markdown from web pages."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket

from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from django.conf import settings as django_settings
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as html_to_md
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument

from llm.tools._text_cleaning import normalize_text
from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)

_ABSOLUTE_MAX_CHARS = 50_000

# Default hard ceiling on bytes downloaded from a (user/LLM-supplied) URL.
# Overridable via settings.WEB_FETCH_MAX_RESPONSE_BYTES.
_DEFAULT_MAX_RESPONSE_BYTES = 10_000_000  # 10 MB


def _max_response_bytes() -> int:
    return getattr(django_settings, "WEB_FETCH_MAX_RESPONSE_BYTES", _DEFAULT_MAX_RESPONSE_BYTES)


class _SSRFBlocked(Exception):
    """Raised when a URL resolves to a private/reserved address."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class _ResponseTooLarge(Exception):
    """Raised when a response body exceeds the configured size cap."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "noscript", "iframe",
    "aside", "svg", "canvas", "object", "embed",
    "meta", "template", "dialog",
]

# Inline style substrings that indicate a visually hidden element.
_HIDDEN_STYLE_MARKERS = [
    "display:none",
    "display: none",
    "visibility:hidden",
    "visibility: hidden",
    "font-size:0",
    "font-size: 0",
    "opacity:0",
    "opacity: 0",
    "clip:rect(0",
]

_HIDDEN_OVERFLOW_RE = re.compile(r"height\s*:\s*0.*overflow\s*:\s*hidden", re.IGNORECASE)


def _strip_hidden_elements(soup: BeautifulSoup) -> None:
    """Remove HTML elements that are invisible to humans.

    Attackers hide prompt-injection payloads in elements styled with
    display:none, visibility:hidden, zero font-size, aria-hidden, etc.
    These are invisible in a browser but survive ``get_text()`` extraction.
    """
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Remove elements hidden via style, attributes, or input type
    for tag in list(soup.find_all(True)):
        # Some malformed/parsed tags expose attrs=None, which would crash
        # tag.has_attr / tag.get with TypeError. Treat them as having no
        # hiding attributes and move on.
        attrs = tag.attrs if isinstance(tag.attrs, dict) else {}

        # Hidden HTML attribute
        if "hidden" in attrs:
            tag.decompose()
            continue

        # aria-hidden="true"
        if str(attrs.get("aria-hidden") or "").lower() == "true":
            tag.decompose()
            continue

        # <input type="hidden">
        if tag.name == "input" and str(attrs.get("type") or "").lower() == "hidden":
            tag.decompose()
            continue

        # Inline style hiding
        style = attrs.get("style") or ""
        if style:
            style_lower = style.lower().replace(" ", "")
            if any(m.replace(" ", "") in style_lower for m in _HIDDEN_STYLE_MARKERS):
                tag.decompose()
                continue
            if _HIDDEN_OVERFLOW_RE.search(style):
                tag.decompose()
                continue

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, link-local, or reserved."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Treat unparseable IPs as blocked
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def _resolve_and_validate(url: str) -> tuple[str | None, str | None]:
    """Resolve a URL's host ONCE and validate every resolved IP is public.

    Returns ``(validated_ip, None)`` on success or ``(None, error_message)``.
    The returned IP is the address the caller MUST connect to — pinning to it
    closes the DNS-rebinding (TOCTOU) gap where a second, independent DNS
    lookup by ``requests`` could land on a private address after the check
    passed.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return None, "No hostname in URL"
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return None, f"DNS resolution failed for {hostname}"
    if not addr_infos:
        return None, f"DNS resolution failed for {hostname}"
    validated_ip: str | None = None
    for _family, _type, _proto, _canon, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            return None, "URL resolves to a private or reserved IP address"
        if validated_ip is None:
            validated_ip = ip_str
    return validated_ip, None


def _check_url_ssrf(url: str) -> str | None:
    """Return an error message if the URL targets a private/internal host, else None.

    Thin wrapper over :func:`_resolve_and_validate` kept for callers/tests that
    only need the boolean-ish verdict.
    """
    return _resolve_and_validate(url)[1]


class _PinnedIPAdapter(HTTPAdapter):
    """Routes the TCP connection to a pre-validated IP while keeping TLS SNI and
    certificate verification bound to the original hostname.

    One instance is mounted on a fresh per-request ``Session`` (see
    :func:`_pinned_get`), so there is no shared/global state — safe under the
    ThreadPoolExecutor that runs tools concurrently. The request URL is left
    untouched, so the ``Host`` header and redirect/``Location`` logic keep
    seeing the real hostname; only the socket target is swapped to the IP.
    """

    def __init__(self, validated_ip: str, hostname: str, *args, **kwargs):
        self._validated_ip = validated_ip
        self._hostname = hostname
        super().__init__(*args, **kwargs)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        # No proxy handling: this tool sets no proxies. If proxy support is ever
        # added, the base class's proxy branch must be reinstated here.
        host_params, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
        host_params["host"] = self._validated_ip  # connect to the validated IP
        if str(request.url).lower().startswith("https"):
            # Verify the cert against the real hostname, and send it as SNI,
            # even though the socket points at the IP.
            pool_kwargs["server_hostname"] = self._hostname
            pool_kwargs["assert_hostname"] = self._hostname
        return self.poolmanager.connection_from_host(**host_params, pool_kwargs=pool_kwargs)


def _enforce_size_and_buffer(resp: requests.Response, max_bytes: int) -> None:
    """Enforce a hard byte ceiling while reading, then buffer the body.

    Fast-rejects when ``Content-Length`` already exceeds the cap, but does not
    trust it (chunked / lying servers): the limit is also enforced while
    streaming. On success the full body is stored on ``resp._content`` so
    ``resp.text`` / ``resp.content`` (and their charset detection) behave
    exactly as a non-streamed response would. Raises ``_ResponseTooLarge`` and
    closes the response when the cap is exceeded.
    """
    content_length = resp.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                resp.close()
                raise _ResponseTooLarge(
                    f"Response too large (Content-Length {content_length} > {max_bytes} bytes)"
                )
        except ValueError:
            pass  # malformed header — fall through to streaming enforcement

    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            resp.close()
            raise _ResponseTooLarge(f"Response exceeded {max_bytes} bytes during download")
        chunks.append(chunk)
    resp._content = b"".join(chunks)
    resp._content_consumed = True


def _pinned_get(url: str, *, timeout: int, headers: dict, max_bytes: int) -> requests.Response:
    """Fetch a single URL (no redirect following) pinned to a validated public IP.

    Resolves + validates the host once, mounts a :class:`_PinnedIPAdapter` on a
    fresh ``Session``, streams the body under a size cap, and returns a response
    whose body is fully buffered. Raises ``_SSRFBlocked`` or ``_ResponseTooLarge``.
    """
    validated_ip, error = _resolve_and_validate(url)
    if error:
        raise _SSRFBlocked(error)
    hostname = urlparse(url).hostname
    session = requests.Session()
    adapter = _PinnedIPAdapter(validated_ip, hostname)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        resp = session.get(
            url, timeout=timeout, headers=headers, allow_redirects=False, stream=True,
        )
        _enforce_size_and_buffer(resp, max_bytes)
        return resp
    finally:
        session.close()


class WebFetchInput(ReasonBaseModel):
    url: str = Field(description="The URL of the web page to fetch.")
    max_chars: int = Field(default=20_000, description="Maximum characters to return (default 20000, max 50000).")


_JS_RENDER_MIN_HTML = 5000
_JS_RENDER_MAX_CONTENT = 200


def _fetch_via_jina(url: str, context=None, reason: str = "") -> dict | None:
    """Fetch a page via Jina Reader API and extract content with readability."""
    api_key = getattr(django_settings, "JINA_API_KEY", "")
    if not api_key:
        return None
    logger.info("web_fetch: falling back to Jina for url=%s reason=%s", url, reason)
    try:
        resp = requests.get(
            f"https://eu.r.jina.ai/{url}",
            timeout=30,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/html",
                "X-Return-Format": "html",
            },
            stream=True,
        )
        resp.raise_for_status()
        # Trusted host (eu.r.jina.ai) — no SSRF pinning needed, but still cap
        # the body so a huge page can't OOM the worker. _ResponseTooLarge is an
        # Exception, so it's caught here and the fallback declines gracefully.
        _enforce_size_and_buffer(resp, _max_response_bytes())
    except Exception:
        logger.info("web_fetch: Jina fallback failed for url=%s", url)
        return None

    raw_html = resp.text
    if not raw_html.strip():
        logger.info("web_fetch: Jina returned empty content for url=%s", url)
        return None

    soup = BeautifulSoup(raw_html, "html.parser")
    _strip_hidden_elements(soup)
    cleaned_html = str(soup)

    title = ""
    text = ""
    try:
        doc = ReadabilityDocument(cleaned_html)
        title = doc.short_title()
        text = html_to_md(doc.summary(), strip=["img"])
        text = normalize_text(text)
    except Exception:
        pass

    if len(text) < _JS_RENDER_MAX_CONTENT:
        logger.debug("web_fetch: readability extracted too little from Jina HTML, using cleaned text")
        text = normalize_text(soup.get_text(separator="\n", strip=True))

    if not text.strip():
        logger.info("web_fetch: Jina returned no extractable content for url=%s", url)
        return None

    logger.info("web_fetch: Jina fallback succeeded for url=%s chars=%d", url, len(text))
    _run_web_scan(text, context)

    return {
        "url": url,
        "title": title,
        "content": text,
        "truncated": False,
        "char_count": len(text),
        "source": "jina",
    }


def _run_web_scan(text: str, context=None) -> None:
    """Fire-and-forget prompt-injection scan (never blocks)."""
    try:
        if text.strip():
            from guardrails.web_content import scan_web_content

            scan_web_content(
                text,
                user_id=context.user_id if context else None,
                thread_id=context.conversation_id if context else None,
                org_id=None,
                source_label="web_fetch",
            )
    except Exception:
        logger.debug("web_fetch: web content scan failed (non-fatal)")


class WebFetchTool(ContextAwareTool):
    """Fetch a web page and extract clean markdown content."""

    name: str = "web_fetch"
    description: str = (
        "Fetch a web page and extract its content as clean markdown. "
        "Use this to read the content of a specific URL, such as articles, "
        "documentation, or other web pages."
    )
    args_schema: type[BaseModel] = WebFetchInput

    def _run(self, url: str, max_chars: int = 20_000, **kwargs) -> str:
        from django.core.cache import cache

        url = url.strip()
        if not url:
            return json.dumps({"error": "Empty URL"})

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed."})

        # Note: SSRF protection (resolve + validate + connection pinning) happens
        # inside _pinned_get, per hop. We don't pre-check here so the host is
        # resolved exactly once per fetch and the validated IP is the one we
        # actually connect to (closes the DNS-rebinding gap).

        max_chars = max(1, min(max_chars, _ABSOLUTE_MAX_CHARS))

        cache_key = "web_fetch:" + hashlib.sha256(url.encode()).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            logger.debug("web_fetch: cache read failed, proceeding without cache")
            cached = None
        if cached is not None:
            logger.debug("Web fetch cache hit for url=%s", url)
            data = json.loads(cached)
            content = data.get("content", "")
            if len(content) > max_chars:
                data["content"] = content[:max_chars]
                data["truncated"] = True
                data["char_count"] = max_chars
            return json.dumps(data)

        # --- Fetch HTML ---
        headers = {
            "User-Agent": f"Mozilla/5.0 (compatible; {django_settings.ASSISTANT_NAME}Bot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        max_bytes = _max_response_bytes()
        current_url = url
        try:
            # Each hop is resolved + SSRF-validated + IP-pinned independently
            # (allow_redirects=False), and the body is streamed under a size cap.
            response = _pinned_get(current_url, timeout=15, headers=headers, max_bytes=max_bytes)

            redirect_count = 0
            while response.is_redirect and redirect_count < 5:
                redirect_url = response.headers.get("Location", "")
                if not redirect_url:
                    break
                # Re-validate the redirect target's scheme. Relative/schemeless
                # Locations fall here (no scheme) and are rejected; even if they
                # slipped through, _resolve_and_validate fails closed on the
                # missing hostname.
                redirect_parsed = urlparse(redirect_url)
                if redirect_parsed.scheme not in ("http", "https"):
                    return json.dumps({
                        "error": f"Invalid redirect scheme: {redirect_parsed.scheme!r}. Only http/https allowed.",
                        "url": redirect_url,
                    })
                response.close()  # release the streamed socket before the next hop
                current_url = redirect_url
                response = _pinned_get(current_url, timeout=15, headers=headers, max_bytes=max_bytes)
                redirect_count += 1

            response.raise_for_status()
        except _SSRFBlocked as exc:
            return json.dumps({"error": exc.message, "url": current_url})
        except _ResponseTooLarge as exc:
            return json.dumps({"error": exc.message, "url": current_url})
        except requests.exceptions.Timeout:
            jina = _fetch_via_jina(url, self.context, reason="timeout")
            if jina:
                return json.dumps(jina)
            return json.dumps({"error": "Request timed out", "url": url})
        except requests.exceptions.ConnectionError:
            jina = _fetch_via_jina(url, self.context, reason="connection_error")
            if jina:
                return json.dumps(jina)
            return json.dumps({"error": "Connection failed", "url": url})
        except requests.exceptions.HTTPError:
            jina = _fetch_via_jina(url, self.context, reason=f"http_{response.status_code}")
            if jina:
                return json.dumps(jina)
            return json.dumps({"error": f"HTTP {response.status_code}", "url": url})
        except requests.exceptions.RequestException:
            jina = _fetch_via_jina(url, self.context, reason="request_error")
            if jina:
                return json.dumps(jina)
            return json.dumps({"error": "Request failed", "url": url})

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not any(ct in content_type for ct in ("text/html", "text/plain", "application/xhtml", "text/xml", "application/xml")):
            return json.dumps({
                "error": f"Non-text content type: {content_type}",
                "url": url,
            })

        logger.info("web_fetch: fetched url=%s status=%d chars=%d", url, response.status_code, len(response.text))
        raw_html = response.text

        # --- Security pre-processing: strip hidden elements ---
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
        except Exception:
            return json.dumps({"error": "Failed to parse HTML", "url": url})
        _strip_hidden_elements(soup)
        cleaned_html = str(soup)

        # --- Extract content with readability, convert to markdown ---
        try:
            doc = ReadabilityDocument(cleaned_html)
            title = doc.short_title()
            article_html = doc.summary()
            text = html_to_md(article_html, strip=["img"])
            text = normalize_text(text)
        except Exception:
            logger.debug("web_fetch: readability extraction failed, falling back to BS4")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            text = soup.get_text(separator="\n", strip=True)
            text = normalize_text(text)

        # --- JS-rendered page detection: fall back to Jina ---
        if len(text) < _JS_RENDER_MAX_CONTENT and len(raw_html) > _JS_RENDER_MIN_HTML:
            logger.info("web_fetch: suspected JS-rendered page url=%s (html=%d, content=%d)", url, len(raw_html), len(text))
            jina = _fetch_via_jina(url, self.context, reason="js_rendered")
            if jina:
                return self._cache_and_return(cache, cache_key, jina, max_chars)

        _run_web_scan(text, self.context)

        result = {
            "url": url,
            "title": title,
            "content": text,
            "truncated": False,
            "char_count": len(text),
        }
        return self._cache_and_return(cache, cache_key, result, max_chars)

    @staticmethod
    def _cache_and_return(cache, cache_key: str, result: dict, max_chars: int) -> str:
        full_result = json.dumps(result)
        try:
            cache.set(cache_key, full_result, timeout=3600)
        except Exception:
            logger.debug("web_fetch: cache write failed, continuing")

        text = result["content"]
        if len(text) > max_chars:
            result = {**result, "content": text[:max_chars], "truncated": True, "char_count": max_chars}
        return json.dumps(result)


__all__ = ["WebFetchTool"]
