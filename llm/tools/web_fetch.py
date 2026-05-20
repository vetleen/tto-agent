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
from django.conf import settings as django_settings
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as html_to_md
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument

from llm.tools._text_cleaning import normalize_text
from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)

_ABSOLUTE_MAX_CHARS = 50_000

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


def _check_url_ssrf(url: str) -> str | None:
    """Return an error message if the URL targets a private/internal host, else None."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return "No hostname in URL"
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return f"DNS resolution failed for {hostname}"
    for family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            return "URL resolves to a private or reserved IP address"
    return None


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
        )
        resp.raise_for_status()
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

        # SSRF protection: block requests to private/internal IPs
        ssrf_error = _check_url_ssrf(url)
        if ssrf_error:
            return json.dumps({"error": ssrf_error, "url": url})

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
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": f"Mozilla/5.0 (compatible; {django_settings.ASSISTANT_NAME}Bot/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                allow_redirects=False,
            )

            redirect_count = 0
            while response.is_redirect and redirect_count < 5:
                redirect_url = response.headers.get("Location", "")
                if not redirect_url:
                    break
                ssrf_error = _check_url_ssrf(redirect_url)
                if ssrf_error:
                    return json.dumps({"error": ssrf_error, "url": redirect_url})
                response = requests.get(
                    redirect_url,
                    timeout=15,
                    headers={
                        "User-Agent": f"Mozilla/5.0 (compatible; {django_settings.ASSISTANT_NAME}Bot/1.0)",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                    allow_redirects=False,
                )
                redirect_count += 1

            response.raise_for_status()
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
