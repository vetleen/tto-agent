"""Web Fetch tool — fetch and extract clean text from web pages."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel, Field

from llm.tools._text_cleaning import normalize_text
from llm.tools.interfaces import ContextAwareTool

logger = logging.getLogger(__name__)

_ABSOLUTE_MAX_CHARS = 50_000

# Tags to decompose beyond the standard script/style/nav set.
_EXTRA_STRIP_TAGS = [
    "aside", "form", "svg", "canvas", "object", "embed",
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
    # Decompose extra non-content tags
    for tag in soup.find_all(_EXTRA_STRIP_TAGS):
        tag.decompose()

    # Remove elements hidden via style, attributes, or input type
    for tag in list(soup.find_all(True)):
        # Hidden HTML attribute
        if tag.has_attr("hidden"):
            tag.decompose()
            continue

        # aria-hidden="true"
        if tag.get("aria-hidden", "").lower() == "true":
            tag.decompose()
            continue

        # <input type="hidden">
        if tag.name == "input" and tag.get("type", "").lower() == "hidden":
            tag.decompose()
            continue

        # Inline style hiding
        style = tag.get("style", "")
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


class WebFetchInput(BaseModel):
    url: str = Field(description="The URL of the web page to fetch.")
    max_chars: int = Field(default=20_000, description="Maximum characters to return (default 20000, max 50000).")


class WebFetchTool(ContextAwareTool):
    """Fetch a web page and extract clean text content."""

    name: str = "web_fetch"
    description: str = (
        "Fetch a web page and extract its text content. "
        "Use this to read the content of a specific URL, such as articles, "
        "documentation, or other web pages."
    )
    args_schema: type[BaseModel] = WebFetchInput

    def _run(self, url: str, max_chars: int = 20_000) -> str:
        from django.core.cache import cache

        url = url.strip()
        if not url:
            return json.dumps({"error": "Empty URL"})

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed."})

        max_chars = max(1, min(max_chars, _ABSOLUTE_MAX_CHARS))

        cache_key = "web_fetch:" + hashlib.sha256(url.encode()).hexdigest()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Web fetch cache hit for url=%s", url)
            # Re-truncate cached content to requested max_chars
            data = json.loads(cached)
            content = data.get("content", "")
            if len(content) > max_chars:
                data["content"] = content[:max_chars]
                data["truncated"] = True
                data["char_count"] = max_chars
            return json.dumps(data)

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; WilfredBot/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return json.dumps({"error": "Request timed out", "url": url})
        except requests.exceptions.ConnectionError:
            return json.dumps({"error": "Connection failed", "url": url})
        except requests.exceptions.HTTPError:
            return json.dumps({"error": f"HTTP {response.status_code}", "url": url})
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Request failed: {e}", "url": url})

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not any(ct in content_type for ct in ("text/html", "text/plain", "application/xhtml", "text/xml", "application/xml")):
            return json.dumps({
                "error": f"Non-text content type: {content_type}",
                "url": url,
            })

        try:
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            return json.dumps({"error": "Failed to parse HTML", "url": url})

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove unwanted tags
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
            tag.decompose()

        # Strip hidden/invisible elements (prompt-injection defense)
        _strip_hidden_elements(soup)

        # Prefer main content area if available
        main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
        source = main if main else soup

        # Extract and normalize text
        text = source.get_text(separator="\n", strip=True)
        text = normalize_text(text)

        # Scan for prompt injection (log only, never blocks)
        try:
            if text.strip():
                from guardrails.web_content import scan_web_content

                scan_web_content(
                    text,
                    user_id=self.context.user_id if self.context else None,
                    thread_id=self.context.conversation_id if self.context else None,
                    org_id=None,
                    source_label="web_fetch",
                )
        except Exception:
            logger.debug("web_fetch: web content scan failed (non-fatal)")

        # Cache full content before truncating for caller
        full_result = json.dumps({
            "url": url,
            "title": title,
            "content": text,
            "truncated": False,
            "char_count": len(text),
        })
        cache.set(cache_key, full_result, timeout=3600)

        # Truncate for this request
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return json.dumps({
            "url": url,
            "title": title,
            "content": text,
            "truncated": truncated,
            "char_count": len(text),
        })


__all__ = ["WebFetchTool"]
