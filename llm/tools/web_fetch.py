"""Web Fetch tool — fetch and extract clean text from web pages."""

from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool

logger = logging.getLogger(__name__)

_ABSOLUTE_MAX_CHARS = 50_000


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
        url = url.strip()
        if not url:
            return json.dumps({"error": "Empty URL"})

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed."})

        max_chars = max(1, min(max_chars, _ABSOLUTE_MAX_CHARS))

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

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Truncate
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
