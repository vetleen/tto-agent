"""Web Fetch tool — fetch and extract clean markdown from web pages."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
import time

from urllib.parse import urljoin, urlparse

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

# trafilatura (the primary content extractor in _extract_content) is chatty: on
# pages it can't cleanly parse it logs per-page chatter such as "discarding data:
# None" at WARNING, which Sentry's WARNING capture turned into events from the
# web_fetch tool (WILFRED-68). Extraction has a fallback chain (readability →
# bs4) so these are non-actionable for us. Raise the library's level to ERROR so
# genuine failures still surface while the per-page chatter stays out of Sentry.
logging.getLogger("trafilatura").setLevel(logging.ERROR)

_ABSOLUTE_MAX_CHARS = 50_000

# Default hard ceiling on bytes downloaded from a (user/LLM-supplied) URL.
# Overridable via settings.WEB_FETCH_MAX_RESPONSE_BYTES.
_DEFAULT_MAX_RESPONSE_BYTES = 10_000_000  # 10 MB


def _max_response_bytes() -> int:
    return getattr(django_settings, "WEB_FETCH_MAX_RESPONSE_BYTES", _DEFAULT_MAX_RESPONSE_BYTES)


# HTML parser for the two BeautifulSoup passes (text + image discovery). lxml is
# several times leaner and faster than the stdlib "html.parser" on the parse
# tree — the transient web-research memory culprit (Aug-2026 R14 tracemalloc
# run). A constant keeps both call sites (and tests) in sync and makes it
# trivially revertible. lxml ships transitively via readability-lxml.
_HTML_PARSER = "lxml"

# High safety cap on the size of decoded HTML handed to the parser. The download
# is already bounded (_DEFAULT_MAX_RESPONSE_BYTES), but a bs4 tree is several
# times the HTML size, so an oversized page can still spike RSS transiently.
# This is an absolute safety net for pathological pages, not a normal-path limit
# — kept high; HTML over it is truncated before parsing (article content sits
# near the top of the document). Overridable via settings.WEB_FETCH_MAX_PARSE_BYTES.
_DEFAULT_MAX_PARSE_BYTES = 8_000_000  # 8 MB


def _max_parse_bytes() -> int:
    return getattr(django_settings, "WEB_FETCH_MAX_PARSE_BYTES", _DEFAULT_MAX_PARSE_BYTES)


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


class _RedirectBlocked(Exception):
    """Raised when a redirect Location uses a non-http(s) scheme."""

    def __init__(self, message: str, url: str):
        self.message = message
        self.url = url
        super().__init__(message)

# Non-content elements removed for BOTH text and image extraction: scripts,
# styling, embeds, and forms/buttons (which carry spam and injection payloads,
# never article content — readability dropped them implicitly; trafilatura keeps
# them, so strip explicitly to stay extractor-independent).
_NONCONTENT_TAGS = [
    "script", "style", "noscript", "iframe",
    "svg", "canvas", "object", "embed",
    "meta", "template", "dialog",
    "form", "button",
]
# Semantic "chrome" regions. Stripped for TEXT extraction (article cleaning) but
# KEPT for image discovery: on portal/homepage layouts the real content images
# live inside nav/header/aside, so stripping them there loses everything.
_CHROME_TAGS = ["nav", "footer", "header", "aside"]
_NOISE_TAGS = _NONCONTENT_TAGS + _CHROME_TAGS

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
    """Remove non-content + chrome tags and elements invisible to humans (text
    extraction path).

    Attackers hide prompt-injection payloads in elements styled with
    display:none, visibility:hidden, zero font-size, aria-hidden, etc.
    These are invisible in a browser but survive ``get_text()`` extraction.
    """
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    _remove_hidden_elements(soup)


def _strip_for_images(soup: BeautifulSoup) -> None:
    """Remove non-content and hidden elements but KEEP semantic chrome
    (nav/header/footer/aside) — the image-discovery path.

    Portal/homepage layouts put their real content images inside the chrome
    regions the text path strips, so image discovery uses this lighter clean.
    Security-critical hidden-element removal still runs (no invisible images).
    """
    for tag in soup.find_all(_NONCONTENT_TAGS):
        tag.decompose()
    # aria-hidden is NOT visual hiding — news sites routinely mark decorative-
    # but-visible teaser <figure>s aria-hidden="true" (their alt duplicates the
    # headline). Honouring it here would drop exactly the content images we want,
    # so the image path skips the aria-hidden rule (visual hiding still applies).
    _remove_hidden_elements(soup, include_aria=False)


def _remove_hidden_elements(soup: BeautifulSoup, *, include_aria: bool = True) -> None:
    """Decompose elements hidden via attributes/style, and strip HTML comments.
    Shared by the text (:func:`_strip_hidden_elements`) and image
    (:func:`_strip_for_images`) cleaning paths.

    ``include_aria`` removes ``aria-hidden="true"`` elements — correct for text
    (screen-reader-hidden content is an injection vector) but wrong for images,
    where aria-hidden marks visible decorative figures."""
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

        # aria-hidden="true" (text path only)
        if include_aria and str(attrs.get("aria-hidden") or "").lower() == "true":
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


# Total wall-clock cap on downloading a single response body. The per-socket
# read timeout (15s) does NOT bound total time: a server that trickles a few
# bytes just inside each read window can hold the connection open indefinitely
# without ever exceeding the byte cap. This cap is generous — a page that needs
# longer than this to send its body is the server's problem, not ours.
_MAX_DOWNLOAD_SECONDS = 30.0


def _enforce_size_and_buffer(
    resp: requests.Response, max_bytes: int, max_seconds: float = _MAX_DOWNLOAD_SECONDS
) -> None:
    """Enforce a hard byte ceiling and total-time cap while reading, then buffer.

    Fast-rejects when ``Content-Length`` already exceeds the cap, but does not
    trust it (chunked / lying servers): the limit is also enforced while
    streaming. A total-elapsed deadline additionally guards against slow-drip
    servers that never trip the per-read socket timeout. On success the full body
    is stored on ``resp._content`` so ``resp.text`` / ``resp.content`` (and their
    charset detection) behave exactly as a non-streamed response would. Raises
    ``_ResponseTooLarge`` or ``requests.exceptions.Timeout`` and closes the
    response when a cap is exceeded.
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
    deadline = time.monotonic() + max_seconds
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        if time.monotonic() > deadline:
            resp.close()
            raise requests.exceptions.Timeout(
                f"Response body download exceeded {max_seconds:.0f}s (slow server); aborted"
            )
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
    # Because the connection targets an IP literal, urllib3 would otherwise send
    # ``Host: <ip>`` — which name-based-vhost CDNs (CloudFront, shared hosts, …)
    # answer with a 404. Send the real hostname explicitly so vhost routing works.
    # SSRF is unaffected: we still connect only to the validated IP. Set (not
    # setdefault) so each redirect hop carries its own host, never a stale one.
    req_headers = dict(headers or {})
    req_headers["Host"] = hostname
    try:
        resp = session.get(
            url, timeout=timeout, headers=req_headers, allow_redirects=False, stream=True,
        )
        _enforce_size_and_buffer(resp, max_bytes)
        return resp
    finally:
        session.close()


def _pinned_get_following_redirects(
    url: str, *, headers: dict, max_bytes: int, timeout: int = 15, max_redirects: int = 5
) -> tuple[requests.Response, str]:
    """Fetch ``url`` following up to ``max_redirects`` hops, returning
    ``(response, final_url)``.

    Each hop is independently DNS-resolved, SSRF-validated, and IP-pinned via
    :func:`_pinned_get` (``allow_redirects=False``), and every body is streamed
    under ``max_bytes``. Relative/schemeless ``Location`` headers are resolved
    against the current URL (RFC 7231) and re-validated so
    ``javascript:``/``ftp:``/``data:`` targets are rejected (``_RedirectBlocked``).
    Shared by the page fetch (``_fetch_core``) and the image fetch
    (``web_image_view``) so both get identical SSRF-safe redirect handling —
    important for CDN image URLs, which redirect constantly.
    """
    current_url = url
    response = _pinned_get(current_url, timeout=timeout, headers=headers, max_bytes=max_bytes)
    redirect_count = 0
    while response.is_redirect and redirect_count < max_redirects:
        redirect_url = response.headers.get("Location", "")
        if not redirect_url:
            break
        redirect_url = urljoin(current_url, redirect_url)
        redirect_parsed = urlparse(redirect_url)
        if redirect_parsed.scheme not in ("http", "https"):
            response.close()
            raise _RedirectBlocked(
                f"Invalid redirect scheme: {redirect_parsed.scheme!r}. Only http/https allowed.",
                redirect_url,
            )
        response.close()  # release the streamed socket before the next hop
        current_url = redirect_url
        response = _pinned_get(current_url, timeout=timeout, headers=headers, max_bytes=max_bytes)
        redirect_count += 1
    return response, current_url


class WebFetchInput(ReasonBaseModel):
    url: str = Field(description="The URL of the web page to fetch.")
    max_chars: int = Field(
        default=10_000,
        description=(
            "Maximum characters to return per call (default 10000, max 50000). "
            "Longer pages are truncated; the response then tells you the "
            "start_index to continue reading from."
        ),
    )
    start_index: int = Field(
        default=0,
        description=(
            "Character offset to continue reading a long page (default 0). "
            "Use the start_index suggested in a previous truncated response."
        ),
    )
    include_images: bool = Field(
        default=False,
        description=(
            "Set true to also list the page's content images (with img-N handles) "
            "so you can then view one with web_image_view. Leave false unless you "
            "need to see or reuse an image. The list appears only on the first "
            "page (start_index=0)."
        ),
    )


_JS_RENDER_MIN_HTML = 5000
_JS_RENDER_MAX_CONTENT = 200

# Jina bug: target-page errors come back as HTTP 200 with the error only in
# the body ("Target URL returned error 404: Not Found").
_JINA_BODY_ERROR_RE = re.compile(r"Target URL returned error\s+\d{3}", re.IGNORECASE)

# Jina 503s are usually transient (rendering cold starts); brief retries only —
# this runs inside an interactive chat turn, so no long waits.
_JINA_503_BACKOFF = [3, 8]

# Markdown image syntax ``![alt](url)`` — stripped from Jina body text (the
# images are surfaced separately via the data.images summary).
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _fetch_via_jina(url: str, context=None, reason: str = "") -> dict | None:
    """Fetch a page as clean markdown via the Jina Reader API.

    Jina renders the page (incl. JS) and runs its own extraction pipeline
    server-side, so no local HTML processing is needed. Also handles PDFs.
    """
    api_key = getattr(django_settings, "JINA_API_KEY", "")
    if not api_key:
        return None
    base_url = getattr(django_settings, "JINA_READER_BASE_URL", "https://r.jina.ai")
    logger.info("web_fetch: falling back to Jina for url=%s reason=%s", url, reason)

    resp = None
    for attempt in range(1 + len(_JINA_503_BACKOFF)):
        try:
            resp = requests.get(
                f"{base_url}/{url}",
                timeout=30,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "X-Return-Format": "markdown",
                    # Server-side equivalent of _strip_hidden_elements: drop
                    # invisible elements before extraction.
                    "X-Detach-Invisibles": "true",
                    # Populate data.images (a {name: url} summary) so image
                    # discovery works on the Jina path too — the dominant path,
                    # since many sites 400/421 our IP-pinned direct fetch. NOTE:
                    # X-Retain-Images:none SUPPRESSES this summary, so it's gone;
                    # inline image markdown is stripped from the text below.
                    "X-With-Images-Summary": "true",
                    "X-Timeout": "25",
                },
                stream=True,
            )
            resp.raise_for_status()
            # Trusted host — no SSRF pinning needed, but still cap the body so
            # a huge page can't OOM the worker. _ResponseTooLarge is an
            # Exception, so it's caught here and the fallback declines gracefully.
            _enforce_size_and_buffer(resp, _max_response_bytes())
            break
        except requests.exceptions.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 503 and attempt < len(_JINA_503_BACKOFF):
                logger.info(
                    "web_fetch: Jina 503 for url=%s (attempt %d), retrying",
                    url, attempt + 1,
                )
                time.sleep(_JINA_503_BACKOFF[attempt])
                continue
            # 402 = InsufficientBalanceError (account out of tokens) — make it
            # visible in logs so a dead fallback doesn't go unnoticed again.
            logger.warning("web_fetch: Jina fallback failed for url=%s status=%s", url, status)
            return None
        except Exception as e:
            logger.warning(
                "web_fetch: Jina fallback failed for url=%s error=%s: %s",
                url, type(e).__name__, str(e)[:200],
            )
            return None
    if resp is None:
        return None

    try:
        payload = resp.json()
    except Exception:
        logger.info("web_fetch: Jina returned non-JSON response for url=%s", url)
        return None

    data = payload.get("data") or {}
    title = normalize_text(data.get("title") or "")
    # Requesting the images summary means Jina keeps inline image markdown in the
    # body; strip it so the text stays clean (the images live in data.images).
    content = normalize_text(_MD_IMAGE_RE.sub("", data.get("content") or ""))

    if not content.strip():
        logger.info("web_fetch: Jina returned no extractable content for url=%s", url)
        return None

    if _JINA_BODY_ERROR_RE.search(content[:200]):
        logger.info("web_fetch: Jina reported in-body target error for url=%s", url)
        return None

    logger.info("web_fetch: Jina fallback succeeded for url=%s chars=%d", url, len(content))
    images = _images_from_jina(data)
    # Alt text (parsed from Jina's image summary) is page-supplied, untrusted
    # content — scan it alongside the body, exactly like the direct path.
    alt_blob = _alt_text_blob(images)
    _run_web_scan(content + (("\n" + alt_blob) if alt_blob else ""), context)

    return {
        "url": url,
        "title": title,
        "content": content,
        "char_count": len(content),
        "source": "jina",
        "images": images,
    }


def _run_web_scan(text: str, context=None) -> None:
    """Fire-and-forget prompt-injection scan (never blocks)."""
    from guardrails.web_content import scan_web_content_from_tool

    scan_web_content_from_tool(text, context, source_label="web_fetch")


def _alt_text_blob(images: list[dict]) -> str:
    """Join image candidates' alt texts for scanning.

    Alt text is page-supplied, untrusted content that ends up in the image
    manifest shown to the model — every path that produces candidates must
    scan it alongside the page body (direct, Jina, and the js-rendered
    fallback merge)."""
    return "\n".join(img["alt"] for img in images if img.get("alt"))


def _extract_content(cleaned_html: str, soup: BeautifulSoup) -> tuple[str, str]:
    """Extract (title, markdown body) from hidden-element-stripped HTML.

    Chain: trafilatura (primary) → readability + markdownify → bs4 text dump.
    Whichever non-last-resort extractor yields more content wins when
    trafilatura's output looks too thin.
    """
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    text = ""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            cleaned_html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
        )
        if extracted:
            text = normalize_text(extracted)
    except Exception:
        logger.debug("web_fetch: trafilatura extraction failed, falling back")

    if len(text) < _JS_RENDER_MAX_CONTENT:
        try:
            doc = ReadabilityDocument(cleaned_html)
            if not title:
                title = doc.short_title()
            fallback_text = normalize_text(html_to_md(doc.summary(), strip=["img"]))
            if len(fallback_text) > len(text):
                text = fallback_text
        except Exception:
            logger.debug("web_fetch: readability extraction failed, falling back to BS4")

    if not text:
        text = normalize_text(soup.get_text(separator="\n", strip=True))
    return title, text


# --- Image candidate extraction ----------------------------------------------

# Cap on how many image candidates a single fetch surfaces in its manifest.
_WEB_IMAGE_MANIFEST_CAP = 10
# Truncation limits for the manifest line (filename stem / alt text).
_IMG_FILENAME_STEM_MAX = 40
_IMG_ALT_MAX = 120
# Class/id/role substrings that mark chrome/icon/tracking images to drop.
_IMG_JUNK_RE = re.compile(r"logo|icon|avatar|sprite|thumb|badge|pixel", re.IGNORECASE)
# URL-path patterns for non-editorial assets: CMS plugin/theme/core directories
# and language-flag icons (e.g. WPML's /wp-content/plugins/.../res/flags/). These
# are chrome the DOM-position filter can't catch on the Jina path (no DOM there).
_JUNK_IMG_PATH_RE = re.compile(
    r"/wp-content/(?:plugins|themes)/|/wp-includes/|/flags/", re.IGNORECASE
)


def _attr_text(value) -> str:
    """Flatten a bs4 attribute (str or multi-valued list) to a plain string."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value or "")


def _int_attr(value) -> int | None:
    """Parse a leading integer from an HTML width/height attr, or None.

    Percentage values (``width="50%"``) are treated as unknown, not tiny."""
    if value is None:
        return None
    s = str(value).strip()
    if "%" in s:
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def _largest_srcset_url(srcset: str) -> str:
    """Return the candidate URL with the largest width/density descriptor, or ""."""
    best_url, best_weight = "", -1.0
    for part in srcset.split(","):
        bits = part.split()
        if not bits:
            continue
        cand = bits[0].strip()
        weight = 0.0
        if len(bits) > 1:
            m = re.match(r"([\d.]+)", bits[1].strip())
            if m:
                try:
                    weight = float(m.group(1))
                except ValueError:
                    weight = 0.0
        if cand and weight > best_weight:
            best_url, best_weight = cand, weight
    return best_url


def _truncate_filename(name: str) -> str:
    """Truncate a filename's stem to _IMG_FILENAME_STEM_MAX chars, keeping any
    short extension (the extension is itself a signal)."""
    name = (name or "").strip()
    if len(name) <= _IMG_FILENAME_STEM_MAX:
        return name
    dot = name.rfind(".")
    if 0 < dot and (len(name) - dot) <= 6:  # plausible short extension
        return name[:dot][:_IMG_FILENAME_STEM_MAX] + "…" + name[dot:]
    return name[:_IMG_FILENAME_STEM_MAX] + "…"


def _extract_image_candidates(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """Return content-image candidates from a chrome-stripped soup.

    Enumerates ``<img>`` in the cleaned DOM (header/footer/nav/aside already
    decomposed by :func:`_strip_hidden_elements`), drops obvious
    chrome/icons/pixels/SVG/data-URIs, scores figure- and alt-bearing images
    higher, dedups by absolute URL, and caps the list. Biased toward recall — a
    spurious entry costs one manifest line and the model is the final filter.
    Each dict: ``{"url", "filename", "alt"}``.
    """
    from urllib.parse import unquote

    seen: set[str] = set()
    scored: list[tuple[int, dict]] = []
    for img in soup.find_all("img"):
        attrs = img.attrs if isinstance(img.attrs, dict) else {}

        # Resolve source: prefer the largest srcset candidate, else src.
        raw_src = ""
        srcset = attrs.get("srcset")
        if isinstance(srcset, str) and srcset.strip():
            raw_src = _largest_srcset_url(srcset)
        if not raw_src:
            raw_src = str(attrs.get("src") or "").strip()
        if not raw_src or raw_src.startswith("data:"):
            continue

        abs_url = urljoin(base_url, raw_src)
        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith(".svg"):
            continue
        # Junk by URL (logo/icon/sprite in the filename, or a CMS plugin/theme/
        # flag asset dir) — kept chrome regions bring in header/footer logos.
        if _IMG_JUNK_RE.search(parsed.path) or _JUNK_IMG_PATH_RE.search(parsed.path):
            continue

        # Junk by class/id/role (logo, icon, sprite, thumbnail, tracking pixel).
        signal = " ".join(_attr_text(attrs.get(k)) for k in ("class", "id", "role"))
        if _IMG_JUNK_RE.search(signal):
            continue

        # Tiny by declared dimensions (icons / 1x1 pixels).
        w, h = _int_attr(attrs.get("width")), _int_attr(attrs.get("height"))
        if w is not None and h is not None and w <= 64 and h <= 64:
            continue

        if abs_url in seen:
            continue
        seen.add(abs_url)

        alt = normalize_text(str(attrs.get("alt") or "")).strip()[:_IMG_ALT_MAX]
        score = 0
        if img.find_parent("figure") is not None:
            score += 2
        if alt:
            score += 1
        filename = _truncate_filename(unquote(parsed.path.rsplit("/", 1)[-1]) or "image")
        scored.append((score, {"url": abs_url, "filename": filename, "alt": alt}))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:_WEB_IMAGE_MANIFEST_CAP]]


def _images_from_jina(data: dict) -> list[dict]:
    """Build image candidates from Jina Reader's ``data.images`` summary — a
    ``{name: url}`` dict in DOM order.

    Jina's names are usually generic (``"Image 1"``) so alt is left empty; URL
    order is preserved (top-of-page first). Same URL-level filtering as the
    direct path (skip data:/svg/logo-icon-sprite paths), deduped and capped.
    """
    from urllib.parse import unquote

    raw = data.get("images")
    if not isinstance(raw, dict):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for name, url in raw.items():
        if not isinstance(url, str):
            continue
        u = url.strip()
        if not u or u.startswith("data:"):
            continue
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith(".svg"):
            continue
        if _IMG_JUNK_RE.search(parsed.path) or _JUNK_IMG_PATH_RE.search(parsed.path):
            continue
        if u in seen:
            continue
        seen.add(u)
        # Jina names look like "Image 1" or "Image 2: <real alt>". Drop the
        # generic "Image N" prefix; keep any real alt that follows.
        nm = str(name).strip()
        m = re.match(r"image\s*\d+\s*[:\-–]?\s*", nm, re.IGNORECASE)
        alt = normalize_text(nm[m.end():] if m else nm)[:_IMG_ALT_MAX]
        filename = _truncate_filename(unquote(parsed.path.rsplit("/", 1)[-1]) or "image")
        out.append({"url": u, "filename": filename, "alt": alt})
        if len(out) >= _WEB_IMAGE_MANIFEST_CAP:
            break
    return out


def _cache_result(cache, cache_key: str, result: dict) -> dict:
    """Cache a successful fetch result dict (full content) and return it.

    Skips an empty extraction (no content) so a transient failure — a
    JS-rendered page, or Jina being down/declined — isn't pinned for the full
    hour; the next fetch of that URL can try again instead of re-serving "0 of 0".
    """
    if not (result.get("content") or "").strip():
        return result
    try:
        cache.set(cache_key, json.dumps(result), timeout=3600)
    except Exception:
        logger.debug("web_fetch: cache write failed, continuing")
    return result


def _fetch_core(url: str, cache, context=None) -> dict:
    """Fetch a URL and return a result dict with the FULL extracted content.

    Returns ``{url, title, content, char_count, source}`` on success or
    ``{"error": ..., "url": ...}`` on failure. Truncation/pagination happens
    at the formatting boundary, never here — the cache always holds the full
    content so paginated re-reads are cache hits.
    """
    url = url.strip()
    if not url:
        return {"error": "Empty URL", "url": url}

    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"error": f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed.", "url": url}

    # Note: SSRF protection (resolve + validate + connection pinning) happens
    # inside _pinned_get, per hop. We don't pre-check here so the host is
    # resolved exactly once per fetch and the validated IP is the one we
    # actually connect to (closes the DNS-rebinding gap).

    # v3: result dicts now carry an "images" candidate list; bump so pre-upgrade
    # cached entries (without it) aren't served for include_images requests.
    cache_key = "web_fetch_v3:" + hashlib.sha256(url.encode()).hexdigest()
    try:
        cached = cache.get(cache_key)
    except Exception:
        logger.debug("web_fetch: cache read failed, proceeding without cache")
        cached = None
    if cached is not None:
        logger.debug("Web fetch cache hit for url=%s", url)
        return json.loads(cached)

    # --- Fetch HTML ---
    headers = {
        "User-Agent": f"Mozilla/5.0 (compatible; {django_settings.ASSISTANT_NAME}Bot/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    max_bytes = _max_response_bytes()
    current_url = url
    try:
        # Each hop is resolved + SSRF-validated + IP-pinned independently and the
        # body is streamed under a size cap (see _pinned_get_following_redirects).
        response, current_url = _pinned_get_following_redirects(
            url, headers=headers, max_bytes=max_bytes, timeout=15
        )
        response.raise_for_status()
    except _RedirectBlocked as exc:
        return {"error": exc.message, "url": exc.url}
    except _SSRFBlocked as exc:
        return {"error": exc.message, "url": current_url}
    except _ResponseTooLarge as exc:
        return {"error": exc.message, "url": current_url}
    except requests.exceptions.Timeout:
        jina = _fetch_via_jina(url, context, reason="timeout")
        if jina:
            return _cache_result(cache, cache_key, jina)
        return {"error": "Request timed out", "url": url}
    except requests.exceptions.ConnectionError:
        jina = _fetch_via_jina(url, context, reason="connection_error")
        if jina:
            return _cache_result(cache, cache_key, jina)
        return {"error": "Connection failed", "url": url}
    except requests.exceptions.HTTPError:
        jina = _fetch_via_jina(url, context, reason=f"http_{response.status_code}")
        if jina:
            return _cache_result(cache, cache_key, jina)
        return {"error": f"HTTP {response.status_code}", "url": url}
    except requests.exceptions.RequestException:
        jina = _fetch_via_jina(url, context, reason="request_error")
        if jina:
            return _cache_result(cache, cache_key, jina)
        return {"error": "Request failed", "url": url}

    # Check content type
    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" in content_type:
        # Jina Reader parses PDFs natively; direct extraction can't.
        jina = _fetch_via_jina(url, context, reason="pdf")
        if jina:
            return _cache_result(cache, cache_key, jina)
        return {"error": "PDF could not be processed (Jina Reader unavailable)", "url": url}
    if not any(ct in content_type for ct in ("text/html", "text/plain", "application/xhtml", "text/xml", "application/xml")):
        return {"error": f"Non-text content type: {content_type}", "url": url}

    logger.info("web_fetch: fetched url=%s status=%d chars=%d", url, response.status_code, len(response.text))
    raw_html = response.text

    # Absolute safety net: a bs4 tree is several times the HTML size, so cap the
    # HTML handed to the parser. Rare — only pathologically large pages hit it;
    # the article content sits near the top, so truncation loses little.
    max_parse = _max_parse_bytes()
    if len(raw_html) > max_parse:
        logger.warning(
            "web_fetch: HTML for url=%s is %d chars (> %d cap); truncating before parse",
            url, len(raw_html), max_parse,
        )
        raw_html = raw_html[:max_parse]

    # --- Security pre-processing on a SINGLE parse tree ---
    # A bs4 tree is several times the HTML size, so building two (text + image)
    # doubles the transient RSS peak (Aug-2026 R14 tracemalloc). The image clean
    # is a strict PREFIX of the text clean, so we parse once: extract image
    # candidates first (chrome + aria still present, base = final URL after
    # redirects), then finish the text clean on the same tree. Output is
    # byte-identical to the old two-parse path.
    try:
        soup = BeautifulSoup(raw_html, _HTML_PARSER)
    except Exception:
        return {"error": "Failed to parse HTML", "url": url}

    # Image-discovery clean: non-content + visually-hidden removed, chrome and
    # aria-hidden kept — on homepage/portal layouts the content images live in
    # the chrome/aria-hidden regions the text path strips.
    _strip_for_images(soup)

    # Read image candidates from the tree at this state (read-only). Non-fatal:
    # a failure here must not lose the page text.
    try:
        images = _extract_image_candidates(soup, current_url)
    except Exception:
        logger.debug("web_fetch: image candidate extraction failed (non-fatal)")
        images = []

    # Finish the text clean on the SAME tree: drop semantic chrome and the
    # aria-hidden elements the image path intentionally kept. Combined with the
    # non-content + hidden removal above, this reproduces _strip_hidden_elements.
    for tag in soup.find_all(_CHROME_TAGS):
        tag.decompose()
    _remove_hidden_elements(soup, include_aria=True)
    cleaned_html = str(soup)

    # --- Extract content as markdown ---
    title, text = _extract_content(cleaned_html, soup)

    # --- JS-rendered page detection: fall back to Jina ---
    if len(text) < _JS_RENDER_MAX_CONTENT and len(raw_html) > _JS_RENDER_MIN_HTML:
        logger.info("web_fetch: suspected JS-rendered page url=%s (html=%d, content=%d)", url, len(raw_html), len(text))
        jina = _fetch_via_jina(url, context, reason="js_rendered")
        if jina:
            # Prefer Jina's own image summary (it rendered the page); fall back to
            # the images we extracted from the direct HTML only if Jina found none.
            if not jina.get("images"):
                jina["images"] = images
                # These direct-extracted alts were not part of the Jina-path
                # scan; scan them before they can reach a manifest.
                alt_blob = _alt_text_blob(images)
                if alt_blob:
                    _run_web_scan(alt_blob, context)
            return _cache_result(cache, cache_key, jina)

    # Alt text is page-supplied, untrusted content — scan it alongside the body.
    alt_blob = _alt_text_blob(images)
    _run_web_scan(text + (("\n" + alt_blob) if alt_blob else ""), context)

    result = {
        "url": url,
        "title": title,
        "content": text,
        "char_count": len(text),
        "source": "direct",
        "images": images,
    }
    return _cache_result(cache, cache_key, result)


def _format_fetch_error(data: dict) -> str:
    """Format a fetch error dict as a short plain-text line."""
    url = data.get("url", "")
    error = data.get("error", "Unknown error")
    if url:
        return f"Error fetching {url}: {error}"
    return f"Error: {error}"


def _build_image_manifest(images: list[dict], page_url: str, context) -> list[str]:
    """Allocate run-monotonic handles for image candidates and return manifest
    lines (containing NO URLs).

    Returns ``[]`` when there are no candidates or no context to register handles
    on. Handles are registered on ``context.web_image_manifest`` so
    ``web_image_view`` can resolve them; they live only for this run.
    """
    if not images or context is None or not hasattr(context, "allocate_web_image_handle"):
        return []
    lines = [
        "",
        "--- Images on this page ---",
        (
            'To look at any of these, call web_image_view with the handle(s) '
            '(e.g. web_image_view(handles=["img-1"])). Handles are valid only '
            "this turn — don't show them to the user."
        ),
    ]
    for img in images:
        handle = context.allocate_web_image_handle({
            "url": img["url"],
            "page_url": page_url,
            "filename": img.get("filename", ""),
            "alt": img.get("alt", ""),
        })
        alt = img.get("alt") or ""
        suffix = f' — "{alt}"' if alt else " — (no alt)"
        lines.append(f'[{handle}] {img.get("filename", "image")}{suffix}')
    return lines


def _format_fetch_result(
    data: dict, max_chars: int, start_index: int, image_manifest_lines: list[str] | None = None
) -> str:
    """Format a fetch result dict as markdown, sliced to the requested window.

    ``image_manifest_lines`` (already allocated by the caller) are rendered
    INSIDE the external-content wrapper — the manifest quotes page-supplied alt
    text, which is untrusted and must stay within the trust boundary.
    """
    from llm.tools._text_cleaning import (
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_END,
        EXTERNAL_CONTENT_NOTE,
    )

    content = data.get("content", "")
    total_len = len(content)

    if 0 < total_len <= start_index:
        return (
            f"Error fetching {data.get('url', '')}: start_index {start_index} is beyond "
            f"the end of the content ({total_len} chars). Use a smaller start_index."
        )

    window = content[start_index:start_index + max_chars]
    end_index = start_index + len(window)
    if end_index < total_len:
        # Truncate at a whitespace boundary so we don't cut mid-word.
        boundary = max(window.rfind(" "), window.rfind("\n"))
        if boundary > max_chars // 2:
            window = window[:boundary]
            end_index = start_index + len(window)

    if end_index < total_len:
        pagination = (
            f"Showing chars {start_index}–{end_index} of {total_len} — "
            f"call again with start_index={end_index} to continue reading"
        )
    else:
        pagination = f"Showing chars {start_index}–{end_index} of {total_len} (complete)"

    lines = [EXTERNAL_CONTENT_BEGIN, EXTERNAL_CONTENT_NOTE, ""]
    if data.get("title"):
        lines.append(f"**{data['title']}**")
    lines.append(f"URL: {data.get('url', '')}")
    if data.get("source") == "jina":
        lines.append("(fetched via Jina Reader)")
    lines.append(pagination)
    lines.append("")
    lines.append(window)
    if image_manifest_lines:
        lines.extend(image_manifest_lines)
    lines.append(EXTERNAL_CONTENT_END)
    return "\n".join(lines)


class WebFetchTool(ContextAwareTool):
    """Fetch a web page and extract clean markdown content."""

    name: str = "web_fetch"
    section: str = "skills"
    subagent_section: str = "chat"
    start_label: str = "Fetching web page..."
    end_label: str = "Fetched web page"
    description: str = (
        "Fetch a web page and extract its content as clean markdown. "
        "Use this to read the content of a specific URL, such as articles, "
        "documentation, PDFs, or other web pages. Long pages are returned "
        "in chunks: pass the suggested start_index to keep reading. "
        "Pass include_images=true to also list the page's content images so you "
        "can view one with web_image_view."
    )
    args_schema: type[BaseModel] = WebFetchInput

    def _run(
        self, url: str, max_chars: int = 10_000, start_index: int = 0,
        include_images: bool = False, **kwargs,
    ) -> str:
        from django.core.cache import cache

        max_chars = max(1, min(max_chars, _ABSOLUTE_MAX_CHARS))
        start_index = max(0, start_index)

        data = _fetch_core(url, cache, context=self.context)
        if "error" in data:
            return _format_fetch_error(data)

        # Allocate image handles only on the first page (start_index==0) so
        # paginated re-reads of the same page don't mint duplicate handles.
        manifest_lines = None
        if include_images and start_index == 0:
            manifest_lines = _build_image_manifest(
                data.get("images") or [], data.get("url", url), self.context
            )
        return _format_fetch_result(
            data, max_chars=max_chars, start_index=start_index,
            image_manifest_lines=manifest_lines,
        )


__all__ = ["WebFetchTool"]
