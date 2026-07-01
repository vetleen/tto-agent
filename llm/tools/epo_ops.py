"""EPO Open Patent Services (OPS) patent tools — search, retrieve, family.

Three skill-gated tools (`patent_epoops_search`, `patent_epoops_get`,
`patent_epoops_family`) backed by the European Patent Office's OPS REST API — the
API behind Espacenet, covering DOCDB bibliographic data, INPADOC families and
legal status, and EP/WO full text. Exposed through the `patent-searcher` seed
subagent skill.

Auth is OAuth2 client-credentials with a single shared Wilfred credential
(read-only public data). The tools register only when EPO_OPS_KEY and
EPO_OPS_SECRET are set.

NOTE ON OPS SPECIFICS: the endpoint paths, CQL field codes, publication-number
formats, and the nested JSON response shapes below follow the OPS v3.2 RESTful
Services Reference Guide. They cannot be verified from this codebase and should
be validated against live OPS responses (staging, with real credentials) before
launch. Every path/field constant and every parser is centralized here and
written defensively (missing/unexpected shapes degrade to partial data, never a
crash) so corrections are localized.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time

import requests
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Rate limiter (copied from brave_search to keep OPS throttling independent).
# --------------------------------------------------------------------------- #
class _TokenBucketRateLimiter:
    """Process-wide token bucket that gates outgoing requests."""

    def __init__(self, requests_per_second: float, burst: int = 1):
        self._rps = requests_per_second
        self._max_tokens = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + (now - self._last_refill) * self._rps,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rps
            time.sleep(wait)


def _rpm() -> int:
    from django.conf import settings

    return int(getattr(settings, "EPO_OPS_RPM", 30) or 30)


_ops_rate_limiter = _TokenBucketRateLimiter(requests_per_second=_rpm() / 60.0, burst=1)
_token_lock = threading.Lock()

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_RATE_LIMIT_BACKOFF_SCHEDULE: list[float] = [5.0, 15.0, 30.0, 60.0]

_TOKEN_CACHE_KEY = "epo_ops_access_token_v1"
_TOKEN_FALLBACK_TTL = 1140  # OPS tokens last ~20min; used if expires_in is absent

# OPS paths (validate against the OPS reference). ``base`` is e.g.
# https://ops.epo.org/3.2 ; REST services live under ``{base}/rest-services``.
_AUTH_PATH = "auth/accesstoken"
_REST_PREFIX = "rest-services"

# Retrieval uses the docdb dotted number format (CC.NUMBER.KIND). Validated live:
# OPS 404s when a kind code is appended to an epodoc number
# (epodoc/EP1000000A1 -> 404), but the docdb form (docdb/EP.1000000.A1) works for
# every authority AND preserves the kind (A1 vs B1 matters for claims/description).
# See _docdb_ref.

# Map a `patent_epoops_get` `parts` value to OPS constituents.
_PART_TO_CONSTITUENT = {
    "biblio": "biblio",
    "abstract": "abstract",
    "claims": "claims",
    "description": "description",
    "all": "biblio,abstract",
}

_ATTRIBUTION = "_Source: EPO / Espacenet (Open Patent Services)._"


# --------------------------------------------------------------------------- #
# Credentials & token.
# --------------------------------------------------------------------------- #
def _get_credentials() -> tuple[str, str]:
    from django.conf import settings

    key = getattr(settings, "EPO_OPS_KEY", "")
    secret = getattr(settings, "EPO_OPS_SECRET", "")
    if not key or not secret:
        raise ValueError("EPO_OPS_KEY / EPO_OPS_SECRET are not configured")
    return key, secret


def _base_url() -> str:
    from django.conf import settings

    return getattr(settings, "EPO_OPS_BASE_URL", "https://ops.epo.org/3.2").rstrip("/")


def _fetch_new_token() -> str | None:
    """Exchange client credentials for a bearer token; cache it. Returns the
    token, or None on failure."""
    from django.core.cache import cache

    key, secret = _get_credentials()
    basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
    try:
        resp = requests.post(
            f"{_base_url()}/{_AUTH_PATH}",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("EPO OPS token request failed: %s", e)
        return None

    token = payload.get("access_token")
    if not token:
        logger.warning("EPO OPS token response had no access_token")
        return None
    try:
        ttl = int(float(payload.get("expires_in", _TOKEN_FALLBACK_TTL))) - 60
    except (TypeError, ValueError):
        ttl = _TOKEN_FALLBACK_TTL
    try:
        cache.set(_TOKEN_CACHE_KEY, token, timeout=max(ttl, 60))
    except Exception:
        logger.debug("EPO OPS: token cache write failed, continuing")
    return token


def _get_access_token(force_refresh: bool = False) -> str | None:
    """Return a cached bearer token, fetching (under a lock) when missing."""
    from django.core.cache import cache

    if not force_refresh:
        try:
            cached = cache.get(_TOKEN_CACHE_KEY)
        except Exception:
            cached = None
        if cached:
            return cached
    with _token_lock:
        # Re-check under the lock so a concurrent fetch isn't duplicated.
        if not force_refresh:
            try:
                cached = cache.get(_TOKEN_CACHE_KEY)
            except Exception:
                cached = None
            if cached:
                return cached
        return _fetch_new_token()


def _invalidate_token() -> None:
    from django.core.cache import cache

    try:
        cache.delete(_TOKEN_CACHE_KEY)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Request core.
# --------------------------------------------------------------------------- #
def _ops_request(path: str, params: dict, tool_name: str, context=None) -> dict:
    """GET an OPS rest-service and return parsed JSON, or ``{"error": ...}``.

    ``path`` is relative to ``{base}/rest-services`` (e.g.
    ``published-data/search/biblio``). Never raises to the caller: HTTP/parse
    failures return a graceful error dict.
    """
    url = f"{_base_url()}/{_REST_PREFIX}/{path.lstrip('/')}"

    last_exc = None
    refreshed = False
    for attempt in range(1 + _MAX_RETRIES):
        token = _get_access_token(force_refresh=refreshed)
        if not token:
            return {"error": "EPO OPS authentication failed. Patent search is temporarily unavailable."}

        try:
            _ops_rate_limiter.acquire()
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                params=params,
                timeout=15,
                stream=True,
            )
            response.raise_for_status()

            from llm.tools.web_fetch import _enforce_size_and_buffer, _max_response_bytes

            _enforce_size_and_buffer(response, _max_response_bytes())
            _log_ops_usage(context, tool_name, len(response.content or b""))
            return response.json()

        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = getattr(response, "status_code", None)
            if status in (401, 403) and not refreshed:
                # Token may have been revoked before its TTL — refresh once.
                logger.info("EPO OPS %s, refreshing token and retrying", status)
                _invalidate_token()
                refreshed = True
                continue
            if status == 429 or (status is not None and status >= 500):
                wait = _RATE_LIMIT_BACKOFF_SCHEDULE[min(attempt, len(_RATE_LIMIT_BACKOFF_SCHEDULE) - 1)]
                logger.warning("EPO OPS %s (attempt %d), waiting %.1fs", status, attempt + 1, wait)
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue
            if status is not None and status < 500:
                body = ""
                try:
                    body = response.text[:300]
                except Exception:
                    pass
                logger.warning("EPO OPS client error %s path=%s body=%s", status, path, body)
                if status == 404:
                    return {"error": "No matching patent record was found (EPO OPS 404)."}
                return {"error": f"EPO OPS request failed ({status}). This will not resolve by retrying."}
        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning("EPO OPS timeout (attempt %d) path=%s", attempt + 1, path)
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("EPO OPS request error (attempt %d) path=%s: %s", attempt + 1, path, e)
        except Exception as e:
            from llm.tools.web_fetch import _ResponseTooLarge

            if isinstance(e, _ResponseTooLarge):
                logger.warning("EPO OPS response too large path=%s: %s", path, e)
                return {"error": "EPO OPS returned an unexpectedly large response."}
            logger.warning("EPO OPS unexpected error path=%s: %s", path, e)
            return {"error": "EPO OPS returned an unreadable response."}

        if attempt < _MAX_RETRIES:
            time.sleep(_BACKOFF_BASE * (2 ** attempt))

    logger.error("EPO OPS failed after retries path=%s: %s", path, last_exc)
    return {"error": "EPO OPS is currently unavailable after retries. Consider reporting this to the user."}


def _log_ops_usage(context, tool_name: str, response_bytes: int) -> None:
    """Best-effort per-org usage log. Never raises into the tool."""
    try:
        user_id = getattr(context, "user_id", None) if context else None
        org_id = None
        if user_id:
            from accounts.models import Membership

            org_id = (
                Membership.objects.filter(user_id=user_id)
                .values_list("org_id", flat=True)
                .first()
            )
        from llm.models import OpsUsageLog

        OpsUsageLog.objects.create(
            org_id=org_id,
            user_id=int(user_id) if user_id else None,
            tool_name=tool_name,
            response_bytes=response_bytes,
        )
    except Exception:
        logger.debug("EPO OPS: usage log write failed (non-fatal)")


# --------------------------------------------------------------------------- #
# Query construction & number normalization.
# --------------------------------------------------------------------------- #
_CQL_STRIP_RE = re.compile(r'["()=/]+')
_MAX_CQL_LEN = 1000


def _sanitize_cql_value(value: str) -> str:
    """Strip CQL-significant characters from a user/model-supplied value.

    v1 keeps this deliberately blunt (strip rather than escape) so a value can
    never inject Boolean operators or field codes into the query.
    """
    if not value:
        return ""
    cleaned = _CQL_STRIP_RE.sub(" ", value)
    return re.sub(r"\s+", " ", cleaned).strip()


def _sanitize_date(value: str, *, is_end: bool) -> str:
    """Coerce a date to OPS's YYYYMMDD form. Accepts YYYY or YYYYMMDD."""
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 4:
        return digits + ("1231" if is_end else "0101")
    if len(digits) == 8:
        return digits
    return ""


def _build_cql(
    keywords: str = "",
    applicant: str = "",
    inventor: str = "",
    cpc: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """Build an OPS CQL query string from structured inputs.

    Field codes (validate against OPS reference): txt (title+abstract+claims),
    pa (applicant), in (inventor), cpc (CPC classification), pd (publication
    date). Clauses are ANDed.
    """
    clauses: list[str] = []
    for field, raw in (
        ("txt", keywords),
        ("pa", applicant),
        ("in", inventor),
        ("cpc", cpc),
    ):
        val = _sanitize_cql_value(raw or "")
        if val:
            clauses.append(f'{field}="{val}"')

    df = _sanitize_date(date_from, is_end=False)
    dt = _sanitize_date(date_to, is_end=True)
    if df and dt:
        clauses.append(f'pd within "{df} {dt}"')
    elif df:
        clauses.append(f'pd within "{df} 30001231"')
    elif dt:
        clauses.append(f'pd within "10000101 {dt}"')

    return " and ".join(clauses)[:_MAX_CQL_LEN]


_PUBNUM_STRIP_RE = re.compile(r"[\s.,/\-]+")


def _normalize_pubnumber(raw: str) -> str:
    """Normalize a publication number to compact uppercase form.

    ``"EP 1 000 000 A1"`` / ``"ep1000000a1"`` / ``"EP.1000000.A1"`` →
    ``"EP1000000A1"``. Kind code (if present) is kept.
    """
    if not raw:
        return ""
    return _PUBNUM_STRIP_RE.sub("", raw.strip().upper())


_DOCDB_SPLIT_RE = re.compile(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$")


def _docdb_ref(raw: str) -> tuple[str, str] | None:
    """Return ``(input_format, path_number)`` for an OPS retrieval path, or None
    if empty.

    Prefers the docdb dotted form ``CC.NUMBER.KIND`` (which OPS accepts with the
    kind code, unlike epodoc — see the note above ``_PART_TO_CONSTITUENT``).
    Falls back to epodoc with any trailing kind stripped for numbers that don't
    split into the standard country/number/kind shape.
    """
    n = _normalize_pubnumber(raw)
    if not n:
        return None
    m = _DOCDB_SPLIT_RE.match(n)
    if m:
        country, number, kind = m.group(1), m.group(2), (m.group(3) or "")
        return "docdb", f"{country}.{number}.{kind}"
    return "epodoc", re.sub(r"([A-Z]{2}\d+)[A-Z]\d?$", r"\1", n)


def _espacenet_url(pubnumber: str) -> str:
    """Stable Espacenet deep link for a publication number (search by pn=)."""
    n = _normalize_pubnumber(pubnumber)
    if not n:
        return ""
    return f"https://worldwide.espacenet.com/patent/search?q=pn%3D{n}"


# --------------------------------------------------------------------------- #
# JSON parsing helpers (defensive; shapes per OPS reference — validate live).
# --------------------------------------------------------------------------- #
def _as_list(node) -> list:
    """OPS returns a single child as a dict and multiples as a list. Normalize."""
    if node is None:
        return []
    return node if isinstance(node, list) else [node]


def _text(node) -> str:
    """Extract text from an OPS value node ({"$": "text"} or a bare string)."""
    if isinstance(node, dict):
        val = node.get("$", "")
        return val if isinstance(val, str) else ""
    if isinstance(node, str):
        return node
    return ""


def _unwrap(data: dict) -> dict:
    if isinstance(data, dict):
        return data.get("ops:world-patent-data", data) or {}
    return {}


def _collect_text(node, acc: list[str]) -> None:
    """Recursively gather text from ``$`` leaves (skipping ``@attributes``).

    Fallback for constituents (claims/description/abstract) whose exact shape
    varies by authority.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$":
                if isinstance(v, str):
                    acc.append(v)
            elif isinstance(k, str) and k.startswith("@"):
                continue
            else:
                _collect_text(v, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_text(item, acc)
    elif isinstance(node, str):
        acc.append(node)


def _first_title(biblio: dict) -> str:
    titles = _as_list(biblio.get("invention-title"))
    en = [t for t in titles if isinstance(t, dict) and t.get("@lang") == "en"]
    chosen = en[0] if en else (titles[0] if titles else "")
    return _text(chosen)


def _party_names(biblio: dict, plural: str, singular: str) -> list[str]:
    parties = biblio.get("parties", {}) or {}
    group = parties.get(plural, {}) or {}
    entries = [e for e in _as_list(group.get(singular)) if isinstance(e, dict)]
    # OPS repeats each party once per data-format (epodoc + original). Prefer the
    # epodoc rendering to avoid near-duplicate names; fall back to all if absent.
    epodoc = [e for e in entries if e.get("@data-format") == "epodoc"]
    chosen = epodoc or entries
    names: list[str] = []
    for entry in chosen:
        name = entry.get(f"{singular}-name", {}) or {}
        text = _text(name.get("name"))
        if text and text not in names:
            names.append(text)
    return names


def _publication_date(biblio: dict) -> str:
    ref = biblio.get("publication-reference", {}) or {}
    for doc_id in _as_list(ref.get("document-id")):
        if isinstance(doc_id, dict) and doc_id.get("date"):
            date = _text(doc_id.get("date"))
            if date:
                return date
    return ""


def _pubnumber_from_attrs(node: dict) -> str:
    country = node.get("@country", "") or ""
    number = node.get("@doc-number", "") or ""
    kind = node.get("@kind", "") or ""
    return f"{country}{number}{kind}"


def _doc_id_pubnumber(doc_id: dict) -> str:
    """Publication number from a single ``document-id`` node.

    Handles both the attribute form (``@country``/``@doc-number``/``@kind`` — as
    in search exchange-documents) and the child-element form
    (``country``/``doc-number``/``kind`` as ``{"$": ...}`` — as in family
    members). For epodoc, ``doc-number`` already carries the country prefix.
    """
    if not isinstance(doc_id, dict):
        return ""

    def part(key: str) -> str:
        attr = doc_id.get("@" + key)
        if attr:
            return str(attr)
        return _text(doc_id.get(key))

    if doc_id.get("@document-id-type") == "epodoc":
        return part("doc-number") + part("kind")
    return f"{part('country')}{part('doc-number')}{part('kind')}"


def _pubnumber_from_doc_ids(doc_ids) -> str:
    """Best publication number from a ``document-id`` list, preferring the docdb
    rendering (country + number + kind), then epodoc, then anything usable."""
    ids = [d for d in _as_list(doc_ids) if isinstance(d, dict)]
    for want in ("docdb", "epodoc"):
        for d in ids:
            if d.get("@document-id-type") == want:
                num = _doc_id_pubnumber(d)
                if num:
                    return num
    for d in ids:
        num = _doc_id_pubnumber(d)
        if num:
            return num
    return ""


def _abstract_text(doc: dict) -> str:
    parts: list[str] = []
    for abstract in _as_list(doc.get("abstract")):
        if not isinstance(abstract, dict):
            continue
        for p in _as_list(abstract.get("p")):
            t = _text(p)
            if t:
                parts.append(t)
    return "\n".join(parts)


def _parse_exchange_document(doc: dict) -> dict:
    biblio = doc.get("bibliographic-data", {}) or {}
    pubnum = _pubnumber_from_attrs(doc)
    if not pubnum:
        ref = biblio.get("publication-reference", {}) or {}
        pubnum = _pubnumber_from_doc_ids(ref.get("document-id"))
    return {
        "publication_number": pubnum,
        "title": _first_title(biblio),
        "applicants": _party_names(biblio, "applicants", "applicant"),
        "inventors": _party_names(biblio, "inventors", "inventor"),
        "date": _publication_date(biblio),
        "abstract": _abstract_text(doc),
    }


def _exchange_documents(root: dict) -> list[dict]:
    """Collect exchange-document nodes from a search or retrieval envelope."""
    docs: list[dict] = []
    # Retrieval: root -> exchange-documents -> exchange-document
    for container in _as_list(root.get("exchange-documents")):
        if isinstance(container, dict):
            docs.extend(d for d in _as_list(container.get("exchange-document")) if isinstance(d, dict))
    # Search: root -> ops:biblio-search -> ops:search-result -> exchange-documents
    search = root.get("ops:biblio-search", {}) or {}
    sr = search.get("ops:search-result", {}) or {}
    for container in _as_list(sr.get("exchange-documents")):
        if isinstance(container, dict):
            docs.extend(d for d in _as_list(container.get("exchange-document")) if isinstance(d, dict))
    return docs


def _parse_search_results(data: dict) -> dict:
    root = _unwrap(data)
    search = root.get("ops:biblio-search", {}) or {}
    total = search.get("@total-result-count", "")
    results = [_parse_exchange_document(d) for d in _exchange_documents(root)]
    return {"total": total, "results": results, "count": len(results)}


def _parse_family(data: dict) -> dict:
    root = _unwrap(data)
    family = root.get("ops:patent-family", {}) or {}
    members = []
    for member in _as_list(family.get("ops:family-member")):
        if not isinstance(member, dict):
            continue
        pub_ref = member.get("publication-reference", {}) or {}
        pubnum = _pubnumber_from_doc_ids(pub_ref.get("document-id"))
        legal: list[str] = []
        for event in _as_list(member.get("ops:legal")):
            if isinstance(event, dict):
                desc = (event.get("@desc") or _text(event.get("ops:law-text")) or "").strip()
                code = (event.get("@code") or "").strip()
                label = " ".join(x for x in (code, desc) if x).strip()
                if label and label not in legal:
                    legal.append(label)
        members.append({"publication_number": pubnum, "legal_events": legal})
    return {"members": members, "count": len(members)}


# Legal-event descriptions that signal current status (surfaced first).
_LEGAL_STATUS_KEYWORDS = (
    "GRANT", "LAPSED", "CEASED", "REVOKED", "WITHDRAWN", "EXPIRED",
    "OPPOSITION", "REFUS", "FEE", "RENEWAL",
)


def _rank_legal(events: list[str]) -> list[str]:
    """Surface status-bearing legal events (grant/lapse/fee/...) ahead of
    routine procedural ones, preserving order within each group."""
    status = [e for e in events if any(k in e.upper() for k in _LEGAL_STATUS_KEYWORDS)]
    other = [e for e in events if e not in status]
    return status + other


# --------------------------------------------------------------------------- #
# Formatters (marker-wrapped markdown + attribution).
# --------------------------------------------------------------------------- #
def _wrap(lines: list[str]) -> str:
    from llm.tools._text_cleaning import (
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_END,
        EXTERNAL_CONTENT_NOTE,
    )

    return "\n".join([EXTERNAL_CONTENT_BEGIN, EXTERNAL_CONTENT_NOTE, "", *lines, "", _ATTRIBUTION, EXTERNAL_CONTENT_END])


def _clean(text: str) -> str:
    from llm.tools._text_cleaning import normalize_text

    return normalize_text(text or "")


def _format_search(data: dict) -> str:
    if "error" in data:
        return f"Patent search error: {data['error']}"
    parsed = _parse_search_results(data)
    if not parsed["results"]:
        return "No matching patents found."
    lines: list[str] = []
    total = parsed.get("total")
    if total:
        lines.append(f"About {total} total results; showing {parsed['count']}.")
        lines.append("")
    for i, r in enumerate(parsed["results"], 1):
        lines.append(f"**[{i}] {_clean(r['title']) or '(no title)'}** — {r['publication_number'] or '(no number)'}")
        url = _espacenet_url(r["publication_number"])
        if url:
            lines.append(f"Espacenet: {url}")
        if r["applicants"]:
            lines.append(f"Applicant(s): {_clean(', '.join(r['applicants']))}")
        if r["date"]:
            lines.append(f"Published: {r['date']}")
        abstract = _clean(r["abstract"])
        if abstract:
            lines.append(abstract[:600] + ("…" if len(abstract) > 600 else ""))
        lines.append("")
    return _wrap(lines)


def _format_get(data: dict, publication_number: str, parts: str) -> str:
    if "error" in data:
        return f"Patent retrieval error: {data['error']}"
    root = _unwrap(data)
    docs = _exchange_documents(root)
    lines: list[str] = [f"Publication: {publication_number} (requested: {parts})"]
    url = _espacenet_url(publication_number)
    if url:
        lines.append(f"Espacenet: {url}")
    lines.append("")
    header_len = len(lines)
    if docs:
        r = _parse_exchange_document(docs[0])
        if r["title"]:
            lines.append(f"**{_clean(r['title'])}**")
        if r["applicants"]:
            lines.append(f"Applicant(s): {_clean(', '.join(r['applicants']))}")
        if r["inventors"]:
            lines.append(f"Inventor(s): {_clean(', '.join(r['inventors']))}")
        if r["date"]:
            lines.append(f"Published: {r['date']}")
        if r["abstract"]:
            lines.append("")
            lines.append("Abstract:")
            lines.append(_clean(r["abstract"]))
    # For claims/description (or when the biblio parse is thin), surface raw text.
    if parts in ("claims", "description") or len(docs) == 0:
        acc: list[str] = []
        _collect_text(root, acc)
        body = _clean("\n".join(acc))
        if body:
            lines.append("")
            lines.append(body[:8000] + ("…" if len(body) > 8000 else ""))
    if len(lines) <= header_len:
        return f"No content found for {publication_number} (part: {parts})."
    return _wrap(lines)


def _format_family(data: dict, publication_number: str) -> str:
    if "error" in data:
        return f"Patent family error: {data['error']}"
    parsed = _parse_family(data)
    if not parsed["members"]:
        return f"No family members found for {publication_number}."
    lines: list[str] = [f"INPADOC family for {publication_number} ({parsed['count']} member(s)):", ""]
    for m in parsed["members"]:
        line = f"- {m['publication_number'] or '(unknown)'}"
        if m["legal_events"]:
            line += f" — legal: {_clean('; '.join(_rank_legal(m['legal_events'])[:6]))}"
        lines.append(line)
    return _wrap(lines)


# --------------------------------------------------------------------------- #
# Tools.
# --------------------------------------------------------------------------- #
class PatentEpoOpsSearchInput(ReasonBaseModel):
    keywords: str = Field(
        default="",
        description="Free-text keywords searched across title, abstract and claims.",
    )
    applicant: str = Field(default="", description="Applicant / assignee name to filter by.")
    inventor: str = Field(default="", description="Inventor name to filter by.")
    cpc: str = Field(default="", description="CPC classification code to filter by (e.g. H01M).")
    date_from: str = Field(default="", description="Earliest publication date, YYYY or YYYYMMDD.")
    date_to: str = Field(default="", description="Latest publication date, YYYY or YYYYMMDD.")
    count: int = Field(default=10, description="Number of results to return (1-25, default 10).")


class PatentEpoOpsSearchTool(ContextAwareTool):
    """Search EPO/Espacenet published patent data."""

    name: str = "patent_epoops_search"
    section: str = "skills"
    audience: str = "shared"
    start_label: str = "Searching patents..."
    end_label: str = "Searched patents"
    description: str = (
        "Search the EPO/Espacenet patent database (Open Patent Services) for published "
        "patents by keywords, applicant, inventor, CPC classification and/or publication "
        "date range. Returns a ranked list of publications with numbers, titles, "
        "applicants and abstract snippets. Use patent_epoops_get to read a specific "
        "publication in full."
    )
    args_schema: type[BaseModel] = PatentEpoOpsSearchInput

    def _run(
        self,
        keywords: str = "",
        applicant: str = "",
        inventor: str = "",
        cpc: str = "",
        date_from: str = "",
        date_to: str = "",
        count: int = 10,
        **kwargs,
    ) -> str:
        from django.core.cache import cache

        cql = _build_cql(keywords, applicant, inventor, cpc, date_from, date_to)
        if not cql:
            return "Patent search error: provide at least one of keywords, applicant, inventor or cpc."
        count = max(1, min(int(count or 10), 25))

        cache_key = "epo_ops_search_v1:" + hashlib.sha256(f"{cql}:{count}".encode()).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            return _format_search(json.loads(cached))

        data = _ops_request(
            "published-data/search/biblio",
            {"q": cql, "Range": f"1-{count}"},
            tool_name=self.name,
            context=self.context,
        )
        if "error" not in data:
            try:
                cache.set(cache_key, json.dumps(data), timeout=900)
            except Exception:
                logger.debug("epo_ops search: cache write failed, continuing")
        return _format_search(data)


class PatentEpoOpsGetInput(ReasonBaseModel):
    publication_number: str = Field(
        description="Publication number to retrieve, e.g. EP1000000A1 or US9876543B2."
    )
    parts: str = Field(
        default="biblio",
        description=(
            "Which part to retrieve: biblio (bibliographic data + abstract), abstract, "
            "claims, description, or all (biblio + abstract). Default biblio. "
            "claims/description are large and only available for some authorities (EP/WO)."
        ),
    )


class PatentEpoOpsGetTool(ContextAwareTool):
    """Retrieve a single EPO/Espacenet publication."""

    name: str = "patent_epoops_get"
    section: str = "skills"
    audience: str = "shared"
    start_label: str = "Retrieving patent..."
    end_label: str = "Retrieved patent"
    description: str = (
        "Retrieve a specific patent publication from EPO/Espacenet by publication number. "
        "Choose parts to control what is returned: biblio (title, applicants, date, "
        "abstract), abstract, claims, description, or all. Use patent_epoops_search first "
        "to find publication numbers."
    )
    args_schema: type[BaseModel] = PatentEpoOpsGetInput

    def _run(self, publication_number: str = "", parts: str = "biblio", **kwargs) -> str:
        from django.core.cache import cache

        ref = _docdb_ref(publication_number)
        if ref is None:
            return "Patent retrieval error: a publication number is required."
        fmt, path_number = ref
        display = _normalize_pubnumber(publication_number)
        parts = (parts or "biblio").strip().lower()
        constituent = _PART_TO_CONSTITUENT.get(parts)
        if constituent is None:
            return (
                "Patent retrieval error: parts must be one of "
                "biblio, abstract, claims, description, all."
            )

        cache_key = "epo_ops_get_v1:" + hashlib.sha256(
            f"{fmt}:{path_number}:{constituent}".encode()
        ).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            return _format_get(json.loads(cached), display, parts)

        data = _ops_request(
            f"published-data/publication/{fmt}/{path_number}/{constituent}",
            {},
            tool_name=self.name,
            context=self.context,
        )
        if "error" not in data:
            try:
                cache.set(cache_key, json.dumps(data), timeout=3600)
            except Exception:
                logger.debug("epo_ops get: cache write failed, continuing")
        return _format_get(data, display, parts)


class PatentEpoOpsFamilyInput(ReasonBaseModel):
    publication_number: str = Field(
        description="Publication number whose patent family to retrieve, e.g. EP1000000A1."
    )


class PatentEpoOpsFamilyTool(ContextAwareTool):
    """Retrieve the INPADOC patent family + legal status for a publication."""

    name: str = "patent_epoops_family"
    section: str = "skills"
    audience: str = "shared"
    start_label: str = "Looking up patent family..."
    end_label: str = "Retrieved patent family"
    description: str = (
        "Retrieve the INPADOC patent family for a publication from EPO/Espacenet: the "
        "related filings in other jurisdictions (where else it was filed) and their legal "
        "status (granted / lapsed / in force). Use this for freedom-to-operate and to "
        "understand a patent's geographic reach."
    )
    args_schema: type[BaseModel] = PatentEpoOpsFamilyInput

    def _run(self, publication_number: str = "", **kwargs) -> str:
        from django.core.cache import cache

        ref = _docdb_ref(publication_number)
        if ref is None:
            return "Patent family error: a publication number is required."
        fmt, path_number = ref
        display = _normalize_pubnumber(publication_number)

        cache_key = "epo_ops_family_v1:" + hashlib.sha256(
            f"{fmt}:{path_number}".encode()
        ).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            return _format_family(json.loads(cached), display)

        data = _ops_request(
            f"family/publication/{fmt}/{path_number}/legal",
            {},
            tool_name=self.name,
            context=self.context,
        )
        if "error" not in data:
            try:
                cache.set(cache_key, json.dumps(data), timeout=3600)
            except Exception:
                logger.debug("epo_ops family: cache write failed, continuing")
        return _format_family(data, display)


__all__ = [
    "PatentEpoOpsSearchTool",
    "PatentEpoOpsGetTool",
    "PatentEpoOpsFamilyTool",
]
