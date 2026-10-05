"""EPO Open Patent Services (OPS) patent tools — search, retrieve, family, CPC.

Four skill-gated tools (`patent_epoops_search`, `patent_epoops_get`,
`patent_epoops_family`, `patent_epoops_classification`) backed by the European
Patent Office's OPS REST API — the API behind Espacenet, covering DOCDB
bibliographic data, INPADOC families and legal status, EP/WO full text and the
CPC scheme. The first three are exposed through the `patent-searcher` seed
subagent skill; the classification tool is for skills that list it.

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
from pydantic import BaseModel, Field, field_validator

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel
from llm.tools._throttle import (
    BACKOFF_BASE as _BACKOFF_BASE,
    MAX_RETRIES as _MAX_RETRIES,
    RATE_LIMIT_BACKOFF_SCHEDULE as _RATE_LIMIT_BACKOFF_SCHEDULE,
    TokenBucketRateLimiter as _TokenBucketRateLimiter,
    deadline_capped_wait as _deadline_capped_wait,
)

logger = logging.getLogger(__name__)


def _rpm() -> int:
    from django.conf import settings

    try:
        rpm = int(getattr(settings, "EPO_OPS_RPM", 30))
    except (TypeError, ValueError):
        return 30
    return rpm if rpm > 0 else 30


_ops_rate_limiter = _TokenBucketRateLimiter(requests_per_second=_rpm() / 60.0, burst=1)
_token_lock = threading.Lock()

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
_FAULT_CODE_RE = re.compile(r"<(?:[\w-]+:)?code>\s*([^<]*?)\s*</", re.IGNORECASE)
_FAULT_MESSAGE_RE = re.compile(r"<(?:[\w-]+:)?message>\s*([^<]*?)\s*</", re.IGNORECASE)

# OPS answers a query it cannot evaluate (malformed CPC, sentinel dates, ...)
# with 500 SERVER.DomainAccess "please try again later" — deterministically, so
# retrying the same query only burns minutes of backoff.
_BAD_QUERY_FAULT = "SERVER.DomainAccess"


def _parse_ops_fault(response) -> tuple[str, str, str]:
    """``(code, message, rejection)`` from an OPS error response.

    ``code``/``message`` come from the OPS XML fault body (e.g.
    ``SERVER.DomainAccess``); ``rejection`` is the ``X-Rejection-Reason`` header
    OPS sets when a fair-use quota or throttle refused the call. Any part may be
    "". Never raises.
    """
    try:
        text = response.text or ""
    except Exception:
        text = ""
    if not isinstance(text, str):
        text = ""
    code_m = _FAULT_CODE_RE.search(text)
    msg_m = _FAULT_MESSAGE_RE.search(text)
    try:
        rejection = (response.headers or {}).get("X-Rejection-Reason", "") or ""
    except Exception:
        rejection = ""
    return (
        code_m.group(1).strip() if code_m else "",
        msg_m.group(1).strip() if msg_m else "",
        str(rejection).strip(),
    )


def _ops_request(
    path: str, params: dict, tool_name: str, context=None, accept: str = "application/json"
) -> dict:
    """GET an OPS rest-service and return parsed JSON, or ``{"error": ...}``.

    ``path`` is relative to ``{base}/rest-services`` (e.g.
    ``published-data/search/biblio``). Never raises to the caller: HTTP/parse
    failures return a graceful error dict. Every exit writes one
    ``OpsUsageLog`` row with the outcome (and OPS's fault code on failure).

    A non-JSON ``accept`` (the CPC scheme service only speaks XML) returns the
    body as ``{"_xml": text}`` instead of parsed JSON.
    """
    url = f"{_base_url()}/{_REST_PREFIX}/{path.lstrip('/')}"
    started = time.monotonic()
    attempts = 0

    def _done(result: dict, outcome: str, *, http_status=None, error_code="", error_message="",
              response_bytes=0) -> dict:
        _log_ops_usage(
            context,
            tool_name,
            response_bytes,
            outcome=outcome,
            http_status=http_status,
            error_code=error_code,
            error_message=error_message,
            request_path=path,
            query=str((params or {}).get("q", "")),
            attempts=max(attempts, 1),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    last_exc = None
    last_status = None
    last_code = ""
    last_message = ""
    refreshed = False
    for attempt in range(1 + _MAX_RETRIES):
        token = _get_access_token(force_refresh=refreshed)
        if not token:
            return _done(
                {"error": "EPO OPS authentication failed. Patent search is temporarily unavailable."},
                "error",
                error_code="AuthFailed",
            )

        attempts += 1
        try:
            _ops_rate_limiter.acquire()
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": accept,
                },
                params=params,
                timeout=15,
                stream=True,
            )
            response.raise_for_status()

            from llm.tools.web_fetch import _enforce_size_and_buffer, _max_response_bytes

            _enforce_size_and_buffer(response, _max_response_bytes())
            data = response.json() if accept == "application/json" else {"_xml": response.text}
            return _done(data, "ok", http_status=response.status_code,
                         response_bytes=len(response.content or b""))

        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = getattr(response, "status_code", None)
            code, message, rejection = _parse_ops_fault(response)
            last_status, last_code, last_message = status, code, message
            if status == 403 and rejection:
                # Fair-use quota / throttle refusal — a fresh token won't help.
                logger.warning("EPO OPS rejected request (%s) path=%s", rejection, path)
                return _done(
                    {"error": (
                        f"EPO OPS refused the request: fair-use quota or throttle reached "
                        f"({rejection}). Try again later and tell the user patent search is limited right now."
                    )},
                    "error",
                    http_status=status,
                    error_code=code or "Rejected",
                    error_message=f"X-Rejection-Reason: {rejection}. {message}".strip(),
                )
            if status in (401, 403) and not refreshed:
                # Token may have been revoked before its TTL — refresh once.
                logger.info("EPO OPS %s, refreshing token and retrying", status)
                _invalidate_token()
                refreshed = True
                continue
            if status == 500 and code == _BAD_QUERY_FAULT:
                logger.warning("EPO OPS could not process query (%s) path=%s", code, path)
                return _done(
                    {"error": (
                        f"EPO OPS could not process this query ({code}). This is almost always "
                        "the query itself, not an outage, so retrying it unchanged will not help: "
                        "use fewer keywords, check the CPC symbol (e.g. A61B8/06), or drop a filter."
                    )},
                    "error",
                    http_status=status,
                    error_code=code,
                    error_message=message,
                )
            if status == 429 or (status is not None and status >= 500):
                wait = _RATE_LIMIT_BACKOFF_SCHEDULE[min(attempt, len(_RATE_LIMIT_BACKOFF_SCHEDULE) - 1)]
                # Per-attempt retry chatter stays at INFO (Sentry breadcrumb);
                # the terminal "failed after retries" WARNING is the event.
                logger.info("EPO OPS %s %s (attempt %d), waiting %.1fs", status, code or "-", attempt + 1, wait)
                if attempt < _MAX_RETRIES:
                    nap, may_retry = _deadline_capped_wait(wait, context)
                    if nap > 0:
                        time.sleep(nap)
                    if may_retry:
                        continue
                    break  # run deadline reached during backoff — stop retrying
            # 429 is transient (rate limit). On the final attempt it must NOT
            # fall into this permanent-client-error branch, or the model is told
            # a rate limit "will not resolve by retrying" and abandons search;
            # let it drop to the terminal "unavailable after retries" message.
            if status is not None and status < 500 and status != 429:
                if status == 404:
                    # A missing record is a normal search outcome, not a fault.
                    logger.info("EPO OPS client error %s path=%s", status, path)
                    return _done(
                        {"error": "No matching patent record was found (EPO OPS 404)."},
                        "no_results",
                        http_status=status,
                        error_code=code,
                        error_message=message,
                    )
                logger.warning("EPO OPS client error %s %s path=%s", status, code or "-", path)
                detail = " ".join(p for p in (str(status), code) if p)
                if message:
                    detail = f"{detail}: {message}"
                return _done(
                    {"error": f"EPO OPS request failed ({detail}). This will not resolve by retrying."},
                    "error",
                    http_status=status,
                    error_code=code,
                    error_message=message,
                )
        except requests.exceptions.Timeout as e:
            last_exc = e
            last_status, last_code, last_message = None, "Timeout", str(e)
            logger.info("EPO OPS timeout (attempt %d) path=%s", attempt + 1, path)
        except requests.exceptions.RequestException as e:
            last_exc = e
            last_status, last_code, last_message = None, type(e).__name__, str(e)
            logger.info("EPO OPS request error (attempt %d) path=%s: %s", attempt + 1, path, e)
        except Exception as e:
            from llm.tools.web_fetch import _ResponseTooLarge

            if isinstance(e, _ResponseTooLarge):
                logger.warning("EPO OPS response too large path=%s: %s", path, e)
                return _done(
                    {"error": "EPO OPS returned an unexpectedly large response."},
                    "error",
                    http_status=200,
                    error_code="TooLarge",
                    error_message=str(e),
                )
            logger.warning("EPO OPS unexpected error path=%s: %s", path, e)
            return _done(
                {"error": "EPO OPS returned an unreadable response."},
                "error",
                error_code="Unreadable",
                error_message=f"{type(e).__name__}: {e}",
            )

        if attempt < _MAX_RETRIES:
            nap, may_retry = _deadline_capped_wait(_BACKOFF_BASE * (2 ** attempt), context)
            if nap > 0:
                time.sleep(nap)
            if not may_retry:
                break  # run deadline reached during backoff — stop retrying

    # WARNING (one Sentry event per exhausted call), not ERROR: the tool
    # degrades gracefully and the model reports the outage to the user.
    logger.warning("EPO OPS failed after retries path=%s code=%s", path, last_code or "-", exc_info=last_exc)
    suffix = f" (last error: {last_code})" if last_code else ""
    return _done(
        {"error": f"EPO OPS is currently unavailable after retries{suffix}. Consider reporting this to the user."},
        "error",
        http_status=last_status,
        error_code=last_code,
        error_message=last_message,
    )


def _log_ops_usage(
    context,
    tool_name: str,
    response_bytes: int = 0,
    *,
    outcome: str = "ok",
    http_status: int | None = None,
    error_code: str = "",
    error_message: str = "",
    request_path: str = "",
    query: str = "",
    attempts: int = 1,
    duration_ms: int = 0,
) -> None:
    """Best-effort per-org usage/outcome log. Never raises into the tool."""
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
            outcome=outcome,
            http_status=http_status,
            error_code=(error_code or "")[:64],
            error_message=(error_message or "")[:500],
            request_path=(request_path or "")[:255],
            query=(query or "")[:1000],
            attempts=min(max(int(attempts), 1), 32767),
            duration_ms=max(int(duration_ms), 0),
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


# Section/class/subclass, optionally main group and subgroup: H01M, A61B8,
# A61B8/06, G01S15/8984. Validated after whitespace is removed.
_CPC_RE = re.compile(r"^[A-HY]\d{2}[A-Z](\d{1,4}(/\d{1,6})?)?$")


def _normalize_cpc(raw: str) -> str:
    """Return a CPC symbol in OPS form (``A61B8/06``), or "" if it isn't one.

    The slash is significant: OPS answers ``cpc="A61B8 06"`` (what the generic
    sanitizer used to produce) with a 500 SERVER.DomainAccess.
    """
    if not raw:
        return ""
    # Only the conventional gaps are dropped ("A61B 8/06", "A61B8 / 06"); a space
    # between digits ("A61B8 06") is ambiguous and rejected, not glued into 806.
    cpc = raw.strip().upper()
    cpc = re.sub(r"^([A-HY]\d{2}[A-Z])\s+(?=\d)", r"\1", cpc)
    cpc = re.sub(r"\s*/\s*", "/", cpc)
    return cpc if _CPC_RE.match(cpc) else ""


_MAX_CPC_CODES = 10
_CPC_SPLIT_RE = re.compile(r"\s*(?:[,;]|\bor\b)\s*", re.IGNORECASE)


def _split_cpc(value) -> list[str]:
    """Raw CPC entries from a list or a free-form string.

    Accepts ``["A61B8/06", "G01S15/8984"]`` as well as the strings models tend
    to send instead: ``"A61B8/06, G01S15/8984"``, ``"A61B8/06 or G01S15/8984"``
    and ``"A61B8/06 G01S15/8984"``. A space inside one symbol (``"A61B 8/06"``)
    is kept together — whitespace only splits when every piece is a symbol.
    """
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        for piece in _CPC_SPLIT_RE.split(str(item or "")):
            piece = piece.strip()
            if not piece:
                continue
            words = piece.split()
            if len(words) > 1 and not _normalize_cpc(piece) and all(_normalize_cpc(w) for w in words):
                out.extend(words)
            else:
                out.append(piece)
    return out


def _cpc_clause(codes: list[str], include_subgroups: bool = True) -> str:
    """CQL for one or more (already valid) CPC symbols, OR-ed.

    ``/low`` widens a subgroup to everything filed beneath it (A61B8/06 also
    matches A61B8/065); main groups and subclasses already include theirs.
    OPS rejects ``/low`` inside ``cpc any "…"`` (400
    CLIENT.InvalidClassificationRelation), hence the parenthesised OR.
    """
    terms = []
    for code in codes:
        sym = _normalize_cpc(code)
        if not sym:
            continue
        if include_subgroups and "/" in sym:
            sym += "/low"
        term = f"cpc={sym}"
        if term not in terms:
            terms.append(term)
    if not terms:
        return ""
    return terms[0] if len(terms) == 1 else "(" + " or ".join(terms) + ")"


def _build_cql(
    keywords: str = "",
    applicant: str = "",
    inventor: str = "",
    cpc: str | list[str] = "",
    date_from: str = "",
    date_to: str = "",
    include_subgroups: bool = True,
) -> str:
    """Build an OPS CQL query string from structured inputs.

    Field codes: txt (title+abstract+claims), pa (applicant), in (inventor),
    cpc (CPC classification), pd (publication date). Clauses are ANDed.
    Verified live against OPS 3.2 (2026-10-05):

    - Multi-word keywords use ``txt all "a b c"`` (every word, any order);
      ``txt="a b c"`` is an exact-phrase search and usually finds nothing.
    - CPC is emitted unquoted with its slash (``cpc=A61B8/06``), several codes
      as ``(cpc=X or cpc=Y)``; invalid symbols are dropped (the tool rejects
      them before calling OPS). See ``_cpc_clause``.
    - A one-sided date range is ``pd>=`` / ``pd<=``. Padding it with a sentinel
      year (``pd within "20000101 30001231"``) makes OPS 500.
    """
    clauses: list[str] = []
    kw = _sanitize_cql_value(keywords or "")
    if kw:
        clauses.append(f'txt all "{kw}"' if " " in kw else f'txt="{kw}"')
    for field, raw in (("pa", applicant), ("in", inventor)):
        val = _sanitize_cql_value(raw or "")
        if val:
            clauses.append(f'{field}="{val}"')
    cpc_clause = _cpc_clause(_split_cpc(cpc)[:_MAX_CPC_CODES], include_subgroups)
    if cpc_clause:
        clauses.append(cpc_clause)

    df = _sanitize_date(date_from, is_end=False)
    dt = _sanitize_date(date_to, is_end=True)
    if df and dt:
        clauses.append(f'pd within "{df} {dt}"')
    elif df:
        clauses.append(f"pd>={df}")
    elif dt:
        clauses.append(f"pd<={dt}")

    return " and ".join(clauses)[:_MAX_CQL_LEN]


_PUBNUM_STRIP_RE = re.compile(r"[\s.,/\-]+")
_PUBNUM_VALID_RE = re.compile(r"^[A-Z0-9]+$")


def _normalize_pubnumber(raw: str) -> str:
    """Normalize a publication number to compact uppercase form.

    ``"EP 1 000 000 A1"`` / ``"ep1000000a1"`` / ``"EP.1000000.A1"`` →
    ``"EP1000000A1"``. Kind code (if present) is kept. Returns ``""`` for input
    that isn't a valid publication number after stripping formatting separators —
    a real number is letters+digits only, so anything else (``?``, ``#``, ``%``,
    …) would inject into the OPS URL path and is rejected here.
    """
    if not raw:
        return ""
    n = _PUBNUM_STRIP_RE.sub("", raw.strip().upper())
    if not _PUBNUM_VALID_RE.match(n):
        return ""
    return n


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
        # Skip an empty kind so a kind-less number yields "EP.1000000", not the
        # trailing-dot "EP.1000000." that OPS rejects with a 404.
        return "docdb", ".".join(part for part in (country, number, kind) if part)
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


def _cpc_codes(biblio: dict) -> dict[str, list[str]]:
    """CPC symbols assigned to a document, split into inventive / additional.

    OPS lists them under ``patent-classifications/patent-classification`` as
    parts (section, class, subclass, main-group, subgroup) with
    ``classification-value`` I (inventive) or A (additional), repeated once per
    generating office — so symbols are de-duplicated, first occurrence wins.
    A symbol seen as inventive anywhere is reported as inventive only.
    """
    inventive: list[str] = []
    additional: list[str] = []
    group = biblio.get("patent-classifications", {}) or {}
    for entry in _as_list(group.get("patent-classification") if isinstance(group, dict) else None):
        if not isinstance(entry, dict):
            continue
        scheme = entry.get("classification-scheme", {}) or {}
        if isinstance(scheme, dict) and not str(scheme.get("@scheme", "CPC")).upper().startswith("CPC"):
            continue
        parts = [_text(entry.get(k)).strip() for k in ("section", "class", "subclass", "main-group", "subgroup")]
        section, cls, subclass, main_group, subgroup = parts
        if not (section and cls and subclass and main_group and subgroup):
            continue
        symbol = f"{section}{cls}{subclass}{main_group}/{subgroup}"
        if _text(entry.get("classification-value")).strip().upper() == "A":
            if symbol not in additional and symbol not in inventive:
                additional.append(symbol)
        elif symbol not in inventive:
            inventive.append(symbol)
            if symbol in additional:
                additional.remove(symbol)
    return {"inventive": inventive, "additional": additional}


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
        "cpc": _cpc_codes(biblio),
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
        cpc_line = _cpc_line(r.get("cpc"), max_inventive=6, max_additional=4)
        if cpc_line:
            lines.append(cpc_line)
        abstract = _clean(r["abstract"])
        if abstract:
            lines.append(abstract[:600] + ("…" if len(abstract) > 600 else ""))
        lines.append("")
    return _wrap(lines)


def _cpc_line(cpc: dict | None, *, max_inventive: int | None = None, max_additional: int | None = None) -> str:
    """``CPC: A61B8/06, A61B8/4254 (additional: G01S15/8979)`` or ""."""
    if not cpc:
        return ""

    def _take(codes: list[str], cap: int | None) -> str:
        if cap is None or len(codes) <= cap:
            return ", ".join(codes)
        return ", ".join(codes[:cap]) + f", … (+{len(codes) - cap})"

    inventive = cpc.get("inventive") or []
    additional = cpc.get("additional") or []
    if not inventive and not additional:
        return ""
    line = "CPC: " + (_take(inventive, max_inventive) if inventive else "(none inventive)")
    if additional:
        line += f" (additional: {_take(additional, max_additional)})"
    return line


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
        cpc_line = _cpc_line(r.get("cpc"))
        if cpc_line:
            lines.append(cpc_line)
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
# CPC classification scheme (classification/cpc/...).
# --------------------------------------------------------------------------- #
_CPC_NS = "http://www.epo.org/cpcexport"
_CPC_ITEM = f"{{{_CPC_NS}}}classification-item"


def _cpc_lookup_symbol(raw: str) -> str:
    """Symbol in the form the scheme service expects, or "".

    The service wants main groups as ``A61B8/00`` (``A61B8`` → 404); subclasses
    (``A61B``) and subgroups are passed through.
    """
    sym = _normalize_cpc(raw)
    if sym and "/" not in sym and len(sym) > 4:
        sym += "/00"
    return sym


def _cpc_item_title(item) -> str:
    """Title of one ``classification-item``: its title-parts joined by "; ",
    each with any explanation (e.g. "A61B8/02 … take precedence") in parens.

    The text sits in ``cpc:text`` or, for many subgroups, ``cpc:comment/cpc:text``
    — so everything that isn't an explanation counts as the title.
    """
    title = item.find(f"{{{_CPC_NS}}}class-title")
    if title is None:
        return ""
    explanation_tag = f"{{{_CPC_NS}}}explanation"
    parts: list[str] = []
    for tp in title.findall(f"{{{_CPC_NS}}}title-part"):
        main_bits: list[str] = []
        expl_bits: list[str] = []
        for child in tp:
            (expl_bits if child.tag == explanation_tag else main_bits).append("".join(child.itertext()))
        main = " ".join(main_bits)
        expl = " ".join(expl_bits)
        main = re.sub(r"\s+", " ", main).strip()
        expl = re.sub(r"\s+", " ", expl).strip()
        if main and expl:
            parts.append(f"{main} ({expl})")
        elif main or expl:
            parts.append(main or expl)
    return "; ".join(parts)


def _cpc_item_info(item) -> dict:
    sym_el = item.find(f"{{{_CPC_NS}}}classification-symbol")
    return {
        "symbol": (sym_el.text or "").strip() if sym_el is not None else "",
        "title": _cpc_item_title(item),
        "has_children": item.get("has-children") == "true",
        "not_allocatable": item.get("not-allocatable") == "true",
    }


def _parse_cpc_scheme(xml_text: str, symbol: str) -> dict | None:
    """Path, entry and direct children for ``symbol`` from a scheme response
    fetched with ``ancestors=true&depth=1``. None when the symbol isn't there."""
    from lxml import etree

    if not xml_text:
        return None
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
        root = etree.fromstring(xml_text.encode("utf-8"), parser)
    except Exception:
        logger.info("EPO OPS: unparseable CPC scheme response for %s", symbol)
        return None

    target = None
    for item in root.iter(_CPC_ITEM):
        sym_el = item.find(f"{{{_CPC_NS}}}classification-symbol")
        if sym_el is not None and (sym_el.text or "").strip() == symbol:
            target = item
            break
    if target is None:
        return None

    path: list[dict] = []
    for anc in reversed(list(target.iterancestors(_CPC_ITEM))):
        info = _cpc_item_info(anc)
        # The scheme repeats a class symbol on two levels (A61: HEALTH /
        # MEDICAL…); fold those into one path step.
        if path and path[-1]["symbol"] == info["symbol"]:
            path[-1]["title"] = "; ".join(t for t in (path[-1]["title"], info["title"]) if t)
        else:
            path.append(info)
    children = [_cpc_item_info(c) for c in target.findall(_CPC_ITEM)]
    return {"path": path, "entry": _cpc_item_info(target), "children": children}


def _cpc_marks(info: dict) -> str:
    marks = []
    if info.get("not_allocatable"):
        marks.append("heading only, not assigned to documents")
    if info.get("has_children"):
        marks.append("has narrower subgroups")
    return f" [{'; '.join(marks)}]" if marks else ""


def _format_cpc_symbol(parsed: dict | None, symbol: str) -> str:
    if not parsed:
        return (
            f"No CPC entry for {symbol}. Check the symbol (e.g. A61B8/06), or look up its "
            "parent group, or find candidates with query=."
        )
    entry = parsed["entry"]
    lines = [f"CPC {entry['symbol']} — {_clean(entry['title'])}{_cpc_marks(entry)}", ""]
    if parsed["path"]:
        lines.append("Place in the scheme (broadest first):")
        for step in parsed["path"]:
            lines.append(f"- {step['symbol']} — {_clean(step['title'])}")
        lines.append("")
    if parsed["children"]:
        lines.append("Narrower groups directly below it:")
        for child in parsed["children"]:
            lines.append(f"- {child['symbol']} — {_clean(child['title'])}{_cpc_marks(child)}")
    else:
        lines.append("No narrower groups below this one.")
    return _wrap(lines)


def _parse_cpc_search(data: dict) -> list[dict]:
    root = _unwrap(data)
    search = root.get("ops:classification-search", {}) or {}
    result = search.get("ops:search-result", {}) or {}
    hits: list[dict] = []
    for stat in _as_list(result.get("ops:classification-statistics") if isinstance(result, dict) else None):
        if not isinstance(stat, dict):
            continue
        symbol = (stat.get("@classification-symbol") or "").strip()
        if not symbol:
            continue
        acc: list[str] = []
        _collect_text(stat.get("cpc:class-title"), acc)
        try:
            score = float(stat.get("@percentage") or 0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append({"symbol": symbol, "title": " ".join(a.strip() for a in acc if a.strip()), "score": score})
    return hits


def _format_cpc_search(data: dict, query: str) -> str:
    hits = _parse_cpc_search(data)
    if not hits:
        return f"No CPC groups found for {query!r}. Try different or fewer technical words."
    lines = [
        f"CPC main groups where documents mentioning {query!r} are most often classified "
        "(statistical, highest score first — candidates to verify, not answers):",
        "",
    ]
    for h in hits:
        lines.append(f"- {h['symbol']} — {_clean(h['title'])} (score {h['score']:.2f})")
    lines.append("")
    lines.append(
        "Next: look up a plausible candidate with symbol= to read its definition and narrower "
        "subgroups; discard unrelated groups."
    )
    return _wrap(lines)


# --------------------------------------------------------------------------- #
# Tools.
# --------------------------------------------------------------------------- #
class PatentEpoOpsSearchInput(ReasonBaseModel):
    keywords: str = Field(
        default="",
        description=(
            "Free-text keywords searched across title, abstract and claims. Every word "
            "must appear (any order), so 2-5 distinctive terms recall far better than a "
            "long sentence; run several searches rather than one long one."
        ),
    )
    applicant: str = Field(default="", description="Applicant / assignee name to filter by.")
    inventor: str = Field(default="", description="Inventor name to filter by.")
    cpc: list[str] = Field(
        default_factory=list,
        description=(
            f"CPC classification symbols to restrict to (up to {_MAX_CPC_CODES}; a document "
            "matching ANY of them qualifies). Levels: subclass 'A61B' (very broad), main group "
            "'A61B8' (broad), subgroup 'A61B8/06' (precise). Take codes from the CPC line of "
            "relevant hits or confirm them with patent_epoops_classification; don't guess "
            "deep subgroups from memory."
        ),
    )
    include_subgroups: bool = Field(
        default=True,
        description=(
            "Also match documents filed in the narrower groups below each cpc subgroup "
            "(A61B8/06 then also matches A61B8/065). Keep true for prior-art recall; set "
            "false to search exactly the listed groups."
        ),
    )
    date_from: str = Field(default="", description="Earliest publication date, YYYY or YYYYMMDD.")
    date_to: str = Field(default="", description="Latest publication date, YYYY or YYYYMMDD.")
    count: int = Field(default=10, description="Number of results to return (1-25, default 10).")

    @field_validator("cpc", mode="before")
    @classmethod
    def _coerce_cpc(cls, value):
        # Models often send "A61B8/06, G01S15/8984" instead of a list.
        return _split_cpc(value)


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
        "date range (all given filters must match). Returns a ranked list with publication "
        "numbers, titles, applicants, dates, each hit's CPC codes and an abstract snippet; "
        "use patent_epoops_get to read one in full.\n"
        "How to aim it:\n"
        "- Broad recall: 2-4 core keywords, optionally one main group (cpc=['A61B8']).\n"
        "- Precision: keywords + 1-3 subgroups (cpc=['A61B8/06', 'G01S15/8984']).\n"
        "- Different wording: search a relevant subgroup with no keywords, or with one broad "
        "term, to catch documents that describe the same idea in other words.\n"
        "- Iterate: the CPC line of a strong hit shows where similar documents are filed — "
        "search those classes next. 'additional' codes are secondary aspects.\n"
        "- Too many results: add a keyword, a narrower subgroup or a date range. Zero results: "
        "drop a keyword or a filter, or move up a CPC level.\n"
        "Unsure which classes fit? Use patent_epoops_classification first."
    )
    args_schema: type[BaseModel] = PatentEpoOpsSearchInput

    def _run(
        self,
        keywords: str = "",
        applicant: str = "",
        inventor: str = "",
        cpc: list[str] | str | None = None,
        include_subgroups: bool = True,
        date_from: str = "",
        date_to: str = "",
        count: int = 10,
        **kwargs,
    ) -> str:
        from django.core.cache import cache

        codes = _split_cpc(cpc)
        invalid = [c for c in codes if not _normalize_cpc(c)]
        if invalid:
            return (
                f"Patent search error: not CPC symbols: {', '.join(repr(c) for c in invalid)}. "
                "Use codes like H01M, A61B8 or A61B8/06 (one per list item)."
            )
        if len(codes) > _MAX_CPC_CODES:
            return (
                f"Patent search error: at most {_MAX_CPC_CODES} CPC codes per search "
                f"(got {len(codes)}); split them over several searches."
            )
        cql = _build_cql(
            keywords, applicant, inventor, codes, date_from, date_to,
            include_subgroups=bool(include_subgroups),
        )
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
        "Choose parts to control what is returned: biblio (title, applicants, inventors, "
        "date, CPC codes, abstract), abstract, claims, description, or all. Use "
        "patent_epoops_search first to find publication numbers. A highly relevant "
        "document's inventive CPC codes are the best classes to search next."
    )
    args_schema: type[BaseModel] = PatentEpoOpsGetInput

    def _run(self, publication_number: str = "", parts: str = "biblio", **kwargs) -> str:
        from django.core.cache import cache

        ref = _docdb_ref(publication_number)
        if ref is None:
            return "Patent retrieval error: a valid publication number is required."
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
            return "Patent family error: a valid publication number is required."
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


class PatentEpoOpsClassificationInput(ReasonBaseModel):
    query: str = Field(
        default="",
        description=(
            "Find candidate CPC groups: 2-5 technical words for ONE search concept (e.g. "
            "'ultrasound blood flow measurement'). Returns main groups only, ranked "
            "statistically. Leave empty when using symbol."
        ),
    )
    symbol: str = Field(
        default="",
        description=(
            "Explain ONE CPC symbol: subclass 'A61B', main group 'A61B8' or subgroup "
            "'A61B8/06'. Returns its title, where it sits in the scheme and the narrower "
            "groups below it. Leave empty when using query."
        ),
    )
    count: int = Field(default=10, description="query mode: number of candidate groups (1-20, default 10).")


class PatentEpoOpsClassificationTool(ContextAwareTool):
    """Look up the CPC classification scheme (EPO OPS classification services)."""

    name: str = "patent_epoops_classification"
    section: str = "skills"
    audience: str = "shared"
    start_label: str = "Looking up patent classification..."
    end_label: str = "Looked up patent classification"
    description: str = (
        "Look up the Cooperative Patent Classification (CPC) to choose the classes to "
        "search with patent_epoops_search(cpc=[...]). Searching the right classes catches "
        "documents that use different wording; the wrong ones miss documents or add noise. "
        "Two modes — pass exactly one:\n"
        "- symbol='A61B8/06': title, place in the scheme and the narrower groups directly "
        "below. Use it to confirm a code means what you think, and to drill down: look up "
        "a main group, pick the subgroup that matches the concept, look that up if it has "
        "narrower groups.\n"
        "- query='technical words': main groups where documents using those words are most "
        "often classified. Statistical and noisy — unrelated groups can rank first — so "
        "treat results as candidates and confirm with symbol= before searching.\n"
        "The most reliable source of codes is the CPC line on highly relevant hits from "
        "patent_epoops_search / patent_epoops_get; use this tool to verify and refine those, "
        "or to get started when you have no relevant hit yet. Run one lookup per search concept."
    )
    args_schema: type[BaseModel] = PatentEpoOpsClassificationInput

    def _run(self, query: str = "", symbol: str = "", count: int = 10, **kwargs) -> str:
        query = (query or "").strip()
        symbol = (symbol or "").strip()
        if bool(query) == bool(symbol):
            return "Classification lookup error: pass exactly one of query or symbol."
        if symbol:
            return self._lookup_symbol(symbol)
        return self._search(query, max(1, min(int(count or 10), 20)))

    def _cached_request(self, cache_key: str, path: str, params: dict, accept: str) -> dict:
        from django.core.cache import cache

        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            return json.loads(cached)
        data = _ops_request(path, params, tool_name=self.name, context=self.context, accept=accept)
        if "error" not in data:
            try:
                # The scheme changes a few times a year.
                cache.set(cache_key, json.dumps(data), timeout=86400)
            except Exception:
                logger.debug("epo_ops classification: cache write failed, continuing")
        return data

    def _lookup_symbol(self, raw: str) -> str:
        lookup = _cpc_lookup_symbol(raw)
        if not lookup:
            return (
                f"Classification lookup error: {raw!r} is not a CPC symbol. Use a subclass "
                "(A61B), main group (A61B8) or subgroup (A61B8/06)."
            )
        data = self._cached_request(
            "epo_ops_cpc_symbol_v1:" + lookup,
            f"classification/cpc/{lookup}",
            {"ancestors": "true", "depth": "1"},
            accept="application/cpc+xml",
        )
        if "error" in data:
            if "404" in data["error"]:
                return _format_cpc_symbol(None, lookup)
            return f"Classification lookup error: {data['error']}"
        return _format_cpc_symbol(_parse_cpc_scheme(data.get("_xml", ""), lookup), lookup)

    def _search(self, query: str, count: int) -> str:
        q = _sanitize_cql_value(query)
        if not q:
            return "Classification lookup error: query needs some technical words."
        data = self._cached_request(
            "epo_ops_cpc_search_v1:" + hashlib.sha256(f"{q}:{count}".encode()).hexdigest(),
            "classification/cpc/search",
            {"q": q, "Range": f"1-{count}"},
            accept="application/json",
        )
        if "error" in data:
            if "404" in data["error"]:
                return _format_cpc_search({}, q)
            return f"Classification lookup error: {data['error']}"
        return _format_cpc_search(data, q)


__all__ = [
    "PatentEpoOpsSearchTool",
    "PatentEpoOpsGetTool",
    "PatentEpoOpsFamilyTool",
    "PatentEpoOpsClassificationTool",
]
