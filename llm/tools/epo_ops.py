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
from typing import Literal

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


# OPS reports, with every response, how many requests per 60 s window this
# client may make per service in the current system state (Reference Guide
# §2.3.3, header ``X-Throttling-Control: idle (retrieval=green:200,
# search=yellow:20, inpadoc=red:30, images=green:200, other=green:1000)``).
# Green = <50 % of that limit used, yellow 50-75 %, red >75 %, black =
# suspended. Instances don't share counters, so it is advisory.
_THROTTLE_HEADER = "X-Throttling-Control"
_THROTTLE_RE = re.compile(r"(\w+)=(\w+):(\d+)")
# Steady rate as a share of the allowance: red starts at 75 % used, so 60 %
# keeps us yellow at worst even when two instances disagree.
_THROTTLE_HEADROOM = 0.6


def _service_for_path(path: str) -> str:
    """The OPS service a rest-services path belongs to (throttling is per service)."""
    p = (path or "").lstrip("/")
    if p.startswith("published-data/search"):
        return "search"
    if p.startswith("published-data/publication"):
        return "retrieval"
    if p.startswith("family/"):
        return "inpadoc"
    return "other"  # classification etc.


class _ServiceThrottle:
    """One token bucket per OPS service, adapted to ``X-Throttling-Control``.

    Every bucket starts at the configured ceiling (``EPO_OPS_RPM``) and is
    lowered to a share of whatever OPS says it allows right now — this is what
    keeps a busy sub-agent out of ``CLIENT.RobotDetected``.
    """

    SERVICES = ("retrieval", "search", "inpadoc", "other")

    def __init__(self, ceiling_rpm: int):
        self.ceiling_rpm = ceiling_rpm
        self._rpm = {s: ceiling_rpm for s in self.SERVICES}
        self._buckets = {
            s: _TokenBucketRateLimiter(requests_per_second=ceiling_rpm / 60.0, burst=1)
            for s in self.SERVICES
        }
        self._state = ""
        self._lock = threading.Lock()

    def acquire(self, path: str = "") -> None:
        self._buckets[_service_for_path(path)].acquire()

    def observe(self, headers) -> None:
        """Apply the throttling header of a response (any response; never raises)."""
        try:
            value = (headers or {}).get(_THROTTLE_HEADER, "") or ""
        except Exception:
            return
        pairs = _THROTTLE_RE.findall(value)
        if not pairs:
            return
        state = value.split("(", 1)[0].strip().lower()
        with self._lock:
            self._state = state
            for name, colour, limit in pairs:
                if name not in self._buckets:
                    continue
                limit = int(limit)
                suspended = colour.lower() == "black" or limit <= 0
                rpm = 1 if suspended else min(self.ceiling_rpm, max(1, int(limit * _THROTTLE_HEADROOM)))
                if rpm == self._rpm[name]:
                    continue
                if suspended:
                    # Worth a Sentry event: OPS has suspended a service for us.
                    logger.warning("EPO OPS %s service suspended (black); throttling to 1/min", name)
                else:
                    logger.info(
                        "EPO OPS throttle: %s -> %d/min (state=%s, %s, %d/min allowed)",
                        name, rpm, state, colour, limit,
                    )
                self._rpm[name] = rpm
                self._buckets[name].set_rate(rpm / 60.0)

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self._state, "rpm": dict(self._rpm)}


_ops_rate_limiter = _ServiceThrottle(_rpm())
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
# EPO's fair-use detector ("recent behaviour implies you are a robot ... try
# again later"): a 403 that is transient, not an auth problem.
_ROBOT_FAULT = "CLIENT.RobotDetected"


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
            _ops_rate_limiter.acquire(path)
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
            # Error responses carry the throttling header too.
            _ops_rate_limiter.observe(getattr(response, "headers", None))
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
            if status == 403 and code == _ROBOT_FAULT:
                # Back off like a 429; a fresh token changes nothing here.
                wait = _RATE_LIMIT_BACKOFF_SCHEDULE[min(attempt, len(_RATE_LIMIT_BACKOFF_SCHEDULE) - 1)]
                logger.info("EPO OPS robot detection (attempt %d), waiting %.1fs", attempt + 1, wait)
                if attempt < _MAX_RETRIES:
                    nap, may_retry = _deadline_capped_wait(wait, context)
                    if nap > 0:
                        time.sleep(nap)
                    if may_retry:
                        continue
                break
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

    if last_code == _ROBOT_FAULT:
        # Worth one Sentry event: being throttled in production matters.
        logger.warning("EPO OPS robot detection persisted after retries path=%s", path)
        return _done(
            {"error": (
                "EPO's fair-use detector is refusing requests right now (CLIENT.RobotDetected). "
                "Wait a few minutes before searching again — do not retry at once — and tell "
                "the user patent search is paused briefly."
            )},
            "error",
            http_status=last_status,
            error_code=last_code,
            error_message=last_message,
        )
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
# OPS accepted 3,430-character queries live (5 concepts × 10 phrases + 10 codes,
# 2026-10-06); its real ceiling is unmeasured. The per-row limits bound a sane
# query well below this — it only stops runaway input; beyond it OPS's own
# 400/414 is surfaced as a client error.
_MAX_CQL_LEN = 8000


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


def _cpc_terms(codes: list[str], include_subgroups: bool = True) -> list[str]:
    """``cpc=…`` terms for (already valid) CPC symbols, de-duplicated.

    ``/low`` widens a subgroup to everything filed beneath it (A61B8/06 also
    matches A61B8/065); main groups and subclasses already include theirs.
    OPS rejects ``/low`` inside ``cpc any "…"`` (400
    CLIENT.InvalidClassificationRelation), so callers OR the terms instead.
    """
    terms: list[str] = []
    for code in codes:
        sym = _normalize_cpc(code)
        if not sym:
            continue
        if include_subgroups and "/" in sym:
            sym += "/low"
        term = f"cpc={sym}"
        if term not in terms:
            terms.append(term)
    return terms


def _or_group(terms: list[str]) -> str:
    if not terms:
        return ""
    return terms[0] if len(terms) == 1 else "(" + " or ".join(terms) + ")"


def _cpc_clause(codes: list[str], include_subgroups: bool = True) -> str:
    """CQL for one or more CPC symbols, OR-ed: ``(cpc=X or cpc=Y)``."""
    return _or_group(_cpc_terms(codes, include_subgroups))


# Keyword fields: the paper's examiner searches use title/abstract; ``txt``
# also reaches full text where OPS has it (txt=hydrochlor?thiazid* 14,127 hits
# vs ta= 557, probed 2026-10-05) plus applicant/inventor names.
# Verified live (2026-10-06): ti / ab / ta / txt exist; "claims" and "desc" are
# not OPS indexes (claims-only search is impossible; full text is the closest).
_KEYWORD_FIELDS = {"title": "ti", "abstract": "ab", "title_abstract": "ta", "full_text": "txt"}
_MAX_TERMS_PER_ROW = 10
_MAX_PHRASE_WORDS = 4
_TERM_STRIP_RE = re.compile(r'["()=/\\<>]+')
_WILDCARD_RE = re.compile(r"[*?#]")


def _keyword_field(value: str) -> str:
    return _KEYWORD_FIELDS.get((value or "").strip().lower(), "ta")


def _check_leading_truncation(word: str, field: str, whole: str) -> None:
    """OPS rule: truncation at the start of a word only works in the title and
    abstract indices, so ``*thiazide`` is fine in ti/ab/ta but a 400 in txt."""
    if field == "txt" and word and word[0] in "*?#":
        raise ValueError(
            f"{whole!r}: leading truncation ({word!r}) only works in title/abstract fields "
            "(an EPO rule): use field title_abstract, title or abstract, or drop the leading "
            "wildcard."
        )


def _keyword_term(raw: str, field: str = "ta") -> str:
    """One synonym → a quoted CQL term (``"bilayer*"``, ``"bi layer*"``).

    Multi-word terms are exact phrases. Truncation ``*`` (any string), ``?``
    (zero or one char) and ``#`` (exactly one char) pass through; OPS needs at
    least 3 real characters in a word that uses ``*`` (``hy*`` → 400
    CLIENT.PrefixTooShort). Raises ValueError with a message for the model.
    """
    val = re.sub(r"\s+", " ", _TERM_STRIP_RE.sub(" ", raw or "")).strip()
    if not val:
        return ""
    words = val.split(" ")
    if len(words) > _MAX_PHRASE_WORDS:
        raise ValueError(
            f"{raw!r} is too long: a keyword is a word or an exact phrase of at most "
            f"{_MAX_PHRASE_WORDS} words. Put alternative wordings in separate keywords and "
            "separate ideas in separate concepts."
        )
    for word in words:
        if "*" in word and len(_WILDCARD_RE.sub("", word)) < 3:
            raise ValueError(f"{raw!r}: '*' needs at least 3 letters in the word (e.g. 'hyd*').")
        _check_leading_truncation(word, field, raw)
    return f'"{val}"'


# ``A NEAR/3 B`` (within 3 words), ``A NEAR B`` / ``A NEAR/S B`` (same sentence),
# ``A NEAR/P B`` (same paragraph). Uppercase only: a lowercase "near" inside a
# phrase ("antenna near field") must stay an ordinary phrase.
_NEAR_RE = re.compile(r"^(\S+)\s+NEAR(?:/(\d+|S|P))?\s+(\S+)$")


def _proximity_word(raw: str, whole: str, field: str = "ta") -> str:
    """One side of a NEAR expression: a single sanitised word (truncation allowed)."""
    word = re.sub(r"\s+", " ", _TERM_STRIP_RE.sub(" ", raw)).strip()
    if not word or " " in word:
        raise ValueError(f"{whole!r}: each side of NEAR is one word (truncation allowed), not a phrase.")
    if "*" in word and len(_WILDCARD_RE.sub("", word)) < 3:
        raise ValueError(f"{whole!r}: '*' needs at least 3 letters in the word (e.g. 'hyd*').")
    _check_leading_truncation(word, field, whole)
    return word


def _proximity_term(raw: str, field: str) -> str | None:
    """A NEAR keyword as a CQL proximity clause, or None if ``raw`` isn't one.

    ``zero NEAR/3 order`` → ``(ta=zero prox/distance<=3 ta=order)``; ``NEAR`` /
    ``NEAR/S`` → ``prox/unit=sentence``; ``NEAR/P`` → ``prox/unit=paragraph``.
    The unquoted form is the one verified live (2026-10-06).
    """
    m = _NEAR_RE.match((raw or "").strip())
    if not m:
        if re.search(r"(^|\s)NEAR(/\S*)?(\s|$)", raw or ""):
            raise ValueError(
                f"{raw!r}: NEAR takes one word on each side, e.g. 'zero NEAR/3 order' "
                "(a phrase side is not possible; use a phrase keyword instead)."
            )
        return None
    left = _proximity_word(m.group(1), raw, field)
    right = _proximity_word(m.group(3), raw, field)
    qual = (m.group(2) or "S").upper()
    if qual == "P":
        op = "prox/unit=paragraph"
    elif qual == "S":
        op = "prox/unit=sentence"
    else:
        if int(qual) < 1:
            raise ValueError(f"{raw!r}: NEAR/N needs N of at least 1.")
        op = f"prox/distance<={int(qual)}"
    return f"({field}={left} {op} {field}={right})"


def _concept_clause(concept: dict, default_field: str, include_subgroups: bool = True) -> str:
    """One search-table column: its keywords and CPC codes, all OR-ed.

    The concept's own ``field`` (title / abstract / title_abstract / full_text)
    overrides the search-wide default.
    """
    field = _keyword_field(concept["field"]) if concept.get("field") else default_field
    terms: list[str] = []
    for kw in (concept.get("keywords") or [])[:_MAX_TERMS_PER_ROW]:
        term = _proximity_term(kw, field)
        if term is None:
            quoted = _keyword_term(kw, field)
            if not quoted:
                continue
            term = f"{field}={quoted}"
        if term not in terms:
            terms.append(term)
    terms.extend(
        t for t in _cpc_terms(_split_cpc(concept.get("cpc"))[:_MAX_TERMS_PER_ROW], include_subgroups)
        if t not in terms
    )
    return _or_group(terms)


_WO_OLD_RE = re.compile(r"^WO(\d{4})(\d{6})$")


def _wo_short_form(number: str) -> str:
    """WO numbers up to 2003 are known to OPS in the short form only:
    ``WO2003059327`` → ``WO03059327`` (the long form 404s); 2004+ stay long."""
    m = _WO_OLD_RE.match(number)
    if m and int(m.group(1)) <= 2003:
        return f"WO{m.group(1)[2:]}{m.group(2)}"
    return number


def _citation_number(raw: str) -> str:
    """Publication number for ``ct=`` (kind code dropped), or ""."""
    n = _normalize_pubnumber(raw)
    if not n:
        return ""
    n = re.sub(r"^([A-Z]{2}\d+)[A-Z]\d?$", r"\1", n)
    return _wo_short_form(n)


def _build_cql(
    keywords: str = "",
    applicant: str = "",
    inventor: str = "",
    cpc: str | list[str] = "",
    date_from: str = "",
    date_to: str = "",
    include_subgroups: bool = True,
    concepts: list[dict] | None = None,
    keyword_field: str = "title_abstract",
    cites: str = "",
    publication: str = "",
    exclude: dict | None = None,
) -> str:
    """Build an OPS CQL query string from structured inputs.

    Field codes: ti / ab / ta / txt for keywords (title, abstract, both, full
    text — per concept or the search default), pa (applicant), in (inventor),
    cpc (CPC), pd (publication date), ct (cites), pn (publication). Everything
    is ANDed; inside one concept everything is ORed; ``exclude`` is NOT-ed off
    the whole query. Verified live against OPS 3.2 (2026-10-05/06):

    - ``concepts`` are search-table columns: ``(ta="a" or ta="b*" or
      cpc=X/low)``; the paper's strategies A-D reproduce exactly this way.
    - ``A NEAR/3 B`` keywords become ``(ta=A prox/distance<=3 ta=B)``.
    - ``x not y`` is OPS's NOT (``and not`` is a syntax error).
    - Legacy ``keywords`` mean "all of these words": ``ta all "a b c"``
      (``ta="a b c"`` would be an exact phrase and usually finds nothing).
    - CPC is emitted unquoted with its slash (``cpc=A61B8/06``); invalid
      symbols are dropped (the tool rejects them before calling OPS).
    - A one-sided date range is ``pd>=`` / ``pd<=``. Padding it with a sentinel
      year (``pd within "20000101 30001231"``) makes OPS 500.

    Raises ValueError (message meant for the model) for an invalid keyword, an
    exclude with nothing to exclude from, or a query over ``_MAX_CQL_LEN`` — a
    truncated structured query is invalid CQL.
    """
    field = _keyword_field(keyword_field)
    clauses: list[str] = []
    for concept in concepts or []:
        clause = _concept_clause(concept, field, include_subgroups)
        if clause:
            clauses.append(clause)
    kw = _sanitize_cql_value(keywords or "")
    if kw:
        for word in kw.split(" "):
            _check_leading_truncation(word, field, keywords)
        clauses.append(f'{field} all "{kw}"' if " " in kw else f'{field}="{kw}"')
    for fld, raw in (("pa", applicant), ("in", inventor)):
        val = _sanitize_cql_value(raw or "")
        if val:
            clauses.append(f'{fld}="{val}"')
    cpc_clause = _cpc_clause(_split_cpc(cpc)[:_MAX_CPC_CODES], include_subgroups)
    if cpc_clause:
        clauses.append(cpc_clause)
    cited = _citation_number(cites)
    if cited:
        clauses.append(f"ct={cited}")
    # pn= matches the family result that contains this publication, whichever
    # member OPS shows for it (probed: pn=WO03059327 AND the case-study union → 1).
    pub = _citation_number(publication)
    if pub:
        clauses.append(f"pn={pub}")

    df = _sanitize_date(date_from, is_end=False)
    dt = _sanitize_date(date_to, is_end=True)
    if df and dt:
        clauses.append(f'pd within "{df} {dt}"')
    elif df:
        clauses.append(f"pd>={df}")
    elif dt:
        clauses.append(f"pd<={dt}")

    cql = " and ".join(clauses)
    excluded = _concept_clause(exclude, field, include_subgroups) if exclude else ""
    if excluded:
        if not cql:
            raise ValueError(
                "exclude needs something to exclude from: add at least one concept or filter."
            )
        positive = f"({cql})" if len(clauses) > 1 else cql
        cql = f"{positive} not {excluded}"
    if len(cql) > _MAX_CQL_LEN:
        raise ValueError(
            f"the query is too long ({len(cql)} characters, max {_MAX_CQL_LEN}); use fewer "
            "keywords or codes, or split it over several searches."
        )
    return cql


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
        if country == "WO":
            number = _wo_short_form(f"WO{number}")[2:]
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


def _cited_references(biblio: dict) -> list[dict]:
    """Backward citations from ``references-cited``.

    Each: ``{"number", "npl", "by", "phase", "category", "claims"}``. Examiner
    citations from a search report carry the category (X: relevant alone,
    Y: relevant in combination, A: background) and the claims they concern;
    ``npl`` holds the text of a non-patent citation instead of a number.
    """
    refs: list[dict] = []
    group = biblio.get("references-cited", {}) or {}
    for cit in _as_list(group.get("citation") if isinstance(group, dict) else None):
        if not isinstance(cit, dict):
            continue
        number = npl = ""
        patcit = cit.get("patcit")
        if isinstance(patcit, dict):
            number = _pubnumber_from_doc_ids(patcit.get("document-id"))
        elif isinstance(cit.get("nplcit"), dict):
            npl = _text(cit["nplcit"].get("text")).strip().lstrip("- ").strip()
        if not number and not npl:
            continue
        refs.append({
            "number": number,
            "npl": npl,
            "by": (cit.get("@cited-by") or "").strip(),
            "phase": (cit.get("@cited-phase") or "").strip(),
            "category": "/".join(_text(c).strip() for c in _as_list(cit.get("category")) if _text(c).strip()),
            "claims": ", ".join(_text(c).strip() for c in _as_list(cit.get("rel-claims")) if _text(c).strip()),
        })
    return refs


def _parse_exchange_document(doc: dict) -> dict:
    biblio = doc.get("bibliographic-data", {}) or {}
    pubnum = _pubnumber_from_attrs(doc)
    if not pubnum:
        ref = biblio.get("publication-reference", {}) or {}
        pubnum = _pubnumber_from_doc_ids(ref.get("document-id"))
    return {
        "publication_number": pubnum,
        "family_id": str(doc.get("@family-id") or "").strip(),
        "title": _first_title(biblio),
        "applicants": _party_names(biblio, "applicants", "applicant"),
        "inventors": _party_names(biblio, "inventors", "inventor"),
        "date": _publication_date(biblio),
        "abstract": _abstract_text(doc),
        "cpc": _cpc_codes(biblio),
        "citations": _cited_references(biblio),
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


_MAX_POSITION = 2000  # OPS serves search positions 1..2000 only
_VIEW_LIMITS = {"list": 100, "abstracts": 50}
_REPRESENTATIVE_PREFIXES = ("EP", "WO")


def _group_families(results: list[dict]) -> list[dict]:
    """Collapse a page's publications into families (OPS ``@family-id``).

    Search hits are publications, so one invention shows up once per family
    member (US, EP, CN, …). Each group: ``{"family_id", "rep", "others"}`` —
    the representative is the first EP, else WO, else first member (EP/WO
    usually have English abstracts and full text); ``others`` are the rest's
    numbers. Order follows the first member's position.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = r.get("family_id") or f"pub:{r.get('publication_number')}"
        groups.setdefault(key, []).append(r)
    out: list[dict] = []
    for key, members in groups.items():
        rep = next(
            (m for p in _REPRESENTATIVE_PREFIXES for m in members
             if (m.get("publication_number") or "").startswith(p)),
            members[0],
        )
        out.append({
            "family_id": "" if key.startswith("pub:") else key,
            "rep": rep,
            "others": [m["publication_number"] for m in members if m is not rep and m.get("publication_number")],
        })
    return out


def _search_header(total: int, offset: int, shown_pubs: int, families: int, cql: str, hidden: int) -> list[str]:
    lines: list[str] = []
    if shown_pubs:
        lines.append(
            f"{total} results — one per patent family, shown under one member's number (maybe not "
            f"the number you know). Positions {offset + 1}-{offset + shown_pubs}. Ordered by family, "
            "newest families first — NOT by relevance."
        )
    end = offset + shown_pubs
    if end < min(total, _MAX_POSITION):
        lines.append(f"More results: repeat the search with offset={end}.")
    if total > _MAX_POSITION:
        lines.append(
            f"Only the first {_MAX_POSITION} positions can ever be listed — narrow the search (another "
            "concept, a narrower CPC subgroup, or a date range) to see the rest."
        )
    if hidden:
        lines.append(f"{hidden} famil{'y' if hidden == 1 else 'ies'} already seen in this conversation hidden (hide_seen).")
    lines.append(f"Query: {cql}")
    lines.append("")
    return lines


def _format_search_groups(
    groups: list[dict],
    *,
    total: int,
    offset: int = 0,
    shown_pubs: int = 0,
    cql: str = "",
    view: str = "abstracts",
    seen: set | frozenset = frozenset(),
    hidden: int = 0,
) -> str:
    lines = _search_header(total, offset, shown_pubs, len(groups) + hidden, cql, hidden)
    if not groups:
        lines.append("Every family on this page was already seen; continue with the next offset.")
        return _wrap(lines)
    for i, g in enumerate(groups, 1):
        r = g["rep"]
        num = r["publication_number"] or "(no number)"
        mark = " [seen]" if g["family_id"] and g["family_id"] in seen else ""
        family = f"family: {', '.join(g['others'])}" if g["others"] else ""
        if view == "list":
            applicant = _clean(r["applicants"][0]) if r["applicants"] else ""
            cpc = _cpc_line({"inventive": (r.get("cpc") or {}).get("inventive") or []}, max_inventive=4)
            parts = [f"[{i}] {num} ({r['date'] or '?'}) {_clean(r['title']) or '(no title)'}{mark}"]
            parts += [p for p in (applicant, cpc) if p]
            lines.append(" — ".join(parts))
            if family:
                lines.append(f"    {family}")
            continue
        lines.append(f"**[{i}] {_clean(r['title']) or '(no title)'}** — {num}{mark}")
        url = _espacenet_url(r["publication_number"])
        if url:
            lines.append(f"Espacenet: {url}")
        if r["applicants"]:
            lines.append(f"Applicant(s): {_clean(', '.join(r['applicants']))}")
        if r["date"]:
            lines.append(f"Published: {r['date']}")
        if family:
            lines.append(family[0].upper() + family[1:])
        cpc_line = _cpc_line(r.get("cpc"), max_inventive=6, max_additional=4)
        if cpc_line:
            lines.append(cpc_line)
        abstract = _clean(r["abstract"])
        if abstract:
            lines.append(abstract[:400] + ("…" if len(abstract) > 400 else ""))
        lines.append("")
    return _wrap(lines)


def _total_count(data: dict) -> int:
    root = _unwrap(data)
    search = root.get("ops:biblio-search", {}) or {}
    try:
        return int(search.get("@total-result-count") or 0)
    except (TypeError, ValueError):
        return 0


def _format_search(data: dict, *, cql: str = "", offset: int = 0, view: str = "abstracts") -> str:
    """Format one search page without seen-tracking (see the tool for that)."""
    if "error" in data:
        return f"Patent search error: {data['error']}"
    parsed = _parse_search_results(data)
    if not parsed["results"]:
        return "No matching patents found."
    groups = _group_families(parsed["results"])
    return _format_search_groups(
        groups, total=_total_count(data), offset=offset, shown_pubs=parsed["count"], cql=cql, view=view
    )


# --------------------------------------------------------------------------- #
# "Already seen" families, per conversation (shared by its sub-agents).
# --------------------------------------------------------------------------- #
_SEEN_TTL = 14 * 86400


def _seen_scope(context) -> str:
    if context is None:
        return ""
    return str(getattr(context, "conversation_id", "") or getattr(context, "run_id", "") or "")


def _seen_key(scope: str, family_id: str) -> str:
    return f"epo_seen_v1:{scope}:{family_id}"


def _seen_lookup(scope: str, family_ids: list[str]) -> set[str]:
    """Family ids already shown in this conversation. Best-effort: {} on error."""
    from django.core.cache import cache

    ids = [f for f in family_ids if f]
    if not scope or not ids:
        return set()
    try:
        found = cache.get_many([_seen_key(scope, f) for f in ids])
    except Exception:
        logger.debug("epo_ops: seen lookup failed, continuing")
        return set()
    return {f for f in ids if _seen_key(scope, f) in found}


def _seen_record(scope: str, family_ids: list[str]) -> None:
    from django.core.cache import cache

    ids = [f for f in family_ids if f]
    if not scope or not ids:
        return
    try:
        cache.set_many({_seen_key(scope, f): 1 for f in ids}, timeout=_SEEN_TTL)
    except Exception:
        logger.debug("epo_ops: seen record failed, continuing")


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


_MAX_CITED_SHOWN = 30
_MAX_NPL_SHOWN = 5


def _citation_lines(refs: list[dict]) -> list[str]:
    """``Cited references`` block: examiner citations first (with category and
    claims), then applicant/other; patents capped, non-patent literature last."""
    if not refs:
        return []
    patents = [r for r in refs if r["number"]]
    npl = [r for r in refs if r["npl"]]
    patents.sort(key=lambda r: 0 if r["by"] == "examiner" else 1)  # stable: keeps OPS order within
    lines = ["Cited references (X = relevant on its own, Y = relevant combined with another, A = background):"]
    seen: set[str] = set()
    shown = 0
    for r in patents:
        if r["number"] in seen:
            continue
        seen.add(r["number"])
        if shown >= _MAX_CITED_SHOWN:
            continue
        shown += 1
        detail = [r["by"] or "unknown"]
        if r["phase"] and r["phase"] != "undefined":
            detail.append(r["phase"].replace("-", " "))
        if r["category"]:
            detail.append(f"category {r['category']}")
        if r["claims"]:
            detail.append(f"claims {r['claims']}")
        lines.append(f"- {r['number']} ({', '.join(detail)})")
    if len(seen) > shown:
        lines.append(f"- … +{len(seen) - shown} more cited patents")
    if npl:
        lines.append(f"Non-patent literature cited: {len(npl)}")
        for r in npl[:_MAX_NPL_SHOWN]:
            cat = f" [category {r['category']}]" if r["category"] else ""
            lines.append(f"- {_clean(r['npl'])[:200]}{cat}")
    return lines


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
        cited = _citation_lines(r.get("citations") or [])
        if cited:
            lines.append("")
            lines.extend(cited)
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
class SearchConcept(BaseModel):
    """One search-table column: alternative keywords and CPC codes for one idea."""

    name: str = Field(default="", description="Short label for the concept, e.g. 'bilayer tablet'.")
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            f"Up to {_MAX_TERMS_PER_ROW} alternative wordings (synonyms, spellings) — a document "
            "matching ANY of them qualifies. Each is one word or an exact phrase of at most "
            f"{_MAX_PHRASE_WORDS} words. Truncation: * any ending (needs 3+ letters: 'bilayer*'), "
            "? zero or one character ('hydrochlor?thiazid*'), # exactly one character; a leading "
            "wildcard ('*thiazide') works in title/abstract fields only, not full_text. "
            "Proximity: 'zero NEAR/3 order' (two words within 3 words of each other, any order), "
            "'zero NEAR order' (same sentence), 'zero NEAR/P order' (same paragraph) — NEAR in "
            "capitals, one word on each side."
        ),
    )
    cpc: list[str] = Field(
        default_factory=list,
        description=(
            f"Up to {_MAX_TERMS_PER_ROW} CPC symbols for this concept (OR-ed with each other and "
            "with the keywords), e.g. 'A61K9/209'."
        ),
    )
    field: Literal["title", "abstract", "title_abstract", "full_text"] | None = Field(
        default=None,
        description=(
            "Where THIS concept's keywords must occur, overriding the search's keyword_field: "
            "'title' (strictest), 'abstract', 'title_abstract', 'full_text'. Leave unset to "
            "use the search default."
        ),
    )

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in re.split(r"[,;]|\s+\bor\b\s+", value, flags=re.IGNORECASE) if v.strip()]
        return value

    @field_validator("cpc", mode="before")
    @classmethod
    def _coerce_cpc(cls, value):
        return _split_cpc(value)


class PatentEpoOpsSearchInput(ReasonBaseModel):
    concepts: list[SearchConcept] = Field(
        default_factory=list,
        description=(
            "The search as search-table columns (typically 2-5; more is rarely useful). Inside "
            "a concept, all keywords and codes are alternatives (OR); a document must match "
            "EVERY concept (AND). Give a concept only its CPC codes, only its keywords, or both."
        ),
    )
    keyword_field: Literal["title", "abstract", "title_abstract", "full_text"] = Field(
        default="title_abstract",
        description=(
            "Default field for keywords (a concept can override it): 'title_abstract' (default, "
            "precise), 'title' (strictest), 'abstract', or 'full_text' (title, abstract, "
            "claims/description where EPO has them — mostly EP/WO — and names; much broader and "
            "noisier, useful when title/abstract searches miss documents). There is no "
            "claims-only field in EPO's service."
        ),
    )
    exclude: SearchConcept | None = Field(
        default=None,
        description=(
            "Documents matching ANY of these keywords/codes are removed from the result (NOT). "
            "Use sparingly and only after seeing what a noisy class or term brings in — it "
            "silently drops documents that also match everything else."
        ),
    )
    keywords: str = Field(
        default="",
        description=(
            "Simple alternative to concepts: words that must ALL appear (any order). Prefer "
            "concepts for structured searches."
        ),
    )
    applicant: str = Field(default="", description="Applicant / assignee name to filter by.")
    inventor: str = Field(default="", description="Inventor name to filter by.")
    cpc: list[str] = Field(
        default_factory=list,
        description=(
            f"Simple alternative to concepts: CPC symbols (up to {_MAX_CPC_CODES}), a document "
            "matching ANY of them qualifies. Levels: subclass 'A61B' (very broad), main group "
            "'A61B8' (broad), subgroup 'A61B8/06' (precise)."
        ),
    )
    include_subgroups: bool = Field(
        default=True,
        description=(
            "Also match documents filed in the narrower groups below each CPC subgroup "
            "(A61B8/06 then also matches A61B8/065). Keep true for recall; false searches "
            "exactly the listed groups."
        ),
    )
    cites: str = Field(
        default="",
        description=(
            "Only documents that cite this publication (forward citations), e.g. 'WO03059327'. "
            "Combine with concepts to find later documents building on a close hit."
        ),
    )
    publication: str = Field(
        default="",
        description=(
            "Restrict to the family of this publication number, e.g. 'WO03059327'. With "
            "count_only=true and your other filters it answers 'is this document in my result "
            "set?' (1 = yes, 0 = no) — needed because each result row shows only one family "
            "member's number."
        ),
    )
    date_from: str = Field(default="", description="Earliest publication date, YYYY or YYYYMMDD.")
    date_to: str = Field(
        default="",
        description="Latest publication date, YYYY or YYYYMMDD (e.g. the day before a priority date).",
    )
    count_only: bool = Field(
        default=False,
        description=(
            "Only return how many results (patent families) match — cheap. Use it to size a "
            "search before listing it, and to record what each strategy contributes."
        ),
    )
    view: Literal["abstracts", "list"] = Field(
        default="abstracts",
        description=(
            "'abstracts' (default, up to 50 per page): title, applicants, date, CPC and a short "
            "abstract. 'list' (up to 100 per page): one line per family — number, date, title, "
            "applicant, CPC — for screening larger sets quickly."
        ),
    )
    count: int = Field(default=25, description="Results per page (abstracts: 1-50, list: 1-100; default 25).")
    offset: int = Field(
        default=0,
        description="Skip this many results (paging): offset=0 is the first page, then use the offset the result suggests.",
    )
    hide_seen: bool = Field(
        default=False,
        description="Leave out families already shown by an earlier search in this conversation (any agent).",
    )

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
        "Search EPO/Espacenet (worldwide published patents) with a search table. Pass "
        "`concepts` — one per column: keywords (synonyms, truncation) and/or CPC codes. Inside "
        "a concept everything is OR-ed; concepts are AND-ed with each other and with "
        "applicant/inventor/dates/cites.\n"
        "Classic strategies for two concepts C1, C2:\n"
        "- codes only: C1 {cpc} AND C2 {cpc}\n"
        "- mixed: C1 {cpc} AND C2 {keywords}, and C1 {keywords} AND C2 {cpc}\n"
        "- keywords only: C1 {keywords} AND C2 {keywords}\n"
        "- union of all four in one search: give each concept both its keywords and its codes.\n"
        "Run the separate strategies with count_only=true to see what each contributes; list "
        "the union to screen the documents.\n"
        "Keywords match in title+abstract by default; set keyword_field, or a concept's own "
        "field, to title / abstract / full_text (no claims-only field exists). Two words that "
        "must sit together: a phrase ('zero order') or NEAR ('zero NEAR/3 order'). Two words "
        "that must both appear anywhere: two concepts. exclude= removes documents (NOT) — "
        "rarely needed.\n"
        "Each result is one patent FAMILY, shown under one member's number — which may differ "
        "from the number you know (the paper's WO03059327 appears as EP1467712A1). To check "
        "whether a known document is in a result set, rerun it with count_only=true and "
        "publication='<number>' (1 = yes).\n"
        "Results are ordered newest family first, NOT by relevance or publication date, so a "
        "large result set is not 'best first': size it with count_only, narrow it (add a "
        "concept, a narrower subgroup, date_to) until it can be read in full, then page with "
        "offset. [seen] marks families shown earlier in this conversation; hide_seen=true "
        "leaves them out.\n"
        "Each hit shows its CPC codes: codes recurring on relevant hits are classes worth "
        "searching. Use patent_epoops_get for a full record and its cited references, "
        "patent_epoops_classification to check a CPC code, and cites= for later documents "
        "citing a close hit."
    )
    args_schema: type[BaseModel] = PatentEpoOpsSearchInput

    def _run(
        self,
        concepts: list | None = None,
        keyword_field: str = "title_abstract",
        keywords: str = "",
        applicant: str = "",
        inventor: str = "",
        cpc: list[str] | str | None = None,
        include_subgroups: bool = True,
        cites: str = "",
        publication: str = "",
        exclude=None,
        date_from: str = "",
        date_to: str = "",
        count_only: bool = False,
        view: str = "abstracts",
        count: int = 25,
        offset: int = 0,
        hide_seen: bool = False,
        **kwargs,
    ) -> str:
        def _as_dict(c):
            return c.model_dump() if hasattr(c, "model_dump") else dict(c or {})

        concept_dicts = [_as_dict(c) for c in (concepts or [])]
        exclude_dict = _as_dict(exclude) if exclude else None
        if exclude_dict and not (exclude_dict.get("keywords") or exclude_dict.get("cpc")):
            exclude_dict = None
        error = _validate_search_input(
            concept_dicts, _split_cpc(cpc), cites, publication, exclude=exclude_dict,
        )
        if error:
            return f"Patent search error: {error}"
        try:
            cql = _build_cql(
                keywords, applicant, inventor, _split_cpc(cpc), date_from, date_to,
                include_subgroups=bool(include_subgroups),
                concepts=concept_dicts,
                keyword_field=keyword_field,
                cites=cites,
                publication=publication,
                exclude=exclude_dict,
            )
        except ValueError as e:
            return f"Patent search error: {e}"
        if not cql:
            return (
                "Patent search error: provide at least one concept, or keywords, applicant, "
                "inventor, cpc or cites."
            )
        if count_only:
            return self._count(cql)

        view = view if view in _VIEW_LIMITS else "abstracts"
        count = max(1, min(int(count or 25), _VIEW_LIMITS[view]))
        offset = max(0, int(offset or 0))
        if offset >= _MAX_POSITION:
            return (
                f"Patent search error: OPS only lists the first {_MAX_POSITION} positions; "
                "narrow the search instead of paging further."
            )
        end = min(offset + count, _MAX_POSITION)

        page = self._fetch_page(cql, offset, end)
        if "error" in page:
            if "404" in page["error"]:
                return f"No matching patents found{' at this offset' if offset else ''}. Query: {cql}"
            return f"Patent search error: {page['error']} Query: {cql}"
        results = page["results"]
        if not results:
            return f"No matching patents found{' at this offset' if offset else ''}. Query: {cql}"

        groups = _group_families(results)
        scope = _seen_scope(self.context)
        family_ids = [g["family_id"] for g in groups]
        seen = _seen_lookup(scope, family_ids)
        hidden = 0
        if hide_seen:
            kept = [g for g in groups if not (g["family_id"] and g["family_id"] in seen)]
            hidden = len(groups) - len(kept)
            groups = kept
        _seen_record(scope, family_ids)
        return _format_search_groups(
            groups,
            total=page["total"],
            offset=offset,
            shown_pubs=len(results),
            cql=cql,
            view=view,
            seen=seen,
            hidden=hidden,
        )

    def _fetch_page(self, cql: str, offset: int, end: int) -> dict:
        """One OPS search page as slim parsed results (cached 15 min).

        A 100-hit biblio page is ~1 MB of JSON; only the fields the formatter
        uses are kept (abstract cut to 400 chars), so the cache entry stays small.
        """
        from django.core.cache import cache

        cache_key = "epo_ops_search_v2:" + hashlib.sha256(f"{cql}:{offset}:{end}".encode()).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            return json.loads(cached)

        data = _ops_request(
            "published-data/search/biblio",
            {"q": cql, "Range": f"{offset + 1}-{end}"},
            tool_name=self.name,
            context=self.context,
        )
        if "error" in data:
            return data
        parsed = _parse_search_results(data)
        page = {
            "total": _total_count(data),
            "results": [
                {
                    "publication_number": r["publication_number"],
                    "family_id": r["family_id"],
                    "title": r["title"],
                    "applicants": r["applicants"][:3],
                    "date": r["date"],
                    "abstract": (r["abstract"] or "")[:420],
                    "cpc": r["cpc"],
                }
                for r in parsed["results"]
            ],
        }
        try:
            cache.set(cache_key, json.dumps(page), timeout=900)
        except Exception:
            logger.debug("epo_ops search: cache write failed, continuing")
        return page

    def _count(self, cql: str) -> str:
        from django.core.cache import cache

        cache_key = "epo_ops_count_v1:" + hashlib.sha256(cql.encode()).hexdigest()
        try:
            total = cache.get(cache_key)
        except Exception:
            total = None
        if total is None:
            data = _ops_request(
                "published-data/search",
                {"q": cql, "Range": "1-1"},
                tool_name=self.name,
                context=self.context,
            )
            if "error" in data:
                if "404" not in data["error"]:
                    return f"Patent search error: {data['error']} Query: {cql}"
                total = 0
            else:
                total = _total_count(data)
            try:
                cache.set(cache_key, total, timeout=900)
            except Exception:
                logger.debug("epo_ops count: cache write failed, continuing")
        return f"{total} results (patent families) match. Query: {cql}"


def _validate_search_input(
    concepts: list[dict], legacy_cpc: list[str], cites: str, publication: str = "",
    exclude: dict | None = None,
) -> str:
    """Error message for invalid structured input, or "" — checked before OPS is called."""
    if len(legacy_cpc) > _MAX_CPC_CODES:
        return (
            f"at most {_MAX_CPC_CODES} CPC codes per search (got {len(legacy_cpc)}); "
            "split them over several searches."
        )
    invalid = [c for c in legacy_cpc if not _normalize_cpc(c)]
    rows = [(c.get("name") or f"concept {i}", c) for i, c in enumerate(concepts, 1)]
    if exclude:
        rows.append(("exclude", exclude))
    for label, concept in rows:
        kws = concept.get("keywords") or []
        codes = _split_cpc(concept.get("cpc"))
        if not kws and not codes:
            return f"{label!r} has neither keywords nor CPC codes."
        if len(kws) > _MAX_TERMS_PER_ROW or len(codes) > _MAX_TERMS_PER_ROW:
            return f"{label!r}: at most {_MAX_TERMS_PER_ROW} keywords and {_MAX_TERMS_PER_ROW} CPC codes."
        invalid += [c for c in codes if not _normalize_cpc(c)]
    if invalid:
        return (
            f"not CPC symbols: {', '.join(repr(c) for c in invalid)}. "
            "Use codes like H01M, A61B8 or A61B8/06 (one per list item)."
        )
    if (cites or "").strip() and not _citation_number(cites):
        return f"{cites!r} is not a publication number (cites)."
    if (publication or "").strip() and not _citation_number(publication):
        return f"{publication!r} is not a publication number (publication)."
    return ""


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
        "date, CPC codes, abstract and cited references), abstract, claims, description, or "
        "all. Claims/description exist mainly for EP and WO documents — for others, find an "
        "EP/WO member with patent_epoops_family. Cited references list what the examiner "
        "(category X: relevant alone, Y: in combination, A: background) and the applicant "
        "cited; for a close document these are prime prior-art candidates, and "
        "patent_epoops_search(cites=...) finds later documents citing it. A highly relevant "
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
        elif parts in ("claims", "description") and "404" in data["error"]:
            # Seen live: EP2252273A1 claims → 404 while its WO member had them.
            return (
                f"No {parts} text at EPO for {display}. EPO holds full text mainly for EP and WO "
                "publications, and an EP application that entered from a PCT application has its "
                "text under the WO number — use patent_epoops_family to find the WO (or another "
                "EP/WO) member and request its claims."
            )
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
