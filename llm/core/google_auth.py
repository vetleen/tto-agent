"""Google Cloud / Vertex AI credentials for Gemini.

Wilfred talks to Gemini through one of two backends:

* **Gemini Developer API / AI Studio** — authenticated by ``GEMINI_API_KEY`` (or
  ``GOOGLE_API_KEY``), picked up from the environment by the SDK. This is the
  local-dev default.
* **Vertex AI** (Google Cloud) — authenticated by a service account, keeping ML
  processing inside the configured region (``eu``). This is the staging/prod path
  (and, when set up locally, dev too).

The backend is selected purely by the **presence of** ``GOOGLE_SERVICE_ACCOUNT_JSON``:
when it is set, :func:`get_vertex_config` returns the kwargs that steer both the chat
factory (``llm/core/model_factory.py``) and the image-generation service
(``llm/service/image_generation_service.py``) onto Vertex. When it is absent, callers
fall back to the API-key path unchanged.

``GOOGLE_SERVICE_ACCOUNT_JSON`` carries the credential in one of two shapes, and this
module accepts both: a **filesystem path** to the key file (convenient locally) or the
**key JSON inline** (required on Heroku, whose config vars can't point at a file on the
ephemeral dyno filesystem).
"""

from __future__ import annotations

import json
import logging
import os
import threading

from llm.service.errors import LLMConfigurationError

logger = logging.getLogger(__name__)

# Vertex needs a scoped credential; this is the standard scope for GCP APIs.
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Vertex location. We use "global" because the Gemini 3.x models this app runs are
# only published at Vertex's global endpoint — the EU multi-region ("eu") is not a
# valid Gemini generateContent endpoint (it 404s), and concrete EU regions
# (europe-west*) currently carry only Gemini 2.5.x. Data is therefore NOT
# guaranteed EU-resident on chat; revisit a europe-west* region if/when Gemini 3.x
# lands there. Overridable via GOOGLE_CLOUD_LOCATION.
_DEFAULT_LOCATION = "global"

# Building a service_account.Credentials object parses the key and sets up signing,
# so we do it once. Identity stability also matters: the model factory keys its
# LangChain client cache on ``json.dumps(kwargs, default=str)``, which stringifies
# the credentials object's repr — a fresh object per call would change that repr
# and thrash the cache. Keyed on the raw JSON so a changed key (or tests) rebuilds.
_credentials_cache: dict[str, object] = {}
_credentials_cache_lock = threading.Lock()


def clear_credentials_cache() -> None:
    """Clear the memoized credentials. Primarily for tests that patch
    ``service_account.Credentials.from_service_account_info``."""
    with _credentials_cache_lock:
        _credentials_cache.clear()


def _load_credentials(service_account_ref: str):
    """Build a service-account Credentials object from a path or inline JSON.

    Raises ``LLMConfigurationError`` if the reference can't be loaded — we
    deliberately do NOT swallow this and fall back to the API key, which would
    silently mask a broken production credential (and in prod there may be no API
    key left to fall back to).
    """
    from google.oauth2 import service_account

    scopes = [_CLOUD_PLATFORM_SCOPE]
    # The value is either a path to the key file (local dev) or the key JSON
    # inline (Heroku). os.path.isfile safely returns False for inline JSON
    # (multi-line strings aren't files); guard the rare OSError/ValueError.
    try:
        is_path = os.path.isfile(service_account_ref)
    except (ValueError, OSError):
        is_path = False

    try:
        if is_path:
            return service_account.Credentials.from_service_account_file(
                service_account_ref, scopes=scopes
            )
        info = json.loads(service_account_ref)
        return service_account.Credentials.from_service_account_info(
            info, scopes=scopes
        )
    except Exception as exc:
        raise LLMConfigurationError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is set but could not be loaded as a "
            "service-account credential (expected a path to a key file, or the "
            "key JSON inline)."
        ) from exc


def _build_credentials(service_account_ref: str):
    """Return a cached (or newly built) service-account Credentials object."""
    creds = _credentials_cache.get(service_account_ref)
    if creds is not None:
        return creds

    with _credentials_cache_lock:
        creds = _credentials_cache.get(service_account_ref)
        if creds is None:
            creds = _load_credentials(service_account_ref)
            _credentials_cache[service_account_ref] = creds
    return creds


def get_vertex_config() -> dict | None:
    """Return Vertex AI kwargs for Gemini clients, or ``None`` for the API-key path.

    ``None`` means no ``GOOGLE_SERVICE_ACCOUNT_JSON`` is configured — callers should
    use their existing API-key authentication. Otherwise returns a dict suitable to
    splat into ``ChatGoogleGenerativeAI`` / ``init_chat_model`` and ``genai.Client``:
    ``{"vertexai": True, "project": ..., "location": ..., "credentials": ...}``.
    """
    service_account_ref = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not service_account_ref:
        return None

    credentials = _build_credentials(service_account_ref)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        # Service-account JSON always carries the project it was issued under.
        project = getattr(credentials, "project_id", None)
    if not project:
        raise LLMConfigurationError(
            "Vertex AI is configured (GOOGLE_SERVICE_ACCOUNT_JSON is set) but no "
            "project could be determined. Set GOOGLE_CLOUD_PROJECT."
        )

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or _DEFAULT_LOCATION

    return {
        "vertexai": True,
        "project": project,
        "location": location,
        "credentials": credentials,
    }


__all__ = ["get_vertex_config", "clear_credentials_cache"]
