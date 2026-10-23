"""Process-wide OpenAI SDK client shared by the services that call OpenAI
directly (transcription, decisions) rather than through LangChain.

One shared client instead of a fresh ``OpenAI()`` per call: reuses the httpx
connection pool across sequential calls (no per-call TLS handshake) and pins
explicit timeouts — the SDK default is a 600 s read timeout, so a single wedged
request could otherwise burn most of a Celery task's time budget. Callers with
tighter needs pass per-request ``options`` (``timeout``/``max_retries``) to the
SDK call instead of building their own client.
"""

from __future__ import annotations

import threading

_openai_client = None
_openai_client_cls = None  # the OpenAI class the cached client was built from
_openai_client_lock = threading.Lock()


def get_openai_client():
    """Return the process-wide OpenAI client (thread-safe, lazily built).

    * ``read=300``: a non-streaming transcription call must cover server-side
      processing of a full ~23-minute file; typical calls finish well under a
      minute, and capping at 5 min halves the worst-case hang.
    * ``write=120``: uploading up to 25 MB on a slow route.
    * ``max_retries=2`` (the SDK default, made explicit): covers 429/5xx,
      which the upload orchestrator's transient retry deliberately does not.

    The cache is keyed on the identity of the ``OpenAI`` class itself: tests
    that ``patch("openai.OpenAI")`` get a fresh client built from their mock
    (each patch is a different class object), while production — where the
    class never changes — builds exactly once.
    """
    global _openai_client, _openai_client_cls
    from openai import OpenAI

    if _openai_client is None or _openai_client_cls is not OpenAI:
        with _openai_client_lock:
            if _openai_client is None or _openai_client_cls is not OpenAI:
                import httpx

                _openai_client = OpenAI(
                    timeout=httpx.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0),
                    max_retries=2,
                )
                _openai_client_cls = OpenAI
    return _openai_client


__all__ = ["get_openai_client"]
