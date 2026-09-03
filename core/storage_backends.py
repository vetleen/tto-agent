"""S3 storage backend that shares one boto3 Session across threads.

Upstream ``S3Boto3Storage`` caches its connection in ``threading.local()`` and
creates a **new ``boto3.Session`` per thread** (django-storages
``storages/backends/s3.py``). Each session has its own botocore ``Loader``, so
every web worker thread that touches S3 loads its own full copy of the S3 service
model (tens of MB of ``OrderedDict``s). On the long-lived, threaded daphne process
these per-thread copies accumulate and drive RSS to R14/R15 (prod incident
2026-08-20).

Sharing a single ``boto3.Session`` shares its botocore ``Loader`` cache, so the S3
service model is loaded **once** regardless of thread count. Per-thread *resources*
are still created from the shared session (boto3 resources aren't thread-safe), but
they reuse the cached model — the heavy part is paid once.

``botocore.session.create_client`` (which ``session.resource()`` funnels into) holds
no lock and lazily instantiates the session's shared caches (service-model loader,
credential resolver, endpoint resolver, config store) on first use. So the session's
caches are warmed **once, single-threaded, under a lock** at creation (see
``_create_session``); concurrent per-thread resource creation then only *reads* those
populated caches, which is safe.
"""

from __future__ import annotations

import threading

from django.contrib.staticfiles.storage import HashedFilesMixin
from storages.backends.s3boto3 import S3Boto3Storage
from whitenoise.storage import CompressedManifestStaticFilesStorage


class SharedSessionS3Storage(S3Boto3Storage):
    """``S3Boto3Storage`` that shares one ``boto3.Session`` across all threads.

    Overriding only ``_create_session`` fixes both ``connection`` and
    ``unsigned_connection`` (both call it) while leaving the per-thread resource
    isolation upstream relies on for thread-safety intact.
    """

    # Class-level: this app runs a single S3 configuration, so one shared session
    # is correct. A class attribute stays out of instance ``__dict__``, so storage
    # pickling (``__getstate__``/``__setstate__``) is unaffected.
    _session_lock = threading.Lock()
    _shared_session = None

    def _create_session(self):
        cls = SharedSessionS3Storage
        if cls._shared_session is None:
            with cls._session_lock:
                if cls._shared_session is None:
                    session = super()._create_session()
                    # Warm the session's lazy, shared caches (S3 service-model
                    # loader, credential resolver, endpoint resolver, config
                    # store) single-threaded, under the lock, before any thread
                    # races to populate them via ``session.resource()``. Mirror
                    # the exact call upstream's ``connection`` makes so the same
                    # components warm. The throwaway resource is discarded; the
                    # heavy caches persist on the session.
                    session.resource(
                        "s3",
                        region_name=self.region_name,
                        use_ssl=self.use_ssl,
                        endpoint_url=self.endpoint_url,
                        config=self.client_config,
                        verify=self.verify,
                    )
                    # Publish only after warm-up succeeds, so a transient
                    # failure doesn't cache a half-initialized session.
                    cls._shared_session = session
        return cls._shared_session


class ManifestStaticStorage(CompressedManifestStaticFilesStorage):
    """Hashed + compressed static storage for production (WhiteNoise).

    Gives every collected file a content hash (``output.<hash>.css``) so WhiteNoise
    serves it ``Cache-Control: public, max-age=31536000, immutable``. The browser then
    reuses it from disk cache with **no revalidation round-trip** on every navigation —
    the fix for the intermittent unstyled-content flash on repeat page loads. A deploy
    changes the content, hence the hash, hence the URL, so caches bust automatically.
    """

    # Don't 500 a page when a template references a static file missing from the
    # manifest; fall back to the unhashed path (served, just not cache-busted). Prudent
    # for the first rollout — can be tightened to True once we've confirmed there are no
    # dangling ``{% static %}`` references.
    manifest_strict = False

    # Drop the default ``*.js`` reference-rewriting pattern while keeping Django's CSS
    # rules (so the self-hosted font ``url()``s still get hashed). Several vendored
    # ``static/js/vendor/*.min.js`` files end with ``//# sourceMappingURL=<name>.map``
    # comments whose ``.map`` targets we don't ship; the default post-processing tries to
    # resolve them and raises ``ValueError``, which would fail ``collectstatic`` (the
    # Heroku release step). None of our served JS relies on intra-file reference
    # rewriting, so dropping only the JS entry is safe. Filtering Django's own tuple keeps
    # us aligned with its CSS regex across versions. File hashing is independent of
    # ``patterns``, so every JS file still gets a hashed name.
    patterns = tuple(p for p in HashedFilesMixin.patterns if p[0] != "*.js")

    # Build-only sources that live under ``static/`` but are never served: the Tailwind
    # source ``src/input.css`` ``@import``s node_modules paths (``flowbite/…``) that
    # aren't collected, so hashing/rewriting it would fail the whole run. Only the
    # compiled ``src/output.css`` is served. Skipping post-processing leaves the file
    # collected but un-hashed and out of the manifest — fine, since nothing links to it —
    # while ``output.css`` keeps full strict validation, so a genuinely broken font
    # reference there still fails loudly.
    _UNPROCESSED_SOURCES = frozenset({"src/input.css"})

    def post_process(self, paths, **options):
        # collectstatic keys ``paths`` with the OS separator (backslashes on Windows);
        # normalize to forward slashes before matching.
        filtered = {
            path: storage_and_name
            for path, storage_and_name in paths.items()
            if path.replace("\\", "/") not in self._UNPROCESSED_SOURCES
        }
        yield from super().post_process(filtered, **options)
