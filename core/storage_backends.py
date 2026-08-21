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

from storages.backends.s3boto3 import S3Boto3Storage


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
