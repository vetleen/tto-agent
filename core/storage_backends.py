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
they reuse the cached model — the heavy part is paid once. boto3 >= 1.35 handles
concurrent resource/client creation from a shared session.
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
                    cls._shared_session = super()._create_session()
        return cls._shared_session
