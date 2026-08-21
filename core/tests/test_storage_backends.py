"""Tests for ``core.storage_backends.SharedSessionS3Storage``.

The backend exists to stop the per-thread boto3-session duplication that drove
web.1 to R14/R15: upstream ``S3Boto3Storage`` creates a new ``boto3.Session``
per thread (each loading its own ~tens-of-MB S3 service model). These tests pin
the fix — one shared session regardless of thread, warmed once under a lock —
without any network access.

The upstream ``_create_session`` is patched to return a ``MagicMock`` session so
the warm-up ``session.resource("s3", ...)`` never touches real boto credential,
region, or endpoint resolution; the mock's ``.resource`` doubles as a spy.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from django.test import SimpleTestCase
from storages.backends.s3boto3 import S3Boto3Storage

from core.storage_backends import SharedSessionS3Storage


def _patch_super_session(side_effect=None):
    """Patch upstream ``_create_session`` to yield fresh mock sessions.

    ``side_effect`` overrides the factory (e.g. to make the first session's
    ``.resource`` raise). By default each call returns a new ``MagicMock``.
    """
    factory = side_effect or (lambda self: mock.MagicMock(name="boto_session"))
    return mock.patch.object(
        S3Boto3Storage, "_create_session", autospec=True, side_effect=factory
    )


class SharedSessionS3StorageTests(SimpleTestCase):
    def setUp(self):
        # Reset the class-level shared session so each test starts clean and
        # doesn't leak a session into unrelated tests.
        SharedSessionS3Storage._shared_session = None
        self.addCleanup(lambda: setattr(SharedSessionS3Storage, "_shared_session", None))

    def test_one_session_shared_across_threads(self):
        storage = SharedSessionS3Storage(bucket_name="test-bucket")
        with _patch_super_session() as super_create:
            with ThreadPoolExecutor(max_workers=8) as pool:
                sessions = list(pool.map(lambda _: storage._create_session(), range(16)))

        first = sessions[0]
        self.assertIsNotNone(first)
        for s in sessions:
            self.assertIs(s, first)  # exactly one boto3.Session for all threads
        # The session is created and warmed exactly once despite the race.
        self.assertEqual(super_create.call_count, 1)
        first.resource.assert_called_once()

    def test_session_shared_across_instances(self):
        # Class-level cache: all storage instances (there is one S3 config in this
        # app) share the single session, so the S3 model loads once process-wide.
        a = SharedSessionS3Storage(bucket_name="bucket-a")
        b = SharedSessionS3Storage(bucket_name="bucket-b")
        with _patch_super_session() as super_create:
            self.assertIs(a._create_session(), b._create_session())
            self.assertEqual(super_create.call_count, 1)

    def test_second_call_reuses_first(self):
        storage = SharedSessionS3Storage(bucket_name="test-bucket")
        with _patch_super_session() as super_create:
            self.assertIs(storage._create_session(), storage._create_session())
            self.assertEqual(super_create.call_count, 1)

    def test_warmup_creates_one_s3_resource(self):
        # Warm-up must mirror upstream ``connection``: a single ``s3`` resource
        # built with the storage's own connection kwargs.
        storage = SharedSessionS3Storage(bucket_name="test-bucket")
        with _patch_super_session():
            session = storage._create_session()

        session.resource.assert_called_once_with(
            "s3",
            region_name=storage.region_name,
            use_ssl=storage.use_ssl,
            endpoint_url=storage.endpoint_url,
            config=storage.client_config,
            verify=storage.verify,
        )

    def test_not_published_when_warmup_fails(self):
        storage = SharedSessionS3Storage(bucket_name="test-bucket")

        # First session's warm-up raises → nothing is published, error propagates.
        boom = mock.MagicMock(name="boom_session")
        boom.resource.side_effect = RuntimeError("transient warm-up failure")
        good = mock.MagicMock(name="good_session")
        sessions = iter([boom, good])

        with _patch_super_session(side_effect=lambda self: next(sessions)):
            with self.assertRaises(RuntimeError):
                storage._create_session()
            self.assertIsNone(SharedSessionS3Storage._shared_session)

            # Next call retries cleanly and publishes the healthy session.
            self.assertIs(storage._create_session(), good)
            self.assertIs(SharedSessionS3Storage._shared_session, good)
