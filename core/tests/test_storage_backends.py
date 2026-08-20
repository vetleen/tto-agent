"""Tests for ``core.storage_backends.SharedSessionS3Storage``.

The backend exists to stop the per-thread boto3-session duplication that drove
web.1 to R14/R15: upstream ``S3Boto3Storage`` creates a new ``boto3.Session``
per thread (each loading its own ~tens-of-MB S3 service model). These tests pin
the fix — one shared session regardless of thread — without any network access
(``boto3.Session`` construction is lazy/offline).
"""

from concurrent.futures import ThreadPoolExecutor

from django.test import SimpleTestCase

from core.storage_backends import SharedSessionS3Storage


class SharedSessionS3StorageTests(SimpleTestCase):
    def setUp(self):
        # Reset the class-level shared session so each test starts clean and
        # doesn't leak a session into unrelated tests.
        SharedSessionS3Storage._shared_session = None
        self.addCleanup(lambda: setattr(SharedSessionS3Storage, "_shared_session", None))

    def test_one_session_shared_across_threads(self):
        storage = SharedSessionS3Storage(bucket_name="test-bucket")
        with ThreadPoolExecutor(max_workers=8) as pool:
            sessions = list(pool.map(lambda _: storage._create_session(), range(16)))

        first = sessions[0]
        self.assertIsNotNone(first)
        for s in sessions:
            self.assertIs(s, first)  # exactly one boto3.Session for all threads

    def test_session_shared_across_instances(self):
        # Class-level cache: all storage instances (there is one S3 config in this
        # app) share the single session, so the S3 model loads once process-wide.
        a = SharedSessionS3Storage(bucket_name="bucket-a")
        b = SharedSessionS3Storage(bucket_name="bucket-b")
        self.assertIs(a._create_session(), b._create_session())

    def test_second_call_reuses_first(self):
        storage = SharedSessionS3Storage(bucket_name="test-bucket")
        self.assertIs(storage._create_session(), storage._create_session())
