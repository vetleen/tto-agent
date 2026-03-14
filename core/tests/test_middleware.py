import logging
import uuid

from django.test import TestCase, RequestFactory

from core.middleware import RequestIDMiddleware, RequestIDFilter, get_request_id, _thread_locals


class RequestIDMiddlewareTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.middleware = RequestIDMiddleware(
            get_response=lambda request: self._make_response(request)
        )
        self._captured_request = None

    def _make_response(self, request):
        from django.http import HttpResponse
        self._captured_request = request
        return HttpResponse("ok")

    def test_uses_header_when_present(self):
        request = self.factory.get("/", HTTP_X_REQUEST_ID="test-id-123")
        response = self.middleware(request)
        self.assertEqual(self._captured_request.request_id, "test-id-123")
        self.assertEqual(response["X-Request-ID"], "test-id-123")

    def test_generates_uuid_when_header_absent(self):
        request = self.factory.get("/")
        response = self.middleware(request)
        request_id = self._captured_request.request_id
        # Should be a valid UUID4
        uuid.UUID(request_id, version=4)
        self.assertEqual(response["X-Request-ID"], request_id)

    def test_cleans_up_thread_local(self):
        request = self.factory.get("/", HTTP_X_REQUEST_ID="cleanup-test")
        self.middleware(request)
        self.assertEqual(get_request_id(), "-")


class RequestIDFilterTest(TestCase):
    def test_injects_request_id_into_record(self):
        _thread_locals.request_id = "filter-test-id"
        try:
            filter_ = RequestIDFilter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test message", args=(), exc_info=None,
            )
            result = filter_.filter(record)
            self.assertTrue(result)
            self.assertEqual(record.request_id, "filter-test-id")
        finally:
            _thread_locals.request_id = "-"

    def test_returns_dash_when_no_request(self):
        _thread_locals.request_id = "-"
        filter_ = RequestIDFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        filter_.filter(record)
        self.assertEqual(record.request_id, "-")
