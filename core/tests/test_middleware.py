import contextvars
import logging
import uuid

from django.test import TestCase, RequestFactory

from core.middleware import RequestIDMiddleware, RequestIDFilter, get_request_id, _request_id_var


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

    def test_cleans_up_context_var(self):
        request = self.factory.get("/", HTTP_X_REQUEST_ID="cleanup-test")
        self.middleware(request)
        self.assertEqual(get_request_id(), "-")

    def test_request_id_visible_inside_view(self):
        seen = {}

        def view(request):
            from django.http import HttpResponse
            seen["request_id"] = get_request_id()
            return HttpResponse("ok")

        middleware = RequestIDMiddleware(get_response=view)
        middleware(self.factory.get("/", HTTP_X_REQUEST_ID="inside-view"))
        self.assertEqual(seen["request_id"], "inside-view")

    def test_concurrent_contexts_do_not_cross_talk(self):
        """Two requests in separate contexts (as under ASGI) keep their own IDs.

        Simulates the interleaving that happens on asgiref's shared sync thread:
        request A's middleware sets its ID, then request B's runs in a different
        context — A's view must still see A's ID, and B's reset must not clobber
        A's value (the threading.local implementation failed both).
        """
        seen = {}

        def view_a(request):
            from django.http import HttpResponse
            # B runs to completion in its own context while A is "suspended".
            ctx_b.run(
                RequestIDMiddleware(get_response=view_b),
                self.factory.get("/", HTTP_X_REQUEST_ID="id-b"),
            )
            seen["a"] = get_request_id()
            return HttpResponse("ok")

        def view_b(request):
            from django.http import HttpResponse
            seen["b"] = get_request_id()
            return HttpResponse("ok")

        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()
        ctx_a.run(
            RequestIDMiddleware(get_response=view_a),
            self.factory.get("/", HTTP_X_REQUEST_ID="id-a"),
        )
        self.assertEqual(seen["a"], "id-a")
        self.assertEqual(seen["b"], "id-b")
        self.assertEqual(get_request_id(), "-")


class RequestIDFilterTest(TestCase):
    def test_injects_request_id_into_record(self):
        token = _request_id_var.set("filter-test-id")
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
            _request_id_var.reset(token)

    def test_returns_dash_when_no_request(self):
        filter_ = RequestIDFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        filter_.filter(record)
        self.assertEqual(record.request_id, "-")
