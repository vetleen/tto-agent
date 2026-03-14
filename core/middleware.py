import logging
import threading
import uuid


_thread_locals = threading.local()


def get_request_id() -> str:
    """Return the current request ID, or '-' if none is set."""
    return getattr(_thread_locals, "request_id", "-")


class RequestIDMiddleware:
    """Capture or generate a request ID for every HTTP request.

    Reads X-Request-ID from the incoming request (set by Heroku router),
    or generates a UUID4 if absent. Stores it in thread-local storage
    for the logging filter and echoes it back in the response header.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = request.META.get("HTTP_X_REQUEST_ID") or str(uuid.uuid4())
        _thread_locals.request_id = request_id
        request.request_id = request_id
        response = self.get_response(request)
        response["X-Request-ID"] = request_id
        _thread_locals.request_id = "-"
        return response


class RequestIDFilter(logging.Filter):
    """Inject request_id into every log record."""

    def filter(self, record):
        record.request_id = get_request_id()
        return True
