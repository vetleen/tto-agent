"""Shared HTTP helpers for JSON API endpoints."""
import json

from django.http import JsonResponse


def parse_json_object(request):
    """Parse a JSON request body that must be a JSON object.

    Returns ``(dict, None)`` on success or ``(None, JsonResponse(400))`` when the
    body is not valid JSON or is not a top-level object. Rejecting non-object
    bodies (``[]``, ``null``, ``5``, ``"x"``) here means callers can safely do
    ``data.get(...)`` without an ``AttributeError``/500.
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None, JsonResponse({"error": "Invalid JSON"}, status=400)
    if not isinstance(data, dict):
        return None, JsonResponse({"error": "Invalid JSON"}, status=400)
    return data, None
