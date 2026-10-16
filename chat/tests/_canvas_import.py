"""Test helper: run a canvas import end to end (view → worker → status).

The import view now returns 202 and queues ``process_canvas_import``; most
import tests care about the finished result. ``post_import`` posts the file,
runs the worker inline instead of queueing (and swallows the socket push), then
reads the status endpoint and returns a response-like object in the old
synchronous shape: 200 + ``{title, content, thread_title?, truncated?,
original_chars?}`` when the job is ready, 400 + ``{error}`` when it failed.
Non-202 responses (validation errors) are returned unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def post_import(client, url, data):
    from chat.canvas_import import run_canvas_import

    with patch("chat.tasks.process_canvas_import.delay", side_effect=lambda job_id: run_canvas_import(job_id)), \
            patch("chat.canvas_import._push"):
        resp = client.post(url, data, format="multipart")
    if resp.status_code != 202:
        return resp
    status = client.get(resp.json()["status_url"]).json()
    if status["state"] == "ready":
        canvas = status["canvas"]
        body = {"title": canvas["title"], "content": canvas["content"], "canvas_id": canvas["canvas_id"]}
        if status.get("thread_title"):
            body["thread_title"] = status["thread_title"]
        if canvas.get("truncated"):
            body["truncated"] = True
            body["original_chars"] = canvas["original_chars"]
        return SimpleNamespace(status_code=200, json=lambda: body, accepted=resp)
    return SimpleNamespace(status_code=400, json=lambda: {"error": status["error"]}, accepted=resp)
