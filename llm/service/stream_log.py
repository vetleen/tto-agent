"""Compact stream-event accumulation for LLM call logging.

``LLMService.stream`` used to retain every ``StreamEvent`` of a run (one
pydantic object per token, plus full tool results) purely so ``log_stream``
could assemble a transcript afterwards — a large per-run memory spike on the
worker. ``StreamLogAccumulator`` replaces that: events are folded as they pass
through, keeping only what the log row needs (joined token text, per-tool-call
sizes and short previews, the terminal ``message_end``/``error`` data).

The fold is deliberately different from the ``run_via_stream`` collapse:

- token text accumulates across ``message_start`` boundaries — the log's
  transcript view and interrupted-token estimation span ALL turns of a
  tool loop, while the collapse resets per turn;
- the FIRST ``message_end`` wins here (``log_stream`` semantics), while the
  collapse keeps the LAST.

Do not merge the two folds.

``LLM_LOG_FULL_PAYLOADS`` (env, read at call time) opts local debugging into
larger — but still capped — payloads.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from llm.types.responses import ChatResponse
    from llm.types.streaming import StreamEvent

LOG_TEXT_PREVIEW_CHARS = 500
LOG_TOOL_ARGS_CHARS = 1024
LOG_TOOL_RESULT_PREVIEW_CHARS = 200
LOG_FULL_PAYLOAD_CAP_CHARS = 100_000
LOG_FULL_RESULT_PREVIEW_CHARS = 10_000


def full_payloads_enabled() -> bool:
    """Whether ``LLM_LOG_FULL_PAYLOADS`` opts into larger (still capped) payloads."""
    return os.environ.get("LLM_LOG_FULL_PAYLOADS", "").strip().lower() in ("1", "true", "yes")


def cap_text(value: str, cap: int) -> str:
    """Truncate ``value`` to ``cap`` chars, appending a size marker when truncated."""
    if len(value) <= cap:
        return value
    return value[:cap] + f" …[truncated: {len(value):,} chars total]"


def _args_size(arguments) -> int:
    try:
        return len(json.dumps(arguments, default=str))
    except Exception:
        return 0


class StreamLogAccumulator:
    """Folds stream events into the compact form ``log_stream`` needs."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._tool_by_id: dict[str, dict] = {}
        self._end_data: dict | None = None
        self._error_data: dict | None = None
        self.event_count: int = 0

    def add(self, event: "StreamEvent") -> None:
        self.event_count += 1
        et = event.event_type
        data = event.data or {}
        if et == "token":
            self._text_parts.append(data.get("text", ""))
        elif et == "tool_start":
            tid = data.get("tool_call_id") or ""
            self._tool_by_id[tid] = {
                "tool_call_id": tid,
                "tool_name": data.get("tool_name", ""),
                "args_size": _args_size(data.get("arguments", {})),
                "result_size": None,
                "result_preview": None,
            }
        elif et == "tool_end":
            tid = data.get("tool_call_id") or ""
            entry = self._tool_by_id.get(tid)
            if entry is None:
                # end-without-start: mirror the old assembler's tolerance
                entry = {
                    "tool_call_id": tid,
                    "tool_name": data.get("tool_name", ""),
                    "args_size": 0,
                    "result_size": None,
                    "result_preview": None,
                }
                self._tool_by_id[tid] = entry
            result = data.get("result")
            if result is not None:
                result_str = result if isinstance(result, str) else json.dumps(result, default=str)
                preview_cap = (
                    LOG_FULL_RESULT_PREVIEW_CHARS if full_payloads_enabled()
                    else LOG_TOOL_RESULT_PREVIEW_CHARS
                )
                entry["result_size"] = len(result_str)
                entry["result_preview"] = cap_text(result_str, preview_cap)
        elif et == "message_end":
            if self._end_data is None:
                self._end_data = dict(data)
        elif et == "error":
            if self._error_data is None:
                self._error_data = dict(data)

    @classmethod
    def from_events(cls, events: Iterable["StreamEvent"]) -> "StreamLogAccumulator":
        acc = cls()
        for event in events:
            acc.add(event)
        return acc

    @property
    def has_events(self) -> bool:
        return self.event_count > 0

    @property
    def has_end_event(self) -> bool:
        """True when a ``message_end`` was seen — even one with empty data.

        ``log_stream``'s interrupted-stream detection must key off this flag,
        not off ``end_data`` truthiness: an empty-data ``message_end`` is still
        a terminal event.
        """
        return self._end_data is not None

    @property
    def end_data(self) -> dict:
        return self._end_data or {}

    @property
    def error_data(self) -> dict | None:
        return self._error_data

    @property
    def streamed_text(self) -> str:
        return "".join(self._text_parts)

    def build_raw_output(self) -> str:
        """Compact JSON replacing the old full-transcript ``raw_output``."""
        text = self.streamed_text
        text_cap = (
            LOG_FULL_PAYLOAD_CAP_CHARS if full_payloads_enabled()
            else LOG_TEXT_PREVIEW_CHARS
        )
        payload = {
            "final_text_preview": cap_text(text, text_cap),
            "tool_calls": list(self._tool_by_id.values()),
            "counts": {
                "events": self.event_count,
                "text_chars": len(text),
                "tool_calls": len(self._tool_by_id),
            },
        }
        return json.dumps(payload)


def slim_response_raw_output(response: "ChatResponse") -> str:
    """Compact ``raw_output`` for the non-streaming path (``log_call``)."""
    if full_payloads_enabled():
        return cap_text(response.model_dump_json(), LOG_FULL_PAYLOAD_CAP_CHARS)
    message = response.message
    content = message.content if isinstance(message.content, str) else json.dumps(
        message.content, default=str,
    )
    tool_calls = [
        {"tool_name": tc.name, "args_size": _args_size(tc.arguments)}
        for tc in (message.tool_calls or [])
    ]
    payload = {
        "final_text_preview": cap_text(content, LOG_TEXT_PREVIEW_CHARS),
        "tool_calls": tool_calls,
        "counts": {"text_chars": len(content), "tool_calls": len(tool_calls)},
    }
    return json.dumps(payload)


__all__ = [
    "StreamLogAccumulator",
    "slim_response_raw_output",
    "cap_text",
    "full_payloads_enabled",
    "LOG_TEXT_PREVIEW_CHARS",
    "LOG_TOOL_ARGS_CHARS",
    "LOG_TOOL_RESULT_PREVIEW_CHARS",
    "LOG_FULL_PAYLOAD_CAP_CHARS",
    "LOG_FULL_RESULT_PREVIEW_CHARS",
]
