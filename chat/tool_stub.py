"""Tool-result stubbing — collapse old, reproducible tool results to a short,
actionable placeholder so they stop consuming context.

Tool results are *reproducible*: the model can always call the tool again. Once a
result is beyond the recency window (between turns) or the request would overflow
the model window (mid-turn), we replace its body with a stub that names the tool
and its arguments, so the model knows exactly how to get the data back. The
matching assistant ``tool_calls`` entry and the result's ``tool_call_id`` are
preserved — only the *body* shrinks — so provider tool-call/result pairing stays
intact (and the orphan-strip in ``_load_history`` keeps the message).

Findings the model wants to keep across pruning belong in the scratchpad
(``scratchpad_append``), which is injected every turn and never stubbed.
"""

from __future__ import annotations

import json

# How much of each argument value to keep in the stub's arg summary. Keeps the
# stub tiny (a stub must never itself be a context cost) while staying precise
# enough to re-issue the call.
_ARG_VALUE_MAXLEN = 80
_ARGS_MAXLEN = 240


def _summarise_args(arguments) -> str:
    """A compact, single-line rendering of tool arguments for the stub.

    Never emits base64/native payloads or huge values — each value is truncated,
    and the whole thing is capped. ``reason`` (the ubiquitous ReasonBaseModel
    field) is dropped as noise.
    """
    if not isinstance(arguments, dict):
        return ""
    parts: list[str] = []
    for key, value in arguments.items():
        if key == "reason":
            continue
        try:
            if isinstance(value, str):
                rendered = value
            else:
                rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        if len(rendered) > _ARG_VALUE_MAXLEN:
            rendered = rendered[: _ARG_VALUE_MAXLEN - 1] + "…"
        parts.append(f"{key}={rendered}")
    summary = ", ".join(parts)
    if len(summary) > _ARGS_MAXLEN:
        summary = summary[: _ARGS_MAXLEN - 1] + "…"
    return summary


def build_tool_result_stub(tool_name: str, arguments=None) -> str:
    """A short placeholder that replaces an evicted tool result.

    ``arguments`` is the tool call's stored argument dict (from the assistant
    message's ``tool_calls`` metadata); it may be ``None`` when unavailable.
    """
    name = tool_name or "a tool"
    args = _summarise_args(arguments)
    call = f"{name}({args})" if args else name
    return (
        f"[Earlier result of {call} was cleared to save context. "
        "Call the tool again if you still need it.]"
    )
