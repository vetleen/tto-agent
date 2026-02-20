"""
Adapter used by llm_chat: exposes call_llm and call_llm_stream backed by llm_service.client.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any, Generator

from llm_service.client import completion
from llm_service.conf import get_default_model, should_send_reasoning_effort
from llm_service.models import LLMCallLog


def _messages(system_instructions: str, user_prompt: str) -> list[dict[str, Any]]:
    msgs = []
    if system_instructions:
        msgs.append({"role": "system", "content": system_instructions})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _text_from_response(response: Any) -> str:
    if not response or not getattr(response, "choices", None) or len(response.choices) == 0:
        return ""
    c = response.choices[0]
    msg = getattr(c, "message", None)
    if not msg:
        return ""
    return getattr(msg, "content", None) or ""


def _delta_from_chunk(chunk: Any) -> str:
    if not chunk or not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
        return ""
    delta = getattr(chunk.choices[0], "delta", None)
    if not delta:
        return ""
    return getattr(delta, "content", None) or ""


class LLMService:
    """
    High-level API for llm_chat: call_llm (sync, optional JSON schema) and call_llm_stream.
    Uses llm_service.client.completion under the hood; logs are written by the client.
    """

    def call_llm(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        system_instructions: str = "",
        user_prompt: str = "",
        tools: Any = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        user: Any = None,
    ) -> Any:
        """
        Single completion. Returns an object with .succeeded, .parsed_json (if json_schema),
        and .call_log (LLMCallLog when available).
        """
        model = model or get_default_model()
        request_id = str(uuid.uuid4())
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _messages(system_instructions, user_prompt),
            "stream": False,
            "metadata": {"request_id": request_id},
            "user": user,
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None and should_send_reasoning_effort(model):
            kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools

        try:
            response = completion(**kwargs)
        except Exception as e:
            log = (
                LLMCallLog.objects.filter(request_id=request_id)
                .order_by("-created_at")
                .first()
            )
            return SimpleNamespace(
                succeeded=False,
                parsed_json=None,
                call_log=log,
                error=str(e),
            )

        text = _text_from_response(response)
        parsed_json = None
        if json_schema is not None and text:
            try:
                parsed_json = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed_json = None

        log = (
            LLMCallLog.objects.filter(request_id=request_id)
            .order_by("-created_at")
            .first()
        )
        return SimpleNamespace(
            succeeded=True,
            parsed_json=parsed_json,
            call_log=log,
        )

    def call_llm_stream(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        system_instructions: str = "",
        user_prompt: str = "",
        tools: Any = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        user: Any = None,
    ) -> Generator[tuple[str, Any], None, None]:
        """
        Streaming completion. Yields ("response.output_text.delta", event_with_delta)
        and ("final", {"call_log": log, "response": {}}).
        """
        model = model or get_default_model()
        request_id = str(uuid.uuid4())
        kwargs = {
            "model": model,
            "messages": _messages(system_instructions, user_prompt),
            "stream": True,
            "metadata": {"request_id": request_id},
            "user": user,
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None and should_send_reasoning_effort(model):
            kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools

        try:
            stream = completion(**kwargs)
            for chunk in stream:
                delta = _delta_from_chunk(chunk)
                if delta:
                    yield "response.output_text.delta", SimpleNamespace(delta=delta)
        except Exception:
            raise

        log = (
            LLMCallLog.objects.filter(request_id=request_id)
            .order_by("-created_at")
            .first()
        )
        yield "final", {"call_log": log, "response": {}}
