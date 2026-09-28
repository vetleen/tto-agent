"""ChatAnthropic subclass that keeps Anthropic's ``input_transformations``.

With the preserved-thinking beta header, every response carries a top-level
``input_transformations`` array listing replayed thinking blocks the API dropped
(e.g. because the request prefix before them was edited). langchain-anthropic
copies only the model name from the stream's ``message_start`` event, so the
field is lost on the streaming path (non-streaming keeps it: every non-content
field lands in ``response_metadata``). This subclass copies it into the
``message_start`` chunk's ``response_metadata``.

Relies on the private ``_make_message_chunk_from_anthropic_event`` (pinned by
``llm/tests/test_anthropic_provider.py``). Only used for models whose registry
entry sets ``binds_thinking_to_prefix`` (see ``model_factory._build_client``).
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic


def _as_dicts(entries) -> list:
    out = []
    for e in entries or []:
        if isinstance(e, dict):
            out.append(e)
        elif hasattr(e, "model_dump"):
            out.append(e.model_dump())
    return out


class WilfredChatAnthropic(ChatAnthropic):
    def _make_message_chunk_from_anthropic_event(self, event, **kwargs):
        msg, block_start_event = super()._make_message_chunk_from_anthropic_event(event, **kwargs)
        if msg is not None and getattr(event, "type", None) == "message_start":
            transforms = getattr(getattr(event, "message", None), "input_transformations", None)
            if transforms is not None:
                msg.response_metadata["input_transformations"] = _as_dicts(transforms)
        return msg, block_start_event


__all__ = ["WilfredChatAnthropic"]
