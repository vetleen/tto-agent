"""Provider replay state on stored assistant replies (Gemini thought signatures).

Gemini signs its reasoning: the final text part of a reply carries a thought
signature (an encrypted form of the model's internal reasoning). Google
recommends sending it back in that same part on later turns, so the model keeps
its reasoning context; it is not validated, and dropping it "may degrade"
quality. The final reply's signed text blocks are stored on the assistant
``ChatMessage.metadata["replay"]`` and echoed back while the thread stays on
the model that produced them.
"""

from __future__ import annotations

from llm.model_registry import canonical_model_id


def _joined_text(blocks: list[dict]) -> str:
    return "".join(str(b.get("text") or "") for b in blocks)


def build_assistant_replay(model: str | None, blocks: list | None, content: str) -> dict | None:
    """Metadata to store with a final assistant reply, or None.

    Only kept when the blocks spell out exactly the stored text, so replay can
    never resurrect content the message no longer has.
    """
    if not model or not blocks or _joined_text(blocks) != content:
        return None
    return {"model": canonical_model_id(model) or model, "blocks": blocks}


def replay_metadata_for(history_msg: dict, model: str | None) -> dict | None:
    """``Message.metadata`` that sends a stored reply back with its signed blocks.

    Skipped when the thread now uses another model (a signature belongs to the
    model that made it) or the content changed since it was stored.
    """
    replay = history_msg.get("replay")
    if history_msg.get("role") != "assistant" or not isinstance(replay, dict) or not model:
        return None
    blocks = replay.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None
    if replay.get("model") != (canonical_model_id(model) or model):
        return None
    if _joined_text(blocks) != history_msg.get("content"):
        return None
    return {"content_blocks": blocks}


__all__ = ["build_assistant_replay", "replay_metadata_for"]
