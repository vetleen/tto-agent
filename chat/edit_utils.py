"""Shared helper for applying targeted find-replace edits safely.

Used by ``document_edit`` (``chat/tools.py``) and ``canvas_edit``
(``chat/canvas_tools.py``). Every edit must match exactly one occurrence in the
ORIGINAL text — not in the progressively-mutated buffer — and accepted edits may
not overlap one another. Resolving against a mutated buffer let an edit match
text a previous edit had just inserted (or miss text a previous edit removed);
resolving against the original snapshot and applying accepted spans right-to-left
avoids both.
"""

from __future__ import annotations

from collections.abc import Iterable


def apply_unique_text_edits(
    original: str, edits: Iterable[tuple[str, str]]
) -> tuple[str, int, list[dict]]:
    """Apply unique-match find-replace ``edits`` against ``original``.

    ``edits`` is an iterable of ``(old_text, new_text)`` pairs. Returns
    ``(new_text, applied, failed)`` where ``failed`` is a list of
    ``{"old_text": <truncated>, "error": <message>}`` dicts (in input order). An
    edit fails if its ``old_text`` is empty, is absent from ``original``, appears
    more than once, or overlaps the span of an already-accepted edit in this call.
    """
    accepted: list[tuple[int, int, str]] = []  # (start, end, new_text)
    failed: list[dict] = []

    for old_text, new_text in edits:
        if not old_text:
            failed.append({"old_text": (old_text or "")[:80], "error": "Empty old_text."})
            continue
        count = original.count(old_text)
        if count == 0:
            failed.append({"old_text": old_text[:80], "error": "Text not found."})
            continue
        if count > 1:
            failed.append({
                "old_text": old_text[:80],
                "error": f"Found {count} matches — add more surrounding text to make it unique.",
            })
            continue
        start = original.find(old_text)
        end = start + len(old_text)
        if any(start < a_end and a_start < end for a_start, a_end, _ in accepted):
            failed.append({"old_text": old_text[:80], "error": "Overlaps another edit in this call."})
            continue
        accepted.append((start, end, new_text))

    # Apply accepted spans right-to-left so earlier offsets stay valid.
    result = original
    for start, end, repl in sorted(accepted, key=lambda span: span[0], reverse=True):
        result = result[:start] + repl + result[end:]

    return result, len(accepted), failed
