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
    ``{"old_text": <truncated>, "new_text_chars": <int>, "error": <message>}``
    dicts (in input order; the size lets callers spot a runaway replacement).
    An edit fails if its ``old_text`` is empty, is absent from ``original``,
    appears more than once, or overlaps the span of an already-accepted edit in
    this call.
    """
    accepted: list[tuple[int, int, str]] = []  # (start, end, new_text)
    failed: list[dict] = []

    def _fail(old_text: str, new_text: str, error: str) -> None:
        failed.append({
            "old_text": (old_text or "")[:80],
            "new_text_chars": len(new_text or ""),
            "error": error,
        })

    for old_text, new_text in edits:
        if not old_text:
            _fail(old_text, new_text, "Empty old_text.")
            continue
        count = original.count(old_text)
        if count == 0:
            _fail(old_text, new_text, "Text not found.")
            continue
        if count > 1:
            _fail(
                old_text, new_text,
                f"Found {count} matches — add more surrounding text to make it unique.",
            )
            continue
        start = original.find(old_text)
        end = start + len(old_text)
        if any(start < a_end and a_start < end for a_start, a_end, _ in accepted):
            _fail(old_text, new_text, "Overlaps another edit in this call.")
            continue
        accepted.append((start, end, new_text))

    # Apply accepted spans right-to-left so earlier offsets stay valid.
    result = original
    for start, end, repl in sorted(accepted, key=lambda span: span[0], reverse=True):
        result = result[:start] + repl + result[end:]

    return result, len(accepted), failed


def append_with_anchor(content: str, text: str, anchor: str = "") -> tuple[str, bool]:
    """Insert ``text`` into ``content``, returning ``(new_content, inserted_after_anchor)``.

    Placement rule shared by the canvas insert tools: if ``anchor`` is given and
    matches **exactly once** in ``content``, ``text`` is inserted immediately
    after that occurrence; otherwise (no anchor, anchor absent, or anchor
    ambiguous with >1 matches) ``text`` is appended at the end. Insertion is
    separated from surrounding content by a blank line. On an empty canvas the
    result is just ``text``. The unique-match requirement mirrors
    :func:`apply_unique_text_edits` so the two tools place content identically.
    """
    text = text or ""
    if not content:
        return text, False

    if anchor and content.count(anchor) == 1:
        end = content.find(anchor) + len(anchor)
        new_content = content[:end] + "\n\n" + text + content[end:]
        return new_content, True

    # No anchor, or it didn't resolve to a single location — append at the end.
    return content + "\n\n" + text, False
