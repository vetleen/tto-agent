"""Show a rendered deck's slides to the model — shared by ``document_view_native``
(data-room versions) and ``chat_attachment_view`` (chat attachments).

Both owners store one JPEG per slide as ``chat.Asset`` rows with
``role=page_render`` + ``page_number`` (see ``documents/services/page_render.py``);
only the owner filter differs. This module holds the owner-agnostic part: which
slides a ``pages`` selection means, reading the blobs, and queueing them on the
run context as native assets — never tokenized (``asset_id: ""``) and visible for
the current reply only. Callers phrase the result for their own tool.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def format_numbers(numbers: list[int]) -> str:
    """``[3, 4, 5]`` → ``3–5``; ``[1, 4, 7]`` → ``1, 4, 7``; ``[]`` → ``''``."""
    if not numbers:
        return ""
    ordered = sorted(set(int(n) for n in numbers))
    if len(ordered) > 1 and ordered[-1] - ordered[0] == len(ordered) - 1:
        return f"{ordered[0]}–{ordered[-1]}"
    return ", ".join(str(n) for n in ordered)


def view_slide_limits() -> tuple[int, int]:
    """``(default_n, max_n)``: slides shown when no ``pages`` is given, and the
    per-call cap (settings ``DOCUMENT_RENDER_VIEW_DEFAULT_SLIDES`` / ``_MAX_SLIDES``)."""
    from django.conf import settings

    default_n = max(1, int(getattr(settings, "DOCUMENT_RENDER_VIEW_DEFAULT_SLIDES", 8)))
    max_n = max(1, int(getattr(settings, "DOCUMENT_RENDER_VIEW_MAX_SLIDES", 12)))
    return default_n, max_n


@dataclass
class SlideViewOutcome:
    """What :func:`queue_slide_images` did, plus the sentence fragments callers
    append to their own message."""

    total: int
    default_n: int
    max_n: int
    shown: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    budget_hit: bool = False
    truncated: bool = False
    selection_note: str = ""

    def missing_note(self) -> str:
        if not self.missing:
            return ""
        plural = len(self.missing) != 1
        return (
            f" Slide{'s' if plural else ''} {format_numbers(self.missing)} "
            f"{'have' if plural else 'has'} no preview."
        )

    def budget_note(self) -> str:
        if not self.budget_hit or not self.shown:
            return ""
        return f" The attachment budget for this run is exhausted after slide {self.shown[-1]}."

    def cap_note(self) -> str:
        return f" At most {self.max_n} slides are shown per call." if self.truncated else ""

    def more_hint(self) -> str:
        """``" Pass pages='9-16' to view more."`` when unseen slides follow the
        last one shown and the budget did not stop the call."""
        if not self.shown or self.budget_hit:
            return ""
        last = max(self.shown)
        if last >= self.total:
            return ""
        return f" Pass pages='{last + 1}-{min(self.total, last + self.default_n)}' to view more."


def select_slides(total: int, pages: str | None, *, default_n: int, max_n: int) -> tuple[list[int], str, bool]:
    """``(wanted, selection_note, truncated)`` — the 1-based slide numbers a
    ``pages`` selection means (in the order given), or the first ``default_n``."""
    from chat.pdf_attach import parse_page_ranges

    selection_note = ""
    wanted: list[int] = []
    if pages:
        wanted = [i + 1 for i in parse_page_ranges(pages, total)]
        if not wanted:
            selection_note = f" (no slides matched pages='{pages}', so the first slides are shown)"
    if not wanted:
        wanted = list(range(1, min(total, default_n) + 1))
    truncated = len(wanted) > max_n
    return wanted[:max_n], selection_note, truncated


def queue_slide_images(
    context,
    *,
    assets,
    filename: str,
    total: int,
    pages: str | None,
    pathway: str,
    log_ref: str,
    default_n: int | None = None,
    max_n: int | None = None,
) -> SlideViewOutcome:
    """Queue the selected slides of a rendered deck as native image assets.

    ``assets`` is the owner's ``role=page_render`` Asset queryset (a version's or
    an attachment's); the slide selection is applied here. Stops at the first
    slide the run's native-asset budget refuses (``budget_hit``); slides with no
    stored render are reported as ``missing``.
    """
    if default_n is None or max_n is None:
        d, m = view_slide_limits()
        default_n = default_n if default_n is not None else d
        max_n = max_n if max_n is not None else m
    outcome = SlideViewOutcome(total=total, default_n=default_n, max_n=max_n)
    if total <= 0:
        return outcome
    wanted, outcome.selection_note, outcome.truncated = select_slides(
        total, pages, default_n=default_n, max_n=max_n,
    )
    by_number = {a.page_number: a for a in assets.filter(page_number__in=wanted)}
    for n in wanted:
        asset = by_number.get(n)
        if asset is None or not asset.blob:
            outcome.missing.append(n)
            continue
        try:
            with asset.blob.open("rb") as fh:
                data = fh.read()
        except Exception:
            logger.exception("slide_view: failed to read slide render %s for %s", n, log_ref)
            outcome.missing.append(n)
            continue
        # No asset_id: rendered slides are agent-only and never get an embed token.
        if not context.try_add_native_asset({
            "kind": "image",
            "asset_id": "",
            "b64": base64.b64encode(data).decode("ascii"),
            "media_type": asset.content_type or "image/jpeg",
            "description": f"'{filename}' slide {n} of {total}",
        }, pathway=pathway):
            outcome.budget_hit = True
            break
        outcome.shown.append(n)
    return outcome


__all__ = [
    "SlideViewOutcome",
    "format_numbers",
    "queue_slide_images",
    "select_slides",
    "view_slide_limits",
]
