"""The edit-point "current deck" block (main agent).

The deck JSON the model sees in its per-turn preamble is a pre-turn snapshot;
after a ``slide_canvas_write`` / ``slide_canvas_edit`` the only current copy is
scattered across tool-call arguments and results. At a mid-turn edit point the
tool loop (llm/pipelines/edit_points.py) calls ``RunContext.deck_block_provider``
and appends what it returns as a trailing user message — ONE authoritative copy,
re-read from the DB — so the slide tools can trim the deck JSON out of their
earlier arguments (chat/slide_tools.py) and the model edits against the real
canonical text instead of falling back to another full rewrite.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def make_deck_block_provider(thread_id: str, *, baseline: tuple | None) -> Callable[[], str]:
    """Return the zero-arg callable the tool loop invokes at edit points.

    ``baseline`` is ``chat.slides.service.deck_version(<deck the preamble
    rendered>)`` — ``None`` when the turn started without a deck. The provider
    returns ``""`` when there is no active deck or it is unchanged since the
    preamble (the snapshot the model already has is exact, so a block would be
    dead weight), and never raises: a failed read is logged and yields no block.
    """

    def provider() -> str:
        from django.db import close_old_connections, connection

        from chat.prompts import render_active_deck_block
        from chat.slides.service import active_deck_for_prompt, deck_version

        try:
            active, decks = active_deck_for_prompt(thread_id)
            if active is None or deck_version(active) == baseline:
                return ""
            return render_active_deck_block(active, decks)
        except Exception:  # noqa: BLE001 — best-effort; never break the turn
            logger.exception("Deck block provider failed for thread %s", thread_id)
            return ""
        finally:
            # Same hygiene as the tool pool (_execute_tool_calls._run_one): this
            # runs on the pipeline's producer thread. Skipped inside an atomic
            # block (tests), where closing would poison the transaction.
            if not connection.in_atomic_block:
                close_old_connections()

    return provider
