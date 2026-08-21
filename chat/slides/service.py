"""DB operations for slide decks (SlideSet) — the deck sibling of chat.services.

Simpler than the canvas service: decks are read-only for the user (no in-browser
edits, so no user-edit snapshot / accept-merge flow). Checkpoints exist for
guardrail rollback + Undo.
"""

from __future__ import annotations

import copy

from django.db import IntegrityError
from django.utils import timezone

from chat.models import SlideSet, SlideSetCheckpoint
from chat.slides import schema


class DeckLimitError(Exception):
    """Raised when the per-thread deck cap is reached."""


def get_active_deck(thread_id):
    return (
        SlideSet.objects.filter(thread_id=thread_id, is_active=True, deleted_at__isnull=True)
        .order_by("-last_activated_at")
        .first()
    )


def resolve_deck(thread_id, name: str | None = None):
    """Return ``(deck, error)``. ``name`` targets by title; else the active deck."""
    if name:
        deck = SlideSet.objects.filter(
            thread_id=thread_id, title=name, deleted_at__isnull=True
        ).first()
        if deck is None:
            available = list(
                SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
                .order_by("created_at")
                .values_list("title", flat=True)
            )
            return None, {"message": f"No deck named '{name}' in this thread.", "available_decks": available}
        return deck, None
    return get_active_deck(thread_id), None


def _next_order(deck) -> int:
    last = deck.checkpoints.order_by("-order").first()
    return (last.order + 1) if last else 0


def create_deck_checkpoint(deck, *, source: str, description: str = ""):
    return SlideSetCheckpoint.objects.create(
        slide_set=deck,
        title=deck.title,
        content=copy.deepcopy(deck.content),
        source=source,
        description=description,
        order=_next_order(deck),
    )


def activate_deck(thread_id, deck):
    """Make ``deck`` the single active deck for the thread."""
    SlideSet.objects.filter(thread_id=thread_id, is_active=True).exclude(pk=deck.pk).update(
        is_active=False
    )
    deck.is_active = True
    deck.last_activated_at = timezone.now()
    deck.save(update_fields=["is_active", "last_activated_at"])


def set_active_decks(thread_id, names: list[str]):
    """Activate the named decks (capped at MAX_ACTIVE_SLIDE_SETS). Returns
    ``(activated, errors)``."""
    found, errors = [], []
    for name in names[: schema.MAX_ACTIVE_SLIDE_SETS]:
        deck = SlideSet.objects.filter(
            thread_id=thread_id, title=name, deleted_at__isnull=True
        ).first()
        if deck is None:
            errors.append(f"No deck named '{name}'.")
        else:
            found.append(deck)
    SlideSet.objects.filter(thread_id=thread_id, is_active=True).update(is_active=False)
    now = timezone.now()
    for deck in found:
        deck.is_active = True
        deck.last_activated_at = now
        deck.save(update_fields=["is_active", "last_activated_at"])
    return found, errors


def write_deck(thread_id, *, title: str, content: dict, deck_name: str = ""):
    """Create or overwrite a deck by title. Returns ``(deck, created, old_content)``.

    Raises :class:`DeckLimitError` when creating past the per-thread cap.
    """
    title = (title or "Untitled deck")[:255]
    lookup = (deck_name or title)[:255]
    deck = SlideSet.objects.filter(
        thread_id=thread_id, title=lookup, deleted_at__isnull=True
    ).first()
    if deck is not None:
        old = deck.content
        deck.title = title
        deck.content = content
        deck.save(update_fields=["title", "content", "updated_at"])
        return deck, False, old

    count = SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True).count()
    if count >= schema.MAX_SLIDE_SETS_PER_THREAD:
        raise DeckLimitError(
            f"Maximum of {schema.MAX_SLIDE_SETS_PER_THREAD} decks per thread reached."
        )
    try:
        deck = SlideSet.objects.create(thread_id=thread_id, title=title, content=content)
        return deck, True, None
    except IntegrityError:
        # Concurrent create of the same title — fall back to overwrite.
        deck = SlideSet.objects.filter(
            thread_id=thread_id, title=title, deleted_at__isnull=True
        ).first()
        old = deck.content if deck else None
        deck.content = content
        deck.save(update_fields=["content", "updated_at"])
        return deck, False, old


def save_deck_content(deck, content: dict):
    deck.content = content
    deck.save(update_fields=["content", "updated_at"])


def soft_delete_deck(thread_id, deck):
    deck.deleted_at = timezone.now()
    deck.is_active = False
    deck.save(update_fields=["deleted_at", "is_active"])


def restore_deck(thread_id, deck):
    deck.deleted_at = None
    deck.save(update_fields=["deleted_at"])


def revert_deck_to_checkpoint_before(deck, before) -> bool:
    """Revert deck content to the newest checkpoint created before ``before``.

    Used by the guardrail rollback path. Returns True if reverted.
    """
    cp = (
        deck.checkpoints.filter(created_at__lt=before).order_by("-order").first()
    )
    if cp is None:
        return False
    deck.content = copy.deepcopy(cp.content)
    deck.title = cp.title or deck.title
    deck.save(update_fields=["content", "title", "updated_at"])
    create_deck_checkpoint(deck, source="redacted", description="Reverted (content blocked)")
    return True
