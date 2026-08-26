"""DB operations for slide decks (SlideSet) — the deck sibling of chat.services.

Simpler than the canvas service: decks are read-only for the user (no in-browser
edits, so no user-edit snapshot / accept-merge flow). Checkpoints exist for
guardrail rollback + Undo.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

from django.db import IntegrityError, transaction
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


def get_deck_by_title(thread_id, title: str):
    """The non-deleted deck with this exact title in the thread, or ``None``."""
    if not title:
        return None
    return SlideSet.objects.filter(
        thread_id=thread_id, title=title[:255], deleted_at__isnull=True
    ).first()


@contextmanager
def locked_deck(deck_pk):
    """Lock a deck row for the whole of a read-modify-write of its ``content``.

    Yields a FRESHLY READ deck inside a transaction. Callers must mutate the yielded
    instance and never the snapshot they resolved before entering: a tool batch runs
    concurrently in a ThreadPoolExecutor (llm/pipelines/simple_chat.py), so two calls
    that each deep-copy their own pre-call snapshot lose one of the two writes — and
    both still report success. Yields ``None`` when the deck was deleted in the race.

    Do NOT call :func:`activate_deck` inside the lock: it bulk-updates the thread's
    *sibling* deck rows, so holding this deck's lock while taking theirs is an inverted
    lock order between two threads working on different decks. Activation is UI focus,
    not part of the content write, so it belongs after the block.

    ``select_for_update`` is a silent no-op on SQLite (the test backend), so real
    locking exists only on Postgres — same caveat as accounts/services.py. Under
    PgBouncer's transaction pooling the lock is taken and released inside one
    transaction, so it behaves normally there.
    """
    with transaction.atomic():
        yield (
            SlideSet.objects.select_for_update()
            .filter(pk=deck_pk, deleted_at__isnull=True)
            .first()
        )


def _next_order(deck) -> int:
    # Callers mutating deck content run this inside locked_deck(), which serializes
    # the read so concurrent checkpoints can't land on the same order.
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
    """Create or overwrite a deck by title, checkpointing the result.

    Returns ``(deck, created, old_content)``. The checkpoint is written here rather
    than by the caller so it lands inside the same row lock as the overwrite and its
    ``order`` stays monotonic under concurrent writes.

    Raises :class:`DeckLimitError` when creating past the per-thread cap.
    """
    title = (title or "Untitled deck")[:255]
    lookup = (deck_name or title)[:255]

    def _overwrite(existing):
        # Re-read under the row lock so ``old`` is the content we actually replace
        # (it feeds changed_slide_ids) and the checkpoint order is serialized. The
        # content itself is last-writer-wins by design — this is a full rewrite.
        with locked_deck(existing.pk) as locked:
            deck = locked if locked is not None else existing
            old = deck.content
            deck.title = title
            deck.content = content
            deck.save(update_fields=["title", "content", "updated_at"])
            create_deck_checkpoint(deck, source="ai_edit", description="Full rewrite")
        return deck, False, old

    deck = SlideSet.objects.filter(
        thread_id=thread_id, title=lookup, deleted_at__isnull=True
    ).first()
    if deck is not None:
        return _overwrite(deck)

    count = SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True).count()
    if count >= schema.MAX_SLIDE_SETS_PER_THREAD:
        raise DeckLimitError(
            f"Maximum of {schema.MAX_SLIDE_SETS_PER_THREAD} decks per thread reached."
        )
    try:
        deck = SlideSet.objects.create(thread_id=thread_id, title=title, content=content)
    except IntegrityError:
        # Concurrent create of the same title — fall back to overwrite. Only the
        # create is guarded, so an unrelated IntegrityError can't be mistaken for
        # a title collision.
        deck = SlideSet.objects.filter(
            thread_id=thread_id, title=title, deleted_at__isnull=True
        ).first()
        if deck is None:
            # The colliding row vanished in the race; surface the real error
            # rather than an AttributeError on None.
            raise
        return _overwrite(deck)
    create_deck_checkpoint(deck, source="original", description="Created deck")
    return deck, True, None


def save_deck_content(deck, content: dict):
    deck.content = content
    deck.save(update_fields=["content", "updated_at"])


def soft_delete_deck(thread_id, deck):
    was_active = deck.is_active
    deck.deleted_at = timezone.now()
    deck.is_active = False
    deck.save(update_fields=["deleted_at", "is_active"])
    # If we just deleted the active deck, promote the newest surviving deck so
    # the thread still has an active deck and its other decks aren't orphaned
    # (the panel can reopen and switch to it instead of going dark).
    if was_active:
        nxt = (
            SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
            .order_by("-created_at")
            .first()
        )
        if nxt is not None:
            activate_deck(thread_id, nxt)


def restore_deck(thread_id, deck):
    deck.deleted_at = None
    # A live deck may have taken this title while this one was deleted; the
    # partial-unique (thread,title) constraint would then reject the restore.
    # Disambiguate up front so Undo/restore degrades gracefully.
    clash = (
        SlideSet.objects.filter(thread_id=thread_id, title=deck.title, deleted_at__isnull=True)
        .exclude(pk=deck.pk).exists()
    )
    if clash:
        base, n = deck.title, 2
        while SlideSet.objects.filter(
            thread_id=thread_id, title=f"{base} ({n})", deleted_at__isnull=True
        ).exists():
            n += 1
        deck.title = f"{base} ({n})"
        deck.save(update_fields=["deleted_at", "title"])
    else:
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
