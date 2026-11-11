"""DB operations for slide decks (SlideSet) — the deck sibling of chat.services.

Simpler than the canvas service: decks are read-only for the user (no in-browser
edits, so no user-edit snapshot / accept-merge flow). Checkpoints exist for
guardrail rollback + Undo.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager

from django.db import IntegrityError, transaction
from django.utils import timezone

from chat.models import SlideSet, SlideSetCheckpoint
from chat.slides import schema


class DeckLimitError(Exception):
    """Raised when the per-thread deck cap is reached."""


class DeckTitleTaken(DeckLimitError):
    """Raised when renaming a deck to a title another live deck in the thread has.

    A subclass of DeckLimitError so a caller that only handles that one still
    returns a readable message instead of a raw IntegrityError."""

    def __init__(self, title: str):
        self.title = title
        super().__init__(f"A deck titled '{title}' already exists in this thread.")


def get_active_deck(thread_id):
    return (
        SlideSet.objects.filter(thread_id=thread_id, is_active=True, deleted_at__isnull=True)
        .order_by("-last_activated_at")
        .first()
    )


def active_deck_for_prompt(thread_id):
    """``(active deck | None, [{"title", "active", "slides"} ...] for every live
    deck)`` — the per-turn preamble shows the active deck's outline (never its
    JSON); the list lets the model know what decks it can switch between with
    slide_canvas_activate. The active deck's entry also carries its ``outline``
    (:func:`deck_outline`), computed here because the prompt is rendered on the
    event loop, where the comment query can't run."""
    active = get_active_deck(thread_id)
    decks = []
    for d in (
        SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
        .order_by("created_at").only("title", "is_active", "content", "agent_seen")
    ):
        entry = {"title": d.title, "active": d.is_active, "slides": len((d.content or {}).get("slides") or [])}
        if active is not None and d.pk == active.pk:
            entry["outline"] = deck_outline(active)
        decks.append(entry)
    return active, decks


def live_deck_titles(thread_id) -> list[str]:
    """Titles of the thread's live (non-deleted) decks, oldest first."""
    return list(
        SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
        .order_by("created_at")
        .values_list("title", flat=True)
    )


def resolve_deck(thread_id, name: str | None = None):
    """Return ``(deck, error)``. ``name`` targets by title; else the active deck."""
    if name:
        deck = SlideSet.objects.filter(
            thread_id=thread_id, title=name, deleted_at__isnull=True
        ).first()
        if deck is None:
            return None, {
                "message": f"No deck named '{name}' in this thread.",
                "available_decks": live_deck_titles(thread_id),
            }
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

    Raises :class:`DeckLimitError` when creating past the per-thread cap, and
    :class:`DeckTitleTaken` when an overwrite would rename the deck to a title
    another live deck has (the deck is left untouched).
    """
    title = (title or "Untitled deck")[:255]
    lookup = (deck_name or title)[:255]

    def _overwrite(existing):
        # Re-read under the row lock so ``old`` is the content we actually replace
        # (it feeds changed_slide_ids) and the checkpoint order is serialized. The
        # content itself is last-writer-wins by design — this is a full rewrite.
        if title != existing.title and SlideSet.objects.filter(
            thread_id=thread_id, title=title, deleted_at__isnull=True,
        ).exclude(pk=existing.pk).exists():
            raise DeckTitleTaken(title)
        try:
            with locked_deck(existing.pk) as locked:
                deck = locked if locked is not None else existing
                old = deck.content
                deck.title = title
                deck.content = content
                deck.save(update_fields=["title", "content", "updated_at"])
                create_deck_checkpoint(deck, source="ai_edit", description="Full rewrite")
        except IntegrityError:
            # Another deck took the title between the check and the save (the
            # partial unique (thread, title) constraint); the atomic block rolled
            # the overwrite back.
            if title != existing.title:
                raise DeckTitleTaken(title) from None
            raise
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


def insert_slide(deck_pk, slide: dict, *, position: int = -1, description: str = ""):
    """Insert ``slide`` into a deck under its row lock, mint ids, validate, checkpoint.

    Returns ``(deck, old_content, content, idx, issues)``. ``deck`` is ``None`` when
    the deck was deleted in the race; non-empty ``issues`` means the insert would make
    the deck invalid and nothing was saved. ``old_content`` is what the insert
    replaced, read under the lock (the caller's pre-lock snapshot may predate a
    sibling call's write and would report its slides as changed again). The slide
    should carry no slide-level id (or one that is unique in the deck) —
    ``mint_ids`` only fills in missing ids.

    Everything that reads and rewrites deck.content happens under the lock, against
    content re-read inside it — a parallel insert in the same tool batch would
    otherwise overwrite this call's slide (and vice versa).
    """
    with locked_deck(deck_pk) as deck:
        if deck is None:
            return None, None, None, None, []

        old = deck.content or {}
        content = copy.deepcopy(old)
        content.setdefault("version", 1)
        content.setdefault("size", {"w": 960, "h": 540})
        slides = content.setdefault("slides", [])
        idx = len(slides) if position < 0 or position > len(slides) else position
        slides.insert(idx, slide)
        schema.mint_ids(content)

        issues = schema.validate_deck(content)
        if issues:
            return deck, old, content, idx, issues

        save_deck_content(deck, content)
        create_deck_checkpoint(deck, source="ai_edit", description=description)
    return deck, old, content, idx, []


# ---------------------------------------------------------------------------
# Per-slide operations (the slide-by-slide agent tools)
# ---------------------------------------------------------------------------
class SlideOpError(Exception):
    """A per-slide operation the model asked for can't be applied (unknown slide
    id, an edit that matched nothing, …). Raised inside the row lock, so the
    transaction rolls back and nothing is saved. ``extra`` rides along into the
    tool's error result (e.g. the failed edits)."""

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


def slide_index(content: dict, slide_id: str) -> int | None:
    for i, s in enumerate((content or {}).get("slides") or []):
        if isinstance(s, dict) and s.get("id") == slide_id:
            return i
    return None


def _clamp_position(position: int, length: int) -> int:
    return length if position < 0 or position > length else position


def _mutate_deck(deck_pk, mutate, *, description: str):
    """Run ``mutate(content)`` on a deep copy of the deck's content under its row
    lock, then validate, mint ids, save and checkpoint — the shared skeleton of
    every per-slide write (same locking rationale as :func:`insert_slide`).

    Returns ``(deck, old_content, new_content, issues)``: ``deck`` is ``None``
    when the deck was deleted in the race; non-empty ``issues`` means the result
    would be an invalid deck and nothing was saved. :class:`SlideOpError` from
    ``mutate`` propagates (nothing saved)."""
    with locked_deck(deck_pk) as deck:
        if deck is None:
            return None, None, None, []
        old = deck.content or {}
        content = copy.deepcopy(old)
        content.setdefault("version", 1)
        content.setdefault("size", {"w": 960, "h": 540})
        content.setdefault("slides", [])
        mutate(content)
        # Validate before minting: mint_ids assumes every slide/element is a dict.
        issues = schema.validate_deck(content)
        if issues:
            return deck, old, content, issues
        schema.mint_ids(content)
        save_deck_content(deck, content)
        create_deck_checkpoint(deck, source="ai_edit", description=description)
    return deck, old, content, []


def replace_slide(deck_pk, slide_id: str, slide: dict, *, description: str = ""):
    """Replace one slide wholesale, keeping its id (the payload's id is ignored)
    and, when the payload doesn't set one, its seed ``layout`` (models rewriting a
    slide rarely carry it over, and the outline would lose it)."""
    def _mutate(content):
        idx = slide_index(content, slide_id)
        if idx is None:
            raise SlideOpError(f"No slide with id '{slide_id}' in this deck.")
        new = copy.deepcopy(slide)
        new["id"] = slide_id
        if not new.get("layout") and content["slides"][idx].get("layout"):
            new["layout"] = content["slides"][idx]["layout"]
        content["slides"][idx] = new

    return _mutate_deck(deck_pk, _mutate, description=description or f"Rewrote slide {slide_id}")


def deck_settings(content: dict) -> dict:
    """The deck-level JSON — everything but ``slides`` (version, size, theme)."""
    return {k: v for k, v in (content or {}).items() if k != "slides"}


def edit_slides(deck_pk, edits: list[tuple[str, str, str]], *, description: str = ""):
    """Find/replace edits scoped to single slides: ``edits`` is
    ``[(slide_id, old_text, new_text), …]``; ``old_text`` must match exactly once in
    THAT slide's canonical text (``schema.canonical_deck_text(slide)``, what
    slides_read returns). ``slide_id == ""`` edits the deck settings instead.

    Like the old whole-deck edit, edits that match are applied and the misses are
    reported; but if any slide's edited text stops being valid JSON, changes a
    slide id, or the deck stops validating, NOTHING is saved. Returns the
    :func:`_mutate_deck` tuple plus ``(applied_ids, applied, failed)``:
    ``applied_ids`` are the slide ids that took at least one edit (a slide whose
    edits all missed is unchanged and must not count as written)."""
    from chat.edit_utils import apply_unique_text_edits

    groups: dict[str, list[tuple[str, str]]] = {}
    for slide_id, old_text, new_text in edits:
        groups.setdefault(slide_id or "", []).append((old_text, new_text))
    stats = {"applied": 0, "failed": [], "applied_ids": []}

    def _mutate(content):
        for slide_id, pairs in groups.items():
            if slide_id:
                idx = slide_index(content, slide_id)
                if idx is None:
                    stats["failed"].extend(
                        {"slide_id": slide_id, "old_text": (o or "")[:80], "error": "No slide with this id."}
                        for o, _n in pairs
                    )
                    continue
                original = schema.canonical_deck_text(content["slides"][idx])
            else:
                original = schema.canonical_deck_text(deck_settings(content))
            new_text, applied, failed = apply_unique_text_edits(original, pairs)
            stats["failed"].extend({"slide_id": slide_id, **f} for f in failed)
            if not applied:
                continue
            where = f"slide '{slide_id}'" if slide_id else "the deck settings"
            try:
                parsed = json.loads(new_text)
            except (ValueError, TypeError) as exc:
                raise SlideOpError(
                    f"Edits to {where} would produce invalid JSON ({exc}); deck left unchanged. "
                    "Widen your old_text so replacements keep the JSON well-formed."
                )
            if not isinstance(parsed, dict):
                raise SlideOpError(f"Edits to {where} must leave a JSON object; deck left unchanged.")
            if slide_id:
                if parsed.get("id") != slide_id:
                    raise SlideOpError(
                        f"Edits to slide '{slide_id}' changed its id; ids are permanent. Deck left unchanged."
                    )
                content["slides"][idx] = parsed
            else:
                if "slides" in parsed:
                    raise SlideOpError("Deck-settings edits can't add a 'slides' key; deck left unchanged.")
                slides = content["slides"]
                content.clear()
                content.update(parsed)
                content["slides"] = slides
            stats["applied"] += applied
            if slide_id:
                stats["applied_ids"].append(slide_id)
        if not stats["applied"]:
            raise SlideOpError("No edits applied.", failed=stats["failed"])

    result = _mutate_deck(
        deck_pk, _mutate, description=description or f"Edited {len(edits)} section(s)"
    )
    return (*result, stats["applied_ids"], stats["applied"], stats["failed"])


def move_slides(deck_pk, slide_ids: list[str], position: int):
    """Move ``slide_ids`` (in the given order) so the first lands at 0-based
    ``position`` of the resulting deck (-1 = the end)."""
    def _mutate(content):
        slides = content["slides"]
        missing = [sid for sid in slide_ids if slide_index(content, sid) is None]
        if missing:
            raise SlideOpError("No slide with id: " + ", ".join(missing) + ".")
        by_id = {s.get("id"): s for s in slides}
        moving = [by_id[sid] for sid in dict.fromkeys(slide_ids)]
        rest = [s for s in slides if s.get("id") not in set(slide_ids)]
        at = _clamp_position(position, len(rest))
        content["slides"] = rest[:at] + moving + rest[at:]

    return _mutate_deck(deck_pk, _mutate, description=f"Moved {len(slide_ids)} slide(s)")


def delete_slides(deck_pk, slide_ids: list[str]):
    """Remove ``slide_ids``. Refuses to empty the deck (delete the deck instead)."""
    def _mutate(content):
        missing = [sid for sid in slide_ids if slide_index(content, sid) is None]
        if missing:
            raise SlideOpError("No slide with id: " + ", ".join(missing) + ".")
        remaining = [s for s in content["slides"] if s.get("id") not in set(slide_ids)]
        if not remaining:
            raise SlideOpError(
                "That would delete every slide. Keep at least one, or delete the deck with slide_canvas_delete."
            )
        content["slides"] = remaining

    return _mutate_deck(deck_pk, _mutate, description=f"Deleted {len(slide_ids)} slide(s)")


def duplicate_slide(deck_pk, slide_id: str, position: int = -1):
    """Copy a slide (a fresh slide id is minted; element ids are per-slide, so they
    stay) to 0-based ``position``; -1 = right after the original. Returns the
    :func:`_mutate_deck` tuple plus the copy's index."""
    out = {"idx": None}

    def _mutate(content):
        idx = slide_index(content, slide_id)
        if idx is None:
            raise SlideOpError(f"No slide with id '{slide_id}' in this deck.")
        copy_ = copy.deepcopy(content["slides"][idx])
        copy_.pop("id", None)
        at = idx + 1 if position < 0 else _clamp_position(position, len(content["slides"]))
        content["slides"].insert(at, copy_)
        out["idx"] = at

    result = _mutate_deck(deck_pk, _mutate, description=f"Duplicated slide {slide_id}")
    return (*result, out["idx"])


# -- What the agent has seen (outline flags) -----------------------------------
def slide_seen_hash(slide: dict) -> str:
    """Hash of one slide's canonical JSON — position and theme deliberately left
    out (the outline shows the theme separately)."""
    return hashlib.sha256(schema.canonical_deck_text(slide).encode("utf-8")).hexdigest()[:16]


def record_agent_slide_view(deck_pk, slides: list[dict], *, wrote: bool = False) -> None:
    """Stamp ``SlideSet.agent_seen`` for slides whose JSON the agent just read or
    wrote (``wrote`` also moves the slide's write time, which addresses the sent
    comments on it). Prunes entries of slides that no longer exist. Uses
    ``.update()`` so ``updated_at`` (the deck's content version) doesn't move."""
    if not slides:
        return
    now = timezone.now().isoformat()
    with locked_deck(deck_pk) as deck:
        if deck is None:
            return
        live = {s.get("id") for s in (deck.content or {}).get("slides") or [] if isinstance(s, dict)}
        seen = {k: v for k, v in (deck.agent_seen or {}).items() if k in live}
        for slide in slides:
            sid = slide.get("id") if isinstance(slide, dict) else None
            if not sid or sid not in live:
                continue
            entry = dict(seen.get(sid) or {})
            entry["h"] = slide_seen_hash(slide)
            if wrote:
                entry["w"] = now
            seen[sid] = entry
        SlideSet.objects.filter(pk=deck.pk).update(agent_seen=seen)


def _parse_time(value):
    from datetime import datetime

    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def slide_title(slide: dict) -> str:
    """The slide's display title: its ``name``, else the first headline text."""
    name = (slide.get("name") or "").strip()
    if name:
        return name
    for el in slide.get("elements") or []:
        if isinstance(el, dict) and el.get("type") == "text" and el.get("class") in ("headline", "title"):
            text = " ".join(
                (r.get("t") or "") for p in el.get("paragraphs") or [] for r in p.get("runs") or []
            ).strip()
            if text:
                return text[:80]
    return "(untitled)"


def deck_outline(deck, *, with_comments: bool = True) -> list[dict]:
    """One entry per slide for the agent: ``{n, id, layout, title, seen, comments}``.

    ``seen`` is ``"never"`` (the agent has no record of this slide's JSON),
    ``"changed"`` (it changed since the agent last read/wrote it) or ``""``.
    ``comments`` are the texts of SENT comments on the slide the agent hasn't
    acted on yet (sent after its last write to the slide)."""
    from chat.models import SlideComment

    slides = [s for s in (deck.content or {}).get("slides") or [] if isinstance(s, dict)]
    seen = getattr(deck, "agent_seen", None) or {}
    pending: dict[str, list[str]] = {}
    if with_comments and slides and getattr(deck, "pk", None):
        rows = (
            SlideComment.objects.filter(
                slide_set_id=deck.pk, status=SlideComment.Status.SENT, sent_at__isnull=False
            ).order_by("created_at").values_list("slide_id", "text", "sent_at")
        )
        for sid, text, sent_at in rows:
            written = _parse_time((seen.get(sid) or {}).get("w"))
            if written and sent_at <= written:
                continue
            pending.setdefault(sid, []).append(text)
    out = []
    for i, slide in enumerate(slides):
        sid = slide.get("id") or ""
        entry = seen.get(sid)
        if not entry:
            state = "never"
        elif entry.get("h") != slide_seen_hash(slide):
            state = "changed"
        else:
            state = ""
        out.append({
            "n": i + 1, "id": sid, "layout": slide.get("layout") or "custom",
            "title": slide_title(slide), "seen": state, "comments": pending.get(sid, []),
        })
    return out


def compact_outline(content: dict) -> list[str]:
    """``["1. s1 title \"Q3 review\"", …]`` — the outline a structural tool result
    carries so the model knows the new order without re-reading the deck."""
    return [
        f'{i + 1}. {s.get("id")} {s.get("layout") or "custom"} "{slide_title(s)}"'
        for i, s in enumerate((content or {}).get("slides") or []) if isinstance(s, dict)
    ]


def unique_deck_title(thread_id, base: str, *, exclude_pk=None) -> str:
    """``base``, or ``base (2)``, ``base (3)``… — the first title no live deck uses."""
    base = (base or "Untitled deck")[:255]
    live = SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
    if exclude_pk is not None:
        live = live.exclude(pk=exclude_pk)
    if not live.filter(title=base).exists():
        return base
    stem = base[:245]  # leave room for the " (n)" suffix within the 255-char title
    n = 2
    while live.filter(title=f"{stem} ({n})").exists():
        n += 1
    return f"{stem} ({n})"


def soft_delete_deck(thread_id, deck):
    """Soft-delete ``deck``; returns the deck promoted to active in its place
    (or ``None`` when it wasn't active, or no other deck remains)."""
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
        return nxt
    return None


def deleted_decks(thread_id, *, limit: int = 10):
    """The thread's soft-deleted decks, most recently deleted first."""
    return list(
        SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=False)
        .order_by("-deleted_at")[:limit]
    )


def restore_deck(thread_id, deck):
    deck.deleted_at = None
    # A live deck may have taken this title while this one was deleted; the
    # partial-unique (thread,title) constraint would then reject the restore.
    # Disambiguate up front so Undo/restore degrades gracefully.
    title = unique_deck_title(thread_id, deck.title, exclude_pk=deck.pk)
    if title != deck.title:
        deck.title = title
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
