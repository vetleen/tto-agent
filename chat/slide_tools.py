"""Slide-deck tools: author and edit a per-thread slide deck as JSON, slide by slide.

The deck sibling of chat/canvas_tools.py. A deck is one canonical JSON document
(chat.slides.schema); the user only ever sees the rendered filmstrip. The agent's
context carries only the active deck's OUTLINE (chat/prompts.py); it reads
slides with ``slides_read`` and writes them with ``slides_write_slide`` /
``slide_canvas_edit`` (per-slide find/replace) / the move, delete and duplicate
tools — ``slide_canvas_write`` stays for full rebuilds. All tools are
audience="main", section="skills" (gated behind the "Slide Deck Tools" seed skill).
"""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO

from pydantic import BaseModel, Field, field_validator

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry
from llm.tools.interfaces import ArgTrim, drop_args, edits_description, omitted_note

# The slides_add_slide layout ids, sourced from the seed catalogue so this
# always matches what actually exists (no hand-maintained list to go stale).
from chat.slides.layouts import layout_catalog

_LAYOUT_IDS = ", ".join(c["id"] for c in layout_catalog())
# A one-line-per-layout menu (id: what it's for) so the model picks confidently.
_LAYOUT_MENU = "\n".join(f'- {c["id"]}: {c["description"]}' for c in layout_catalog())


# ---------------------------------------------------------------------------
# Argument / result trimming (edit points / between turns)
# ---------------------------------------------------------------------------
# No deck JSON is ever injected into the prompt (the agent sees an outline and
# reads slides with slides_read), so slide JSON in an OLD call's arguments or
# result is always recoverable by a re-read — the canvas model
# (chat/canvas_tools.py). Recency still protects recent calls as usual.
SLIDES_READ_WHERE = "Re-read the current slides with slides_read."


def trim_slide_result_fields(result: str, fields: dict) -> str | None:
    """Shrink a deck tool's RESULT: remove the big JSON field(s) in ``fields``
    (``{key: what-it-was}``) of a ``status: ok`` result and say so in an
    ``omitted_fields_note`` field. The note never takes the value's place: the
    model copies slide JSON out of these results into its writes. Every other
    field is kept; ``None`` leaves the result alone (idempotent)."""
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("status") != "ok":
        return None
    big = {key: what for key, what in fields.items() if data.get(key) not in (None, "", [])}
    if not big:
        return None
    kept = {k: v for k, v in data.items() if k not in big}
    kept["omitted_fields_note"] = omitted_note(
        f"{' and '.join(f'the {what}' for what in big.values())} this result carried "
        f"({', '.join(big)})", SLIDES_READ_WHERE,
    )
    return json.dumps(kept, ensure_ascii=False)


def _slide_ref(content: dict, slide_id: str) -> str:
    """``'slide 4 · Revenue by quarter'`` for labels and results."""
    from chat.slides import service

    idx = service.slide_index(content, slide_id)
    if idx is None:
        return f"slide {slide_id}"
    return f"slide {idx + 1} · {service.slide_title(content['slides'][idx])}"


def _numbers_label(numbers: list[int]) -> str:
    """``[3, 4, 5]`` -> ``"3–5"``, ``[2, 7]`` -> ``"2, 7"``."""
    numbers = sorted(set(numbers))
    if len(numbers) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}–{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def _deck_result(deck, old: dict | None, new: dict, **extra) -> dict:
    """The fields every deck-mutating result carries: the consumer turns them into
    a ``slidedeck.updated`` event (filmstrip + render queue); ``outline`` tells
    the model the resulting order without a re-read."""
    from chat.slides import schema, service

    slide_ids = [s.get("id") for s in new.get("slides") or []]
    return {
        "status": "ok",
        "deck_id": str(deck.pk),
        "title": deck.title,
        **extra,
        "slide_ids": slide_ids,
        "changed_slide_ids": schema.changed_slide_ids(old, new),
        "slide_count": len(slide_ids),
        "outline": service.compact_outline(new),
    }


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------
class ActivateDeckInput(ReasonBaseModel):
    deck_names: list[str] = Field(
        description="Deck title(s) to make active (max 1). The active deck's outline is in your context.",
    )


class CreateDeckInput(ReasonBaseModel):
    title: str = Field(description="Title for the new deck.")
    layouts: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered layout ids, one per slide — the deck's storyboard. Each one of: "
            + _LAYOUT_IDS + ". Omit or pass [] to create an empty deck."
        ),
    )


class WriteDeckInput(ReasonBaseModel):
    content: dict | str | None = Field(
        default=None,
        description=(
            "The full replacement deck as a JSON OBJECT — pass it directly as structured JSON, "
            "do NOT wrap it in a string or escape the quotes: "
            '{"version":1,"size":{"w":960,"h":540},"slides":[...]}. Coordinates are points; '
            "the slide is 960x540. KEEP the ids of slides/elements you carry over (the user "
            "comments by id); leave ids off anything new and they are minted."
        ),
    )
    deck_name: str = Field(
        default="",
        description="Title of the deck to rewrite. Empty = the active deck.",
    )
    rename_to: str = Field(
        default="",
        description=(
            "New title for the deck. Empty (the default) = keep its title. Only for renaming — "
            "never needed to write content, and it must not be another deck's title."
        ),
    )


class DeckEditItem(BaseModel):
    slide_id: str = Field(
        default="",
        description=(
            "Id of the slide this edit applies to (from the outline / slides_read). "
            "Empty = the deck settings (version, size, theme)."
        ),
    )
    old_text: str = Field(
        description="Exact JSON text to find — must be unique within that slide's JSON as slides_read returned it.",
    )
    new_text: str = Field(description="Replacement JSON text.")
    reason: str = Field(default="", description="Brief reason for this edit.")


class EditDeckInput(ReasonBaseModel):
    edits: list[DeckEditItem] = Field(
        description="Targeted find-replace edits, each scoped to one slide's canonical JSON text.",
    )
    slide_id: str = Field(
        default="",
        description="Slide for the edits that don't name their own `slide_id` (all edits on one slide).",
    )
    deck_name: str = Field(default="", description="Deck title to edit. Empty = the active deck.")

    @field_validator("edits", mode="before")
    @classmethod
    def _coerce_edits(cls, value):
        # Tolerate a JSON-stringified array and drop non-actionable items — same
        # leniency as canvas_edit (WILFRED-40 / WILFRED-6A).
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return value
        if isinstance(value, list):
            return [
                item
                for item in value
                if not isinstance(item, dict) or (item.get("old_text") and "new_text" in item)
            ]
        return value


class DeleteDeckInput(ReasonBaseModel):
    deck_name: str = Field(description="Exact title of the deck to delete.")


class AddSlideInput(ReasonBaseModel):
    layout: str = Field(
        description="Layout id to seed. One of: " + _LAYOUT_IDS + ".",
    )
    position: int = Field(
        default=-1,
        description="0-based insertion index; -1 (default) appends at the end.",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class ReadSlidesInput(ReasonBaseModel):
    slide_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the slides to read (from the outline). Takes precedence over start/end.",
    )
    start: int | None = Field(
        default=None, ge=1,
        description="First slide NUMBER to read (1-based, as in the outline). Default 1.",
    )
    end: int | None = Field(
        default=None, ge=1,
        description="Last slide number to read (inclusive). Default: as many as fit in one page.",
    )
    deck_settings: bool = Field(
        default=False,
        description="Also return the deck-level JSON (version, size, theme overrides).",
    )
    include_render: bool = Field(
        default=False,
        description="Also render the returned slides and show you the images (max 4, slower).",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class WriteSlideInput(ReasonBaseModel):
    content: dict | str | None = Field(
        default=None,
        description=(
            "The full JSON of ONE slide as a JSON OBJECT (not a string): "
            '{"name": "...", "elements": [...], ...}. Coordinates are points on a 960x540 slide.'
        ),
    )
    slide_id: str = Field(
        default="",
        description="Id of the slide to replace (keeps its id and position). Empty = insert a new slide.",
    )
    position: int = Field(
        default=-1,
        description="For a new slide: 0-based insertion index (slide number N is position N-1); -1 appends.",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class MoveSlidesInput(ReasonBaseModel):
    slide_ids: list[str] = Field(description="Ids of the slides to move, in the order they should end up.")
    position: int = Field(
        description="0-based index in the resulting deck where the first moved slide lands; -1 = the end.",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class DeleteSlidesInput(ReasonBaseModel):
    slide_ids: list[str] = Field(description="Ids of the slides to delete.")
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class DuplicateSlideInput(ReasonBaseModel):
    slide_id: str = Field(description="Id of the slide to copy.")
    position: int = Field(
        default=-1,
        description="0-based index for the copy; -1 (default) = right after the original.",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class PreviewSlidesInput(ReasonBaseModel):
    slide_ids: list[str] = Field(
        default_factory=list,
        description="Slide ids to render and look at (max 4). Empty = the whole deck (capped).",
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


class ListThemesInput(ReasonBaseModel):
    pass


class SetThemeInput(ReasonBaseModel):
    theme: str = Field(
        description="Theme name or id to apply (from slides_list_themes). 'forest' is the built-in default."
    )
    deck_name: str = Field(default="", description="Deck title. Empty = the active deck.")


def _thread_id(tool) -> str | None:
    return tool.context.conversation_id if tool.context else None


def _tool_org(tool):
    """The acting user's organization (for org-default seeding), or ``None``.

    RunContext carries ``user_id`` but not the org, so resolve it here. Best-effort:
    any lookup failure (no context, no membership) yields ``None`` -> no org default.
    """
    user_id = tool.context.user_id if tool.context else None
    if not user_id:
        return None
    try:
        from accounts.models import Membership

        m = (
            Membership.objects.filter(user_id=user_id)
            .select_related("org")
            .first()
        )
        return m.org if m else None
    except Exception:  # noqa: BLE001 — seeding is best-effort
        return None


def _tool_user(tool):
    """The acting user object (for user-scoped themes), or ``None`` (best-effort)."""
    user_id = tool.context.user_id if tool.context else None
    if not user_id:
        return None
    try:
        from django.contrib.auth import get_user_model

        return get_user_model().objects.filter(id=user_id).first()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class ActivateDeckTool(ContextAwareTool):
    name: str = "slide_canvas_activate"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Opening deck..."
    end_label: str = "Deck opened"
    description: str = (
        "Switch which of the thread's decks is the active one you work on. Only one deck is active "
        "at a time; the active deck's outline is in your context (read slides with slides_read), "
        "and deck tools called without deck_name work on it. "
        "Activating a deck deactivates the others, but they stay saved. Runs before the other "
        "calls in the same batch."
    )
    args_schema: type[BaseModel] = ActivateDeckInput

    def runs_first(self, args: dict) -> bool:
        return True  # the batch's other deck calls default to the active deck

    def _run(self, deck_names: list[str], **kwargs) -> str:
        from chat.slides import service, schema

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})
        if len(deck_names) > schema.MAX_ACTIVE_SLIDE_SETS:
            return json.dumps({
                "status": "error",
                "message": f"You can activate at most {schema.MAX_ACTIVE_SLIDE_SETS} deck(s) at a time.",
            })
        activated, errors = service.set_active_decks(thread_id, deck_names)
        result = {
            "status": "ok",
            "activated": [{"title": d.title, "deck_id": str(d.pk)} for d in activated],
        }
        if errors:
            result["errors"] = errors
        return json.dumps(result)


class CreateDeckTool(ContextAwareTool):
    name: str = "slides_create_deck"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Creating deck..."
    end_label: str = "Created the deck"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") == "ok":
            n = result.get("slide_count", 0)
            return f"Created a deck with {n} slide(s)" if n else "Created an empty deck"
        return None

    description: str = (
        "Create a new slide deck — the only way a deck comes into existence — and make it the "
        "active one (the user sees a rendered preview, never the JSON). Pass `layouts`, an ordered "
        "list of pre-designed layout ids (one per slide), as the deck's storyboard: plan the "
        "narrative first, then pick one layout per point. The slides are seeded with placeholder "
        "text and the user watches them appear; the result lists each slide's id and layout. Then "
        "fill the slides in one at a time — slides_write_slide with the full slide, or slides_read "
        "+ slide_canvas_edit for small changes. Runs before the other calls in the same batch, so "
        "they can already target the new deck. Keep one deck per conversation unless the user "
        "asks for more; to start over, rework this deck (slide_canvas_write) rather than creating "
        "another. Available layouts:\n" + _LAYOUT_MENU
    )
    args_schema: type[BaseModel] = CreateDeckInput

    def runs_first(self, args: dict) -> bool:
        return True

    def _run(self, title: str, layouts=None, **kwargs) -> str:
        from chat.slides import layouts as layouts_mod
        from chat.slides import schema, service
        from chat.slides import theme as theme_mod

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        if service.get_deck_by_title(thread_id, title):
            return json.dumps({
                "status": "error",
                "message": f'A deck titled "{title}" already exists. Pick a different title, or '
                           "rework the existing deck with slide_canvas_edit / slide_canvas_write.",
            })

        # All-or-nothing on layout ids: a partial deck from a typo'd storyboard is
        # worse than an error the model can correct and retry.
        wanted = list(layouts or [])
        slides = []
        unknown = []
        for lid in wanted:
            seed = layouts_mod.get_layout(lid)
            if seed is None:
                unknown.append(lid)
            else:
                slides.append(seed)
        if unknown:
            return json.dumps({
                "status": "error",
                "message": "Unknown layout(s): " + ", ".join(unknown) + ". No deck was created.",
                "available_layouts": [c["id"] for c in layouts_mod.layout_catalog()],
            })

        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": slides}
        schema.mint_ids(deck)

        # New-deck default theme: user's default -> org default -> Forest. Forest is
        # the base theme, so leave the deck theme unset for it (a plain deck stays
        # lean); only a non-default org/user theme is written.
        default = theme_mod.default_theme_for(_tool_user(self), _tool_org(self))
        if default.get("id") != "forest":
            deck["theme"] = theme_mod.theme_to_deck_override(default)

        issues = schema.validate_deck(deck)
        if issues:
            return json.dumps({
                "status": "error",
                "message": "The assembled deck is invalid.",
                "issues": issues[:20],
            })

        try:
            obj, _created, _old = service.write_deck(thread_id, title=title, content=deck)
        except service.DeckLimitError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        service.activate_deck(thread_id, obj)
        # No slide JSON is returned: the agent reads (or writes) each slide next.
        return json.dumps(_deck_result(obj, None, deck), ensure_ascii=False)


class WriteDeckTool(ContextAwareTool):
    name: str = "slide_canvas_write"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Writing slides..."
    end_label: str = "Wrote the slide deck"
    description: str = (
        "Completely restructure an EXISTING deck by replacing its entire JSON — a rework tool for "
        "when the deck needs rebuilding around a new narrative, not how decks are created (use "
        "slides_create_deck for that) and not for changes to some slides (use slides_write_slide, "
        "slide_canvas_edit, or the move/delete/duplicate tools). Read every slide you carry over "
        "with slides_read first, and KEEP the ids of slides/elements you carry over — the user "
        "comments by id. The deck/element JSON format is documented in the Slide Deck Tools skill."
    )
    args_schema: type[BaseModel] = WriteDeckInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        if args.get("content") is None:
            return None
        return drop_args(args, {"content": "the deck JSON"}, SLIDES_READ_WHERE)

    def runs_first(self, args: dict) -> bool:
        # A rename changes the title the batch's other calls target by.
        return bool(((args or {}).get("rename_to") or "").strip())

    def _run(self, content=None, deck_name: str = "", rename_to: str = "", **kwargs) -> str:
        # ``title`` was the old rename parameter: it renamed the target deck
        # silently whenever it differed (2026-11-11 prod: the agent passed another
        # deck's title, renamed its real deck, then deleted it as "the old one").
        # An old call replaying it is ignored; renames go through rename_to only.
        from chat.slides import schema, service

        rename_to = (rename_to or "").strip()[:255]
        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        target, err = service.resolve_deck(thread_id, deck_name or None)
        if err:
            return json.dumps({"status": "error", **err})
        if target is None:
            return json.dumps({
                "status": "error",
                "message": "No deck exists for this thread. Create one with slides_create_deck first.",
            })

        # Accept the deck as a native object (preferred — no escaping) or a JSON
        # string (back-compat, under either name).
        if content is None:
            content = kwargs.get("content_json")
        if content is None:
            return json.dumps({
                "status": "error",
                "message": "content is required — pass the full replacement deck JSON. "
                           "(New decks are created with slides_create_deck.)",
            })
        if isinstance(content, str):
            try:
                deck = json.loads(content)
            except (ValueError, TypeError) as exc:
                return json.dumps({"status": "error", "message": f"content is not valid JSON: {exc}"})
        elif isinstance(content, dict):
            deck = content
        else:
            return json.dumps({"status": "error", "message": "content must be a JSON object (the full deck)."})

        # Validate the raw model-supplied deck BEFORE minting ids: mint_ids assumes
        # every slide/element is a dict, so a malformed deck (e.g. a slide sent as a
        # bare string) must be rejected with structured issues here, not crash mid-mint.
        # Mirrors slide_canvas_edit, which already validates before it mints.
        issues = schema.validate_deck(deck)
        if issues:
            return json.dumps({"status": "error", "message": "Deck JSON is invalid.", "issues": issues[:20]})

        schema.mint_ids(deck)
        # Carry each kept slide's seed layout over (outline only; models drop it).
        old_layouts = {
            s.get("id"): s.get("layout") for s in (target.content or {}).get("slides") or []
            if isinstance(s, dict) and s.get("layout")
        }
        for s in deck.get("slides") or []:
            if not s.get("layout") and old_layouts.get(s.get("id")):
                s["layout"] = old_layouts[s["id"]]

        # Keep the deck's theme across a rewrite when the model didn't set one (so a
        # user's picker choice survives); a theme-less deck falls back to the default
        # like at creation. A deck that carries an explicit theme is left untouched.
        # (The theme applied here is a trusted stored/builtin override, so it does not
        # need re-validation.)
        if not deck.get("theme"):
            from chat.slides import theme as theme_mod

            prev_theme = (target.content or {}).get("theme")
            if prev_theme:
                deck["theme"] = prev_theme
            else:
                default = theme_mod.default_theme_for(_tool_user(self), _tool_org(self))
                if default.get("id") != "forest":
                    deck["theme"] = theme_mod.theme_to_deck_override(default)

        try:
            # write_deck checkpoints the result itself, inside the row lock it takes
            # for the overwrite path, so the checkpoint order stays monotonic. Looking
            # the target up by its current title pins the overwrite path; an empty
            # `rename_to` keeps that name.
            old_title = target.title
            obj, created, old = service.write_deck(
                thread_id, title=(rename_to or old_title), content=deck, deck_name=old_title
            )
        except service.DeckTitleTaken as exc:
            return json.dumps({
                "status": "error",
                "message": (
                    f"A deck titled '{exc.title}' already exists in this thread, so '{old_title}' "
                    "was not renamed and nothing was written. Pick another title for rename_to, or "
                    f"write to that deck instead with deck_name='{exc.title}'."
                ),
                "available_decks": service.live_deck_titles(thread_id),
            }, ensure_ascii=False)
        except service.DeckLimitError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        service.activate_deck(thread_id, obj)
        service.record_agent_slide_view(obj.pk, deck.get("slides") or [], wrote=True)
        extra = {"renamed_from": old_title} if obj.title != old_title else {}
        return json.dumps(_deck_result(obj, None if created else old, deck, **extra), ensure_ascii=False)


def _resolve_target(tool, deck_name: str = ""):
    """``(thread_id, deck, error_json)`` for a tool that works on one deck."""
    from chat.slides import service

    thread_id = _thread_id(tool)
    if not thread_id:
        return None, None, json.dumps({"status": "error", "message": "No thread context available."})
    deck, err = service.resolve_deck(thread_id, deck_name or None)
    if err:
        return thread_id, None, json.dumps({"status": "error", **err})
    if deck is None:
        return thread_id, None, json.dumps({
            "status": "error",
            "message": "No deck exists for this thread. Create one with slides_create_deck first.",
        })
    return thread_id, deck, None


def _op_error(exc) -> str:
    return json.dumps({"status": "error", "message": exc.message, **exc.extra}, ensure_ascii=False)


def _parse_slide_payload(content):
    """``(slide_dict, error_json)`` from a tool's ``content`` (object or JSON string)."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (ValueError, TypeError) as exc:
            return None, json.dumps({"status": "error", "message": f"content is not valid JSON: {exc}"})
    if not isinstance(content, dict) or not content:
        return None, json.dumps({"status": "error", "message": "content must be a JSON object (one slide)."})
    if "slides" in content:
        return None, json.dumps({
            "status": "error",
            "message": "content is a whole deck; pass ONE slide (or use slide_canvas_write for a full rebuild).",
        })
    return content, None


class EditDeckTool(ContextAwareTool):
    name: str = "slide_canvas_edit"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Editing slides..."
    end_label: str = "Edited the slide deck"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return None
        edited = result.get("edited") or []
        if len(edited) == 1:
            return f"Edited {edited[0]}"
        if edited:
            return f"Edited {len(edited)} slides"
        return "Edited the deck settings"

    description: str = (
        "Make targeted find-replace edits to slides of the active deck. Each edit names its "
        "`slide_id` (empty = the deck settings) and its old_text must match exactly once in THAT "
        "slide's canonical JSON as slides_read returned it — read the slide first, and include "
        "surrounding text to make it unique. Edits that would produce invalid JSON or an invalid "
        "deck are rejected and the deck is left unchanged. IDs are permanent: never change an "
        "existing slide's or element's `id` (the user comments by id); leave ids off anything new "
        "and they are minted. Colours accept a theme name (`accent1`, `dk1`, …) or a literal "
        "`#RRGGBB`; fonts follow the text `class` unless a run sets `font`."
    )
    args_schema: type[BaseModel] = EditDeckInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        if args.get("edits") is None:
            return None
        return drop_args(args, {"edits": edits_description(args["edits"])}, SLIDES_READ_WHERE)

    def _run(self, edits, deck_name: str = "", slide_id: str = "", **kwargs) -> str:
        from chat.slides import service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error

        def _get(e, key):
            return (e.get(key, "") if isinstance(e, dict) else getattr(e, key, "")) or ""

        # A top-level slide_id applies to edits that don't name one (a live model
        # put it there and every edit then missed against the deck settings).
        triples = [
            (_get(e, "slide_id") or slide_id, _get(e, "old_text"), _get(e, "new_text")) for e in edits
        ]
        # Matching and rewriting happen under the row lock against content re-read
        # inside it (service._mutate_deck), so a parallel deck tool in the same batch
        # can't have its write reverted here.
        try:
            locked, old, new, issues, applied, failed = service.edit_slides(deck.pk, triples)
        except service.SlideOpError as exc:
            return _op_error(exc)
        if locked is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({
                "status": "error", "applied": 0, "failed": failed,
                "message": "Edits would produce an invalid deck; deck left unchanged.",
                "issues": issues[:20], "deck_id": str(locked.pk),
            }, ensure_ascii=False)

        service.activate_deck(thread_id, locked)
        touched = list(dict.fromkeys(sid for sid, _o, _n in triples if sid))
        live = [new["slides"][i] for i in (service.slide_index(new, sid) for sid in touched) if i is not None]
        service.record_agent_slide_view(locked.pk, live, wrote=True)
        return json.dumps(_deck_result(
            locked, old, new, applied=applied, failed=failed,
            edited=[_slide_ref(new, s["id"]) for s in live],
        ), ensure_ascii=False)


class ReadSlidesTool(ContextAwareTool):
    name: str = "slides_read"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Reading slides..."
    end_label: str = "Read slides"
    stub_despite_recency: bool = True  # big, and a re-read is a local query

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return None
        numbers = [s.get("n") for s in result.get("slides") or [] if s.get("n")]
        if not numbers:
            return "Read the deck settings" if result.get("deck_settings") else None
        word = "slide" if len(numbers) == 1 else "slides"
        return f"Read {word} {_numbers_label(numbers)}"

    description: str = (
        "Read slides of the active deck as JSON — the exact canonical text slide_canvas_edit "
        "matches against. Your context only carries the deck's outline (ids, layouts, titles), "
        "so read a slide before editing it. Pick slides by `slide_ids` or a `start`/`end` range "
        "of slide numbers; a read is paged by size (`next_start` says where to continue). "
        "`deck_settings` adds the deck-level JSON (size, theme overrides); `include_render` also "
        "shows you the rendered images (max 4) — slower, use it to check a layout visually."
    )
    args_schema: type[BaseModel] = ReadSlidesInput

    def _run(self, slide_ids=None, start=None, end=None, deck_settings: bool = False,
             include_render: bool = False, deck_name: str = "", **kwargs) -> str:
        from core.tokens import count_tokens
        from llm.context_budget import canvas_read_token_cap
        from chat.slides import schema, service

        _thread, deck, error = _resolve_target(self, deck_name)
        if error:
            return error
        content = deck.content or {}
        slides = [s for s in content.get("slides") or [] if isinstance(s, dict)]
        total = len(slides)

        unknown: list[str] = []
        if slide_ids:
            wanted = []
            for sid in dict.fromkeys(slide_ids):
                idx = service.slide_index(content, sid)
                if idx is None:
                    unknown.append(sid)
                else:
                    wanted.append(idx)
            if not wanted and not deck_settings:
                return json.dumps({
                    "status": "error", "message": "None of those slide ids are in this deck.",
                    "unknown_slide_ids": unknown, "outline": service.compact_outline(content),
                })
        elif deck_settings and start is None and end is None:
            wanted = []
        else:
            first = max(1, start or 1)
            last = min(total, end or total)
            wanted = list(range(first - 1, last))

        cap = canvas_read_token_cap(getattr(self.context, "max_context_tokens", None))
        used = 0
        out, cut_at = [], None
        for pos, idx in enumerate(wanted):
            slide = slides[idx]
            text = schema.canonical_deck_text(slide)
            tokens = count_tokens(text)
            if out and used + tokens > cap:  # always return at least one slide
                cut_at = pos
                break
            used += tokens
            out.append({"n": idx + 1, "id": slide.get("id"), "title": service.slide_title(slide), "json": text})

        result = {
            "status": "ok", "deck_id": str(deck.pk), "title": deck.title, "slide_count": total,
            "slides": out,
        }
        if deck_settings:
            result["deck_settings"] = schema.canonical_deck_text(service.deck_settings(content))
        if unknown:
            result["unknown_slide_ids"] = unknown
        if cut_at is not None:
            rest = wanted[cut_at:]
            if slide_ids:
                result["not_included_slide_ids"] = [slides[i].get("id") for i in rest]
            else:
                result["next_start"] = rest[0] + 1
            result["note"] = f"Stopped at the per-read size cap (~{cap:,} tokens); read the rest next."
        if include_render and out:
            result["render"] = _render_and_attach(
                self, deck, [s["id"] for s in out][: schema.MAX_PREVIEW_SLIDES_PER_CALL]
            )
        service.record_agent_slide_view(deck.pk, [slides[s["n"] - 1] for s in out])
        return json.dumps(result, ensure_ascii=False)


class WriteSlideTool(ContextAwareTool):
    name: str = "slides_write_slide"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Writing slide..."
    end_label: str = "Wrote a slide"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok" or not result.get("slide"):
            return None
        verb = "Added" if result.get("inserted") else "Wrote"
        return f"{verb} {result['slide']}"

    description: str = (
        "Write ONE slide's full JSON: replace an existing slide (`slide_id` — it keeps its id and "
        "position; carry over the element ids you keep) or insert a new one (no `slide_id`; "
        "`position`). The natural way to fill in a storyboard slide by slide — the user sees each "
        "slide render as you write it. For a small change to a slide, slides_read + "
        "slide_canvas_edit is cheaper."
    )
    args_schema: type[BaseModel] = WriteSlideInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        if args.get("content") is None:
            return None
        return drop_args(args, {"content": "the slide JSON"}, SLIDES_READ_WHERE)

    def _run(self, content=None, slide_id: str = "", position: int = -1, deck_name: str = "", **kwargs) -> str:
        from chat.slides import service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error
        slide, error = _parse_slide_payload(content)
        if error:
            return error

        if slide_id:
            try:
                locked, old, new, issues = service.replace_slide(deck.pk, slide_id, slide)
            except service.SlideOpError as exc:
                return _op_error(exc)
            target = slide_id
        else:
            slide = {k: v for k, v in slide.items() if k != "id"}
            old = deck.content or {}
            locked, new, idx, issues = service.insert_slide(
                deck.pk, slide, position=position, description="Added a slide"
            )
            target = new["slides"][idx].get("id") if new and not issues else ""
        if locked is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({
                "status": "error", "message": "The slide is invalid; nothing was saved.", "issues": issues[:20],
            }, ensure_ascii=False)

        service.activate_deck(thread_id, locked)
        written = new["slides"][service.slide_index(new, target)]
        service.record_agent_slide_view(locked.pk, [written], wrote=True)
        return json.dumps(_deck_result(
            locked, old, new, slide_id=target, slide=_slide_ref(new, target), inserted=not slide_id,
        ), ensure_ascii=False)


class MoveSlidesTool(ContextAwareTool):
    name: str = "slides_move_slides"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Moving slides..."
    end_label: str = "Moved slides"

    def end_label_for_result(self, result: dict) -> str | None:
        moved = result.get("moved") or []
        if result.get("status") != "ok" or not moved:
            return None
        if len(moved) == 1:
            return f"Moved {moved[0]}"
        return f"Moved {len(moved)} slides"

    description: str = (
        "Reorder the active deck: move one or more slides (by id, in the order they should end up) "
        "so the first lands at `position` (0-based; -1 = the end). Use this instead of rewriting "
        "the deck to reorder it."
    )
    args_schema: type[BaseModel] = MoveSlidesInput

    def _run(self, slide_ids, position: int, deck_name: str = "", **kwargs) -> str:
        from chat.slides import service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error
        if not slide_ids:
            return json.dumps({"status": "error", "message": "slide_ids is empty."})
        try:
            locked, old, new, issues = service.move_slides(deck.pk, list(slide_ids), position)
        except service.SlideOpError as exc:
            return _op_error(exc)
        if locked is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({"status": "error", "message": "The move made the deck invalid.", "issues": issues[:10]})
        service.activate_deck(thread_id, locked)
        moved = [_slide_ref(new, sid) for sid in dict.fromkeys(slide_ids)]
        return json.dumps(_deck_result(locked, old, new, moved=moved), ensure_ascii=False)


class DeleteSlidesTool(ContextAwareTool):
    name: str = "slides_delete_slides"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Deleting slides..."
    end_label: str = "Deleted slides"

    def end_label_for_result(self, result: dict) -> str | None:
        deleted = result.get("deleted") or []
        if result.get("status") != "ok" or not deleted:
            return None
        if len(deleted) == 1:
            return f"Deleted {deleted[0]}"
        return f"Deleted {len(deleted)} slides"

    description: str = (
        "Delete slides from the active deck by id. The deck keeps a checkpoint, so this can be "
        "undone. A deck can't be emptied this way — delete the whole deck with slide_canvas_delete."
    )
    args_schema: type[BaseModel] = DeleteSlidesInput

    def _run(self, slide_ids, deck_name: str = "", **kwargs) -> str:
        from chat.slides import service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error
        if not slide_ids:
            return json.dumps({"status": "error", "message": "slide_ids is empty."})
        before = deck.content or {}
        refs = [_slide_ref(before, sid) for sid in dict.fromkeys(slide_ids)]
        try:
            locked, old, new, issues = service.delete_slides(deck.pk, list(slide_ids))
        except service.SlideOpError as exc:
            return _op_error(exc)
        if locked is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({"status": "error", "message": "The delete made the deck invalid.", "issues": issues[:10]})
        service.activate_deck(thread_id, locked)
        return json.dumps(_deck_result(locked, old, new, deleted=refs), ensure_ascii=False)


class DuplicateSlideTool(ContextAwareTool):
    name: str = "slides_duplicate_slide"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Duplicating slide..."
    end_label: str = "Duplicated a slide"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok" or not result.get("source"):
            return None
        return f"Duplicated {result['source']}"

    description: str = (
        "Copy a slide of the active deck (same layout and content, a fresh slide id) — e.g. to "
        "repeat a layout for the next market or product. The copy goes right after the original "
        "unless you give a 0-based `position`. Then change the copy with slides_read + "
        "slide_canvas_edit or slides_write_slide."
    )
    args_schema: type[BaseModel] = DuplicateSlideInput

    def _run(self, slide_id: str, position: int = -1, deck_name: str = "", **kwargs) -> str:
        from chat.slides import service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error
        try:
            locked, old, new, issues, idx = service.duplicate_slide(deck.pk, slide_id, position)
        except service.SlideOpError as exc:
            return _op_error(exc)
        if locked is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({"status": "error", "message": "The copy made the deck invalid.", "issues": issues[:10]})
        service.activate_deck(thread_id, locked)
        copy_id = new["slides"][idx].get("id")
        # Same JSON as the original, which the agent may have read — not "seen" yet
        # under its new id, so it shows as unread in the outline until read.
        return json.dumps(_deck_result(
            locked, old, new, source=_slide_ref(old, slide_id), slide_id=copy_id,
            position=idx,
        ), ensure_ascii=False)


class AddSlideTool(ContextAwareTool):
    name: str = "slides_add_slide"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Adding slide..."
    end_label: str = "Added a slide"

    def end_label_for_result(self, result: dict) -> str | None:
        layout = result.get("layout")
        return f"Added a {layout} slide" if layout else None
    description: str = (
        "Insert a pre-designed layout slide into the active deck (create the deck first with "
        "slides_create_deck), then fill its placeholder text with slide_canvas_edit or "
        "slides_write_slide. Returns the new slide's JSON and id. Available layouts:\n" + _LAYOUT_MENU
    )
    args_schema: type[BaseModel] = AddSlideInput

    def trim_result_at_edit_point(self, result: str, args: dict, *, later_calls: list) -> str | None:
        return trim_slide_result_fields(result, {"slide_json": "seeded slide JSON"})

    def _run(self, layout: str, position: int = -1, deck_name: str = "", **kwargs) -> str:
        from chat.slides import layouts, schema, service

        thread_id, deck, error = _resolve_target(self, deck_name)
        if error:
            return error

        seed = layouts.get_layout(layout)
        if seed is None:
            return json.dumps({
                "status": "error",
                "message": f"Unknown layout '{layout}'.",
                "available_layouts": [c["id"] for c in layouts.layout_catalog()],
            })

        old = deck.content or {}
        deck, content, idx, issues = service.insert_slide(
            deck.pk, seed, position=position, description=f"Added a '{layout}' slide"
        )
        if deck is None:
            return json.dumps({"status": "error", "message": "The deck was deleted."})
        if issues:
            return json.dumps({"status": "error", "message": "Adding the slide made the deck invalid.", "issues": issues[:10]})

        service.activate_deck(thread_id, deck)
        inserted = content["slides"][idx]
        service.record_agent_slide_view(deck.pk, [inserted], wrote=True)
        return json.dumps(_deck_result(
            deck, old, content, slide_id=inserted.get("id"), position=idx, layout=layout,
            slide_json=schema.canonical_deck_text(inserted),
        ), ensure_ascii=False)


class DeleteDeckTool(ContextAwareTool):
    name: str = "slide_canvas_delete"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Deleting deck..."
    end_label: str = "Deleted deck"
    description: str = (
        "Soft-delete a slide deck from this conversation. Content and history are "
        "preserved, and a mistaken delete can be undone with slides_restore_deck. Provide the "
        "exact title. Runs before the other calls in the same batch."
    )
    args_schema: type[BaseModel] = DeleteDeckInput

    def runs_first(self, args: dict) -> bool:
        return True

    def _run(self, deck_name: str, **kwargs) -> str:
        from chat.slides import service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deck, err = service.resolve_deck(thread_id, deck_name)
        if err:
            return json.dumps({"status": "error", **err})
        if deck is None:
            return json.dumps({"status": "error", "message": "deck_name is required: the exact title of the deck to delete."})
        title = deck.title
        slide_count = len((deck.content or {}).get("slides") or [])
        service.soft_delete_deck(thread_id, deck)
        # The deck the "no deck_name" tools now target: the one promoted in its
        # place, or the one that was already active when this one wasn't.
        active = service.get_active_deck(thread_id)
        return json.dumps({
            "status": "ok", "deck_id": str(deck.pk), "deck_title": title,
            # What was lost and what the "no deck_name" tools now target, so a
            # delete of the wrong deck is noticed (2026-11-11 prod: the agent
            # deleted its finished 14-slide deck as a stale duplicate).
            "slide_count": slide_count,
            "active_deck": active.title if active else None,
            "undo": f"slides_restore_deck(deck_name={title!r})",
        }, ensure_ascii=False)


class RestoreDeckInput(ReasonBaseModel):
    deck_name: str = Field(
        default="",
        description="Exact title of the deleted deck to restore. Empty = list the deleted decks.",
    )


class RestoreDeckTool(ContextAwareTool):
    name: str = "slides_restore_deck"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Restoring deck..."
    end_label: str = "Restored deck"
    description: str = (
        "Bring back a deck deleted with slide_canvas_delete — content, slide ids and history "
        "intact — and make it the active deck. Call without deck_name to list this "
        "conversation's deleted decks (newest first, with slide counts). Runs before the other "
        "calls in the same batch."
    )
    args_schema: type[BaseModel] = RestoreDeckInput

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") == "ok" and result.get("deleted_decks") is not None:
            return "Listed deleted decks"
        return None

    def runs_first(self, args: dict) -> bool:
        return True

    def _run(self, deck_name: str = "", **kwargs) -> str:
        from chat.models import SlideSet
        from chat.slides import schema, service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deleted = service.deleted_decks(thread_id)
        listing = [
            {
                "title": d.title, "deck_id": str(d.pk),
                "slide_count": len((d.content or {}).get("slides") or []),
                "deleted_at": d.deleted_at.isoformat() if d.deleted_at else None,
            }
            for d in deleted
        ]
        name = (deck_name or "").strip()
        if not name:
            return json.dumps({
                "status": "ok", "deleted_decks": listing,
                "message": ("Call again with deck_name to restore one." if listing
                            else "No deleted decks in this conversation."),
            }, ensure_ascii=False)

        deck = next((d for d in deleted if d.title == name), None)
        if deck is None:
            return json.dumps({
                "status": "error",
                "message": f"No deleted deck named '{name}' in this conversation.",
                "deleted_decks": [d["title"] for d in listing],
            }, ensure_ascii=False)

        live = SlideSet.objects.filter(thread_id=thread_id, deleted_at__isnull=True).count()
        if live >= schema.MAX_SLIDE_SETS_PER_THREAD:
            return json.dumps({
                "status": "error",
                "message": (
                    f"This conversation already has the maximum of {schema.MAX_SLIDE_SETS_PER_THREAD} "
                    "decks. Delete one you no longer need first."
                ),
                "available_decks": service.live_deck_titles(thread_id),
            }, ensure_ascii=False)

        old_title = deck.title
        service.restore_deck(thread_id, deck)
        service.activate_deck(thread_id, deck)
        content = deck.content or {}
        extra = {"restored": True}
        if deck.title != old_title:
            extra["renamed_to"] = deck.title  # a live deck had taken the title meanwhile
        result = _deck_result(deck, None, content, **extra)
        # Nothing changed on the slides themselves; the panel reloads the deck
        # from slidedeck.state (chat/consumers.py), not a re-render.
        result["changed_slide_ids"] = []
        return json.dumps(result, ensure_ascii=False)


class PreviewSlidesTool(ContextAwareTool):
    name: str = "slides_preview_slide"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Rendering slides..."
    end_label: str = "Rendered slides"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") == "ok":
            n = result.get("previewed_count", 0)
            return f"Rendered {n} slide(s)" if n else "Rendered slides"
        if result.get("status") == "unavailable":
            return "Preview unavailable"
        return None

    description: str = (
        "Render specific slides to images and view them so you can check text overflow, "
        "element collisions, and contrast before finishing. Preview only the slides you "
        "changed (max 4). Empty = the whole deck (capped)."
    )
    args_schema: type[BaseModel] = PreviewSlidesInput

    def _run(self, slide_ids=None, deck_name: str = "", **kwargs) -> str:
        from chat.slides import schema

        _thread, deck, error = _resolve_target(self, deck_name)
        if error:
            return error

        all_ids = [s.get("id") for s in (deck.content or {}).get("slides") or []]
        all_set = set(all_ids)
        ids = [sid for sid in (slide_ids or []) if sid in all_set]
        ids = (ids or all_ids)[: schema.MAX_PREVIEW_SLIDES_PER_CALL]
        if not ids:
            return json.dumps({"status": "error", "message": "The deck has no slides to preview."})
        return json.dumps(_render_and_attach(self, deck, ids))


def _render_and_attach(tool, deck, ids: list[str]) -> dict:
    """Render ``ids`` on the worker (an AGENT_PREVIEW run), wait for it, and queue
    the images as native assets for the current reply. Returns the preview
    result dict (``status`` ok / unavailable) — shared by slides_preview_slide
    and slides_read(include_render=True)."""
    from django.conf import settings

    from chat.models import SlideRenderRun
    from chat.tasks import render_deck_task

    run = SlideRenderRun.objects.create(
        slide_set=deck, purpose=SlideRenderRun.Purpose.AGENT_PREVIEW, slide_ids=ids
    )
    try:
        task = render_deck_task.delay(str(run.id))
        run.celery_task_id = task.id
        run.save(update_fields=["celery_task_id"])
    except Exception:
        SlideRenderRun.objects.filter(pk=run.id).update(
            status=SlideRenderRun.Status.FAILED, error="Could not enqueue render."
        )
        return {
            "status": "unavailable",
            "message": "Couldn't start a render; proceed without a visual preview.",
        }

    timeout = int(getattr(settings, "SLIDE_RENDER_TIMEOUT", 120))
    if tool.context and tool.context.deadline_seconds:
        timeout = min(timeout, max(5, tool.context.deadline_seconds - 5))
    deadline = time.monotonic() + timeout
    run.refresh_from_db()
    terminal = (SlideRenderRun.Status.COMPLETED, SlideRenderRun.Status.FAILED)
    while run.status not in terminal:
        if time.monotonic() >= deadline:
            return {
                "status": "unavailable",
                "message": f"Preview still rendering after {timeout}s; proceed without it.",
            }
        time.sleep(2)
        run.refresh_from_db()

    if run.status == SlideRenderRun.Status.FAILED:
        return {
            "status": "unavailable",
            "message": "Rendering isn't available right now; proceed without a visual preview.",
        }

    attached = _attach_images(tool, run, ids)
    if attached == 0:
        return {
            "status": "unavailable",
            "message": (
                "The render produced no images, or the attachment budget for "
                "this run is exhausted; proceed without a visual preview."
            ),
        }
    return {
        "status": "ok",
        "previewed_count": attached,
        "previewed_slide_ids": ids,
        "message": (
            "Rendered slides attached below. Inspect them for text overflow, element "
            "collisions, and low-contrast text; fix any issues before you finish."
        ),
    }


def _attach_images(tool, run, ids) -> int:
    from chat.assets import image_asset_source
    from chat.models import Asset

    ctx = tool.context
    if ctx is None:
        return 0
    by_id = {s["slide_id"]: s for s in (run.result or {}).get("slides", [])}
    attached = 0
    for sid in ids:
        info = by_id.get(sid)
        if not info or not info.get("asset_id"):
            continue
        try:
            asset = Asset.objects.filter(pk=info["asset_id"]).first()
            if asset is None:
                continue
            source, ct = image_asset_source(asset)
            if source is None:
                continue
            with source.open("rb") as fh:
                data = fh.read()
            data, ct = _downscale_png(data, ct)
            if not ctx.try_add_native_asset({
                "asset_id": sid,  # label only — unused by the pipeline drain
                "b64": base64.b64encode(data).decode("ascii"),
                "media_type": ct or "image/png",
                "description": f"Rendered slide {sid} (page {info.get('page')})",
            }):
                continue
            attached += 1
        except Exception:
            continue
    return attached


def _downscale_png(data: bytes, ct: str, max_w: int = 1024):
    """Shrink a preview PNG for the agent copy (keeps the base64 payload small)."""
    try:
        from PIL import Image

        im = Image.open(BytesIO(data))
        if im.width > max_w:
            ratio = max_w / im.width
            im = im.convert("RGB").resize((max_w, max(1, int(im.height * ratio))))
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except Exception:
        pass
    return data, ct


def _theme_summary(t: dict) -> str:
    """A one-line summary of a theme's palette + footer for the agent."""
    c = t.get("colors") or {}
    footer = t.get("footer") or {}
    used = [s.get("content") for s in (footer.get("sections") or []) if s.get("content") not in (None, "none")]
    fbits = ("footer: " + " / ".join(used)) if used else "no footer content"
    if footer.get("bg_color"):
        fbits += " (banded)"
    logo = "; has logo" if t.get("logo_ext") else ""
    return f"bg {c.get('lt1', '?')}, headings {c.get('dk2', '?')}, accent {c.get('accent1', '?')}; {fbits}{logo}"


class ListThemesTool(ContextAwareTool):
    name: str = "slides_list_themes"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Listing themes..."
    end_label: str = "Listed themes"
    description: str = (
        "List the slide themes available for the current deck — the five built-in themes "
        "(Forest, Slate, Warm sand, Monochrome, Ocean) plus any the organization and the user "
        "have saved. Each has an id, a name, a scope (builtin/org/user) and a short summary of "
        "its palette and footer, and the deck's current theme is flagged. A new deck already "
        "carries the user's (else the org's) default theme, so only look for another when the "
        "user wants a different look. Apply one with slides_set_theme; you cannot create or save "
        "named themes — the user manages those from the deck panel. The summary is not the "
        "deck's theme JSON: to change single colours, read it with slides_read(deck_settings=true) "
        "first."
    )
    args_schema: type[BaseModel] = ListThemesInput

    def _run(self, **kwargs) -> str:
        from chat.slides import service
        from chat.slides import theme as theme_mod

        user, org = _tool_user(self), _tool_org(self)
        themes = theme_mod.list_available_themes(user, org)
        current_id = ""
        thread_id = _thread_id(self)
        if thread_id:
            deck, _err = service.resolve_deck(thread_id, None)
            if deck is not None:
                # No theme override == the base Forest theme.
                current_id = ((deck.content or {}).get("theme") or {}).get("_theme_id", "") or "forest"
        out = [{
            "id": t["id"], "name": t["label"], "scope": t.get("scope", ""),
            "summary": _theme_summary(t), "current": t["id"] == current_id,
        } for t in themes]
        return json.dumps({"status": "ok", "themes": out, "current_theme_id": current_id})


class SetThemeTool(ContextAwareTool):
    name: str = "slides_set_theme"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Applying theme..."
    end_label: str = "Applied the theme"

    def end_label_for_result(self, result: dict) -> str | None:
        name = result.get("applied")
        return f"Applied the '{name}' theme" if name else None

    description: str = (
        "Set the active deck's theme (palette, fonts, tables and footer) by name or id — get the "
        "choices from slides_list_themes. Applies to the whole deck; you can still fine-tune "
        "individual colours afterwards with slide_canvas_edit on the deck settings (slide_id "
        "empty), after reading them with slides_read(deck_settings=true). If the theme carries a logo, place it "
        "where it fits (e.g. the cover) with the token [[image:company-logo]]."
    )
    args_schema: type[BaseModel] = SetThemeInput

    def _run(self, theme: str, deck_name: str = "", **kwargs) -> str:
        from chat.slides import service
        from chat.slides import theme as theme_mod

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        user, org = _tool_user(self), _tool_org(self)
        avail = theme_mod.list_available_themes(user, org)
        chosen = next((t for t in avail if t["id"] == theme), None)
        if chosen is None:
            key = (theme or "").strip().lower()
            chosen = next((t for t in avail if t["label"].lower() == key), None)
        if chosen is None:
            return json.dumps({
                "status": "error",
                "message": f"Unknown theme '{theme}'. Call slides_list_themes for the choices.",
                "available": [{"id": t["id"], "name": t["label"]} for t in avail],
            })

        deck, err = service.resolve_deck(thread_id, deck_name or None)
        if err:
            return json.dumps({"status": "error", **err})
        if deck is None:
            return json.dumps({
                "status": "error",
                "message": "No deck to theme. Create one with slides_create_deck first.",
            })

        override = theme_mod.theme_to_deck_override(chosen)
        with service.locked_deck(deck.pk) as deck:
            if deck is None:
                return json.dumps({"status": "error", "message": "The deck was deleted."})
            content = deck.content or {}
            content["theme"] = theme_mod.keep_deck_effects(content.get("theme"), override)
            service.save_deck_content(deck, content)
            service.create_deck_checkpoint(
                deck, source="ai_edit", description=f"Applied '{chosen['label']}' theme"
            )
        service.activate_deck(thread_id, deck)
        slide_ids = [s.get("id") for s in (content.get("slides") or [])]
        res = {
            "status": "ok", "applied": chosen["label"], "theme_id": chosen["id"],
            "deck_id": str(deck.pk), "title": deck.title,
            "slide_ids": slide_ids, "changed_slide_ids": slide_ids, "slide_count": len(slide_ids),
        }
        if chosen.get("logo_ext"):
            res["note"] = ("This theme has a logo — place it where it fits (e.g. the cover) "
                           "with the token [[image:company-logo]].")
        return json.dumps(res)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(ActivateDeckTool())
_registry.register_tool(CreateDeckTool())
_registry.register_tool(WriteDeckTool())
_registry.register_tool(EditDeckTool())
_registry.register_tool(AddSlideTool())
_registry.register_tool(ReadSlidesTool())
_registry.register_tool(WriteSlideTool())
_registry.register_tool(MoveSlidesTool())
_registry.register_tool(DeleteSlidesTool())
_registry.register_tool(DuplicateSlideTool())
_registry.register_tool(DeleteDeckTool())
_registry.register_tool(RestoreDeckTool())
_registry.register_tool(PreviewSlidesTool())
_registry.register_tool(ListThemesTool())
_registry.register_tool(SetThemeTool())
