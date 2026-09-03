"""Slide-deck tools: author and edit a per-thread slide deck as JSON.

The deck sibling of chat/canvas_tools.py. The assistant writes/edits one
canonical JSON document (chat.slides.schema); the user only ever sees the
rendered filmstrip. All tools are audience="main", section="skills" (gated
behind the "Slide Deck Collaborator" seed skill).

``slides_preview_slide`` (the dispatch+poll visual feedback tool) lives here too
but is wired to the worker render task in a later phase.
"""

from __future__ import annotations

import base64
import copy
import json
import time
from io import BytesIO

from pydantic import BaseModel, Field, field_validator

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

# The slides_add_slide layout ids, sourced from the seed catalogue so this
# always matches what actually exists (no hand-maintained list to go stale).
from chat.slides.layouts import layout_catalog

_LAYOUT_IDS = ", ".join(c["id"] for c in layout_catalog())
# A one-line-per-layout menu (id: what it's for) so the model picks confidently.
_LAYOUT_MENU = "\n".join(f'- {c["id"]}: {c["description"]}' for c in layout_catalog())


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------
class ActivateDeckInput(ReasonBaseModel):
    deck_names: list[str] = Field(
        description="Deck title(s) to make active (max 1). The active deck's JSON is in your context.",
    )


class WriteDeckInput(ReasonBaseModel):
    title: str = Field(description="Title for the deck.")
    content: dict | str | None = Field(
        default=None,
        description=(
            "The full deck as a JSON OBJECT — pass it directly as structured JSON, do NOT "
            "wrap it in a string or escape the quotes: "
            '{"version":1,"size":{"w":960,"h":540},"slides":[...]}. Coordinates are points; '
            "the slide is 960x540; omit slide/element ids and they are minted. OMIT this "
            "entirely to create an EMPTY deck, then build it up from the pre-designed layouts "
            "with slides_add_slide and fill them in with slide_canvas_edit."
        ),
    )
    deck_name: str = Field(
        default="",
        description="Target an existing deck by title. Empty = create/overwrite by the given title.",
    )


class DeckEditItem(BaseModel):
    old_text: str = Field(description="Exact JSON text to find (must be unique in the deck).")
    new_text: str = Field(description="Replacement JSON text.")
    reason: str = Field(default="", description="Brief reason for this edit.")


class EditDeckInput(ReasonBaseModel):
    edits: list[DeckEditItem] = Field(
        description="Targeted find-replace edits over the deck's canonical JSON text.",
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
        "at a time; the active deck's full JSON + theme is included in your context. "
        "Activating a deck deactivates the others, but they stay saved."
    )
    args_schema: type[BaseModel] = ActivateDeckInput

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


class WriteDeckTool(ContextAwareTool):
    name: str = "slide_canvas_write"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Writing slides..."
    end_label: str = "Wrote the slide deck"
    description: str = (
        "Create or completely rewrite a slide deck — this is how a deck comes into existence, and "
        "it becomes the active deck (the user sees a rendered preview, never the JSON). RECOMMENDED: "
        "call it with just a `title` (omit `content`) to create an EMPTY deck, then build it up from "
        "the pre-designed layouts with `slides_add_slide` and fill them in with `slide_canvas_edit`. "
        "Alternatively, pass a full deck JSON in `content` for a bespoke deck or a full restructure. "
        "For targeted changes to an existing deck use `slide_canvas_edit`. The deck/element JSON "
        "format is documented in the Slide Deck Collaborator skill."
    )
    args_schema: type[BaseModel] = WriteDeckInput

    def _run(self, title: str, content=None, deck_name: str = "", **kwargs) -> str:
        from chat.slides import schema, service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # Accept the deck as a native object (preferred — no escaping) or a JSON
        # string (back-compat, under either name). An OMITTED deck creates an
        # empty deck to build up from layouts with slides_add_slide.
        if content is None:
            content = kwargs.get("content_json")
        if content is None:
            deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": []}
        elif isinstance(content, str):
            try:
                deck = json.loads(content)
            except (ValueError, TypeError) as exc:
                return json.dumps({"status": "error", "message": f"content is not valid JSON: {exc}"})
        elif isinstance(content, dict):
            deck = content
        else:
            return json.dumps({"status": "error", "message": "content must be a JSON object (the full deck)."})

        schema.mint_ids(deck)

        # Seed the deck's theme when the model didn't set one: keep an existing
        # deck's theme (so a user's picker choice survives a full rewrite),
        # otherwise inherit the org's default slide theme. A deck that carries an
        # explicit theme is left untouched.
        if not deck.get("theme"):
            from chat.slides import theme as theme_mod

            existing = service.get_deck_by_title(thread_id, deck_name or title)
            prev_theme = (existing.content or {}).get("theme") if existing else None
            if prev_theme:
                deck["theme"] = prev_theme
            else:
                # New-deck default theme: user's default -> org default -> Forest.
                # Forest is the base theme, so leave the deck theme unset for it (a
                # plain deck stays lean); only a non-default org/user theme is written.
                default = theme_mod.default_theme_for(_tool_user(self), _tool_org(self))
                if default.get("id") != "forest":
                    deck["theme"] = theme_mod.theme_to_deck_override(default)

        issues = schema.validate_deck(deck)
        if issues:
            return json.dumps({"status": "error", "message": "Deck JSON is invalid.", "issues": issues[:20]})

        try:
            # write_deck checkpoints the result itself, inside the row lock it takes
            # for the overwrite path, so the checkpoint order stays monotonic.
            obj, created, old = service.write_deck(
                thread_id, title=title, content=deck, deck_name=deck_name
            )
        except service.DeckLimitError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        service.activate_deck(thread_id, obj)

        changed = schema.changed_slide_ids(None if created else old, deck)
        return json.dumps({
            "status": "ok",
            "deck_id": str(obj.pk),
            "title": obj.title,
            "slide_ids": [s.get("id") for s in deck.get("slides", [])],
            "changed_slide_ids": changed,
            "slide_count": len(deck.get("slides", [])),
        })


class EditDeckTool(ContextAwareTool):
    name: str = "slide_canvas_edit"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Editing slides..."
    end_label: str = "Edited the slide deck"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") == "ok":
            n = result.get("applied", 0)
            return f"Edited {n} section(s)" if n else "Edited the slide deck"
        return None

    description: str = (
        "Make targeted find-replace edits to the active deck's JSON. Each edit's old_text must "
        "match exactly once in the canonical deck JSON shown in your context — include surrounding "
        "text to make it unique. Edits that would produce invalid deck JSON are rejected and the "
        "deck is left unchanged. IDs are permanent: never change an existing slide's or element's "
        "`id` (the user comments by id); leave ids off anything new and they are minted. Colours "
        "accept a theme name (`accent1`, `dk1`, …) or a literal `#RRGGBB`; fonts follow the text "
        "`class` unless a run sets `font`."
    )
    args_schema: type[BaseModel] = EditDeckInput

    def _run(self, edits, deck_name: str = "", **kwargs) -> str:
        from chat.edit_utils import apply_unique_text_edits
        from chat.slides import schema, service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deck, err = service.resolve_deck(thread_id, deck_name or None)
        if err:
            return json.dumps({"status": "error", **err})
        if deck is None:
            return json.dumps({
                "status": "error",
                "message": "No deck exists for this thread. Use slide_canvas_write to create one first.",
            })

        pairs = [
            (
                e.get("old_text", "") if isinstance(e, dict) else e.old_text,
                e.get("new_text", "") if isinstance(e, dict) else e.new_text,
            )
            for e in edits
        ]

        # Match and rewrite under the row lock, against content re-read inside it, so a
        # parallel deck tool in the same batch can't have its write reverted here. An
        # edit whose old_text no longer matches the fresh content now fails cleanly with
        # "No edits applied." (actionable for the model) instead of silently clobbering.
        with service.locked_deck(deck.pk) as deck:
            if deck is None:
                return json.dumps({"status": "error", "message": "The deck was deleted."})

            original = schema.canonical_deck_text(deck.content)
            new_text, applied, failed = apply_unique_text_edits(original, pairs)
            if applied == 0:
                return json.dumps({
                    "status": "error", "applied": 0, "failed": failed,
                    "message": "No edits applied.", "deck_id": str(deck.pk), "title": deck.title,
                })

            # JSON guard: the edited text must still be a valid deck, or we reject
            # the whole edit set and leave the deck untouched.
            try:
                new_deck = json.loads(new_text)
            except (ValueError, TypeError) as exc:
                return json.dumps({
                    "status": "error", "applied": 0, "failed": failed,
                    "message": f"Edits would produce invalid JSON ({exc}); deck left unchanged. "
                               "Widen your old_text so replacements keep the JSON well-formed.",
                    "deck_id": str(deck.pk),
                })
            issues = schema.validate_deck(new_deck)
            if issues:
                return json.dumps({
                    "status": "error", "applied": 0, "failed": failed,
                    "message": "Edits would produce an invalid deck; deck left unchanged.",
                    "issues": issues[:20], "deck_id": str(deck.pk),
                })

            schema.mint_ids(new_deck)
            changed = schema.changed_slide_ids(deck.content, new_deck)
            service.save_deck_content(deck, new_deck)
            service.create_deck_checkpoint(deck, source="ai_edit", description=f"Edited {applied} section(s)")

        service.activate_deck(thread_id, deck)
        slide_ids = [s.get("id") for s in new_deck.get("slides", [])]
        return json.dumps({
            "status": "ok", "applied": applied, "failed": failed,
            "changed_slide_ids": changed, "deck_id": str(deck.pk), "title": deck.title,
            # Full ordered id list so an edit that adds a slide rebuilds the client
            # filmstrip with the new slide, matching slide_canvas_write.
            "slide_ids": slide_ids, "slide_count": len(slide_ids),
        })


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
        "slide_canvas_write), then edit its placeholder text with slide_canvas_edit. Returns the "
        "new slide's JSON and id. Available layouts:\n" + _LAYOUT_MENU
    )
    args_schema: type[BaseModel] = AddSlideInput

    def _run(self, layout: str, position: int = -1, deck_name: str = "", **kwargs) -> str:
        from chat.slides import layouts, schema, service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deck, err = service.resolve_deck(thread_id, deck_name or None)
        if err:
            return json.dumps({"status": "error", **err})
        if deck is None:
            return json.dumps({
                "status": "error",
                "message": "No active deck. Create one with slide_canvas_write first.",
            })

        seed = layouts.get_layout(layout)
        if seed is None:
            return json.dumps({
                "status": "error",
                "message": f"Unknown layout '{layout}'.",
                "available_layouts": [c["id"] for c in layouts.layout_catalog()],
            })

        # Everything that reads and rewrites deck.content happens under the row lock,
        # against content re-read inside it — a parallel slides_add_slide in the same
        # tool batch would otherwise overwrite this call's slide (and vice versa).
        with service.locked_deck(deck.pk) as deck:
            if deck is None:
                return json.dumps({"status": "error", "message": "The deck was deleted."})

            content = copy.deepcopy(deck.content or {})
            content.setdefault("version", 1)
            content.setdefault("size", {"w": 960, "h": 540})
            slides = content.setdefault("slides", [])
            idx = len(slides) if position < 0 or position > len(slides) else position
            slides.insert(idx, seed)
            schema.mint_ids(content)

            issues = schema.validate_deck(content)
            if issues:
                return json.dumps({"status": "error", "message": "Adding the slide made the deck invalid.", "issues": issues[:10]})

            service.save_deck_content(deck, content)
            service.create_deck_checkpoint(deck, source="ai_edit", description=f"Added a '{layout}' slide")

        service.activate_deck(thread_id, deck)

        inserted = content["slides"][idx]
        slide_ids = [s.get("id") for s in content["slides"]]
        return json.dumps({
            "status": "ok",
            "deck_id": str(deck.pk),
            "title": deck.title,
            "slide_id": inserted.get("id"),
            "position": idx,
            "layout": layout,
            "slide_json": schema.canonical_deck_text(inserted),
            # The client rebuilds its filmstrip from slide_ids, so a new slide only
            # shows a queued/rendering placeholder (instead of appearing out of
            # nowhere when the render lands) if we return the full ordered id list.
            "slide_ids": slide_ids,
            "changed_slide_ids": [inserted.get("id")],
            "slide_count": len(slide_ids),
        })


class DeleteDeckTool(ContextAwareTool):
    name: str = "slide_canvas_delete"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Deleting deck..."
    end_label: str = "Deleted deck"
    description: str = (
        "Soft-delete a slide deck from this conversation. Content and history are "
        "preserved and the user can undo, so this is safe. Provide the exact title."
    )
    args_schema: type[BaseModel] = DeleteDeckInput

    def _run(self, deck_name: str, **kwargs) -> str:
        from chat.slides import service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deck, err = service.resolve_deck(thread_id, deck_name)
        if err:
            return json.dumps({"status": "error", **err})
        title = deck.title
        service.soft_delete_deck(thread_id, deck)
        return json.dumps({"status": "ok", "deck_id": str(deck.pk), "deck_title": title})


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
        from django.conf import settings

        from chat.models import SlideRenderRun
        from chat.slides import schema, service
        from chat.tasks import render_deck_task

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        deck, err = service.resolve_deck(thread_id, deck_name or None)
        if err:
            return json.dumps({"status": "error", **err})
        if deck is None:
            return json.dumps({"status": "error", "message": "No active deck to preview."})

        all_ids = [s.get("id") for s in (deck.content or {}).get("slides") or []]
        all_set = set(all_ids)
        ids = [sid for sid in (slide_ids or []) if sid in all_set]
        ids = (ids or all_ids)[: schema.MAX_PREVIEW_SLIDES_PER_CALL]
        if not ids:
            return json.dumps({"status": "error", "message": "The deck has no slides to preview."})

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
            return json.dumps({
                "status": "unavailable",
                "message": "Couldn't start a render; proceed without a visual preview.",
            })

        timeout = int(getattr(settings, "SLIDE_RENDER_TIMEOUT", 120))
        if self.context and self.context.deadline_seconds:
            timeout = min(timeout, max(5, self.context.deadline_seconds - 5))
        deadline = time.monotonic() + timeout
        run.refresh_from_db()
        terminal = (SlideRenderRun.Status.COMPLETED, SlideRenderRun.Status.FAILED)
        while run.status not in terminal:
            if time.monotonic() >= deadline:
                return json.dumps({
                    "status": "unavailable",
                    "message": f"Preview still rendering after {timeout}s; proceed without it.",
                })
            time.sleep(2)
            run.refresh_from_db()

        if run.status == SlideRenderRun.Status.FAILED:
            return json.dumps({
                "status": "unavailable",
                "message": "Rendering isn't available right now; proceed without a visual preview.",
            })

        attached = self._attach_images(run, ids)
        if attached == 0:
            return json.dumps({
                "status": "unavailable",
                "message": "The render produced no images; proceed without a visual preview.",
            })
        return json.dumps({
            "status": "ok",
            "previewed_count": attached,
            "previewed_slide_ids": ids,
            "message": (
                "Rendered slides attached below. Inspect them for text overflow, element "
                "collisions, and low-contrast text; fix any issues with slide_canvas_edit "
                "before you finish."
            ),
        })

    def _attach_images(self, run, ids) -> int:
        from chat.assets import image_asset_source
        from chat.models import Asset

        ctx = self.context
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
                ctx.pending_native_assets.append({
                    "asset_id": sid,  # label only — unused by the pipeline drain
                    "b64": base64.b64encode(data).decode("ascii"),
                    "media_type": ct or "image/png",
                    "description": f"Rendered slide {sid} (page {info.get('page')})",
                })
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
        "List the slide themes available for the current deck — the built-in Forest, the "
        "organization's themes, and the user's own themes. Each has an id, a name, a scope "
        "(builtin/org/user) and a short summary of its palette and footer, and the deck's current "
        "theme is flagged. Apply one with slides_set_theme; the user manages their own themes from "
        "the deck panel."
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
        "individual colours afterwards with slide_canvas_edit. If the theme carries a logo, place it "
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
                "message": "No deck to theme. Create one with slide_canvas_write first.",
            })

        override = theme_mod.theme_to_deck_override(chosen)
        with service.locked_deck(deck.pk) as deck:
            if deck is None:
                return json.dumps({"status": "error", "message": "The deck was deleted."})
            content = deck.content or {}
            content["theme"] = override
            service.save_deck_content(deck, content)
            service.create_deck_checkpoint(
                deck, source="ai_edit", description=f"Applied '{chosen['label']}' theme"
            )
        service.activate_deck(thread_id, deck)
        slide_ids = [s.get("id") for s in (content.get("slides") or [])]
        res = {
            "status": "ok", "applied": chosen["label"], "theme_id": chosen["id"],
            "deck_id": str(deck.pk), "title": deck.title,
            "slide_ids": slide_ids, "changed_slide_ids": slide_ids,
        }
        if chosen.get("logo_ext"):
            res["note"] = ("This theme has a logo — place it where it fits (e.g. the cover) "
                           "with the token [[image:company-logo]].")
        return json.dumps(res)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(ActivateDeckTool())
_registry.register_tool(WriteDeckTool())
_registry.register_tool(EditDeckTool())
_registry.register_tool(AddSlideTool())
_registry.register_tool(DeleteDeckTool())
_registry.register_tool(PreviewSlidesTool())
_registry.register_tool(ListThemesTool())
_registry.register_tool(SetThemeTool())
