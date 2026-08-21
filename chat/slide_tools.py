"""Slide-deck tools: author and edit a per-thread slide deck as JSON.

The deck sibling of chat/canvas_tools.py. The assistant writes/edits one
canonical JSON document (chat.slides.schema); the user only ever sees the
rendered filmstrip. All tools are audience="main", section="skills" (gated
behind the "Slide Deck Collaborator" seed skill).

``slides_preview_slide`` (the dispatch+poll visual feedback tool) lives here too
but is wired to the worker render task in a later phase.
"""

from __future__ import annotations

import copy
import json

from pydantic import BaseModel, Field, field_validator

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------
class ActivateDeckInput(ReasonBaseModel):
    deck_names: list[str] = Field(
        description="Deck title(s) to make active (max 1). The active deck's JSON is in your context.",
    )


class WriteDeckInput(ReasonBaseModel):
    title: str = Field(description="Title for the deck.")
    content_json: str = Field(
        description=(
            "The FULL deck as a JSON string (see the schema in your instructions): "
            '{"version":1,"size":{"w":960,"h":540},"slides":[...]}. Coordinates are '
            "points; the slide is 960x540. Omit slide/element ids and they are minted."
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
        description="Layout id to seed (title/section/bullets/two_col/image_right/table/metric/quote/closing/blank).",
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


def _thread_id(tool) -> str | None:
    return tool.context.conversation_id if tool.context else None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class ActivateDeckTool(ContextAwareTool):
    name: str = "slide_canvas_activate"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Opening deck..."
    end_label: str = "Updated the active deck"
    description: str = (
        "Set which slide deck is active. The active deck's full JSON is included in "
        "your context so you can edit it. Activating a deck deactivates the others."
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
        "Create or completely rewrite a slide deck from a full JSON document. Use for "
        "a new deck or a full restructure; for targeted changes use slide_canvas_edit. "
        "The user sees a rendered preview, never the JSON."
    )
    args_schema: type[BaseModel] = WriteDeckInput

    def _run(self, title: str, content_json: str, deck_name: str = "", **kwargs) -> str:
        from chat.slides import schema, service

        thread_id = _thread_id(self)
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        try:
            deck = json.loads(content_json)
        except (ValueError, TypeError) as exc:
            return json.dumps({"status": "error", "message": f"content_json is not valid JSON: {exc}"})
        if not isinstance(deck, dict):
            return json.dumps({"status": "error", "message": "content_json must be a JSON object."})

        schema.mint_ids(deck)
        issues = schema.validate_deck(deck)
        if issues:
            return json.dumps({"status": "error", "message": "Deck JSON is invalid.", "issues": issues[:20]})

        try:
            obj, created, old = service.write_deck(
                thread_id, title=title, content=deck, deck_name=deck_name
            )
        except service.DeckLimitError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        service.create_deck_checkpoint(
            obj, source="original" if created else "ai_edit",
            description="Created deck" if created else "Full rewrite",
        )
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
        "Make targeted find-replace edits to the active deck's JSON. Each edit's "
        "old_text must match exactly once in the canonical deck JSON shown in your "
        "context — include surrounding text to make it unique. Edits that would "
        "produce invalid deck JSON are rejected and the deck is left unchanged."
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

        original = schema.canonical_deck_text(deck.content)
        pairs = [
            (
                e.get("old_text", "") if isinstance(e, dict) else e.old_text,
                e.get("new_text", "") if isinstance(e, dict) else e.new_text,
            )
            for e in edits
        ]
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
        return json.dumps({
            "status": "ok", "applied": applied, "failed": failed,
            "changed_slide_ids": changed, "deck_id": str(deck.pk), "title": deck.title,
        })


class AddSlideTool(ContextAwareTool):
    name: str = "slides_add_slide"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Adding slide..."
    end_label: str = "Added a slide"
    description: str = (
        "Insert a pre-designed layout slide into the active deck, then edit its "
        "placeholder text with slide_canvas_edit. Returns the new slide's JSON and id."
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
        return json.dumps({
            "status": "ok",
            "deck_id": str(deck.pk),
            "slide_id": inserted.get("id"),
            "position": idx,
            "slide_json": schema.canonical_deck_text(inserted),
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


# Register on import
_registry = get_tool_registry()
_registry.register_tool(ActivateDeckTool())
_registry.register_tool(WriteDeckTool())
_registry.register_tool(EditDeckTool())
_registry.register_tool(AddSlideTool())
_registry.register_tool(DeleteDeckTool())
