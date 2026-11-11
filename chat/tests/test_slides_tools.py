"""Tests for the slide-deck tools (DB-backed)."""

from __future__ import annotations

import base64
import json
import uuid
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatThread, SlideSet, SlideRenderRun
from chat.slides.schema import canonical_deck_text
from chat.slide_tools import (
    ActivateDeckTool, AddSlideTool, CreateDeckTool, DeleteDeckTool, EditDeckTool,
    ListThemesTool, PreviewSlidesTool, SetThemeTool, WriteDeckTool,
)
from llm.types.context import RunContext

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _deck_json():
    return json.dumps({
        "version": 1, "size": {"w": 960, "h": 540},
        "slides": [
            {"name": "Title", "skip_footer": True, "elements": [
                {"type": "text", "x": 80, "y": 210, "w": 800, "h": 100, "class": "headline",
                 "paragraphs": [{"align": "center", "runs": [{"t": "Hello Deck"}]}]},
            ]},
        ],
    })


class SlideToolTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"t+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.ctx = RunContext(run_id="r1", conversation_id=str(self.thread.id), user_id=str(self.user.id))

    def _run(self, tool, **kw):
        tool.set_context(self.ctx)
        return json.loads(tool._run(**kw))

    def test_create_result_carries_an_outline_not_the_deck_json(self):
        """The agent reads or writes each slide next; creation returns ids + an
        outline, never the deck JSON."""
        r = self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets"])
        self.assertNotIn("deck_json", r)
        self.assertNotIn("Presentation title", json.dumps(r))
        self.assertEqual(r["outline"], ['1. s1 title "Title"', '2. s2 bullets "Bullets"'])

    def test_create_empty_deck(self):
        r = self._run(CreateDeckTool(), title="Scratch")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], [])
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertTrue(deck.is_active)
        self.assertEqual(deck.content["slides"], [])
        self.assertEqual([c.source for c in deck.checkpoints.all()], ["original"])
        # ...and it's a valid deck we can immediately seed a layout into.
        seed = self._run(AddSlideTool(), layout="title")
        self.assertEqual(seed["status"], "ok")

    def test_create_with_layouts_storyboard(self):
        """The ordered layouts list seeds one slide per id, in order, each tagged
        with its layout."""
        r = self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets", "closing"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], ["s1", "s2", "s3"])
        self.assertEqual(r["changed_slide_ids"], ["s1", "s2", "s3"])
        self.assertEqual(r["slide_count"], 3)
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual([s["layout"] for s in deck.content["slides"]], ["title", "bullets", "closing"])
        # The agent hasn't seen any slide JSON yet.
        self.assertEqual(deck.agent_seen, {})
        self.assertTrue(deck.is_active)
        self.assertEqual(len(deck.content["slides"]), 3)
        self.assertEqual(deck.content["slides"][0]["name"], "Title")

    def test_create_unknown_layout_is_all_or_nothing(self):
        r = self._run(CreateDeckTool(), title="Deck A", layouts=["title", "nope"])
        self.assertEqual(r["status"], "error")
        self.assertIn("bullets", r["available_layouts"])
        self.assertFalse(SlideSet.objects.filter(thread=self.thread).exists())

    def test_create_duplicate_title_errors(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(CreateDeckTool(), title="Deck A")
        self.assertEqual(r["status"], "error")
        self.assertEqual(SlideSet.objects.filter(thread=self.thread).count(), 1)

    def test_create_respects_deck_limit(self):
        from chat.slides.schema import MAX_SLIDE_SETS_PER_THREAD

        for i in range(MAX_SLIDE_SETS_PER_THREAD):
            self.assertEqual(self._run(CreateDeckTool(), title=f"Deck {i}")["status"], "ok")
        r = self._run(CreateDeckTool(), title="One more")
        self.assertEqual(r["status"], "error")

    def test_write_rewrites_active_deck(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(WriteDeckTool(), content_json=_deck_json())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], ["s1"])
        self.assertEqual(r["title"], "Deck A")  # empty title keeps the name
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content["slides"][0]["id"], "s1")
        self.assertEqual(
            [c.source for c in deck.checkpoints.order_by("order")], ["original", "ai_edit"]
        )

    def test_write_can_rename(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(WriteDeckTool(), rename_to="Deck B", content_json=_deck_json())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["title"], "Deck B")
        self.assertEqual(r["renamed_from"], "Deck A")

    def test_write_without_rename_keeps_title_and_says_nothing(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(WriteDeckTool(), content_json=_deck_json())
        self.assertEqual(r["title"], "Deck A")
        self.assertNotIn("renamed_from", r)

    def test_legacy_title_arg_does_not_rename(self):
        """2026-11-11 prod: `title` silently renamed the target deck. An old call
        replaying it writes the content and keeps the name."""
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(WriteDeckTool(), title="Deck B", content_json=_deck_json())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["title"], "Deck A")
        self.assertNotIn("title", WriteDeckTool().args_schema.model_fields)

    def test_rename_onto_another_decks_title_is_refused_and_writes_nothing(self):
        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        self._run(CreateDeckTool(), title="Deck B", layouts=["bullets"])  # now active
        before = SlideSet.objects.get(thread=self.thread, title="Deck B")
        r = self._run(WriteDeckTool(), rename_to="Deck A", content_json=_deck_json())
        self.assertEqual(r["status"], "error")
        self.assertIn("already exists", r["message"])
        self.assertIn("deck_name='Deck A'", r["message"])
        self.assertEqual(sorted(r["available_decks"]), ["Deck A", "Deck B"])
        after = SlideSet.objects.get(pk=before.pk)
        self.assertEqual(after.title, "Deck B")
        self.assertEqual(after.content, before.content)
        self.assertEqual(after.checkpoints.count(), before.checkpoints.count())

    def test_rename_collision_in_a_race_is_still_structured(self):
        """The unique constraint catches a title taken between check and save."""
        from django.db import IntegrityError

        self._run(CreateDeckTool(), title="Deck A")
        with mock.patch("chat.slides.service.create_deck_checkpoint", side_effect=IntegrityError("dup")):
            r = self._run(WriteDeckTool(), rename_to="Deck Z", content_json=_deck_json())
        self.assertEqual(r["status"], "error")
        self.assertIn("'Deck Z' already exists", r["message"])
        self.assertEqual(SlideSet.objects.get(thread=self.thread).title, "Deck A")

    def test_write_runs_first_only_when_renaming(self):
        self.assertTrue(WriteDeckTool().runs_first({"rename_to": "New"}))
        self.assertFalse(WriteDeckTool().runs_first({"rename_to": "  "}))
        self.assertFalse(WriteDeckTool().runs_first({"content": {}}))

    def test_write_without_deck_errors(self):
        """Creation moved to slides_create_deck — a rewrite with no deck must point there."""
        r = self._run(WriteDeckTool(), content_json=_deck_json())
        self.assertEqual(r["status"], "error")
        self.assertIn("slides_create_deck", r["message"])

    def test_write_requires_content(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(WriteDeckTool())
        self.assertEqual(r["status"], "error")
        self.assertIn("content", r["message"])

    def test_write_accepts_native_object(self):
        """The deck may be passed as a native JSON object (no string escaping) —
        the primary path for the model, which avoids double-escaping errors."""
        self._run(CreateDeckTool(), title="Obj")
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [
            {"name": "Title", "skip_footer": True, "elements": [
                {"type": "text", "x": 80, "y": 210, "w": 800, "h": 100, "class": "headline",
                 "paragraphs": [{"align": "center", "runs": [{"t": "Object Deck"}]}]},
            ]},
        ]}
        r = self._run(WriteDeckTool(), content=deck)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], ["s1"])
        self.assertEqual(SlideSet.objects.get(pk=r["deck_id"]).content["slides"][0]["id"], "s1")

    def test_write_invalid_json(self):
        self._run(CreateDeckTool(), title="X")
        r = self._run(WriteDeckTool(), content_json="{not json")
        self.assertEqual(r["status"], "error")

    def test_write_invalid_deck(self):
        self._run(CreateDeckTool(), title="X")
        bad = json.dumps({"slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 1}]}]})
        r = self._run(WriteDeckTool(), content_json=bad)
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["issues"])

    def test_write_string_slide_returns_error_not_crash(self):
        """WILFRED-81: a slide sent as a bare string used to crash mint_ids (it runs
        before validation). It must now come back as a structured validation error."""
        self._run(CreateDeckTool(), title="X")
        bad = json.dumps({"version": 1, "size": {"w": 960, "h": 540}, "slides": ["not a dict"]})
        r = self._run(WriteDeckTool(), content_json=bad)
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["issues"])

    def test_add_slide(self):
        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        r = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_id"], "s2")
        self.assertIn('"id": "s2"', r["slide_json"])

    def test_add_and_edit_return_full_slide_ids(self):
        """add_slide / edit must return the full ordered slide_ids (and add names the
        new one in changed_slide_ids). The client rebuilds its filmstrip from this list,
        so without it a newly added slide never appears live — it needed a manual F5."""
        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        added = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(added["slide_ids"], ["s1", "s2"])
        self.assertEqual(added["changed_slide_ids"], ["s2"])
        self.assertEqual(added["slide_count"], 2)
        edited = self._run(EditDeckTool(), edits=[
            {"slide_id": "s1", "old_text": '"Presentation title"', "new_text": '"Hi"'},
        ])
        self.assertEqual(edited["slide_ids"], ["s1", "s2"])

    def test_add_slide_unknown_layout(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(AddSlideTool(), layout="nope")
        self.assertEqual(r["status"], "error")
        self.assertIn("bullets", r["available_layouts"])

    def test_layout_catalog_ids(self):
        from chat.slides.layouts import layout_catalog

        ids = [c["id"] for c in layout_catalog()]
        for lid in ("title", "title_split", "photo", "section", "bullets", "two_col", "chart",
                    "image_bleed", "swimlane", "roadmap_gantt", "issue_tree", "closing", "blank"):
            self.assertIn(lid, ids)
        # Consolidated 2026-09: one cover + one divider (their extras live in the
        # slide comment); table/comparison/process/cycle/timeline/matrix/ecosystem
        # moved into the skill as element recipes; "team" removed pending a redesign.
        for gone in ("title_01", "title_02", "title_03", "section_02", "table", "comparison",
                     "process", "cycle", "timeline", "matrix_2x2", "ecosystem", "team"):
            self.assertNotIn(gone, ids)

    def test_layout_descriptions_say_when_to_use(self):
        """Every catalogue line tells the model WHEN to pick the layout."""
        from chat.slides.layouts import layout_catalog

        for c in layout_catalog():
            self.assertIn("use ", c["description"].lower(), c["id"])

    def test_retired_layout_elements_survive_as_comments(self):
        """The optional elements of the retired cover/divider/comparison seeds are
        pre-designed in the surviving layouts' comments (exact geometry included)."""
        from chat.slides.layouts import get_layout

        title = get_layout("title")["comment"]
        self.assertIn("[[image:company-logo]]", title)
        self.assertIn("y1=462", title)          # the hairline info band
        section = get_layout("section")["comment"]
        self.assertIn("'01'", section)          # optional section number
        self.assertIn("y=336", section)         # optional sub-heading
        two_col = get_layout("two_col")["comment"]
        self.assertIn("x1=474", two_col)        # the comparison variant's divider
        chart = get_layout("chart")["comment"]
        self.assertIn("RIGHT", chart)           # how to flip the chart side

    def test_set_theme_keeps_deck_effects(self):
        """A named-theme switch must not reset the deck-wide shadow choice."""
        r = self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        deck = SlideSet.objects.get(pk=r["deck_id"])
        content = deck.content
        content.setdefault("theme", {})["effects"] = {"shape_shadow": True}
        deck.content = content
        deck.save(update_fields=["content"])
        s = self._run(SetThemeTool(), theme="slate")
        self.assertEqual(s["status"], "ok")
        deck.refresh_from_db()
        self.assertEqual(deck.content["theme"]["_theme_id"], "slate")
        self.assertTrue(deck.content["theme"]["effects"]["shape_shadow"])

    def test_add_slide_carries_layout_comment(self):
        """A seed's authoring ``comment`` travels into the inserted slide JSON."""
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(AddSlideTool(), layout="image_right")
        self.assertEqual(r["status"], "ok")
        self.assertIn('"comment"', r["slide_json"])
        self.assertIn("either side", r["slide_json"])

    def test_edit_valid(self):
        w = self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        r = self._run(EditDeckTool(), edits=[
            {"slide_id": "s1", "old_text": '"Presentation title"', "new_text": '"Welcome"'},
        ])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["applied"], 1)
        self.assertEqual(r["changed_slide_ids"], ["s1"])
        self.assertEqual(r["edited"], ["slide 1 · Title"])
        self.assertEqual(EditDeckTool().end_label_for_result(r), "Edited slide 1 · Title")
        deck = SlideSet.objects.get(pk=w["deck_id"])
        self.assertIn("Welcome", json.dumps(deck.content))
        self.assertIn("s1", deck.agent_seen)  # an edit is a write: the slide counts as seen

    def test_edit_json_guard_rejects_and_preserves(self):
        w = self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        deck = SlideSet.objects.get(pk=w["deck_id"])
        before = json.dumps(deck.content, sort_keys=True)
        r = self._run(EditDeckTool(), edits=[
            {"slide_id": "s1", "old_text": '"skip_footer": true', "new_text": '"skip_footer": true OOPS'},
        ])
        self.assertEqual(r["status"], "error")
        self.assertIn("invalid JSON", r["message"])
        deck.refresh_from_db()
        self.assertEqual(json.dumps(deck.content, sort_keys=True), before)
        # no ai_edit checkpoint created for the rejected edit
        self.assertEqual(deck.checkpoints.filter(source="ai_edit").count(), 0)

    def test_edit_nonunique_reports_failure(self):
        json_deck = json.dumps({"version": 1, "size": {"w": 960, "h": 540}, "slides": [
            {"elements": [{"type": "text", "x": 1, "y": 1, "w": 1, "h": 1,
                           "paragraphs": [{"runs": [{"t": "dup"}]}, {"runs": [{"t": "dup"}]}]}]},
        ]})
        self._run(CreateDeckTool(), title="Deck A")
        self._run(WriteDeckTool(), content_json=json_deck)
        r = self._run(EditDeckTool(), edits=[{"slide_id": "s1", "old_text": '"t": "dup"', "new_text": '"t": "x"'}])
        self.assertEqual(r["status"], "error")
        self.assertTrue(any("matches" in f["error"] for f in r["failed"]))

    def test_edit_text_shared_by_two_slides_is_unique_per_slide(self):
        self._run(CreateDeckTool(), title="Deck A", layouts=["bullets", "bullets"])
        r = self._run(EditDeckTool(), edits=[
            {"slide_id": "s2", "old_text": '"t": "Slide title"', "new_text": '"t": "Zweite"'},
        ])
        self.assertEqual(r["status"], "ok", r)
        deck = SlideSet.objects.get(title="Deck A")
        self.assertNotIn("Zweite", json.dumps(deck.content["slides"][0]))
        self.assertIn("Zweite", json.dumps(deck.content["slides"][1]))

    def test_edit_deck_settings(self):
        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        r = self._run(EditDeckTool(), edits=[{"slide_id": "", "old_text": '"version": 1', "new_text": '"version": 1'}])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(EditDeckTool().end_label_for_result(r), "Edited the deck settings")

    # -- slides_read ------------------------------------------------------------
    def test_read_by_ids_returns_canonical_slide_json_and_stamps_seen(self):
        from chat.slide_tools import ReadSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets", "closing"])
        r = self._run(ReadSlidesTool(), slide_ids=["s3", "nope"])
        self.assertEqual([s["n"] for s in r["slides"]], [3])
        self.assertEqual(r["unknown_slide_ids"], ["nope"])
        deck = SlideSet.objects.get(title="Deck A")
        self.assertEqual(r["slides"][0]["json"], canonical_deck_text(deck.content["slides"][2]))
        self.assertEqual(set(deck.agent_seen), {"s3"})
        self.assertEqual(ReadSlidesTool().end_label_for_result(r), "Read slide 3")
        # The JSON it returned is exactly what the edit tool matches against.
        e = self._run(EditDeckTool(), edits=[
            {"slide_id": "s3", "old_text": r["slides"][0]["json"][:40], "new_text": r["slides"][0]["json"][:40]},
        ])
        self.assertEqual(e["status"], "ok")

    def test_read_range_and_label(self):
        from chat.slide_tools import ReadSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets", "bullets", "closing"])
        r = self._run(ReadSlidesTool(), start=2, end=4)
        self.assertEqual([s["n"] for s in r["slides"]], [2, 3, 4])
        self.assertNotIn("next_start", r)
        self.assertEqual(ReadSlidesTool().end_label_for_result(r), "Read slides 2–4")

    def test_read_pages_at_the_token_cap(self):
        from chat.slide_tools import ReadSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["bullets"] * 5)
        with mock.patch("llm.context_budget.canvas_read_token_cap", return_value=1):
            r = self._run(ReadSlidesTool())
        self.assertEqual([s["n"] for s in r["slides"]], [1])  # at least one slide
        self.assertEqual(r["next_start"], 2)
        with mock.patch("llm.context_budget.canvas_read_token_cap", return_value=1):
            by_id = self._run(ReadSlidesTool(), slide_ids=["s4", "s5"])
        self.assertEqual(by_id["not_included_slide_ids"], ["s5"])

    def test_read_deck_settings_only(self):
        from chat.slide_tools import ReadSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        r = self._run(ReadSlidesTool(), deck_settings=True)
        self.assertEqual(r["slides"], [])
        self.assertIn('"size"', r["deck_settings"])
        self.assertNotIn('"slides"', r["deck_settings"])

    def test_read_include_render_uses_the_preview_helper(self):
        from chat.slide_tools import ReadSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets"])
        with mock.patch("chat.slide_tools._render_and_attach", return_value={"status": "ok"}) as render:
            r = self._run(ReadSlidesTool(), slide_ids=["s2"], include_render=True)
        self.assertEqual(r["render"], {"status": "ok"})
        self.assertEqual(render.call_args.args[2], ["s2"])

    # -- slides_write_slide / move / delete / duplicate ----------------------------
    def _slide(self, title):
        return {"name": title, "elements": [
            {"type": "text", "x": 40, "y": 40, "w": 880, "h": 60, "class": "headline",
             "paragraphs": [{"runs": [{"t": title}]}]},
        ]}

    def test_write_slide_replaces_in_place(self):
        from chat.slide_tools import WriteSlideTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets"])
        r = self._run(WriteSlideTool(), slide_id="s2", content=self._slide("Market size"))
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["slide_ids"], ["s1", "s2"])
        self.assertEqual(r["changed_slide_ids"], ["s2"])
        self.assertEqual(WriteSlideTool().end_label_for_result(r), "Wrote slide 2 · Market size")
        deck = SlideSet.objects.get(title="Deck A")
        self.assertIn("w", deck.agent_seen["s2"])

    def test_write_slide_inserts_and_accepts_a_string(self):
        from chat.slide_tools import WriteSlideTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        r = self._run(WriteSlideTool(), content=json.dumps(self._slide("Agenda")), position=0)
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["slide_ids"][0], r["slide_id"])
        self.assertEqual(WriteSlideTool().end_label_for_result(r), "Added slide 1 · Agenda")

    def test_write_slide_rejects_a_deck_and_invalid_slides(self):
        from chat.slide_tools import WriteSlideTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        self.assertEqual(self._run(WriteSlideTool(), content={"slides": []})["status"], "error")
        bad = self._run(WriteSlideTool(), slide_id="s1", content={"elements": [{"type": "bogus"}]})
        self.assertEqual(bad["status"], "error")
        self.assertTrue(bad["issues"])
        self.assertEqual(self._run(WriteSlideTool(), slide_id="s9", content=self._slide("x"))["status"], "error")

    def test_move_delete_duplicate(self):
        from chat.slide_tools import DeleteSlidesTool, DuplicateSlideTool, MoveSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets", "closing"])
        m = self._run(MoveSlidesTool(), slide_ids=["s3"], position=0)
        self.assertEqual(m["slide_ids"], ["s3", "s1", "s2"])
        self.assertEqual(MoveSlidesTool().end_label_for_result(m), "Moved slide 1 · Closing")
        self.assertEqual(m["outline"][0], '1. s3 closing "Closing"')
        dup = self._run(DuplicateSlideTool(), slide_id="s2")
        self.assertEqual(dup["slide_ids"][3], dup["slide_id"])
        self.assertEqual(dup["changed_slide_ids"], [dup["slide_id"]])
        self.assertEqual(DuplicateSlideTool().end_label_for_result(dup), "Duplicated slide 3 · Bullets")
        d = self._run(DeleteSlidesTool(), slide_ids=["s1"])
        self.assertEqual(d["slide_ids"], ["s3", "s2", dup["slide_id"]])
        self.assertEqual(DeleteSlidesTool().end_label_for_result(d), "Deleted slide 2 · Title")
        everything = self._run(DeleteSlidesTool(), slide_ids=d["slide_ids"])
        self.assertEqual(everything["status"], "error")

    def test_move_only_reports_slides_whose_pixels_change(self):
        """With the default theme the page number is drawn, so the moved slides and
        the ones they jumped over change — skip_footer slides (title/closing) don't."""
        from chat.slide_tools import MoveSlidesTool

        self._run(CreateDeckTool(), title="Deck A", layouts=["title", "bullets", "bullets", "closing"])
        m = self._run(MoveSlidesTool(), slide_ids=["s3"], position=1)
        self.assertEqual(m["slide_ids"], ["s1", "s3", "s2", "s4"])
        self.assertEqual(sorted(m["changed_slide_ids"]), ["s2", "s3"])

    # -- trimming ------------------------------------------------------------------
    def test_trim_hooks_point_at_slides_read(self):
        from chat.slide_tools import WriteSlideTool

        for tool, args in (
            (WriteDeckTool(), {"content": {"slides": []}}),
            (EditDeckTool(), {"edits": [{"slide_id": "s1", "old_text": "a", "new_text": "b"}]}),
            (WriteSlideTool(), {"content": {"name": "x"}, "slide_id": "s1"}),
        ):
            trim = tool.trim_args_at_edit_point(args, later_calls=[])
            self.assertIn("slides_read", trim.note, tool.name)
            self.assertFalse({"content", "edits"} & set(trim.args), tool.name)
            self.assertFalse(tool.trim_despite_recency, tool.name)

    def test_add_slide_result_hook_drops_slide_json(self):
        raw_tool = AddSlideTool()
        raw_tool.set_context(self.ctx)
        self._run(CreateDeckTool(), title="Deck A")
        raw = raw_tool._run(layout="bullets")
        out = json.loads(raw_tool.trim_result_at_edit_point(raw, {}, later_calls=[]))
        self.assertNotIn("slide_json", out)
        self.assertIn("slides_read", out["omitted_fields_note"])
        self.assertEqual(out["slide_id"], json.loads(raw)["slide_id"])
        self.assertIsNone(raw_tool.trim_result_at_edit_point(json.dumps(out), {}, later_calls=[]))

    def test_activate_and_delete(self):
        self._run(CreateDeckTool(), title="Deck A")
        r = self._run(ActivateDeckTool(), deck_names=["Deck A"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["activated"][0]["title"], "Deck A")
        d = self._run(DeleteDeckTool(), deck_name="Deck A")
        self.assertEqual(d["status"], "ok")
        self.assertIsNotNone(SlideSet.objects.get(title="Deck A").deleted_at)

    def test_delete_active_promotes_survivor(self):
        """Deleting the active deck when others remain promotes the newest survivor
        so the thread keeps an active deck (the panel can switch instead of orphaning)."""
        self._run(CreateDeckTool(), title="Deck A")
        self._run(CreateDeckTool(), title="Deck B")
        # Deck B is active (created last); delete it and Deck A should take over.
        self.assertTrue(SlideSet.objects.get(title="Deck B").is_active)
        self._run(DeleteDeckTool(), deck_name="Deck B")
        self.assertIsNotNone(SlideSet.objects.get(title="Deck B").deleted_at)
        self.assertTrue(SlideSet.objects.get(title="Deck A").is_active)

    def test_edit_no_deck(self):
        r = self._run(EditDeckTool(), edits=[{"old_text": "a", "new_text": "b"}])
        self.assertEqual(r["status"], "error")


class DeckConcurrentWriteTests(TestCase):
    """A tool must mutate the deck it re-reads under the lock, never the snapshot
    it resolved beforehand.

    Tool batches run concurrently in a ThreadPoolExecutor (llm/pipelines/simple_chat.py),
    so two ``slides_add_slide`` calls used to deep-copy the same pre-call content and the
    later save silently dropped the earlier slide — while both returned ``status: ok``.
    A stale deck instance reproduces exactly that lost update deterministically; real row
    locks can't be exercised here because ``select_for_update`` is a no-op on SQLite.
    """

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"c+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.ctx = RunContext(run_id="r1", conversation_id=str(self.thread.id), user_id=str(self.user.id))

    def _run(self, tool, **kw):
        tool.set_context(self.ctx)
        return json.loads(tool._run(**kw))

    def _stale(self, deck_id):
        """A deck instance holding the content as it was *before* the next call."""
        return SlideSet.objects.get(pk=deck_id)

    def test_add_slide_reads_fresh_content_under_lock(self):
        w = self._run(CreateDeckTool(), title="Deck A")
        stale = self._stale(w["deck_id"])  # zero slides
        self.assertEqual(stale.content["slides"], [])

        first = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(first["status"], "ok")

        with mock.patch("chat.slides.service.resolve_deck", return_value=(stale, None)):
            second = self._run(AddSlideTool(), layout="title")
        self.assertEqual(second["status"], "ok")

        deck = SlideSet.objects.get(pk=w["deck_id"])
        slides = deck.content["slides"]
        self.assertEqual(len(slides), 2, "the concurrent add clobbered the first slide")
        ids = [s["id"] for s in slides]
        self.assertEqual(len(set(ids)), 2, f"slide ids collided: {ids}")
        self.assertNotEqual(first["slide_id"], second["slide_id"])

    def test_edit_deck_reads_fresh_content_under_lock(self):
        w = self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        stale = self._stale(w["deck_id"])  # one slide

        added = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(added["status"], "ok")

        with mock.patch("chat.slides.service.resolve_deck", return_value=(stale, None)):
            r = self._run(EditDeckTool(), edits=[
                {"slide_id": "s1", "old_text": '"Presentation title"', "new_text": '"Welcome"'},
            ])

        deck = SlideSet.objects.get(pk=w["deck_id"])
        # Whether the edit applied or cleanly failed, the concurrently added slide
        # must survive — the stale snapshot must never be written back.
        self.assertEqual(len(deck.content["slides"]), 2, "the edit reverted the added slide")
        if r["status"] == "ok":
            self.assertIn("Welcome", json.dumps(deck.content))
        else:
            self.assertEqual(r["applied"], 0)

    def test_checkpoint_order_strictly_increases(self):
        """Regression guard on checkpoint ordering (Undo + guardrail rollback read it).

        Passes on the unlocked code: ``_next_order`` re-queries the DB, so it is immune
        to the stale-instance trick above. The real race needs true concurrency, which
        only the row lock closes.
        """
        w = self._run(CreateDeckTool(), title="Deck A", layouts=["title"])
        self._run(AddSlideTool(), layout="bullets")
        self._run(AddSlideTool(), layout="closing")
        self._run(EditDeckTool(), edits=[
            {"slide_id": "s1", "old_text": '"Presentation title"', "new_text": '"Welcome"'},
        ])

        deck = SlideSet.objects.get(pk=w["deck_id"])
        orders = list(deck.checkpoints.order_by("id").values_list("order", flat=True))
        self.assertEqual(len(orders), 4)
        self.assertEqual(orders, sorted(set(orders)), f"checkpoint order not monotonic: {orders}")


class BuildResolverTests(TestCase):
    """build_pptx image-token resolution + the owner-scoping leak guard."""

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"i+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")

    def _img_deck(self, token, title="D"):
        return SlideSet.objects.create(
            thread=self.thread, title=title,
            content={"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"id": "s1", "elements": [
                    {"id": "e1", "type": "image", "x": 100, "y": 100, "w": 300, "h": 200, "token": token},
                ]},
            ]},
        )

    def test_owner_image_embeds(self):
        from chat.assets import store_thread_image
        from chat.slides.pptx_build import build_pptx

        asset = store_thread_image(self.thread, img_bytes=_PNG, content_type="image/png")
        data, warnings = build_pptx(self._img_deck(f"[[image:{asset.id}]]"))
        self.assertEqual(warnings, [])
        import io
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue(any(n.startswith("ppt/media/") for n in z.namelist()))

    def test_resolver_batch_loads_assets_in_one_query(self):
        """WILFRED-86: the resolver pre-loads every referenced asset in ONE query
        instead of one SELECT per distinct token during render."""
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        from chat.assets import store_thread_image
        from chat.slides.pptx_build import make_image_resolver

        a1 = store_thread_image(self.thread, img_bytes=_PNG, content_type="image/png")
        a2 = store_thread_image(self.thread, img_bytes=_PNG + b"x", content_type="image/png")
        a3 = store_thread_image(self.thread, img_bytes=_PNG + b"yy", content_type="image/png")
        deck = SlideSet.objects.create(
            thread=self.thread, title="multi",
            content={"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"id": "s1", "elements": [
                    {"id": f"e{i}", "type": "image", "x": 1, "y": 1, "w": 10, "h": 10,
                     "token": f"[[image:{a.id}]]"}
                    for i, a in enumerate((a1, a2, a3))
                ]},
            ]},
        )
        with CaptureQueriesContext(connection) as cap:
            resolve = make_image_resolver(deck)
            results = [resolve(f"[[image:{a.id}]]") for a in (a1, a2, a3)]
        asset_selects = [
            q for q in cap.captured_queries
            if "chat_asset" in q["sql"].lower() and q["sql"].lstrip().lower().startswith("select")
        ]
        self.assertEqual(len(asset_selects), 1, asset_selects)
        self.assertTrue(all(r is not None for r in results))

    def test_cross_owner_token_is_blocked(self):
        from chat.assets import store_thread_image
        from chat.slides.pptx_build import build_pptx

        User = get_user_model()
        other = User.objects.create_user(email=f"o+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        other_thread = ChatThread.objects.create(created_by=other, title="o")
        foreign = store_thread_image(other_thread, img_bytes=_PNG + b"x", content_type="image/png")
        _data, warnings = build_pptx(self._img_deck(f"[[image:{foreign.id}]]"))
        self.assertTrue(any("unavailable" in w for w in warnings))

    def _dataroom_image_token(self):
        """A data-room image asset (blob-less, version-owned) + its token.

        Returns ``(data_room, token)``; the data room is NOT yet attached to the
        deck's thread, so the caller controls whether the image is resolvable.
        """
        from django.core.files.base import ContentFile

        from chat.assets import get_or_create_version_image_token
        from documents.models import (
            DataRoom, DataRoomDocument, DataRoomDocumentVersion,
        )

        room = DataRoom.objects.create(
            name="Aldera DOFI", slug=f"r{uuid.uuid4().hex[:6]}", created_by=self.user,
        )
        doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="ntnu.png",
            mime_type="image/png", doc_index=1,
            status=DataRoomDocument.Status.READY,
        )
        doc.original_file.save("ntnu.png", ContentFile(_PNG), save=True)
        version = DataRoomDocumentVersion.objects.create(
            document=doc, version_index=0,
            origin=DataRoomDocumentVersion.Origin.UPLOADED, mime_type="image/png",
        )
        token = get_or_create_version_image_token(version_id=version.id, mime="image/png")
        return room, token

    def test_attached_data_room_image_embeds(self):
        """A data-room image renders in the deck when its room is attached."""
        import io
        import zipfile

        from chat.models import ChatThreadDataRoom
        from chat.slides.pptx_build import build_pptx

        room, token = self._dataroom_image_token()
        ChatThreadDataRoom.objects.create(thread=self.thread, data_room=room)
        data, warnings = build_pptx(self._img_deck(token))
        self.assertEqual(warnings, [])
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue(any(n.startswith("ppt/media/") for n in z.namelist()))

    def test_unattached_data_room_image_is_blocked(self):
        """The same token does NOT resolve when the room isn't attached — the
        attached-room gate is the ACL against referencing arbitrary UUIDs."""
        from chat.slides.pptx_build import build_pptx

        _room, token = self._dataroom_image_token()
        _data, warnings = build_pptx(self._img_deck(token))
        self.assertTrue(any("unavailable" in w for w in warnings))


def _pptx_parts(data):
    """Slide XML + media parts of a .pptx (core props carry a timestamp)."""
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(data))
    return {n: z.read(n) for n in z.namelist() if n.startswith(("ppt/slides/", "ppt/media/"))}


class PrefetchedResolverTests(TestCase):
    """prepare_pptx_build does every DB/storage read, so build_deck_pptx is pure."""

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"pf+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")

    def _deck(self, *tokens):
        return SlideSet.objects.create(
            thread=self.thread, title="D",
            content={"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"id": "s1", "elements": [
                    {"id": f"e{i}", "type": "image", "x": 100, "y": 100, "w": 300, "h": 200, "token": t}
                    for i, t in enumerate(tokens)
                ]},
            ]},
        )

    def test_build_after_prepare_touches_no_db_or_storage(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        from chat.assets import store_thread_image
        from chat.slides import pptx_build

        asset = store_thread_image(self.thread, img_bytes=_PNG, content_type="image/png")
        with mock.patch.object(pptx_build, "_footer_logo_bytes", return_value=_PNG):
            kwargs = pptx_build.prepare_pptx_build(self._deck(f"[[image:{asset.id}]]", "[[image:company-logo]]"))
        with CaptureQueriesContext(connection) as cap, \
                mock.patch("django.core.files.storage.default_storage.open", side_effect=AssertionError("storage read")), \
                mock.patch.object(pptx_build, "_footer_logo_bytes", side_effect=AssertionError("logo read")):
            data, warnings = pptx_build.build_deck_pptx(**kwargs)
        self.assertEqual(cap.captured_queries, [])
        self.assertEqual(warnings, [])
        self.assertEqual(sum(1 for n in _pptx_parts(data) if n.startswith("ppt/media/")), 1)

    def test_foreign_token_still_unavailable(self):
        from chat.assets import store_thread_image
        from chat.slides.pptx_build import build_deck_pptx, prepare_pptx_build

        other = get_user_model().objects.create_user(email=f"po+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        foreign = store_thread_image(
            ChatThread.objects.create(created_by=other, title="o"), img_bytes=_PNG, content_type="image/png",
        )
        _data, warnings = build_deck_pptx(**prepare_pptx_build(self._deck(f"[[image:{foreign.id}]]")))
        self.assertTrue(any("unavailable" in w for w in warnings))

    def test_same_output_as_the_live_resolver(self):
        from chat.assets import store_thread_image
        from chat.slides.pptx_build import build_deck_pptx, build_pptx, make_image_resolver

        a1 = store_thread_image(self.thread, img_bytes=_PNG, content_type="image/png")
        a2 = store_thread_image(self.thread, img_bytes=_PNG + b"x", content_type="image/png")
        deck = self._deck(f"[[image:{a1.id}]]", f"[[image:{a2.id}]]", f"[[image:{uuid.uuid4()}]]")
        new, new_w = build_pptx(deck)
        old, old_w = build_deck_pptx(deck.content, image_resolver=make_image_resolver(deck))
        self.assertEqual(new_w, old_w)
        self.assertEqual(_pptx_parts(new), _pptx_parts(old))


class SlidesExportViewTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email="sxv@ex.com", password="pass")
        self.client.login(email="sxv@ex.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.deck = SlideSet.objects.create(thread=self.thread, title="Q3 review", content=json.loads(_deck_json()))
        self.url = f"/chat/threads/{self.thread.id}/slides/{self.deck.id}/export/"

    def test_downloads_pptx(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        self.assertIn("Q3 review", resp["Content-Disposition"])
        self.assertTrue(b"".join(resp.streaming_content).startswith(b"PK"))

    def test_build_runs_off_the_request_thread(self):
        import threading

        from chat.slides import pptx_build

        seen = {}
        real = pptx_build.build_deck_pptx

        def spy(*a, **kw):
            seen["thread"] = threading.current_thread()
            return real(*a, **kw)

        with mock.patch.object(pptx_build, "build_deck_pptx", spy):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNot(seen["thread"], threading.main_thread())

    def test_busy_when_no_render_slot_frees_up(self):
        import threading

        import chat.views as views

        full = threading.BoundedSemaphore(1)
        full.acquire()
        with mock.patch.object(views, "_EXPORT_RENDER_SLOTS", full), \
                self.settings(EXPORT_RENDER_SLOT_WAIT_SECONDS=0):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("busy", resp.json()["error"])

    def test_build_failure_is_a_json_500(self):
        with mock.patch("chat.slides.pptx_build.build_deck_pptx", side_effect=RuntimeError("boom")):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 500)
        self.assertIn("error", resp.json())

    def test_other_users_and_deleted_decks_404(self):
        from django.utils import timezone

        other = get_user_model().objects.create_user(email="sxo@ex.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other, title="o")
        other_deck = SlideSet.objects.create(thread=other_thread, title="X", content=json.loads(_deck_json()))
        self.assertEqual(self.client.get(f"/chat/threads/{other_thread.id}/slides/{other_deck.id}/export/").status_code, 404)
        SlideSet.objects.filter(pk=self.deck.pk).update(deleted_at=timezone.now())
        self.assertEqual(self.client.get(self.url).status_code, 404)


class PreviewToolTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"p+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.ctx = RunContext(run_id="r1", conversation_id=str(self.thread.id), user_id=str(self.user.id))
        CreateDeckTool().set_context(self.ctx)._run(title="Deck A", layouts=["blank", "blank"])

    def _fake_delay(self, run_id_str):
        """Fabricate a completed render (no LibreOffice) for the polling tool."""
        from chat.assets import store_slide_set_image
        from chat.models import SlideRender

        run = SlideRenderRun.objects.get(pk=run_id_str)
        deck = run.slide_set
        slides = []
        for i, sid in enumerate(run.slide_ids):
            asset = store_slide_set_image(deck, img_bytes=_PNG)
            SlideRender.objects.update_or_create(
                slide_set=deck, slide_id=sid,
                defaults={"content_hash": "h", "asset": asset, "width": 100, "height": 56},
            )
            slides.append({"slide_id": sid, "asset_id": str(asset.id), "width": 100, "height": 56, "page": i + 1})
        run.status = SlideRenderRun.Status.COMPLETED
        run.result = {"slides": slides, "warnings": []}
        run.save(update_fields=["status", "result"])
        return mock.Mock(id="fake-task")

    def test_preview_attaches_images(self):
        tool = PreviewSlidesTool()
        tool.set_context(self.ctx)
        with mock.patch("chat.tasks.render_deck_task.delay", side_effect=self._fake_delay):
            r = json.loads(tool._run(slide_ids=["s1"]))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["previewed_count"], 1)
        self.assertEqual(len(self.ctx.pending_native_assets), 1)
        item = self.ctx.pending_native_assets[0]
        self.assertEqual(item["media_type"], "image/png")
        self.assertIn("Rendered slide s1", item["description"])

    def test_preview_broker_failure_is_graceful(self):
        tool = PreviewSlidesTool()
        tool.set_context(self.ctx)
        with mock.patch("chat.tasks.render_deck_task.delay", side_effect=RuntimeError("broker down")):
            r = json.loads(tool._run(slide_ids=["s1"]))
        self.assertEqual(r["status"], "unavailable")

    def test_preview_failed_render_is_graceful(self):
        def fail_delay(run_id_str):
            run = SlideRenderRun.objects.get(pk=run_id_str)
            run.status = SlideRenderRun.Status.FAILED
            run.error = "no soffice"
            run.save(update_fields=["status", "error"])
            return mock.Mock(id="x")

        tool = PreviewSlidesTool()
        tool.set_context(self.ctx)
        with mock.patch("chat.tasks.render_deck_task.delay", side_effect=fail_delay):
            r = json.loads(tool._run(slide_ids=["s1"]))
        self.assertEqual(r["status"], "unavailable")


class DeckThemeSeedingTests(TestCase):
    """A new deck inherits the org's default slide theme when the model omits
    one; an explicit theme is kept, and a user's picker choice survives a
    full rewrite."""

    def setUp(self):
        from accounts.models import Membership, Organization

        User = get_user_model()
        self.user = User.objects.create_user(email=f"s+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.org = Organization.objects.create(
            name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}",
            preferences={"slide_theme": {"name": "slate"}},
        )
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.ctx = RunContext(run_id="r1", conversation_id=str(self.thread.id), user_id=str(self.user.id))

    def _create(self, title="Deck", layouts=None):
        tool = CreateDeckTool()
        tool.set_context(self.ctx)
        return json.loads(tool._run(title=title, layouts=layouts or []))

    def _write(self, deck, title="", deck_name=""):
        tool = WriteDeckTool()
        tool.set_context(self.ctx)
        return json.loads(tool._run(title=title, content=deck, deck_name=deck_name))

    def _bare_deck(self, theme=None):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [
            {"name": "T", "elements": [
                {"type": "text", "x": 80, "y": 210, "w": 800, "h": 100, "class": "headline",
                 "paragraphs": [{"runs": [{"t": "Hi"}]}]},
            ]},
        ]}
        if theme is not None:
            deck["theme"] = theme
        return deck

    def test_new_deck_inherits_org_default_theme(self):
        # The org's legacy single style becomes its default theme; the deck's
        # theme override carries that palette (full v2 shape + provenance).
        r = self._create()
        deck = SlideSet.objects.get(pk=r["deck_id"])
        th = deck.content.get("theme") or {}
        self.assertEqual(th.get("colors", {}).get("dk1"), "#1E293B")  # slate body colour
        self.assertIn("footer", th)
        self.assertTrue(th.get("_theme_id"))

    def test_explicit_theme_is_untouched(self):
        custom = {"colors": {"accent1": "#123456"}}
        self._create()
        r = self._write(self._bare_deck(theme=custom))
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content["theme"], custom)

    def test_rewrite_preserves_existing_theme(self):
        from chat.slides.theme import preset_theme_override

        # Create the deck (gets the org default), then simulate the user picking
        # "ocean" from the slide panel, then a full rewrite that omits the theme.
        r = self._create()
        deck = SlideSet.objects.get(pk=r["deck_id"])
        deck.content["theme"] = preset_theme_override("ocean")
        deck.save(update_fields=["content"])

        self._write(self._bare_deck())
        deck.refresh_from_db()
        self.assertEqual(deck.content["theme"], preset_theme_override("ocean"))

    def test_no_membership_leaves_deck_theme_unset(self):
        # A user with no org gets no seeded theme (deck resolves to base forest).
        from accounts.models import Membership

        Membership.objects.filter(user=self.user).delete()
        r = self._create()
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertNotIn("theme", deck.content)


class ThemeToolTests(SlideToolTests):
    """slides_list_themes / slides_set_theme (v2 named themes)."""

    def _make_deck(self):
        return self._run(CreateDeckTool(), title="Deck")["deck_id"]

    def test_new_deck_seeded_forest_and_listed_current(self):
        self._make_deck()
        r = self._run(ListThemesTool())
        self.assertEqual(r["status"], "ok")
        forest = next((t for t in r["themes"] if t["id"] == "forest"), None)
        self.assertIsNotNone(forest)
        self.assertTrue(forest["current"])  # CreateDeck seeds the default (Forest)

    def test_set_forest_applies(self):
        self._make_deck()
        r = self._run(SetThemeTool(), theme="forest")
        self.assertEqual(r["status"], "ok")
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content["theme"]["_theme_id"], "forest")

    def test_set_unknown_theme_errors_with_choices(self):
        self._make_deck()
        r = self._run(SetThemeTool(), theme="does-not-exist")
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["available"])

    def test_set_user_theme_by_label_carries_footer_and_logo_hint(self):
        from accounts.models import UserSettings
        from chat.slides import theme as T

        th = T.slide_theme_defaults()
        th["id"], th["label"], th["logo_ext"] = "tuser1", "My Brand", "png"
        th["footer"] = {"bg_color": "dk2", "text": "x", "size": 9, "color": "dk2", "sections": [
            {"colspan": 4, "align": "left", "content": "logo"},
            {"colspan": 4, "align": "center", "content": "none"},
            {"colspan": 4, "align": "right", "content": "page"}]}
        UserSettings.objects.update_or_create(user=self.user, defaults={"preferences": {"slide_themes": [th]}})

        self._make_deck()
        r = self._run(SetThemeTool(), theme="My Brand")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["theme_id"], "tuser1")
        self.assertIn("company-logo", r.get("note", ""))
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content["theme"]["_theme_id"], "tuser1")
        self.assertEqual(deck.content["theme"]["footer"]["sections"][0]["content"], "logo")


class RunsFirstTests(TestCase):
    """Deck lifecycle calls finish before the rest of their batch
    (ContextAwareTool.runs_first; the pipeline side is in
    llm/tests/test_simple_chat_pipeline.py)."""

    def test_lifecycle_tools_run_first(self):
        for tool in (CreateDeckTool(), DeleteDeckTool(), ActivateDeckTool()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.runs_first({}))

    def test_content_tools_do_not(self):
        for tool in (EditDeckTool(), AddSlideTool(), SetThemeTool(), PreviewSlidesTool()):
            with self.subTest(tool=tool.name):
                self.assertFalse(tool.runs_first({}))

    def test_skill_resource_load_runs_first_only_for_a_deck(self):
        from agent_skills.tools import LoadTemplateToCanvasTool

        tool = LoadTemplateToCanvasTool()
        self.assertTrue(tool.runs_first({"target": "deck"}))
        self.assertFalse(tool.runs_first({"target": "canvas"}))
        self.assertFalse(tool.runs_first({}))
