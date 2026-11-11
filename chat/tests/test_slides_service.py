"""Tests for the per-slide deck service helpers and the agent outline."""

from __future__ import annotations

import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatThread, SlideComment, SlideSet
from chat.slides import layouts, schema, service


def _text_slide(title: str, body: str = "Body") -> dict:
    return {"name": title, "elements": [
        {"type": "text", "x": 40, "y": 40, "w": 880, "h": 60, "class": "headline",
         "paragraphs": [{"runs": [{"t": title}]}]},
        {"type": "text", "x": 40, "y": 140, "w": 880, "h": 300,
         "paragraphs": [{"runs": [{"t": body}]}]},
    ]}


class SlideServiceTests(TestCase):
    def setUp(self):
        user = get_user_model().objects.create_user(email=f"s+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=user, title="t")
        content = {"version": 1, "size": {"w": 960, "h": 540},
                   "slides": [_text_slide("One"), _text_slide("Two"), _text_slide("Three")]}
        schema.mint_ids(content)
        self.deck, _c, _o = service.write_deck(self.thread.id, title="Deck", content=content)

    def _ids(self):
        self.deck.refresh_from_db()
        return [s["id"] for s in self.deck.content["slides"]]

    # -- replace / edit -------------------------------------------------------
    def test_replace_slide_keeps_id_and_position(self):
        deck, old, new, issues = service.replace_slide(self.deck.pk, "s2", {"id": "zz", **_text_slide("New")})
        self.assertEqual(issues, [])
        self.assertEqual(self._ids(), ["s1", "s2", "s3"])
        self.assertEqual(self.deck.content["slides"][1]["name"], "New")
        self.assertEqual(self.deck.checkpoints.order_by("-order").first().source, "ai_edit")

    def test_replace_unknown_slide_raises_and_saves_nothing(self):
        before = self.deck.checkpoints.count()
        with self.assertRaises(service.SlideOpError):
            service.replace_slide(self.deck.pk, "nope", _text_slide("X"))
        self.assertEqual(self.deck.checkpoints.count(), before)

    def test_replace_invalid_slide_returns_issues(self):
        _d, _o, _n, issues = service.replace_slide(self.deck.pk, "s1", {"elements": [{"type": "bogus"}]})
        self.assertTrue(issues)
        self.deck.refresh_from_db()
        self.assertEqual(self.deck.content["slides"][0]["name"], "One")

    def test_edit_is_scoped_to_one_slide(self):
        # "Body" appears in every slide — unique within s2 is enough.
        *_r, applied, failed = service.edit_slides(self.deck.pk, [("s2", '"t": "Body"', '"t": "Changed"')])
        self.assertEqual((applied, failed), (1, []))
        self.deck.refresh_from_db()
        bodies = [s["elements"][1]["paragraphs"][0]["runs"][0]["t"] for s in self.deck.content["slides"]]
        self.assertEqual(bodies, ["Body", "Changed", "Body"])

    def test_edit_partial_misses_are_reported(self):
        *_r, applied, failed = service.edit_slides(self.deck.pk, [
            ("s1", '"t": "One"', '"t": "Uno"'),
            ("s1", "not there", "x"),
            ("s9", "x", "y"),
        ])
        self.assertEqual(applied, 1)
        self.assertEqual({f["slide_id"] for f in failed}, {"s1", "s9"})

    def test_edit_reports_only_the_slides_that_took_an_edit(self):
        *_r, applied_ids, applied, failed = service.edit_slides(self.deck.pk, [
            ("s1", '"t": "One"', '"t": "Uno"'),
            ("s2", "not there", "x"),
            ("", '"w": 960', '"w": 961'),
        ])
        self.assertEqual(applied_ids, ["s1"])  # s2 missed; the deck settings are not a slide
        self.assertEqual(applied, 2)
        self.assertEqual([f["slide_id"] for f in failed], ["s2"])

    def test_edit_nothing_applied_raises(self):
        with self.assertRaises(service.SlideOpError) as cm:
            service.edit_slides(self.deck.pk, [("s1", "not there", "x")])
        self.assertEqual(cm.exception.extra["failed"][0]["error"], "Text not found.")

    def test_edit_invalid_json_saves_nothing(self):
        with self.assertRaises(service.SlideOpError):
            service.edit_slides(self.deck.pk, [
                ("s2", '"t": "Two"', '"t": "Zwei"'),
                ("s1", '"name": "One"', '"name": "One'),
            ])
        self.deck.refresh_from_db()
        self.assertEqual(self.deck.content["slides"][1]["name"], "Two")

    def test_edit_cannot_change_slide_id(self):
        with self.assertRaises(service.SlideOpError):
            service.edit_slides(self.deck.pk, [("s1", '"id": "s1"', '"id": "s7"')])

    def test_edit_deck_settings(self):
        *_r, applied, _f = service.edit_slides(self.deck.pk, [("", '"w": 960', '"w": 961')])
        self.assertEqual(applied, 1)
        self.deck.refresh_from_db()
        self.assertEqual(self.deck.content["size"]["w"], 961)
        self.assertEqual(self._ids(), ["s1", "s2", "s3"])

    # -- structure -------------------------------------------------------------
    def test_move_slides(self):
        service.move_slides(self.deck.pk, ["s3"], 0)
        self.assertEqual(self._ids(), ["s3", "s1", "s2"])
        service.move_slides(self.deck.pk, ["s3", "s1"], -1)
        self.assertEqual(self._ids(), ["s2", "s3", "s1"])

    def test_move_unknown_raises(self):
        with self.assertRaises(service.SlideOpError):
            service.move_slides(self.deck.pk, ["s9"], 0)

    def test_delete_slides(self):
        service.delete_slides(self.deck.pk, ["s1", "s3"])
        self.assertEqual(self._ids(), ["s2"])

    def test_delete_refuses_to_empty_the_deck(self):
        with self.assertRaises(service.SlideOpError):
            service.delete_slides(self.deck.pk, ["s1", "s2", "s3"])
        self.assertEqual(self._ids(), ["s1", "s2", "s3"])

    def test_duplicate_slide_mints_a_fresh_id_after_the_original(self):
        *_r, idx = service.duplicate_slide(self.deck.pk, "s1")
        ids = self._ids()
        self.assertEqual(idx, 1)
        self.assertEqual(len(ids), 4)
        self.assertNotIn(ids[1], ("s1", "s2", "s3"))
        self.assertEqual(self.deck.content["slides"][1]["name"], "One")

    # -- outline + seen state --------------------------------------------------
    def test_outline_seen_states(self):
        outline = service.deck_outline(self.deck)
        self.assertEqual([o["seen"] for o in outline], ["never", "never", "never"])
        service.record_agent_slide_view(self.deck.pk, self.deck.content["slides"][:2])
        self.deck.refresh_from_db()
        self.assertEqual([o["seen"] for o in service.deck_outline(self.deck)], ["", "", "never"])
        service.replace_slide(self.deck.pk, "s2", _text_slide("Two", body="Edited elsewhere"))
        self.deck.refresh_from_db()
        self.assertEqual([o["seen"] for o in service.deck_outline(self.deck)], ["", "changed", "never"])

    def test_outline_layout_and_title(self):
        service.insert_slide(self.deck.pk, layouts.get_layout("bullets"))
        untitled = {"elements": [{"type": "text", "x": 1, "y": 1, "w": 10, "h": 10, "class": "headline",
                                  "paragraphs": [{"runs": [{"t": "From the headline"}]}]}]}
        service.insert_slide(self.deck.pk, untitled)
        self.deck.refresh_from_db()
        outline = service.deck_outline(self.deck)
        self.assertEqual(outline[0]["layout"], "custom")
        self.assertEqual(outline[3]["layout"], "bullets")
        self.assertEqual(outline[4]["title"], "From the headline")
        self.assertEqual(outline[4]["n"], 5)

    def test_outline_comments_until_the_agent_writes_the_slide(self):
        sent = timezone.now()
        SlideComment.objects.create(slide_set=self.deck, slide_id="s2", text="Shorter please",
                                    status=SlideComment.Status.SENT, sent_at=sent)
        SlideComment.objects.create(slide_set=self.deck, slide_id="s1", text="draft, unsent")
        outline = service.deck_outline(self.deck)
        self.assertEqual(outline[1]["comments"], ["Shorter please"])
        self.assertEqual(outline[0]["comments"], [])
        # A read doesn't address it; a write does.
        service.record_agent_slide_view(self.deck.pk, [self.deck.content["slides"][1]])
        self.deck.refresh_from_db()
        self.assertEqual(service.deck_outline(self.deck)[1]["comments"], ["Shorter please"])
        service.record_agent_slide_view(self.deck.pk, [self.deck.content["slides"][1]], wrote=True)
        self.deck.refresh_from_db()
        self.assertEqual(service.deck_outline(self.deck)[1]["comments"], [])
        # A newer comment shows again.
        SlideComment.objects.create(slide_set=self.deck, slide_id="s2", text="Now bigger",
                                    status=SlideComment.Status.SENT, sent_at=timezone.now() + timedelta(seconds=1))
        self.assertEqual(service.deck_outline(self.deck)[1]["comments"], ["Now bigger"])

    def test_record_view_does_not_bump_updated_at_and_prunes(self):
        before = SlideSet.objects.get(pk=self.deck.pk).updated_at
        service.record_agent_slide_view(self.deck.pk, self.deck.content["slides"])
        service.delete_slides(self.deck.pk, ["s3"])
        after_delete = SlideSet.objects.get(pk=self.deck.pk).updated_at
        service.record_agent_slide_view(self.deck.pk, [self.deck.content["slides"][0]])
        deck = SlideSet.objects.get(pk=self.deck.pk)
        self.assertEqual(deck.updated_at, after_delete)
        self.assertNotEqual(before, after_delete)
        self.assertEqual(set(deck.agent_seen), {"s1", "s2"})

    def test_compact_outline(self):
        lines = service.compact_outline(self.deck.content)
        self.assertEqual(lines[0], '1. s1 custom "One"')
