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
    ActivateDeckTool, AddSlideTool, DeleteDeckTool, EditDeckTool,
    PreviewSlidesTool, WriteDeckTool,
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

    def test_write_creates_activates_checkpoints(self):
        r = self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], ["s1"])
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertTrue(deck.is_active)
        self.assertEqual(deck.content["slides"][0]["id"], "s1")
        self.assertEqual([c.source for c in deck.checkpoints.all()], ["original"])

    def test_write_empty_deck_when_content_omitted(self):
        """Omitting content creates an empty, active deck to build up from layouts."""
        r = self._run(WriteDeckTool(), title="Scratch")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], [])
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertTrue(deck.is_active)
        self.assertEqual(deck.content["slides"], [])
        # ...and it's a valid deck we can immediately seed a layout into.
        seed = self._run(AddSlideTool(), layout="title_01")
        self.assertEqual(seed["status"], "ok")

    def test_write_accepts_native_object(self):
        """The deck may be passed as a native JSON object (no string escaping) —
        the primary path for the model, which avoids double-escaping errors."""
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [
            {"name": "Title", "skip_footer": True, "elements": [
                {"type": "text", "x": 80, "y": 210, "w": 800, "h": 100, "class": "headline",
                 "paragraphs": [{"align": "center", "runs": [{"t": "Object Deck"}]}]},
            ]},
        ]}
        r = self._run(WriteDeckTool(), title="Obj", content=deck)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_ids"], ["s1"])
        self.assertEqual(SlideSet.objects.get(pk=r["deck_id"]).content["slides"][0]["id"], "s1")

    def test_write_invalid_json(self):
        r = self._run(WriteDeckTool(), title="X", content_json="{not json")
        self.assertEqual(r["status"], "error")

    def test_write_invalid_deck(self):
        bad = json.dumps({"slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 1}]}]})
        r = self._run(WriteDeckTool(), title="X", content_json=bad)
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["issues"])

    def test_add_slide(self):
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        r = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["slide_id"], "s2")
        self.assertIn('"id": "s2"', r["slide_json"])

    def test_add_and_edit_return_full_slide_ids(self):
        """add_slide / edit must return the full ordered slide_ids (and add names the
        new one in changed_slide_ids). The client rebuilds its filmstrip from this list,
        so without it a newly added slide never appears live — it needed a manual F5."""
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        added = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(added["slide_ids"], ["s1", "s2"])
        self.assertEqual(added["changed_slide_ids"], ["s2"])
        self.assertEqual(added["slide_count"], 2)
        edited = self._run(EditDeckTool(), edits=[{"old_text": '"Hello Deck"', "new_text": '"Hi"'}])
        self.assertEqual(edited["slide_ids"], ["s1", "s2"])

    def test_add_slide_unknown_layout(self):
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        r = self._run(AddSlideTool(), layout="nope")
        self.assertEqual(r["status"], "error")
        self.assertIn("bullets", r["available_layouts"])

    def test_layout_catalog_ids(self):
        from chat.slides.layouts import layout_catalog

        ids = [c["id"] for c in layout_catalog()]
        for lid in ("title_01", "title_02", "title_03", "section", "section_02", "image_bleed"):
            self.assertIn(lid, ids)
        self.assertNotIn("team", ids)  # removed pending a redesign

    def test_add_slide_carries_layout_comment(self):
        """A seed's authoring ``comment`` travels into the inserted slide JSON."""
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        r = self._run(AddSlideTool(), layout="image_right")
        self.assertEqual(r["status"], "ok")
        self.assertIn('"comment"', r["slide_json"])
        self.assertIn("either side", r["slide_json"])

    def test_edit_valid(self):
        w = self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        r = self._run(EditDeckTool(), edits=[{"old_text": '"Hello Deck"', "new_text": '"Welcome"'}])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["applied"], 1)
        self.assertEqual(r["changed_slide_ids"], ["s1"])
        deck = SlideSet.objects.get(pk=w["deck_id"])
        self.assertIn("Welcome", json.dumps(deck.content))

    def test_edit_json_guard_rejects_and_preserves(self):
        w = self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        deck = SlideSet.objects.get(pk=w["deck_id"])
        before = json.dumps(deck.content, sort_keys=True)
        r = self._run(EditDeckTool(), edits=[{"old_text": '"version": 1', "new_text": '"version": 1 OOPS'}])
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["applied"], 0)
        deck.refresh_from_db()
        self.assertEqual(json.dumps(deck.content, sort_keys=True), before)
        # no ai_edit checkpoint created for the rejected edit
        self.assertEqual(deck.checkpoints.filter(source="ai_edit").count(), 0)

    def test_edit_nonunique_reports_failure(self):
        json_deck = json.dumps({"version": 1, "size": {"w": 960, "h": 540}, "slides": [
            {"elements": [{"type": "text", "x": 1, "y": 1, "w": 1, "h": 1,
                           "paragraphs": [{"runs": [{"t": "dup"}]}, {"runs": [{"t": "dup"}]}]}]},
        ]})
        self._run(WriteDeckTool(), title="Deck A", content_json=json_deck)
        r = self._run(EditDeckTool(), edits=[{"old_text": '"t": "dup"', "new_text": '"t": "x"'}])
        self.assertEqual(r["status"], "error")
        self.assertTrue(any("matches" in f["error"] for f in r["failed"]))

    def test_activate_and_delete(self):
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        r = self._run(ActivateDeckTool(), deck_names=["Deck A"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["activated"][0]["title"], "Deck A")
        d = self._run(DeleteDeckTool(), deck_name="Deck A")
        self.assertEqual(d["status"], "ok")
        self.assertIsNotNone(SlideSet.objects.get(title="Deck A").deleted_at)

    def test_delete_active_promotes_survivor(self):
        """Deleting the active deck when others remain promotes the newest survivor
        so the thread keeps an active deck (the panel can switch instead of orphaning)."""
        self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        self._run(WriteDeckTool(), title="Deck B", content_json=_deck_json())
        # Deck B is active (written last); delete it and Deck A should take over.
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
        w = self._run(WriteDeckTool(), title="Deck A")
        stale = self._stale(w["deck_id"])  # zero slides
        self.assertEqual(stale.content["slides"], [])

        first = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(first["status"], "ok")

        with mock.patch("chat.slides.service.resolve_deck", return_value=(stale, None)):
            second = self._run(AddSlideTool(), layout="title_01")
        self.assertEqual(second["status"], "ok")

        deck = SlideSet.objects.get(pk=w["deck_id"])
        slides = deck.content["slides"]
        self.assertEqual(len(slides), 2, "the concurrent add clobbered the first slide")
        ids = [s["id"] for s in slides]
        self.assertEqual(len(set(ids)), 2, f"slide ids collided: {ids}")
        self.assertNotEqual(first["slide_id"], second["slide_id"])

    def test_edit_deck_reads_fresh_content_under_lock(self):
        w = self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        stale = self._stale(w["deck_id"])  # one slide

        added = self._run(AddSlideTool(), layout="bullets")
        self.assertEqual(added["status"], "ok")

        with mock.patch("chat.slides.service.resolve_deck", return_value=(stale, None)):
            r = self._run(EditDeckTool(), edits=[{"old_text": '"Hello Deck"', "new_text": '"Welcome"'}])

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
        w = self._run(WriteDeckTool(), title="Deck A", content_json=_deck_json())
        self._run(AddSlideTool(), layout="bullets")
        self._run(AddSlideTool(), layout="title_01")
        self._run(EditDeckTool(), edits=[{"old_text": '"Hello Deck"', "new_text": '"Welcome"'}])

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


class PreviewToolTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"p+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.ctx = RunContext(run_id="r1", conversation_id=str(self.thread.id), user_id=str(self.user.id))
        WriteDeckTool().set_context(self.ctx)._run(
            title="Deck A",
            content_json=json.dumps({"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"name": "A", "elements": []}, {"name": "B", "elements": []},
            ]}),
        )

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

    def _write(self, deck, title="Deck", deck_name=""):
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
        from chat.slides.theme import preset_theme_override

        r = self._write(self._bare_deck())
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content.get("theme"), preset_theme_override("slate"))

    def test_explicit_theme_is_untouched(self):
        custom = {"colors": {"accent1": "#123456"}}
        r = self._write(self._bare_deck(theme=custom))
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertEqual(deck.content["theme"], custom)

    def test_rewrite_preserves_existing_theme(self):
        from chat.slides.theme import preset_theme_override

        # Create the deck (gets the org default), then simulate the user picking
        # "ocean" from the slide panel, then a full rewrite that omits the theme.
        r = self._write(self._bare_deck(), title="Deck")
        deck = SlideSet.objects.get(pk=r["deck_id"])
        deck.content["theme"] = preset_theme_override("ocean")
        deck.save(update_fields=["content"])

        self._write(self._bare_deck(), title="Deck")
        deck.refresh_from_db()
        self.assertEqual(deck.content["theme"], preset_theme_override("ocean"))

    def test_no_membership_leaves_deck_theme_unset(self):
        # A user with no org gets no seeded theme (deck resolves to base forest).
        from accounts.models import Membership

        Membership.objects.filter(user=self.user).delete()
        r = self._write(self._bare_deck())
        deck = SlideSet.objects.get(pk=r["deck_id"])
        self.assertNotIn("theme", deck.content)
