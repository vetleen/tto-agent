"""LLMCallLog rows written by the document pipeline carry the document and
data-room ids, and deleting either redacts the rows' content
(documents/signals.py)."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from documents.models import DataRoom, DataRoomDocument
from llm.models import LLMCallLog
from llm.retention import document_log_ids
from llm.service.logger import log_error
from llm.types import ChatRequest, Message, RunContext

User = get_user_model()

PROMPT = [{"role": "user", "content": "document text"}]


def _make_log(user, **fields):
    defaults = dict(
        user=user,
        model="gpt-5-mini",
        prompt=PROMPT,
        raw_output="description",
        tools=[{"name": "tool_a"}],
        input_tokens=5,
        output_tokens=7,
        cost_usd=Decimal("0.25"),
    )
    defaults.update(fields)
    return LLMCallLog.objects.create(**defaults)


@patch("documents.services.vector_store.delete_vectors_for_document")
class DocumentDeleteRedactsLLMCallLogTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="doclog@example.com", password="testpass")
        self.room = DataRoom.objects.create(name="Room", slug="doclog-room", created_by=self.user)
        self.doc = self._make_doc("a.txt")

    def _make_doc(self, name, room=None):
        return DataRoomDocument.objects.create(
            data_room=room or self.room,
            uploaded_by=self.user,
            original_filename=name,
            status=DataRoomDocument.Status.UPLOADED,
        )

    def _doc_log(self, doc):
        return _make_log(self.user, document_id=doc.pk, data_room_id=doc.data_room_id)

    def assertRedacted(self, log):
        log.refresh_from_db()
        self.assertEqual(log.prompt, {"redacted": True})
        self.assertEqual(log.raw_output, "")
        self.assertIsNone(log.tools)

    def assertIntact(self, log):
        log.refresh_from_db()
        self.assertEqual(log.prompt, PROMPT)
        self.assertEqual(log.raw_output, "description")

    def test_document_delete_redacts_its_rows(self, _vectors):
        log = self._doc_log(self.doc)

        self.doc.delete()

        self.assertRedacted(log)

    def test_metadata_kept(self, _vectors):
        log = self._doc_log(self.doc)
        doc_id, created_at = self.doc.pk, log.created_at

        self.doc.delete()

        log.refresh_from_db()
        self.assertEqual(log.user_id, self.user.pk)
        self.assertEqual(log.document_id, doc_id)
        self.assertEqual(log.model, "gpt-5-mini")
        self.assertEqual(log.input_tokens, 5)
        self.assertEqual(log.output_tokens, 7)
        self.assertEqual(log.cost_usd, Decimal("0.25"))
        self.assertEqual(log.created_at, created_at)

    def test_other_documents_and_unlinked_rows_untouched(self, _vectors):
        sibling = self._make_doc("b.txt")
        sibling_log = self._doc_log(sibling)
        room_log = _make_log(self.user, data_room_id=self.room.pk)
        unlinked = _make_log(self.user)

        self.doc.delete()

        self.assertIntact(sibling_log)
        self.assertIntact(room_log)
        self.assertIntact(unlinked)

    def test_bulk_queryset_delete_redacts(self, _vectors):
        sibling = self._make_doc("b.txt")
        logs = [self._doc_log(self.doc), self._doc_log(sibling)]

        DataRoomDocument.objects.filter(pk__in=[self.doc.pk, sibling.pk]).delete()

        for log in logs:
            self.assertRedacted(log)

    def test_data_room_delete_redacts_document_and_room_rows(self, _vectors):
        doc_log = self._doc_log(self.doc)
        room_log = _make_log(self.user, data_room_id=self.room.pk)
        other_room = DataRoom.objects.create(name="Other", slug="doclog-other", created_by=self.user)
        other_log = self._doc_log(self._make_doc("c.txt", room=other_room))

        self.room.delete()

        self.assertRedacted(doc_log)
        self.assertRedacted(room_log)
        self.assertIntact(other_log)

    def test_user_cascade_redacts_rows_without_a_user(self, _vectors):
        log = _make_log(None, document_id=self.doc.pk, data_room_id=self.room.pk)

        self.user.delete()

        self.assertRedacted(log)


class DocumentLogIdsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ids@example.com", password="testpass")
        self.room = DataRoom.objects.create(name="Room", slug="ids-room", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user, original_filename="a.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )

    def test_document_log_ids(self):
        self.assertEqual(
            document_log_ids(self.doc),
            {"document_id": self.doc.pk, "data_room_id": self.room.pk},
        )

    def test_non_document_yields_no_ids(self):
        self.assertEqual(
            document_log_ids(MagicMock()), {"document_id": None, "data_room_id": None},
        )
        self.assertEqual(
            document_log_ids(None), {"document_id": None, "data_room_id": None},
        )

    def test_logger_writes_ids_from_context(self):
        context = RunContext.create(user_id=self.user.pk, **document_log_ids(self.doc))
        request = ChatRequest(
            messages=[Message(role="user", content="x")], model="gpt-5-mini",
            stream=False, tools=[], context=context,
        )

        log_error(request, ValueError("boom"), 1)

        log = LLMCallLog.objects.get()
        self.assertEqual(log.document_id, self.doc.pk)
        self.assertEqual(log.data_room_id, self.room.pk)

    def test_logger_leaves_ids_null_without_them(self):
        request = ChatRequest(
            messages=[Message(role="user", content="x")], model="gpt-5-mini",
            stream=False, tools=[], context=RunContext.create(user_id=self.user.pk),
        )

        log_error(request, ValueError("boom"), 1)

        log = LLMCallLog.objects.get()
        self.assertIsNone(log.document_id)
        self.assertIsNone(log.data_room_id)

    @patch("llm.get_llm_service")
    def test_description_call_is_tagged(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text

        captured = []

        def fake(request, schema):
            captured.append(request.context)
            raise RuntimeError("stop here")

        mock_get_service.return_value.run_structured.side_effect = fake
        try:
            generate_description_and_tags_from_text(
                "some text", user_id=self.user.pk, data_room_id=self.room.pk,
                document_id=self.doc.pk,
            )
        except RuntimeError:
            pass

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].document_id, self.doc.pk)
        self.assertEqual(captured[0].data_room_id, self.room.pk)

    @patch("documents.services.pii_scan._get_decisions_service")
    def test_pii_window_call_is_tagged(self, mock_service):
        from documents.services.pii_scan import scan_pii_window

        captured = []

        def fake(*args, **kwargs):
            captured.append(kwargs.get("context"))
            raise RuntimeError("stop here")

        with patch("llm.service.decisions_fallback.decide_with_fallback", side_effect=fake):
            try:
                scan_pii_window(
                    "some text", user_id=self.user.pk, data_room_id=self.room.pk,
                    document_id=self.doc.pk,
                )
            except RuntimeError:
                pass

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].document_id, self.doc.pk)
        self.assertEqual(captured[0].data_room_id, self.room.pk)
