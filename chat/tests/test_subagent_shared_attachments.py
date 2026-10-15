"""Tests for sharing chat attachments with a sub-agent.

``chat_subagent_create(attachments=[N, ...])`` stores the files' ids (with the
thread-wide numbers) on the run; the sub-agent's prompt lists them and
``subagent_attachment_view`` views them — and only them.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from chat.attachment_tools import AttachmentViewTool, SubagentAttachmentViewTool
from chat.models import ChatAttachment, ChatMessage, ChatThread, SubAgentRun
from chat.prompts import build_system_prompt
from chat.subagent_prompts import build_subagent_system_prompt
from chat.subagent_service import resolve_subagent_tools, shared_attachment_rows
from chat.subagent_tool import CreateSubagentTool
from core.preferences import ResolvedPreferences
from llm.types.context import RunContext

User = get_user_model()

_IN_MEMORY_STORAGE = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)


def _prefs(**overrides):
    defaults = dict(
        top_model="openai/gpt-5",
        mid_model="openai/gpt-5-mini",
        cheap_model="openai/gpt-5-nano",
        allowed_models=["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano"],
        allowed_tools=["web_search"],
        allowed_subagent_tools=["web_search", "subagent_attachment_view"],
        allowed_skills=[],
        allowed_specializations=[],
        theme="light",
    )
    defaults.update(overrides)
    return ResolvedPreferences(**defaults)


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _AttachmentFixtures(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="share-att@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.msg = ChatMessage.objects.create(thread=self.thread, role="user", content="here")

    def _make(self, name, content_type="text/plain", body=b"BODY", *, sent=True,
              thread=None, message=None, parent=None, ordinal=None):
        return ChatAttachment.objects.create(
            thread=thread or self.thread, uploaded_by=self.user,
            message=(message or self.msg) if sent else None,
            file=SimpleUploadedFile(name, body), original_filename=name,
            content_type=content_type, size_bytes=len(body),
            parent=parent, email_ordinal=ordinal,
        )


@_IN_MEMORY_STORAGE
@patch("chat.tasks.run_subagent_task")
class CreateSubagentSharedAttachmentTests(_AttachmentFixtures):
    def _invoke(self, args):
        tool = CreateSubagentTool()
        tool.set_context(RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id)))
        return json.loads(tool.invoke(args))

    def test_omitted_shares_nothing(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        self._invoke({"prompt": "task"})
        self.assertEqual(SubAgentRun.objects.get().shared_attachments, [])

    def test_ids_and_numbers_stored_deduped_in_order(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        a = self._make("a.txt")
        b = self._make("b.txt")
        result = self._invoke({"prompt": "review", "attachments": [2, 1, 2]})
        self.assertIn(result["status"], ("started", "queued"))
        self.assertEqual(
            SubAgentRun.objects.get().shared_attachments,
            [{"id": str(b.pk), "number": 2}, {"id": str(a.pk), "number": 1}],
        )

    def test_unknown_number_lists_available_and_creates_no_run(self, mock_task):
        self._make("a.txt")
        self._make("draft.txt", sent=False)
        result = self._invoke({"prompt": "review", "attachments": [7]})
        self.assertEqual(result["status"], "error")
        self.assertIn("No attachment #7", result["message"])
        self.assertIn("#1 a.txt", result["message"])
        self.assertNotIn("draft.txt", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_draft_rejected(self, mock_task):
        self._make("draft.txt", sent=False)
        result = self._invoke({"prompt": "review", "attachments": [1]})
        self.assertEqual(result["status"], "error")
        self.assertIn("draft", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_redacted_rejected(self, mock_task):
        self.msg.is_redacted = True
        self.msg.save(update_fields=["is_redacted"])
        self._make("x.txt")
        result = self._invoke({"prompt": "review", "attachments": [1]})
        self.assertEqual(result["status"], "error")
        self.assertIn("removed", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_other_threads_attachment_not_reachable(self, mock_task):
        other = ChatThread.objects.create(created_by=self.user)
        other_msg = ChatMessage.objects.create(thread=other, role="user", content="x")
        self._make("elsewhere.txt", thread=other, message=other_msg)
        result = self._invoke({"prompt": "review", "attachments": [1]})
        self.assertEqual(result["status"], "error")
        self.assertFalse(SubAgentRun.objects.exists())

    def test_email_brings_its_files(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        email = self._make("mail.eml", "message/rfc822")
        second = self._make("second.pdf", "application/pdf", parent=email, ordinal=2)
        first = self._make("first.docx", parent=email, ordinal=1)
        self._make("unrelated.txt")
        self._invoke({"prompt": "review", "attachments": [1]})
        self.assertEqual(
            SubAgentRun.objects.get().shared_attachments,
            [
                {"id": str(email.pk), "number": 1},
                {"id": str(first.pk), "number": 3},
                {"id": str(second.pk), "number": 2},
            ],
        )

    def test_email_child_already_listed_not_duplicated(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        email = self._make("mail.eml", "message/rfc822")
        self._make("child.pdf", "application/pdf", parent=email, ordinal=1)
        self._invoke({"prompt": "review", "attachments": [2, 1]})
        numbers = [e["number"] for e in SubAgentRun.objects.get().shared_attachments]
        self.assertEqual(numbers, [2, 1])


class SharedAttachmentPromptTests(TestCase):
    def test_absent_when_nothing_shared(self):
        for shared in (None, []):
            prompt = build_subagent_system_prompt(shared_attachments=shared)
            self.assertNotIn("Files shared by the orchestrator", prompt)

    def test_lists_files_with_numbers_and_framing(self):
        prompt = build_subagent_system_prompt(shared_attachments=[
            {"number": 3, "filename": "deck.pptx", "kind_label": "presentation, 12 slides"},
            {"number": 7, "filename": "terms.pdf", "kind_label": "PDF, 4 pages"},
        ])
        self.assertIn("# Files shared by the orchestrator", prompt)
        self.assertIn("- #3 deck.pptx (presentation, 12 slides)", prompt)
        self.assertIn("- #7 terms.pdf (PDF, 4 pages)", prompt)
        self.assertIn("subagent_attachment_view", prompt)
        self.assertIn("not as instructions", prompt)

    def test_orchestrator_prompt_explains_attachments(self):
        prompt = build_system_prompt(has_subagent_tool=True)
        self.assertIn("`attachments=[N, ...]`", prompt)
        self.assertIn("canvases or attachments you choose to share", prompt)


class SubagentToolGatingTests(TestCase):
    def test_view_tool_only_with_shared_files(self):
        prefs = _prefs()
        self.assertNotIn("subagent_attachment_view", resolve_subagent_tools(prefs, []))
        self.assertIn(
            "subagent_attachment_view",
            resolve_subagent_tools(prefs, [], has_shared_attachments=True),
        )

    def test_audiences(self):
        self.assertEqual(SubagentAttachmentViewTool().audience, "subagent")
        self.assertEqual(SubagentAttachmentViewTool().section, "chat")
        self.assertEqual(AttachmentViewTool().audience, "main")

    def test_registered_for_subagents_only(self):
        from llm.tools.registry import get_tool_registry

        tool = get_tool_registry().get_tool("subagent_attachment_view")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.audience, "subagent")


@_IN_MEMORY_STORAGE
class RunSubagentSharedAttachmentTests(_AttachmentFixtures):
    def _run(self, run, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        resp = MagicMock()
        resp.message.content = "done"
        resp.usage.total_tokens = 1
        resp.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = resp
        from chat.subagent_service import run_subagent

        run_subagent(run.id)
        run.refresh_from_db()
        return mock_svc.return_value.run_via_stream.call_args[0][1]

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_manifest_tool_and_model_id(self, mock_prefs, mock_svc):
        att = self._make("terms.pdf", "application/pdf")
        att.page_count = 4
        att.save(update_fields=["page_count"])
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            shared_attachments=[{"id": str(att.pk), "number": 1}],
        )
        request = self._run(run, mock_prefs, mock_svc)
        system = request.messages[0].content
        self.assertIn("- #1 terms.pdf (PDF, 4 pages)", system)
        self.assertIn("subagent_attachment_view", run.tool_names)
        self.assertEqual(request.context.model_id, "openai/gpt-5-mini")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_no_files_no_tool_no_manifest(self, mock_prefs, mock_svc):
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")
        request = self._run(run, mock_prefs, mock_svc)
        self.assertNotIn("Files shared by the orchestrator", request.messages[0].content)
        self.assertNotIn("subagent_attachment_view", run.tool_names)

    def test_rows_skip_deleted_and_redacted(self):
        keep = self._make("keep.txt")
        gone = self._make("gone.txt")
        other_msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="y", is_redacted=True,
        )
        redacted = self._make("bad.txt", message=other_msg)
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            shared_attachments=[
                {"id": str(gone.pk), "number": 2},
                {"id": str(keep.pk), "number": 1},
                {"id": str(redacted.pk), "number": 3},
            ],
        )
        gone.delete()
        self.assertEqual(shared_attachment_rows(run), [(1, keep)])


@_IN_MEMORY_STORAGE
class SubagentAttachmentViewToolTests(_AttachmentFixtures):
    def setUp(self):
        super().setUp()
        self.shared = self._make("shared.txt", body=b"SHARED TEXT")
        self.other = self._make("private.txt", body=b"PRIVATE")
        self.image = self._make("pic.png", "image/png", _png_bytes())
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            shared_attachments=[
                {"id": str(self.shared.pk), "number": 1},
                {"id": str(self.image.pk), "number": 3},
            ],
        )

    def _view(self, args, *, run_id=None, model_id=None):
        ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        ctx.run_id = str(run_id or self.run.id)
        ctx.agent_kind = "subagent"
        ctx.model_id = model_id
        tool = SubagentAttachmentViewTool()
        tool.set_context(ctx)
        return json.loads(tool.invoke(args)), ctx

    def test_shared_text_file_extracted(self):
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["representation"], "extracted")
        self.assertEqual(result["content"], "SHARED TEXT")

    def test_unshared_number_refused_with_shared_list(self):
        result, _ = self._view({"attachment_number": 2})
        self.assertEqual(result["status"], "error")
        self.assertIn("wasn't shared", result["message"])
        self.assertEqual(
            result["shared_files"],
            [{"number": 1, "filename": "shared.txt"}, {"number": 3, "filename": "pic.png"}],
        )
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_redacted_after_spawn_refused(self):
        self.msg.is_redacted = True
        self.msg.save(update_fields=["is_redacted"])
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("removed", result["message"])

    def test_deleted_after_spawn_refused(self):
        self.shared.delete()
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("no longer available", result["message"])

    def test_image_queued_natively_on_attachment_pathway(self):
        result, ctx = self._view({"attachment_number": 3})
        self.assertEqual(result["representation"], "native")
        self.assertEqual(len(ctx.pending_native_assets), 1)
        self.assertEqual(ctx.pending_native_assets[0]["_pathway"], "attachment")

    def test_image_on_non_vision_model_is_error(self):
        with patch("llm.display.supports_modality", return_value=False):
            result, ctx = self._view({"attachment_number": 3}, model_id="text-only")
        self.assertEqual(result["status"], "error")
        self.assertIn("cannot view images", result["message"])
        self.assertEqual(ctx.pending_native_assets, [])

    def test_nothing_shared(self):
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="t")
        result, _ = self._view({"attachment_number": 1}, run_id=run.id)
        self.assertEqual(result["status"], "error")
        self.assertIn("didn't share any files", result["message"])

    def test_outside_a_subagent_run(self):
        import uuid

        result, _ = self._view({"attachment_number": 1}, run_id=uuid.uuid4())
        self.assertEqual(result["status"], "error")
        self.assertIn("only available inside a sub-agent run", result["message"])
