"""WebSocket consumer for chat with LLM streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

MAX_HISTORY_TOKENS = 20_000  # legacy default; overridden by dynamic budget when model is known
OVERLAP_TOKENS = 2_000  # legacy default; overridden by dynamic budget when model is known


class ChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for chat with LLM streaming and optional RAG via data rooms."""

    async def connect(self):
        self.user = self.scope.get("user")
        self.resolved_prefs = None
        self.data_room_ids: list[int] = []
        self.active_skill_id: str | None = None
        self._cancel_event: threading.Event | None = None
        self._stream_finished = asyncio.Event()
        self._guardrail_task: asyncio.Task | None = None
        self._guardrail_intercepted: bool = False
        self._guardrail_warn_verdict: object | None = None  # stored for post-stream delivery

        # Reject unauthenticated users
        if not self.user or self.user.is_anonymous:
            await self.close(code=4401)
            return

        # Check if user is suspended in their org
        is_suspended = await self._check_suspension()
        if is_suspended:
            await self.accept()
            await self.send(text_data=json.dumps({
                "event_type": "guardrail.suspended",
                "data": {"message": "Your account has been suspended. Please contact your system administrator."},
            }))
            await self.close(code=4403)
            return

        # Resolve user/org/system preferences
        self.resolved_prefs = await self._resolve_preferences()

        await self.accept()

    @database_sync_to_async
    def _resolve_preferences(self):
        from core.preferences import get_preferences
        return get_preferences(self.user)

    @database_sync_to_async
    def _check_suspension(self) -> bool:
        from accounts.models import Membership
        return Membership.objects.filter(
            user=self.user, is_suspended=True,
        ).exists()

    @database_sync_to_async
    def _get_org_id(self) -> int | None:
        from accounts.models import Membership
        membership = Membership.objects.filter(user=self.user).first()
        return membership.org_id if membership else None

    def _has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available given current prefs."""
        if self.resolved_prefs:
            return tool_name in self.resolved_prefs.allowed_tools
        from llm.tools.registry import get_tool_registry
        return tool_name in get_tool_registry().list_tools()

    @database_sync_to_async
    def _get_organization_name(self) -> str | None:
        from accounts.models import Membership
        membership = (
            Membership.objects
            .filter(user=self.user)
            .select_related("org")
            .first()
        )
        return membership.org.name if membership else None


    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return

        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({"error": "Invalid JSON"}))
            return

        msg_type = data.get("type", "")

        if msg_type == "chat.message":
            await self._handle_chat_message(data)
        elif msg_type == "chat.attach_data_room":
            await self._handle_attach_data_room(data)
        elif msg_type == "chat.detach_data_room":
            await self._handle_detach_data_room(data)
        elif msg_type == "chat.load_thread":
            await self._handle_load_thread(data)
        elif msg_type == "chat.attach_skill":
            await self._handle_attach_skill(data)
        elif msg_type == "chat.detach_skill":
            await self._handle_detach_skill(data)
        elif msg_type == "chat.canvas_save":
            await self._handle_canvas_save(data)
        elif msg_type == "chat.canvas_open":
            await self._handle_canvas_open(data)
        elif msg_type == "chat.canvas_accept":
            await self._handle_canvas_accept(data)
        elif msg_type == "chat.canvas_revert":
            await self._handle_canvas_revert(data)
        elif msg_type == "chat.canvas_save_version":
            await self._handle_canvas_save_version(data)
        elif msg_type == "chat.canvas_restore_version":
            await self._handle_canvas_restore_version(data)
        elif msg_type == "chat.canvas_get_checkpoints":
            await self._handle_canvas_get_checkpoints(data)
        elif msg_type == "chat.canvas_switch":
            await self._handle_canvas_switch(data)
        elif msg_type == "chat.stop":
            await self._handle_stop(data)
        elif msg_type == "pong":
            pass  # heartbeat acknowledgment

    async def _handle_attach_data_room(self, data):
        """Attach a data room to the current session."""
        data_room_id = data.get("data_room_id")
        thread_id = data.get("thread_id")
        if not data_room_id:
            await self.send(text_data=json.dumps({"error": "data_room_id required"}))
            return

        # Validate ownership
        room = await self._validate_data_room(data_room_id)
        if not room:
            await self.send(text_data=json.dumps({"error": "Data room not found or access denied"}))
            return

        if data_room_id not in self.data_room_ids:
            self.data_room_ids.append(data_room_id)

        # Persist M2M if thread exists
        if thread_id:
            await self._persist_data_room_link(thread_id, data_room_id)

        await self.send(text_data=json.dumps({
            "event_type": "data_room.attached",
            "data_room_id": data_room_id,
            "data_room_name": room["name"],
        }))

    async def _handle_detach_data_room(self, data):
        """Detach a data room from the current session."""
        data_room_id = data.get("data_room_id")
        thread_id = data.get("thread_id")
        if not data_room_id:
            return

        if data_room_id in self.data_room_ids:
            self.data_room_ids.remove(data_room_id)

        # Remove M2M if thread exists
        if thread_id:
            await self._remove_data_room_link(thread_id, data_room_id)

        await self.send(text_data=json.dumps({
            "event_type": "data_room.detached",
            "data_room_id": data_room_id,
        }))

    async def _handle_attach_skill(self, data):
        """Attach a skill to the current session."""
        skill_id = data.get("skill_id")
        thread_id = data.get("thread_id")
        if not skill_id:
            await self.send(text_data=json.dumps({"error": "skill_id required"}))
            return

        skill = await self._validate_skill(skill_id)
        if not skill:
            await self.send(text_data=json.dumps({"error": "Skill not found or access denied"}))
            return

        self.active_skill_id = str(skill["id"])

        if thread_id:
            await self._persist_thread_skill(thread_id, skill["id"])

        await self.send(text_data=json.dumps({
            "event_type": "skill.attached",
            "skill_id": str(skill["id"]),
            "skill_name": skill["name"],
        }))

    async def _handle_detach_skill(self, data):
        """Detach the skill from the current session."""
        thread_id = data.get("thread_id")

        self.active_skill_id = None

        if thread_id:
            await self._persist_thread_skill(thread_id, None)

        await self.send(text_data=json.dumps({
            "event_type": "skill.detached",
        }))

    async def _handle_load_thread(self, data):
        """Load a thread's data rooms, skill, and canvas into the session."""
        thread_id = data.get("thread_id")
        if not thread_id:
            return

        thread_data = await self._load_thread_data_rooms(thread_id)
        if thread_data is not None:
            self.data_room_ids = thread_data["data_room_ids"]

            # Load skill
            skill_data = await self._load_thread_skill(thread_id)
            self.active_skill_id = skill_data["skill_id"] if skill_data else None

            # Load thread cost
            thread_cost = await self._get_thread_cost(thread_id)

            await self.send(text_data=json.dumps({
                "event_type": "thread.loaded",
                "thread_id": thread_id,
                "data_rooms": thread_data["data_rooms"],
                "skill": skill_data,
                "thread_cost_usd": thread_cost,
            }))
            # Send canvas state if any exist
            canvases_data = await self._load_all_canvases(thread_id)
            if canvases_data:
                await self.send(text_data=json.dumps({
                    "event_type": "canvases.loaded",
                    "canvases": canvases_data["tabs"],
                    "active_canvas": canvases_data["active"],
                }))
            # Send task plan state if tasks exist
            task_list = await self._get_thread_tasks(thread_id)
            if task_list:
                await self.send(text_data=json.dumps({
                    "event_type": "tasks.loaded",
                    "tasks": [{
                        "id": str(t["id"]),
                        "title": t["title"],
                        "status": t["status"],
                    } for t in task_list],
                }))

    async def _handle_canvas_save(self, data):
        """Save user edits to the canvas."""
        from chat.services import CANVAS_MAX_CHARS

        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        title = data.get("title", "Untitled document")
        content = data.get("content", "")
        content = content[:CANVAS_MAX_CHARS]
        if thread_id:
            await self._save_canvas(thread_id, title, content, canvas_id=canvas_id)

    async def _handle_canvas_open(self, data):
        """Return existing canvases or create a blank one."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        if not thread_id:
            return
        canvases_data = await self._load_all_canvases(thread_id)
        if canvases_data:
            await self.send(text_data=json.dumps({
                "event_type": "canvases.loaded",
                "canvases": canvases_data["tabs"],
                "active_canvas": canvases_data["active"],
            }))
        else:
            # Create a blank canvas
            canvas_data = await self._get_or_create_canvas(thread_id)
            await self.send(text_data=json.dumps({
                "event_type": "canvases.loaded",
                "canvases": [{"id": canvas_data["id"], "title": canvas_data["title"], "is_active": True}],
                "active_canvas": canvas_data,
            }))

    async def _handle_canvas_accept(self, data):
        """Set accepted_checkpoint to the latest checkpoint."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        if not thread_id:
            return
        result = await self._canvas_accept(thread_id, canvas_id=canvas_id)
        if result:
            event = {
                "event_type": "canvas.accepted",
                "accepted_content": result["accepted_content"],
            }
            if result.get("canvas_id"):
                event["canvas_id"] = result["canvas_id"]
            await self.send(text_data=json.dumps(event))
        else:
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "Could not accept canvas changes — no checkpoint found."},
            }))

    async def _handle_canvas_revert(self, data):
        """Restore canvas to its accepted checkpoint."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        if not thread_id:
            return
        result = await self._canvas_revert(thread_id, canvas_id=canvas_id)
        if result:
            event = {
                "event_type": "canvas.reverted",
                "title": result["title"],
                "content": result["content"],
            }
            if result.get("canvas_id"):
                event["canvas_id"] = result["canvas_id"]
            await self.send(text_data=json.dumps(event))
        else:
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "Could not revert canvas — no accepted version found."},
            }))

    async def _handle_canvas_save_version(self, data):
        """Save current canvas state as a user checkpoint and set as accepted."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        title = data.get("title", "")
        content = data.get("content", "")
        if not thread_id:
            return
        result = await self._canvas_save_version(thread_id, title, content, canvas_id=canvas_id)
        if result:
            event = {
                "event_type": "canvas.version_saved",
                "accepted_content": result["accepted_content"],
            }
            if result.get("canvas_id"):
                event["canvas_id"] = result["canvas_id"]
            await self.send(text_data=json.dumps(event))

    async def _handle_canvas_restore_version(self, data):
        """Restore canvas to a specific checkpoint by ID."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        checkpoint_id = data.get("checkpoint_id")
        if not thread_id or not checkpoint_id:
            return
        result = await self._canvas_restore_version(thread_id, checkpoint_id, canvas_id=canvas_id)
        if result:
            event = {
                "event_type": "canvas.restored",
                "title": result["title"],
                "content": result["content"],
            }
            if result.get("canvas_id"):
                event["canvas_id"] = result["canvas_id"]
            await self.send(text_data=json.dumps(event))

    async def _handle_canvas_get_checkpoints(self, data):
        """Return list of checkpoints for the canvas."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        if not thread_id:
            return
        checkpoints = await self._canvas_get_checkpoints(thread_id, canvas_id=canvas_id)
        await self.send(text_data=json.dumps({
            "event_type": "canvas.checkpoints",
            "checkpoints": checkpoints,
        }))

    async def _handle_canvas_switch(self, data):
        """Switch the active canvas to a different tab."""
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        canvas_id = data.get("canvas_id")
        if not thread_id or not canvas_id:
            return
        result = await self._switch_canvas(thread_id, canvas_id)
        if result:
            event = {
                "event_type": "canvas.loaded",
                "canvas_id": result["id"],
                "title": result["title"],
                "content": result["content"],
            }
            if result.get("accepted_content") is not None:
                event["accepted_content"] = result["accepted_content"]
            await self.send(text_data=json.dumps(event))

    async def _send_heartbeats(self, interval=30):
        """Send periodic heartbeat events to keep the connection alive during long operations."""
        try:
            while True:
                await asyncio.sleep(interval)
                await self.send(text_data=json.dumps({"event_type": "heartbeat"}))
        except asyncio.CancelledError:
            pass

    async def _handle_stop(self, data):
        """Handle a stop request from the client."""
        # Signal the streaming loop to stop
        if self._cancel_event:
            self._cancel_event.set()

        # Cancel the guardrail pipeline task if still running
        if self._guardrail_task and not self._guardrail_task.done():
            self._guardrail_task.cancel()

        # Cancel any active subagents for this thread
        thread_id = data.get("thread_id") or getattr(self, "_active_thread_id", None)
        if thread_id:
            await self._cancel_active_subagents(thread_id)

        await self.send(text_data=json.dumps({"event_type": "stream.cancelled"}))

    @database_sync_to_async
    def _cancel_active_subagents(self, thread_id):
        """Cancel all active subagent runs for a thread."""
        from django.utils import timezone

        from chat.models import SubAgentRun

        active_runs = SubAgentRun.objects.filter(
            thread_id=thread_id,
            status__in=[SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING],
        )
        for run in active_runs:
            if run.celery_task_id:
                from celery.result import AsyncResult
                AsyncResult(run.celery_task_id).revoke(terminate=True)
            run.status = SubAgentRun.Status.FAILED
            run.error = "Cancelled by user."
            run.completed_at = timezone.now()
            run.save(update_fields=["status", "error", "completed_at"])

    async def _handle_chat_message(self, data):
        content = (data.get("content") or "").strip()
        if not content:
            await self.send(text_data=json.dumps({"error": "Empty message"}))
            return

        if content.startswith("/"):
            await self._handle_slash_command(content, data)
            return

        thread_id = data.get("thread_id")
        attachment_ids = data.get("attachment_ids") or []

        # Per-message model and thinking overrides
        requested_model = data.get("model") or None
        thinking_level = data.get("thinking_level", "off")
        if thinking_level not in ("off", "low", "medium", "high"):
            thinking_level = "off"

        # Allow payload to specify data_room_ids (e.g. for new threads)
        payload_room_ids = data.get("data_room_ids")
        if payload_room_ids and isinstance(payload_room_ids, list):
            self.data_room_ids = payload_room_ids

        # Allow payload to specify skill_id
        payload_skill_id = data.get("skill_id")
        if payload_skill_id:
            self.active_skill_id = str(payload_skill_id)
        elif payload_skill_id == "":
            self.active_skill_id = None

        try:
            # Get or create thread
            thread, created = await self._get_or_create_thread(thread_id)
            self._active_thread_id = str(thread.id)

            if created:
                # Persist session data_room_ids as M2M for new threads
                if self.data_room_ids:
                    await self._persist_data_room_links(thread.id, self.data_room_ids)
                # Persist skill FK for new threads
                if self.active_skill_id:
                    await self._persist_thread_skill(str(thread.id), self.active_skill_id)

                await self.send(text_data=json.dumps({
                    "event_type": "thread.created",
                    "thread_id": str(thread.id),
                }))

            # Persist user message
            user_metadata = {}
            if attachment_ids:
                user_metadata["attachment_ids"] = attachment_ids
            user_message = await self._create_message(
                thread, "user", content, metadata=user_metadata or None,
            )

            # Link attachments to the saved message
            if attachment_ids:
                await self._link_attachments(attachment_ids, thread, user_message)

            # Check if user is suspended mid-session
            if await self._check_suspension():
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.suspended",
                    "data": {"message": "Your account has been suspended. Please contact your system administrator."},
                }))
                return

            # --- Guardrail check: Layer 0 (heuristic, instant) ---
            from guardrails.service import check_heuristics, run_classifier_pipeline

            org_id = await self._get_org_id()
            heuristic_verdict = await check_heuristics(
                text=content,
                user=self.user,
                thread_id=str(thread.id),
                org_id=org_id,
            )

            if heuristic_verdict.action == "block":
                await self._redact_messages(thread, redact_assistant=False)
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.blocked",
                    "data": {"message": heuristic_verdict.message, "redact": True},
                }))
                return

            # Prepare cancel_event before launching parallel tasks
            self._cancel_event = threading.Event()
            self._stream_finished.clear()
            self._guardrail_intercepted = False
            self._guardrail_warn_verdict = None
            self._modified_canvas_ids = set()

            # Launch classifier+reviewer pipeline in parallel with the LLM stream
            self._guardrail_task = asyncio.create_task(
                self._run_guardrail_pipeline(
                    content, heuristic_verdict.heuristic_result,
                    thread, org_id, self._cancel_event,
                )
            )

            # Resolve model early for dynamic history budget
            prefs = self.resolved_prefs
            if requested_model and prefs and requested_model in prefs.allowed_models:
                model = requested_model
            else:
                model = prefs.top_model if prefs else None

            max_context_tokens = prefs.max_context_tokens if prefs else None

            # Load conversation history (token-aware, model-aware budget)
            history_result = await self._load_history(thread, model=model, max_context_tokens=max_context_tokens)
            history = history_result["messages"]
            meta = history_result["meta"]

            # Gather document context for the system prompt
            doc_context = None
            if self.data_room_ids:
                doc_context = await self._get_document_context(
                    self.data_room_ids, content,
                )

            # Build system prompt (split into static/semi-static/dynamic for caching)
            from chat.prompts import (
                build_dynamic_context,
                build_semi_static_prompt,
                build_static_system_prompt,
            )
            data_rooms = None
            if self.data_room_ids:
                data_rooms = await self._get_data_room_info(self.data_room_ids)
            org_name = await self._get_organization_name()
            try:
                canvases_info = await self._get_canvases_for_prompt(str(thread.id))
            except Exception:
                logger.exception("Failed to load canvases for thread %s", thread.id)
                canvases_info = None
            skill_obj = None
            if self.active_skill_id:
                skill_obj = await self._load_skill(self.active_skill_id)
            tasks = await self._get_thread_tasks(str(thread.id))
            subagent_runs = await self._get_subagent_runs(str(thread.id)) if self._has_tool("create_subagent") else None
            parallel_subagents = prefs.parallel_subagents if prefs else True
            static_system = build_static_system_prompt(
                organization_name=org_name,
                has_subagent_tool=self._has_tool("create_subagent"),
                has_task_tool=self._has_tool("update_tasks"),
                parallel_subagents=parallel_subagents,
            )
            semi_static_system = build_semi_static_prompt(
                data_rooms=data_rooms,
                canvases=canvases_info["canvases"] if canvases_info else None,
                skill=skill_obj,
            )
            dynamic_context = build_dynamic_context(
                doc_context=doc_context,
                active_canvas=canvases_info["active_canvas"] if canvases_info else None,
                tasks=tasks,
                subagent_runs=subagent_runs if subagent_runs else None,
                history_meta=meta,
                data_rooms=data_rooms,
            )

            # Mark undelivered completed sub-agent results as delivered
            if subagent_runs:
                undelivered_ids = [
                    r["id"] for r in subagent_runs
                    if r["status"] == "completed" and not r["result_delivered"]
                ]
                if undelivered_ids:
                    await self._mark_subagent_results_delivered(undelivered_ids)

            # Stream LLM response (cancel_event already created above)
            await self._stream_response(
                thread, static_system, history,
                semi_static_system=semi_static_system,
                dynamic_context=dynamic_context,
                requested_model=requested_model, thinking_level=thinking_level,
                resolved_model=model,
                cancel_event=self._cancel_event,
            )

            # Wait for guardrail pipeline to finish
            if self._guardrail_task and not self._guardrail_task.done():
                await self._guardrail_task

            # If the guardrail pipeline intercepted, redact messages and skip post-stream work
            if self._guardrail_intercepted:
                await self._redact_messages(thread)
                redacted_ids = await self._redact_canvases(thread)
                for cid in redacted_ids:
                    canvas_data = await self._get_canvas_for_redaction_event(thread, cid)
                    if canvas_data:
                        await self.send(text_data=json.dumps(canvas_data))
                return

            # Send guardrail warning after stream if the pipeline flagged but allowed
            if self._guardrail_warn_verdict:
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.warning",
                    "data": {"message": self._guardrail_warn_verdict.message},
                }))

            # Send updated thread cost
            thread_cost = await self._get_thread_cost(str(thread.id))
            await self.send(text_data=json.dumps({
                "event_type": "thread.cost_updated",
                "thread_cost_usd": thread_cost,
            }))

            # Auto-generate title for new threads
            if created:
                await self._generate_thread_title(thread, content)

            # Trigger summarization if history exceeds budget
            if meta.get("needs_summary"):
                await self._trigger_summarization(thread, model=model, max_context_tokens=max_context_tokens)

        except Exception:
            logger.exception("Error handling chat message")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "An error occurred processing your message."},
            }))

    async def _stream_response(
        self, thread, system_prompt, history,
        semi_static_system="",
        dynamic_context="",
        requested_model=None, thinking_level="off", resolved_model=None,
        cancel_event=None,
    ):
        from llm import get_llm_service
        from llm.service.errors import LLMConfigurationError, LLMPolicyDenied, LLMProviderError
        from llm.types import ChatRequest, Message, RunContext

        # System message contains ONLY the static prompt (never changes).
        # Semi-static content (date, skill, data rooms, canvas metadata) is
        # injected into the last user message alongside dynamic context, so
        # the system message + conversation history prefix always caches.
        messages = [Message(role="system", content=system_prompt)]
        for msg in history:
            tool_calls = None
            if msg.get("tool_calls"):
                from llm.types.messages import ToolCall
                tool_calls = [
                    ToolCall(
                        id=tc["id"], name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                    for tc in msg["tool_calls"]
                ]
            messages.append(Message(
                role=msg["role"],
                content=msg["content"],
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=tool_calls,
            ))

        # Enrich user messages that have image attachments with multimodal content blocks
        await self._enrich_with_attachments(messages, history, resolved_model)

        # Deduplicate tool results from prior turns to reduce token waste
        from chat.dedup import deduplicate_tool_results
        messages = deduplicate_tool_results(messages, dynamic_context=dynamic_context)

        # Inject semi-static + dynamic context into the last user message.
        # This keeps the system message (static) + conversation history prefix
        # fully cacheable. Semi-static content (date, skill, data rooms, canvas
        # metadata) changes rarely; dynamic content changes every turn.
        injected_context = ""
        if semi_static_system and dynamic_context:
            injected_context = semi_static_system + "\n\n" + dynamic_context
        elif semi_static_system:
            injected_context = semi_static_system
        elif dynamic_context:
            injected_context = dynamic_context

        if injected_context:
            for i in range(len(messages) - 1, 0, -1):
                if messages[i].role == "user":
                    original = messages[i].content
                    if isinstance(original, str):
                        messages[i] = messages[i].model_copy(
                            update={"content": injected_context + "\n\n" + original}
                        )
                    elif isinstance(original, list):
                        # Multimodal content (images): prepend a text block
                        context_block = {"type": "text", "text": injected_context}
                        messages[i] = messages[i].model_copy(
                            update={"content": [context_block] + list(original)}
                        )
                    break

        context = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
            data_room_ids=self.data_room_ids,
        )

        prefs = self.resolved_prefs

        # Use pre-resolved model from _handle_chat_message, or resolve here
        model = resolved_model
        if model is None:
            if requested_model and prefs and requested_model in prefs.allowed_models:
                model = requested_model
            else:
                model = prefs.top_model if prefs else None

        # Web tools always available; document tools only with data rooms; canvas tools always
        from llm.tools.registry import get_tool_registry
        doc_tools = {"search_documents", "read_document"}
        all_tools = prefs.allowed_tools if prefs else list(get_tool_registry().list_tools().keys())
        if self.data_room_ids:
            tools = list(all_tools)
        else:
            tools = [t for t in all_tools if t not in doc_tools]

        # Extend with skill-specific tools (filtered through prefs.allowed_skills)
        if self.active_skill_id and prefs and prefs.allowed_skills:
            for s in prefs.allowed_skills:
                if s["id"] == self.active_skill_id:
                    for t in s["tool_names"]:
                        if t not in tools:
                            tools.append(t)
                    break
            # If the skill is not in allowed_skills, it was disabled by the
            # org — do NOT fall back to raw tool_names as that would bypass
            # org-level filtering.
        elif self.active_skill_id:
            # No prefs available (e.g. no org) — fall back to raw tool_names
            skill_tool_names = await self._get_skill_tool_names(self.active_skill_id)
            for t in skill_tool_names:
                if t not in tools:
                    tools.append(t)

        if cancel_event is not None:
            self._cancel_event = cancel_event
        elif self._cancel_event is None:
            self._cancel_event = threading.Event()

        request = ChatRequest(
            messages=messages,
            model=model,
            stream=True,
            tools=tools,
            context=context,
            params={"thinking_level": thinking_level, "_cancel_event": self._cancel_event},
        )

        service = get_llm_service()
        accumulated_content = ""
        accumulated_thinking = ""
        pending_tool_calls = []
        pending_tool_results = []
        heartbeat_task = asyncio.create_task(self._send_heartbeats())

        try:
            async for event in service.astream("simple_chat", request, cancel_event=self._cancel_event):
                event_data = event.model_dump()
                await self.send(text_data=json.dumps(event_data))

                # Accumulate assistant text from token events
                if event.event_type == "token":
                    token_text = event.data.get("text", "")
                    accumulated_content += token_text

                # Accumulate thinking/reasoning content
                elif event.event_type == "thinking":
                    accumulated_thinking += event.data.get("text", "")

                # Track tool calls for persistence
                elif event.event_type == "tool_start":
                    pending_tool_calls.append(event.data)
                elif event.event_type == "tool_end":
                    pending_tool_results.append(event.data)
                    # Intercept canvas tool results and broadcast canvas.updated
                    tool_name = event.data.get("tool_name", "")
                    if tool_name in ("write_canvas", "edit_canvas", "show_skill_field_in_canvas", "load_template_to_canvas"):
                        try:
                            result = json.loads(event.data.get("result", "{}"))
                            if result.get("status") == "ok":
                                canvas_event = {
                                    "event_type": "canvas.updated",
                                    "title": result.get("title", ""),
                                    "content": result.get("content", ""),
                                }
                                if "accepted_content" in result:
                                    canvas_event["accepted_content"] = result["accepted_content"]
                                if "canvas_id" in result:
                                    canvas_event["canvas_id"] = result["canvas_id"]
                                    self._modified_canvas_ids.add(result["canvas_id"])
                                await self.send(text_data=json.dumps(canvas_event))
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    # Intercept task tool results and broadcast tasks.updated
                    if tool_name == "update_tasks":
                        try:
                            result = json.loads(event.data.get("result", "{}"))
                            if result.get("status") == "ok":
                                await self.send(text_data=json.dumps({
                                    "event_type": "tasks.updated",
                                    "tasks": result.get("tasks", []),
                                }))
                        except (json.JSONDecodeError, AttributeError):
                            pass
                elif event.event_type == "message_start" and pending_tool_calls:
                    # Tool loop completed, new LLM turn starting — persist intermediate messages
                    await self._persist_tool_loop_messages(
                        thread, pending_tool_calls, pending_tool_results,
                    )
                    pending_tool_calls = []
                    pending_tool_results = []
                    accumulated_content = ""
                    accumulated_thinking = ""

            # Persist any remaining tool loop messages (if stream ends after tools)
            if pending_tool_calls:
                await self._persist_tool_loop_messages(
                    thread, pending_tool_calls, pending_tool_results,
                )

            # Persist final assistant message (with thinking in metadata if present)
            if accumulated_content.strip():
                metadata = {}
                if accumulated_thinking:
                    metadata["thinking"] = accumulated_thinking
                await self._create_message(
                    thread, "assistant", accumulated_content, metadata=metadata,
                )

        except LLMConfigurationError:
            logger.exception("LLM misconfigured for streaming response")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "AI service is not configured. Please contact support."},
            }))
        except LLMPolicyDenied:
            logger.exception("LLM policy denied streaming response")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "This request is not allowed by the current policy."},
            }))
        except LLMProviderError as exc:
            logger.exception("LLM provider error during streaming response")
            error_data = {"message": str(exc) or "The AI service encountered an error. Please try again."}
            if hasattr(exc, "error_code"):
                error_data["error_code"] = exc.error_code
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": error_data,
            }))
        except Exception:
            logger.exception("Unexpected error streaming LLM response")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "Failed to get AI response."},
            }))
        finally:
            heartbeat_task.cancel()
            self._stream_finished.set()
            self._cancel_event = None

    async def _run_guardrail_pipeline(
        self, text, heuristic_result, thread, org_id, cancel_event,
    ):
        """Run classifier+reviewer pipeline concurrently with the LLM stream.

        If the verdict is block/suspend and the stream is still running,
        cancel the stream and send the guardrail event to the client.
        """
        from guardrails.service import STREAM_INTERCEPT_ACTIONS, run_classifier_pipeline

        try:
            verdict = await run_classifier_pipeline(
                text=text,
                user=self.user,
                heuristic_result=heuristic_result,
                thread_id=str(thread.id),
                org_id=org_id,
            )
        except asyncio.CancelledError:
            logger.debug("guardrail: pipeline cancelled (user stopped)")
            return
        except Exception:
            logger.exception("guardrail: pipeline error, failing open")
            return

        if verdict.action in STREAM_INTERCEPT_ACTIONS and not self._stream_finished.is_set():
            # Stream still running — intercept it
            self._guardrail_intercepted = True
            cancel_event.set()

            if verdict.action == "block":
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.blocked",
                    "data": {"message": verdict.message, "redact": True},
                }))
            elif verdict.action == "suspend":
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.suspended",
                    "data": {"message": verdict.message, "redact": True},
                }))
        elif verdict.action in STREAM_INTERCEPT_ACTIONS:
            # Stream already finished — redact persisted messages and notify frontend
            self._guardrail_intercepted = True
            await self._redact_messages(thread)
            redacted_ids = await self._redact_canvases(thread)
            for cid in redacted_ids:
                canvas_data = await self._get_canvas_for_redaction_event(thread, cid)
                if canvas_data:
                    await self.send(text_data=json.dumps(canvas_data))

            if verdict.action == "block":
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.blocked",
                    "data": {"message": verdict.message, "redact": True},
                }))
            elif verdict.action == "suspend":
                await self.send(text_data=json.dumps({
                    "event_type": "guardrail.suspended",
                    "data": {"message": verdict.message, "redact": True},
                }))
        elif verdict.action == "warn":
            # Store for post-stream delivery
            self._guardrail_warn_verdict = verdict

    # -- Summarization helpers --

    async def _trigger_summarization(self, thread, model=None, max_context_tokens=None):
        """Summarise messages outside the token window and save to thread."""
        try:
            from chat.services import generate_summary

            thread_data = await self._get_thread_summary_data(thread)
            messages_to_summarise = await self._get_messages_to_summarise(thread, model=model, max_context_tokens=max_context_tokens)

            if not messages_to_summarise:
                return

            summary_text = await generate_summary(
                messages_to_summarise,
                existing_summary=thread_data["summary"],
                user_id=self.user.pk,
                conversation_id=str(thread.id),
            )

            last_msg = messages_to_summarise[-1]
            new_count = thread_data["summary_message_count"] + len(messages_to_summarise)
            await self._save_summary(
                thread, summary_text, last_msg.id, new_count,
            )
        except Exception:
            logger.exception("Failed to generate conversation summary")

    @database_sync_to_async
    def _get_thread_summary_data(self, thread):
        from chat.models import ChatThread

        t = ChatThread.objects.get(pk=thread.pk)
        return {
            "summary": t.summary,
            "summary_token_count": t.summary_token_count,
            "summary_up_to_message_id": t.summary_up_to_message_id,
            "summary_message_count": t.summary_message_count,
        }

    @database_sync_to_async
    def _get_messages_to_summarise(self, thread, model=None, max_context_tokens=None):
        """Return unsummarised messages that fall outside the token budget window.

        The overlap window is always preserved as raw context and is never summarised.
        """
        from chat.models import ChatMessage, ChatThread
        from llm.model_info import get_history_budget

        max_history_tokens = get_history_budget(model, max_context_tokens=max_context_tokens) if model else MAX_HISTORY_TOKENS
        overlap_tokens = min(4_000, max_history_tokens // 10)

        t = ChatThread.objects.get(pk=thread.pk)

        # All unsummarised messages, newest first
        qs = ChatMessage.objects.filter(thread=thread).order_by("-created_at")
        if t.summary_up_to_message_id:
            cutoff_msg = ChatMessage.objects.filter(
                id=t.summary_up_to_message_id,
            ).first()
            if cutoff_msg:
                qs = qs.filter(created_at__gt=cutoff_msg.created_at)

        all_msgs = list(qs)

        # Reserve the overlap window — never summarise those messages
        overlap_used = 0
        overlap_count = 0
        for msg in all_msgs:
            overlap_used += msg.token_count
            overlap_count += 1
            if overlap_used >= overlap_tokens:
                break

        # Candidates for summarisation: everything beyond the overlap window
        non_overlap = all_msgs[overlap_count:]

        # Of those, keep what fits in the remaining budget
        remaining_budget = max(0, max_history_tokens - t.summary_token_count - overlap_used)
        keep_count = 0
        used = 0
        for msg in non_overlap:
            used += msg.token_count
            if used > remaining_budget:
                break
            keep_count += 1

        to_summarise = non_overlap[keep_count:]
        to_summarise.reverse()  # chronological order
        return to_summarise

    @database_sync_to_async
    def _save_summary(self, thread, text, last_msg_id, count):
        from chat.models import ChatThread
        from core.tokens import count_tokens

        ChatThread.objects.filter(pk=thread.pk).update(
            summary=text,
            summary_token_count=count_tokens(text),
            summary_up_to_message_id=last_msg_id,
            summary_message_count=count,
        )

    # -- Document context helpers --

    async def _get_document_context(self, data_room_ids, user_message):
        """Search data room documents and return context for the system prompt."""
        try:
            doc_context = await self._search_and_build_doc_context(
                data_room_ids, user_message,
            )
            return doc_context
        except Exception:
            logger.exception("Failed to build document context")
            total = await self._count_data_room_documents(data_room_ids)
            return {"total_doc_count": total, "documents": []}

    @database_sync_to_async
    def _search_and_build_doc_context(self, data_room_ids, user_message):
        from documents.models import DataRoomDocument
        from documents.services.retrieval import hybrid_search_chunks

        total_count = DataRoomDocument.objects.filter(
            data_room_id__in=data_room_ids,
            status=DataRoomDocument.Status.READY,
            is_archived=False,
        ).count()

        if total_count == 0:
            return {"total_doc_count": 0, "documents": []}

        # Run hybrid search to find relevant chunks
        try:
            results = hybrid_search_chunks(
                data_room_ids=data_room_ids, query=user_message, k=10,
            )
        except Exception:
            logger.exception("Document context: hybrid search failed")
            results = []

        # Collect unique documents from results (up to 5)
        seen_doc_ids = []
        for r in results:
            doc_id = r.get("document_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.append(doc_id)
            if len(seen_doc_ids) >= 5:
                break

        # Fetch document metadata
        documents = []
        if seen_doc_ids:
            from documents.models import DataRoomDocumentTag

            docs = DataRoomDocument.objects.filter(
                pk__in=seen_doc_ids,
                status=DataRoomDocument.Status.READY,
                is_archived=False,
            ).select_related("data_room").values(
                "pk", "doc_index", "original_filename", "description",
                "token_count", "data_room__name", "data_room_id",
            )

            doc_map = {d["pk"]: d for d in docs}

            # Fetch document_type tags
            doc_type_map = dict(
                DataRoomDocumentTag.objects.filter(
                    document_id__in=seen_doc_ids, key="document_type",
                ).values_list("document_id", "value")
            )

            multi_room = len(data_room_ids) > 1
            for doc_id in seen_doc_ids:
                d = doc_map.get(doc_id)
                if d:
                    entry = {
                        "doc_index": d["doc_index"],
                        "filename": d["original_filename"],
                        "description": d["description"] or "",
                        "token_count": d["token_count"],
                        "document_type": doc_type_map.get(doc_id, ""),
                    }
                    if multi_room:
                        entry["data_room_name"] = d["data_room__name"]
                    documents.append(entry)

        return {"total_doc_count": total_count, "documents": documents}

    @database_sync_to_async
    def _count_data_room_documents(self, data_room_ids):
        from documents.models import DataRoomDocument

        return DataRoomDocument.objects.filter(
            data_room_id__in=data_room_ids,
            status=DataRoomDocument.Status.READY,
            is_archived=False,
        ).count()

    # -- Data room helpers --

    @database_sync_to_async
    def _validate_data_room(self, data_room_id):
        from documents.models import DataRoom

        try:
            room = DataRoom.objects.get(pk=data_room_id)
        except DataRoom.DoesNotExist:
            return None
        if room.created_by_id != self.user.pk:
            return None
        return {"id": room.pk, "name": room.name}

    @database_sync_to_async
    def _get_data_room_info(self, data_room_ids):
        from documents.models import DataRoom

        rooms = DataRoom.objects.filter(pk__in=data_room_ids).values("pk", "name", "description")
        return [{"id": r["pk"], "name": r["name"], "description": r["description"] or ""} for r in rooms]

    @database_sync_to_async
    def _persist_data_room_link(self, thread_id, data_room_id):
        from chat.models import ChatThreadDataRoom

        ChatThreadDataRoom.objects.get_or_create(
            thread_id=thread_id, data_room_id=data_room_id,
        )

    @database_sync_to_async
    def _persist_data_room_links(self, thread_id, data_room_ids):
        from chat.models import ChatThreadDataRoom

        for room_id in data_room_ids:
            ChatThreadDataRoom.objects.get_or_create(
                thread_id=thread_id, data_room_id=room_id,
            )

    @database_sync_to_async
    def _remove_data_room_link(self, thread_id, data_room_id):
        from chat.models import ChatThreadDataRoom

        ChatThreadDataRoom.objects.filter(
            thread_id=thread_id, data_room_id=data_room_id,
        ).delete()

    @database_sync_to_async
    def _load_thread_data_rooms(self, thread_id):
        from chat.models import ChatThread

        try:
            thread = ChatThread.objects.get(pk=thread_id, created_by=self.user)
        except ChatThread.DoesNotExist:
            return None

        rooms = list(
            thread.data_rooms.values("pk", "name")
        )
        return {
            "data_room_ids": [r["pk"] for r in rooms],
            "data_rooms": [{"id": r["pk"], "name": r["name"]} for r in rooms],
        }

    # -- Cost helpers --

    @database_sync_to_async
    def _get_thread_cost(self, thread_id):
        from django.db.models import Sum

        from llm.models import LLMCallLog

        result = LLMCallLog.objects.filter(
            conversation_id=str(thread_id),
        ).aggregate(total=Sum("cost_usd"))
        total = result["total"]
        return float(total) if total is not None else 0.0

    # -- Task helpers --

    @database_sync_to_async
    def _get_thread_tasks(self, thread_id):
        from chat.models import ThreadTask
        return list(
            ThreadTask.objects.filter(thread_id=thread_id)
            .order_by("order", "created_at")
            .values("id", "title", "status")
        )

    # -- Sub-agent helpers --

    @database_sync_to_async
    def _get_subagent_runs(self, thread_id):
        from chat.models import SubAgentRun
        return list(
            SubAgentRun.objects.filter(thread_id=thread_id)
            .order_by("-created_at")[:20]
            .values("id", "status", "prompt", "model_tier", "result", "error", "result_delivered")
        )

    @database_sync_to_async
    def _mark_subagent_results_delivered(self, run_ids):
        from chat.models import SubAgentRun
        if run_ids:
            SubAgentRun.objects.filter(pk__in=run_ids).update(result_delivered=True)

    # -- Canvas helpers --

    @database_sync_to_async
    def _load_all_canvases(self, thread_id):
        """Load all canvases for a thread, returning tabs list + active canvas detail."""
        from chat.models import ChatCanvas, ChatThread
        try:
            thread = ChatThread.objects.get(pk=thread_id)
        except ChatThread.DoesNotExist:
            return None
        canvases = list(
            ChatCanvas.objects.filter(thread_id=thread_id)
            .select_related("accepted_checkpoint")
            .order_by("created_at")
        )
        if not canvases:
            return None
        active_id = thread.active_canvas_id
        tabs = []
        for c in canvases:
            tabs.append({
                "id": str(c.pk),
                "title": c.title,
                "is_active": str(c.pk) == str(active_id) if active_id else False,
            })
        # If no active canvas set, default to first
        active_canvas = None
        if active_id:
            active_canvas = next((c for c in canvases if c.pk == active_id), None)
        if not active_canvas:
            active_canvas = canvases[0]
            tabs[0]["is_active"] = True
        accepted_content = (
            active_canvas.accepted_checkpoint.content
            if active_canvas.accepted_checkpoint else None
        )
        return {
            "tabs": tabs,
            "active": {
                "id": str(active_canvas.pk),
                "title": active_canvas.title,
                "content": active_canvas.content,
                "accepted_content": accepted_content,
            },
        }

    @database_sync_to_async
    def _get_canvases_for_prompt(self, thread_id):
        """Load canvases info for the system prompt."""
        from chat.models import ChatCanvas, ChatThread
        canvases = list(
            ChatCanvas.objects.filter(thread_id=thread_id)
            .select_related("accepted_checkpoint")
            .order_by("created_at")
        )
        if not canvases:
            return None
        try:
            thread = ChatThread.objects.get(pk=thread_id)
        except ChatThread.DoesNotExist:
            return None
        active_id = thread.active_canvas_id
        active_canvas = None
        if active_id:
            active_canvas = next((c for c in canvases if c.pk == active_id), None)
        if not active_canvas:
            active_canvas = canvases[0]
        canvases_info = []
        for c in canvases:
            canvases_info.append({
                "title": c.title,
                "chars": len(c.content),
                "is_active": c.pk == active_canvas.pk,
            })
        return {"canvases": canvases_info, "active_canvas": active_canvas}

    def _resolve_canvas_id(self, thread_id, canvas_id=None):
        """Resolve a canvas by ID or fall back to active canvas. Sync helper."""
        from chat.models import ChatCanvas, ChatThread
        if canvas_id:
            try:
                return ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    pk=canvas_id, thread_id=thread_id,
                )
            except ChatCanvas.DoesNotExist:
                return None
        try:
            thread = ChatThread.objects.get(pk=thread_id)
        except ChatThread.DoesNotExist:
            return None
        if thread.active_canvas_id:
            try:
                return ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    pk=thread.active_canvas_id,
                )
            except ChatCanvas.DoesNotExist:
                pass
        # Fall back to first canvas
        return (
            ChatCanvas.objects.filter(thread_id=thread_id)
            .select_related("accepted_checkpoint")
            .order_by("created_at")
            .first()
        )

    @database_sync_to_async
    def _save_canvas(self, thread_id, title, content, canvas_id=None):
        from chat.models import ChatCanvas
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if canvas:
            canvas.title = title
            canvas.content = content
            canvas.save(update_fields=["title", "content", "updated_at"])
        else:
            from chat.services import set_active_canvas
            canvas = ChatCanvas.objects.create(
                thread_id=thread_id, title=title, content=content,
            )
            set_active_canvas(thread_id, canvas)

    @database_sync_to_async
    def _get_or_create_canvas(self, thread_id):
        from chat.models import ChatCanvas
        from chat.services import set_active_canvas
        canvas = (
            ChatCanvas.objects.filter(thread_id=thread_id)
            .select_related("accepted_checkpoint")
            .order_by("created_at")
            .first()
        )
        if not canvas:
            canvas = ChatCanvas.objects.create(
                thread_id=thread_id, title="Untitled document", content="",
            )
            set_active_canvas(thread_id, canvas)
        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else None
        return {
            "id": str(canvas.pk),
            "title": canvas.title,
            "content": canvas.content,
            "accepted_content": accepted_content,
        }

    @database_sync_to_async
    def _canvas_accept(self, thread_id, canvas_id=None):
        from chat.models import CanvasCheckpoint
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if not canvas:
            return None
        latest = CanvasCheckpoint.objects.filter(canvas=canvas).order_by("-order").first()
        if not latest:
            return None
        canvas.accepted_checkpoint = latest
        canvas.save(update_fields=["accepted_checkpoint"])
        return {"accepted_content": latest.content, "canvas_id": str(canvas.pk)}

    @database_sync_to_async
    def _canvas_revert(self, thread_id, canvas_id=None):
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if not canvas or not canvas.accepted_checkpoint:
            return None
        canvas.title = canvas.accepted_checkpoint.title
        canvas.content = canvas.accepted_checkpoint.content
        canvas.save(update_fields=["title", "content", "updated_at"])
        return {"title": canvas.title, "content": canvas.content, "canvas_id": str(canvas.pk)}

    @database_sync_to_async
    def _canvas_save_version(self, thread_id, title, content, canvas_id=None):
        from chat.models import CanvasCheckpoint
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if not canvas:
            return None
        # Skip if content matches latest checkpoint
        latest = CanvasCheckpoint.objects.filter(canvas=canvas).order_by("-order").first()
        if latest and latest.content == content and latest.title == title:
            return {"accepted_content": content, "canvas_id": str(canvas.pk)}
        content = content[:CANVAS_MAX_CHARS]
        canvas.title = title or canvas.title
        canvas.content = content
        canvas.save(update_fields=["title", "content", "updated_at"])
        cp = create_canvas_checkpoint(canvas, source="user_save", description="User saved version")
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])
        return {"accepted_content": cp.content, "canvas_id": str(canvas.pk)}

    @database_sync_to_async
    def _canvas_restore_version(self, thread_id, checkpoint_id, canvas_id=None):
        from chat.models import CanvasCheckpoint
        from chat.services import create_canvas_checkpoint
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if not canvas:
            return None
        try:
            checkpoint = CanvasCheckpoint.objects.get(pk=checkpoint_id, canvas=canvas)
        except CanvasCheckpoint.DoesNotExist:
            return None
        if checkpoint.source == "redacted":
            return None
        canvas.title = checkpoint.title
        canvas.content = checkpoint.content
        canvas.save(update_fields=["title", "content", "updated_at"])
        cp = create_canvas_checkpoint(canvas, source="restore", description="Restored to checkpoint #%d" % checkpoint.order)
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])
        return {"title": canvas.title, "content": canvas.content, "canvas_id": str(canvas.pk)}

    @database_sync_to_async
    def _canvas_get_checkpoints(self, thread_id, canvas_id=None):
        from chat.models import CanvasCheckpoint
        canvas = self._resolve_canvas_id(thread_id, canvas_id)
        if not canvas:
            return []
        checkpoints = CanvasCheckpoint.objects.filter(canvas=canvas).order_by("-order")
        return [
            {
                "id": cp.pk,
                "source": cp.source,
                "description": cp.description,
                "order": cp.order,
                "created_at": cp.created_at.isoformat(),
            }
            for cp in checkpoints
        ]

    @database_sync_to_async
    def _switch_canvas(self, thread_id, canvas_id):
        from chat.models import ChatCanvas
        from chat.services import set_active_canvas
        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                pk=canvas_id, thread_id=thread_id,
            )
        except ChatCanvas.DoesNotExist:
            return None
        set_active_canvas(thread_id, canvas)
        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else None
        return {
            "id": str(canvas.pk),
            "title": canvas.title,
            "content": canvas.content,
            "accepted_content": accepted_content,
        }

    # -- Skill helpers --

    @database_sync_to_async
    def _validate_skill(self, skill_id):
        from agent_skills.services import get_skill_for_user

        skill = get_skill_for_user(self.user, skill_id)
        if not skill:
            return None
        return {"id": str(skill.pk), "name": skill.name}

    @database_sync_to_async
    def _persist_thread_skill(self, thread_id, skill_id):
        from chat.models import ChatThread

        ChatThread.objects.filter(pk=thread_id, created_by=self.user).update(
            skill_id=skill_id
        )

    @database_sync_to_async
    def _load_thread_skill(self, thread_id):
        from chat.models import ChatThread

        try:
            thread = ChatThread.objects.select_related("skill").get(
                pk=thread_id, created_by=self.user
            )
        except ChatThread.DoesNotExist:
            return None
        if thread.skill and thread.skill.is_active:
            return {"skill_id": str(thread.skill.pk), "skill_name": thread.skill.name}
        return None

    @database_sync_to_async
    def _load_skill(self, skill_id):
        from agent_skills.models import AgentSkill

        try:
            return (
                AgentSkill.objects
                .prefetch_related("templates")
                .get(pk=skill_id, is_active=True)
            )
        except AgentSkill.DoesNotExist:
            return None

    @database_sync_to_async
    def _get_skill_tool_names(self, skill_id):
        from agent_skills.models import AgentSkill

        try:
            skill = AgentSkill.objects.get(pk=skill_id, is_active=True)
            return skill.tool_names or []
        except AgentSkill.DoesNotExist:
            return []

    # -- Database helpers --

    @database_sync_to_async
    def _get_or_create_thread(self, thread_id):
        from chat.models import ChatThread

        if thread_id:
            try:
                thread = ChatThread.objects.get(
                    id=thread_id,
                    created_by=self.user,
                )
                return thread, False
            except ChatThread.DoesNotExist:
                pass

        thread = ChatThread.objects.create(
            created_by=self.user,
        )
        return thread, True

    @database_sync_to_async
    def _create_message(self, thread, role, content, tool_call_id=None, metadata=None):
        from chat.models import ChatMessage
        from core.tokens import count_tokens

        return ChatMessage.objects.create(
            thread=thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
            token_count=count_tokens(content),
        )

    @database_sync_to_async
    def _redact_messages(self, thread, *, redact_user=True, redact_assistant=True):
        """Overwrite content and mark messages as redacted after a guardrail block.

        Finds the most recent user message and (optionally) all assistant/tool
        messages created at or after it, then wipes their content.
        """
        from chat.models import ChatMessage

        REDACTED_TEXT = "[This message was removed by the content safety system.]"

        last_user = (
            ChatMessage.objects.filter(thread=thread, role="user")
            .order_by("-created_at")
            .first()
        )
        if not last_user:
            return

        if redact_user:
            ChatMessage.objects.filter(pk=last_user.pk).update(
                content=REDACTED_TEXT,
                is_redacted=True,
                metadata={},
                token_count=0,
            )

        if redact_assistant:
            ChatMessage.objects.filter(
                thread=thread,
                role__in=["assistant", "tool"],
                created_at__gte=last_user.created_at,
            ).update(
                content=REDACTED_TEXT,
                is_redacted=True,
                metadata={},
                token_count=0,
            )

    @database_sync_to_async
    def _redact_canvases(self, thread):
        """Redact canvases modified during a guardrail-blocked turn."""
        from chat.models import CanvasCheckpoint, ChatCanvas, ChatMessage

        REDACTED_TEXT = "[This message was removed by the content safety system.]"

        last_user = (
            ChatMessage.objects.filter(thread=thread, role="user")
            .order_by("-created_at")
            .first()
        )
        if not last_user:
            return []

        cutoff = last_user.created_at

        # Find canvases via checkpoints created during this turn
        turn_cps = CanvasCheckpoint.objects.filter(
            canvas__thread=thread,
            source__in=["ai_edit", "original"],
            created_at__gte=cutoff,
        )
        affected_ids = set(str(cp.canvas_id) for cp in turn_cps)
        affected_ids |= getattr(self, "_modified_canvas_ids", set())

        if not affected_ids:
            return []

        redacted = []
        for canvas_id in affected_ids:
            try:
                canvas = ChatCanvas.objects.get(pk=canvas_id, thread=thread)
            except ChatCanvas.DoesNotExist:
                continue

            # Mark turn checkpoints as redacted
            CanvasCheckpoint.objects.filter(
                canvas=canvas,
                source__in=["ai_edit", "original"],
                created_at__gte=cutoff,
            ).update(
                content=REDACTED_TEXT,
                source="redacted",
                description="Redacted by content safety system",
            )

            # Roll back to last pre-turn checkpoint, or redact content
            pre_turn_cp = (
                CanvasCheckpoint.objects.filter(canvas=canvas, created_at__lt=cutoff)
                .exclude(source="redacted")
                .order_by("-order")
                .first()
            )
            if pre_turn_cp:
                canvas.content = pre_turn_cp.content
                canvas.title = pre_turn_cp.title
                canvas.accepted_checkpoint = pre_turn_cp
            else:
                canvas.content = REDACTED_TEXT
                canvas.accepted_checkpoint = None

            canvas.save(update_fields=["content", "title", "accepted_checkpoint", "updated_at"])
            redacted.append(str(canvas.pk))

        return redacted

    @database_sync_to_async
    def _get_canvas_for_redaction_event(self, thread, canvas_id):
        """Build a canvas.updated event payload for a redacted canvas."""
        from chat.models import ChatCanvas

        try:
            c = ChatCanvas.objects.get(pk=canvas_id, thread=thread)
        except ChatCanvas.DoesNotExist:
            return None
        accepted = c.accepted_checkpoint
        return {
            "event_type": "canvas.updated",
            "canvas_id": str(c.pk),
            "title": c.title,
            "content": c.content,
            "accepted_content": accepted.content if accepted else c.content,
        }

    @database_sync_to_async
    def _link_attachments(self, attachment_ids, thread, message):
        from chat.models import ChatAttachment

        ChatAttachment.objects.filter(
            id__in=attachment_ids,
            thread=thread,
            uploaded_by=self.user,
            message__isnull=True,
        ).update(message=message)

    async def _enrich_with_attachments(self, messages, history, model):
        """Replace plain-text content with multimodal content blocks for messages with attachments."""
        import base64

        from chat.services import (
            SUPPORTED_DOCX_TYPES,
            SUPPORTED_IMAGE_TYPES,
            SUPPORTED_PDF_TYPES,
            SUPPORTED_TEXT_TYPES,
            build_image_content_block,
            build_pdf_content_block,
            build_text_content_block,
            extract_docx_text,
        )

        # Collect all attachment IDs from history
        all_ids = []
        for msg in history:
            ids = msg.get("attachment_ids") or []
            all_ids.extend(ids)
        if not all_ids:
            return

        # Load attachment records
        attachments_by_id = await self._load_attachments(all_ids)
        if not attachments_by_id:
            return

        # Determine provider from model
        provider = ""
        if model and "/" in model:
            provider = model.split("/", 1)[0].lower()

        # messages[0] is system prompt, so history[i] corresponds to messages[i+1]
        for i, msg in enumerate(history):
            ids = msg.get("attachment_ids") or []
            if not ids:
                continue
            message_obj = messages[i + 1]  # offset by system message
            if message_obj.role != "user":
                continue

            content_blocks = []
            text = message_obj.content if isinstance(message_obj.content, str) else ""
            if text:
                content_blocks.append({"type": "text", "text": text})

            for att_id in ids:
                att = attachments_by_id.get(str(att_id))
                if not att:
                    continue
                try:
                    file_bytes = await self._read_attachment_file(att)
                    ct = att.content_type

                    if ct in SUPPORTED_IMAGE_TYPES:
                        b64 = base64.b64encode(file_bytes).decode("ascii")
                        block = build_image_content_block(b64, ct, provider)
                    elif ct in SUPPORTED_PDF_TYPES:
                        b64 = base64.b64encode(file_bytes).decode("ascii")
                        block = build_pdf_content_block(b64, att.original_filename, provider)
                    elif ct in SUPPORTED_DOCX_TYPES:
                        extracted = extract_docx_text(file_bytes)
                        block = build_text_content_block(extracted, att.original_filename)
                    elif ct in SUPPORTED_TEXT_TYPES:
                        decoded = file_bytes.decode("utf-8", errors="replace")
                        block = build_text_content_block(decoded, att.original_filename)
                    else:
                        continue

                    content_blocks.append(block)
                except Exception:
                    logger.exception("Failed to read attachment %s", att_id)

            if len(content_blocks) > 1:  # has at least text + one attachment
                message_obj.content = content_blocks

    @database_sync_to_async
    def _load_attachments(self, attachment_ids):
        from chat.models import ChatAttachment

        atts = ChatAttachment.objects.filter(id__in=attachment_ids)
        return {str(a.id): a for a in atts}

    @database_sync_to_async
    def _read_attachment_file(self, attachment):
        attachment.file.open("rb")
        try:
            return attachment.file.read()
        finally:
            attachment.file.close()

    async def _persist_tool_loop_messages(self, thread, tool_calls, tool_results):
        """Persist assistant tool-call message and tool result messages."""
        # 1. Assistant message requesting tools (content empty, tool_calls in metadata)
        tc_data = [
            {
                "id": tc.get("tool_call_id", ""),
                "name": tc.get("tool_name", ""),
                "arguments": tc.get("arguments", {}),
            }
            for tc in tool_calls
        ]
        await self._create_message(
            thread, "assistant", "", metadata={"tool_calls": tc_data},
        )
        # 2. Tool result messages
        for tr in tool_results:
            await self._create_message(
                thread, "tool", tr.get("result", ""),
                tool_call_id=tr.get("tool_call_id", ""),
            )

    async def _generate_thread_title(self, thread, first_user_message):
        """Generate a short title for a new thread using a cheap LLM call."""
        try:
            from llm import get_llm_service
            from llm.types import ChatRequest, Message, RunContext

            prompt = (
                "Generate a short title (max 5 words) for a chat that starts with: "
                f"{first_user_message[:500]}. Reply with ONLY the title."
            )
            context = RunContext.create(
                user_id=self.user.pk,
                conversation_id=str(thread.id),
            )
            prefs = self.resolved_prefs
            cheap_model = prefs.cheap_model if prefs else None

            request = ChatRequest(
                messages=[Message(role="user", content=prompt)],
                model=cheap_model or None,
                stream=False,
                tools=[],
                context=context,
            )
            service = get_llm_service()
            response = await service.arun("simple_chat", request)
            title = response.message.content.strip().strip('"').strip("'")[:255]
            if title:
                await self._update_thread_title(thread, title)
                await self.send(text_data=json.dumps({
                    "event_type": "thread.title_updated",
                    "thread_id": str(thread.id),
                    "title": title,
                }))
        except Exception:
            logger.exception("Failed to generate thread title")

    @database_sync_to_async
    def _update_thread_title(self, thread, title):
        from chat.models import ChatThread

        ChatThread.objects.filter(pk=thread.pk).update(title=title)

    @database_sync_to_async
    def _load_history(self, thread, model=None, max_context_tokens=None):
        """Load token-aware conversation history with a recency overlap window.

        The most recent *overlap_tokens* worth of messages are always included as
        raw messages (even if they are already covered by the summary).  Any
        remaining budget is filled with older unsummarised messages.

        When *model* is provided the token budget scales with the model's context
        window (up to 150k).  Falls back to the legacy ``MAX_HISTORY_TOKENS``
        constant when unknown.

        Returns a dict with:
        - ``messages``: list of message dicts to send to the LLM
        - ``meta``: dict with total_messages, included_messages, has_summary,
          needs_summary
        """
        from chat.models import ChatMessage, ChatThread
        from llm.model_info import get_history_budget

        max_history_tokens = get_history_budget(model, max_context_tokens=max_context_tokens) if model else MAX_HISTORY_TOKENS
        overlap_tokens = min(4_000, max_history_tokens // 10)

        t = ChatThread.objects.get(pk=thread.pk)
        total_messages = ChatMessage.objects.filter(thread=thread).exclude(is_redacted=True).count()

        # Load ALL messages newest-first (needed to build the overlap window).
        all_msgs = list(
            ChatMessage.objects.filter(thread=thread).exclude(is_redacted=True).order_by("-created_at")
        )

        if not all_msgs:
            return {
                "messages": [],
                "meta": {
                    "total_messages": 0,
                    "included_messages": 0,
                    "has_summary": False,
                    "needs_summary": False,
                },
            }

        # 1. Build overlap window: newest messages up to overlap_tokens.
        #    Always includes at least one message regardless of size.
        overlap: list = []
        overlap_tokens_used = 0
        for msg in all_msgs:
            overlap_tokens_used += msg.token_count
            overlap.append(msg)
            if overlap_tokens_used >= overlap_tokens:
                break
        # overlap is newest-first; oldest_overlap is the boundary
        oldest_overlap = overlap[-1]

        # 2. Fill remaining budget with unsummarised messages between the
        #    summary cutoff and the start of the overlap window.
        remaining_budget = max(
            0, max_history_tokens - t.summary_token_count - overlap_tokens_used
        )
        add_qs = ChatMessage.objects.filter(
            thread=thread,
            created_at__lt=oldest_overlap.created_at,
        ).exclude(is_redacted=True).order_by("-created_at")
        if t.summary_up_to_message_id:
            cutoff_msg = ChatMessage.objects.filter(
                id=t.summary_up_to_message_id,
            ).first()
            if cutoff_msg:
                add_qs = add_qs.filter(created_at__gt=cutoff_msg.created_at)

        add_msgs = list(add_qs)
        additional: list = []
        used = 0
        for msg in add_msgs:
            used += msg.token_count
            if used > remaining_budget:
                break
            additional.append(msg)

        needs_summary = len(additional) < len(add_msgs)

        # 3. Combine into chronological order: additional + overlap (both reversed)
        included = list(reversed(additional)) + list(reversed(overlap))

        # 4. Build message list
        messages: list[dict] = []
        if t.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{t.summary}",
            })
        for m in included:
            msg_dict = {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            }
            if m.metadata and m.metadata.get("tool_calls"):
                msg_dict["tool_calls"] = m.metadata["tool_calls"]
            if m.metadata and m.metadata.get("attachment_ids"):
                msg_dict["attachment_ids"] = m.metadata["attachment_ids"]
            messages.append(msg_dict)

        return {
            "messages": messages,
            "meta": {
                "total_messages": total_messages,
                "included_messages": len(included),
                "has_summary": bool(t.summary),
                "needs_summary": needs_summary,
            },
        }

    # -- Slash command handling --

    async def _handle_slash_command(self, content, data):
        """Dispatch slash commands typed by the user."""
        parts = content.split(None, 1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/clear": self._cmd_clear,
            "/new": self._cmd_clear,
            "/cost": self._cmd_cost,
            "/tag": self._cmd_tag,
            "/untag": self._cmd_untag,
            "/compact": self._cmd_compact,
        }

        handler = handlers.get(command)
        if handler:
            await handler(args, data)
        else:
            available = ", ".join(sorted(handlers.keys()))
            await self._send_command_result(
                command, "error",
                f"Unknown command {command}. Available commands: {available}",
            )

    async def _send_command_result(self, command, status, message, extra=None):
        """Send a command.result event to the client."""
        payload = {
            "event_type": "command.result",
            "command": command,
            "status": status,
            "message": message,
        }
        if extra:
            payload.update(extra)
        await self.send(text_data=json.dumps(payload))

    async def _cmd_clear(self, args, data):
        """Handle /clear and /new — navigate to new chat."""
        await self._send_command_result(
            "/clear", "ok", "Starting new chat...",
            extra={"action": "navigate"},
        )

    async def _cmd_cost(self, args, data):
        """Handle /cost — show total LLM cost for the thread."""
        thread_id = data.get("thread_id")
        if not thread_id:
            await self._send_command_result("/cost", "ok", "Thread cost: $0.00")
            return
        cost = await self._get_thread_cost(thread_id)
        if cost < 0.01:
            formatted = f"${cost:.4f}"
        else:
            formatted = f"${cost:.2f}"
        await self._send_command_result("/cost", "ok", f"Thread cost: {formatted}")

    async def _cmd_tag(self, args, data):
        """Handle /tag — set or auto-pick a thread emoji."""
        thread_id = data.get("thread_id")
        if not thread_id:
            await self._send_command_result(
                "/tag", "error", "No active thread to tag.",
            )
            return

        thread = await self._get_thread_by_id(thread_id)
        if not thread:
            await self._send_command_result(
                "/tag", "error", "Thread not found.",
            )
            return

        hint = args.strip()[:100] if args else None
        try:
            emoji = await self._auto_pick_emoji(thread, hint=hint)
            await self._update_thread_emoji(thread_id, emoji)
            await self._send_command_result(
                "/tag", "ok", f"Tagged thread with {emoji}",
                extra={"emoji": emoji, "thread_id": str(thread_id)},
            )
        except Exception:
            logger.exception("Failed to auto-pick emoji")
            await self._send_command_result(
                "/tag", "error", "Failed to pick emoji.",
            )

    async def _cmd_untag(self, args, data):
        """Handle /untag — remove the thread emoji."""
        thread_id = data.get("thread_id")
        if not thread_id:
            await self._send_command_result(
                "/untag", "error", "No active thread to untag.",
            )
            return

        thread = await self._get_thread_by_id(thread_id)
        if not thread:
            await self._send_command_result(
                "/untag", "error", "Thread not found.",
            )
            return

        await self._update_thread_emoji(thread_id, "")
        await self._send_command_result(
            "/untag", "ok", "Removed thread tag.",
            extra={"emoji": "", "thread_id": str(thread_id)},
        )

    async def _auto_pick_emoji(self, thread, *, hint=None):
        """Pick an emoji for a thread using a cheap LLM call."""
        from llm import get_llm_service
        from llm.types import ChatRequest, Message, RunContext

        # Get recent messages for context
        from chat.models import ChatMessage

        recent = await database_sync_to_async(
            lambda: list(
                ChatMessage.objects.filter(thread=thread)
                .order_by("-created_at")[:5]
                .values_list("content", flat=True)
            )
        )()
        context_text = "\n".join(reversed(recent))[:500]

        prompt = (
            "Pick a single emoji that best represents this conversation. "
            "Reply with ONLY the emoji, nothing else.\n\n"
        )
        if hint:
            prompt += f"The user asked that you tag it with: {hint}\n\n"
        prompt += context_text
        context = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
        )
        prefs = self.resolved_prefs
        cheap_model = prefs.cheap_model if prefs else None

        request = ChatRequest(
            messages=[Message(role="user", content=prompt)],
            model=cheap_model or None,
            stream=False,
            tools=[],
            context=context,
        )
        service = get_llm_service()
        response = await service.arun("simple_chat", request)
        emoji = response.message.content.strip()[:10]
        return emoji

    async def _cmd_compact(self, args, data):
        """Handle /compact — force summarization of all unsummarised messages."""
        thread_id = data.get("thread_id")
        if not thread_id:
            await self._send_command_result(
                "/compact", "error", "No active thread to compact.",
            )
            return

        thread = await self._get_thread_by_id(thread_id)
        if not thread:
            await self._send_command_result(
                "/compact", "error", "Thread not found.",
            )
            return

        try:
            from chat.services import generate_summary
            from core.tokens import count_tokens

            thread_data = await self._get_thread_summary_data(thread)
            messages = await self._get_all_unsummarised_messages(thread)

            if not messages:
                await self._send_command_result(
                    "/compact", "ok", "Nothing to compact — all messages already summarised.",
                )
                return

            summary_text = await generate_summary(
                messages,
                existing_summary=thread_data["summary"],
                user_id=self.user.pk,
                conversation_id=str(thread.id),
            )

            last_msg = messages[-1]
            new_count = thread_data["summary_message_count"] + len(messages)
            await self._save_summary(thread, summary_text, last_msg.id, new_count)

            token_count = count_tokens(summary_text)
            await self._send_command_result(
                "/compact", "ok",
                f"Compacted {len(messages)} messages into {token_count}-token summary.",
            )
        except Exception:
            logger.exception("Failed to compact thread")
            await self._send_command_result(
                "/compact", "error", "Failed to compact conversation.",
            )

    @database_sync_to_async
    def _get_thread_by_id(self, thread_id):
        """Get a ChatThread owned by the current user."""
        from chat.models import ChatThread

        return ChatThread.objects.filter(
            pk=thread_id, created_by=self.user,
        ).first()

    @database_sync_to_async
    def _update_thread_emoji(self, thread_id, emoji):
        from chat.models import ChatThread

        ChatThread.objects.filter(pk=thread_id).update(emoji=emoji)

    @database_sync_to_async
    def _get_all_unsummarised_messages(self, thread):
        """Return ALL unsummarised messages (for /compact)."""
        from chat.models import ChatMessage, ChatThread

        t = ChatThread.objects.get(pk=thread.pk)
        qs = ChatMessage.objects.filter(thread=thread).order_by("created_at")
        if t.summary_up_to_message_id:
            cutoff_msg = ChatMessage.objects.filter(
                id=t.summary_up_to_message_id,
            ).first()
            if cutoff_msg:
                qs = qs.filter(created_at__gt=cutoff_msg.created_at)
        return list(qs)
