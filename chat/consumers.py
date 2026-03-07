"""WebSocket consumer for chat with LLM streaming."""

from __future__ import annotations

import json
import logging

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

MAX_HISTORY_TOKENS = 20_000
OVERLAP_TOKENS = 2_000  # always show at least this many recent tokens as raw messages


class ChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for chat with LLM streaming and optional RAG via data rooms."""

    async def connect(self):
        self.user = self.scope.get("user")
        self.resolved_prefs = None
        self.data_room_ids: list[int] = []

        # Reject unauthenticated users
        if not self.user or self.user.is_anonymous:
            await self.close(code=4401)
            return

        # Resolve user/org/system preferences
        self.resolved_prefs = await self._resolve_preferences()

        await self.accept()

    @database_sync_to_async
    def _resolve_preferences(self):
        from core.preferences import get_preferences
        return get_preferences(self.user)

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

    async def _handle_load_thread(self, data):
        """Load a thread's data rooms into the session."""
        thread_id = data.get("thread_id")
        if not thread_id:
            return

        thread_data = await self._load_thread_data_rooms(thread_id)
        if thread_data is not None:
            self.data_room_ids = thread_data["data_room_ids"]
            await self.send(text_data=json.dumps({
                "event_type": "thread.loaded",
                "thread_id": thread_id,
                "data_rooms": thread_data["data_rooms"],
            }))

    async def _handle_chat_message(self, data):
        content = (data.get("content") or "").strip()
        if not content:
            await self.send(text_data=json.dumps({"error": "Empty message"}))
            return

        thread_id = data.get("thread_id")

        # Per-message model and thinking overrides
        requested_model = data.get("model") or None
        thinking = bool(data.get("thinking", False))

        # Allow payload to specify data_room_ids (e.g. for new threads)
        payload_room_ids = data.get("data_room_ids")
        if payload_room_ids and isinstance(payload_room_ids, list):
            self.data_room_ids = payload_room_ids

        try:
            # Get or create thread
            thread, created = await self._get_or_create_thread(thread_id)

            if created:
                # Persist session data_room_ids as M2M for new threads
                if self.data_room_ids:
                    await self._persist_data_room_links(thread.id, self.data_room_ids)

                await self.send(text_data=json.dumps({
                    "event_type": "thread.created",
                    "thread_id": str(thread.id),
                }))

            # Persist user message
            await self._create_message(thread, "user", content)

            # Load conversation history (token-aware)
            history_result = await self._load_history(thread)
            history = history_result["messages"]
            meta = history_result["meta"]

            # Gather document context for the system prompt
            doc_context = None
            if self.data_room_ids:
                doc_context = await self._get_document_context(
                    self.data_room_ids, content,
                )

            # Build system prompt
            from chat.prompts import build_system_prompt
            data_rooms = None
            if self.data_room_ids:
                data_rooms = await self._get_data_room_info(self.data_room_ids)
            org_name = await self._get_organization_name()
            system_prompt = build_system_prompt(
                data_rooms=data_rooms,
                history_meta=meta,
                doc_context=doc_context,
                organization_name=org_name,
            )

            # Stream LLM response
            await self._stream_response(
                thread, system_prompt, history,
                requested_model=requested_model, thinking=thinking,
            )

            # Auto-generate title for new threads
            if created:
                await self._generate_thread_title(thread, content)

            # Trigger summarization if history exceeds budget
            if meta.get("needs_summary"):
                await self._trigger_summarization(thread)

        except Exception:
            logger.exception("Error handling chat message")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "An error occurred processing your message."},
            }))

    async def _stream_response(
        self, thread, system_prompt, history,
        requested_model=None, thinking=False,
    ):
        from llm import get_llm_service
        from llm.service.errors import LLMConfigurationError, LLMPolicyDenied, LLMProviderError
        from llm.types import ChatRequest, Message, RunContext

        messages = [Message(role="system", content=system_prompt)]
        for msg in history:
            messages.append(Message(
                role=msg["role"],
                content=msg["content"],
                tool_call_id=msg.get("tool_call_id"),
            ))

        context = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
            data_room_ids=self.data_room_ids,
        )

        prefs = self.resolved_prefs

        # Validate requested model against user's allowed models
        if requested_model and prefs and requested_model in prefs.allowed_models:
            model = requested_model
        else:
            model = prefs.primary_model if prefs else None

        # Only include document tools when data rooms are attached
        if self.data_room_ids:
            tools = prefs.allowed_tools if prefs else ["search_documents", "read_document"]
        else:
            tools = []

        request = ChatRequest(
            messages=messages,
            model=model,
            stream=True,
            tools=tools,
            context=context,
            params={"thinking": thinking},
        )

        service = get_llm_service()
        accumulated_content = ""

        try:
            async for event in service.astream("simple_chat", request):
                event_data = event.model_dump()
                await self.send(text_data=json.dumps(event_data))

                # Accumulate assistant text from token events
                if event.event_type == "token":
                    token_text = event.data.get("text", "")
                    accumulated_content += token_text

            # Persist assistant message
            if accumulated_content.strip():
                await self._create_message(thread, "assistant", accumulated_content)

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
        except LLMProviderError:
            logger.exception("LLM provider error during streaming response")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "The AI service encountered an error. Please try again."},
            }))
        except Exception:
            logger.exception("Unexpected error streaming LLM response")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "Failed to get AI response."},
            }))

    # -- Summarization helpers --

    async def _trigger_summarization(self, thread):
        """Summarise messages outside the token window and save to thread."""
        try:
            from chat.services import generate_summary

            thread_data = await self._get_thread_summary_data(thread)
            messages_to_summarise = await self._get_messages_to_summarise(thread)

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
    def _get_messages_to_summarise(self, thread):
        """Return unsummarised messages that fall outside the token budget window.

        The overlap window (newest OVERLAP_TOKENS worth of messages) is always
        preserved as raw context and is never summarised.
        """
        from chat.models import ChatMessage, ChatThread

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
            if overlap_used >= OVERLAP_TOKENS:
                break

        # Candidates for summarisation: everything beyond the overlap window
        non_overlap = all_msgs[overlap_count:]

        # Of those, keep what fits in the remaining budget
        remaining_budget = max(0, MAX_HISTORY_TOKENS - t.summary_token_count - overlap_used)
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
            docs = DataRoomDocument.objects.filter(
                pk__in=seen_doc_ids,
                status=DataRoomDocument.Status.READY,
                is_archived=False,
            ).select_related("data_room").values(
                "pk", "doc_index", "original_filename", "description",
                "token_count", "data_room__name", "data_room_id",
            )

            doc_map = {d["pk"]: d for d in docs}
            multi_room = len(data_room_ids) > 1
            for doc_id in seen_doc_ids:
                d = doc_map.get(doc_id)
                if d:
                    entry = {
                        "doc_index": d["doc_index"],
                        "filename": d["original_filename"],
                        "description": d["description"] or "",
                        "token_count": d["token_count"],
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

        rooms = DataRoom.objects.filter(pk__in=data_room_ids).values("pk", "name")
        return [{"id": r["pk"], "name": r["name"]} for r in rooms]

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
    def _create_message(self, thread, role, content, tool_call_id=None):
        from chat.models import ChatMessage
        from core.tokens import count_tokens

        return ChatMessage.objects.create(
            thread=thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(content),
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
    def _load_history(self, thread):
        """Load token-aware conversation history with a recency overlap window.

        The most recent OVERLAP_TOKENS worth of messages are always included as
        raw messages (even if they are already covered by the summary).  Any
        remaining budget is filled with older unsummarised messages.

        Returns a dict with:
        - ``messages``: list of message dicts to send to the LLM
        - ``meta``: dict with total_messages, included_messages, has_summary,
          needs_summary
        """
        from chat.models import ChatMessage, ChatThread

        t = ChatThread.objects.get(pk=thread.pk)
        total_messages = ChatMessage.objects.filter(thread=thread).count()

        # Load ALL messages newest-first (needed to build the overlap window).
        all_msgs = list(
            ChatMessage.objects.filter(thread=thread).order_by("-created_at")
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

        # 1. Build overlap window: newest messages up to OVERLAP_TOKENS.
        #    Always includes at least one message regardless of size.
        overlap: list = []
        overlap_tokens_used = 0
        for msg in all_msgs:
            overlap_tokens_used += msg.token_count
            overlap.append(msg)
            if overlap_tokens_used >= OVERLAP_TOKENS:
                break
        # overlap is newest-first; oldest_overlap is the boundary
        oldest_overlap = overlap[-1]

        # 2. Fill remaining budget with unsummarised messages between the
        #    summary cutoff and the start of the overlap window.
        remaining_budget = max(
            0, MAX_HISTORY_TOKENS - t.summary_token_count - overlap_tokens_used
        )
        add_qs = ChatMessage.objects.filter(
            thread=thread,
            created_at__lt=oldest_overlap.created_at,
        ).order_by("-created_at")
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
            messages.append({
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            })

        return {
            "messages": messages,
            "meta": {
                "total_messages": total_messages,
                "included_messages": len(included),
                "has_summary": bool(t.summary),
                "needs_summary": needs_summary,
            },
        }
