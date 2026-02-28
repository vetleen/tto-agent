"""WebSocket consumer for project chat with LLM streaming."""

from __future__ import annotations

import json
import logging

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


class ProjectChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for per-project chat with LLM streaming and RAG."""

    async def connect(self):
        self.project_id = str(self.scope["url_route"]["kwargs"]["project_id"])
        self.project = None
        self.user = self.scope.get("user")

        # Reject unauthenticated users
        if not self.user or self.user.is_anonymous:
            await self.close(code=4401)
            return

        # Validate project exists and user owns it
        self.project = await self._get_project()
        if self.project is None:
            await self.close(code=4404)
            return

        await self.accept()

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
        elif msg_type == "pong":
            pass  # heartbeat acknowledgment

    async def _handle_chat_message(self, data):
        content = (data.get("content") or "").strip()
        if not content:
            await self.send(text_data=json.dumps({"error": "Empty message"}))
            return

        thread_id = data.get("thread_id")

        try:
            # Get or create thread
            thread, created = await self._get_or_create_thread(thread_id)

            if created:
                await self.send(text_data=json.dumps({
                    "event_type": "thread.created",
                    "thread_id": str(thread.id),
                }))

            # Persist user message
            await self._create_message(thread, "user", content)

            # Load conversation history
            history = await self._load_history(thread)

            # Build system prompt
            system_prompt = (
                "You are a helpful assistant for the project "
                f'"{self.project.name}". '
                "You have access to a search_documents tool that can search "
                "the project's uploaded documents. Use it when the user asks "
                "about document contents or needs specific information from their files. "
                "Answer concisely and accurately."
            )

            # Stream LLM response
            await self._stream_response(thread, system_prompt, history)

        except Exception:
            logger.exception("Error handling chat message")
            await self.send(text_data=json.dumps({
                "event_type": "error",
                "data": {"message": "An error occurred processing your message."},
            }))

    async def _stream_response(self, thread, system_prompt, history):
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
            conversation_id=self.project.pk,
        )

        request = ChatRequest(
            messages=messages,
            stream=True,
            tools=["search_documents"],
            context=context,
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

    # -- Database helpers --

    @database_sync_to_async
    def _get_project(self):
        from documents.models import Project

        try:
            project = Project.objects.get(uuid=self.project_id)
        except Project.DoesNotExist:
            return None
        if project.created_by_id != self.user.pk:
            return None
        return project

    @database_sync_to_async
    def _get_or_create_thread(self, thread_id):
        from chat.models import ChatThread

        if thread_id:
            try:
                thread = ChatThread.objects.get(
                    id=thread_id,
                    project=self.project,
                    created_by=self.user,
                )
                return thread, False
            except ChatThread.DoesNotExist:
                pass

        thread = ChatThread.objects.create(
            project=self.project,
            created_by=self.user,
        )
        return thread, True

    @database_sync_to_async
    def _create_message(self, thread, role, content, tool_call_id=None):
        from chat.models import ChatMessage

        return ChatMessage.objects.create(
            thread=thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
        )

    @database_sync_to_async
    def _load_history(self, thread, limit=50):
        from chat.models import ChatMessage

        messages = list(
            ChatMessage.objects.filter(thread=thread)
            .order_by("-created_at")[:limit]
        )
        messages.reverse()
        return [
            {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            }
            for m in messages
        ]
