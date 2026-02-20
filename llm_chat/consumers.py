from __future__ import annotations

import threading
import time

from asgiref.sync import async_to_sync
from channels.generic.websocket import JsonWebsocketConsumer
from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404

from .constants import CHAT_DEFAULT_MODEL
from .models import ChatMessage, ChatThread
from .services import ChatService
from .views import _get_user_chat_model


User = get_user_model()

# Heartbeat configuration
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_TIMEOUT = 60  # seconds - if no pong received within this time, close connection


class ChatConsumer(JsonWebsocketConsumer):
    """
    WebSocket consumer for a single chat thread.

    Responsibilities:
    - Auth: ensure the user owns the thread before connecting
    - Group: join a per-thread group (thread_<thread_id>)
    - Receive "user.message" events from the client
    - Delegate streaming to ChatService.stream_reply
    - Forward stream events back to the client
    - Heartbeat: send periodic ping messages and expect pong responses

    NOTE: ChatService will own persistence of ChatMessage instances; this
    consumer stays focused on transport concerns only.
    """

    def connect(self):
        user = self.scope.get("user")
        if not user or not user.is_authenticated:
            self.close(code=4001)  # Unauthorized
            return

        thread_id = self.scope["url_route"]["kwargs"].get("thread_id")
        if not thread_id:
            self.close(code=4003)  # Invalid thread ID
            return

        try:
            # Authorization: verify user owns the thread
            thread = ChatThread.objects.get(id=thread_id, user=user)
        except ChatThread.DoesNotExist:
            # Thread doesn't exist or user doesn't own it
            self.close(code=4004)  # Thread not found or access denied
            return

        self.thread = thread
        self.group_name = f"thread_{thread.id}"
        self.heartbeat_timer = None
        self.last_pong_received = time.time()
        self._heartbeat_active = True

        async_to_sync(self.channel_layer.group_add)(self.group_name, self.channel_name)
        self.accept()
        
        # Check for pending user messages that need streaming
        # This handles the case where POST creates a thread and sends group_send
        # before the WebSocket connects (race condition)
        self._check_and_process_pending_messages()
        
        # Start heartbeat after connection is established
        self._start_heartbeat()

    def disconnect(self, close_code):
        # Stop heartbeat when disconnecting
        self._stop_heartbeat()
        if hasattr(self, "group_name"):
            async_to_sync(self.channel_layer.group_discard)(self.group_name, self.channel_name)

    def _check_and_process_pending_messages(self):
        """
        Check if there are user messages without assistant responses that need streaming.
        This handles the race condition where POST sends group_send before WebSocket connects.
        """
        from .models import ChatMessage
        
        # Check for user messages without a corresponding assistant message
        # (or with a streaming assistant message that hasn't completed)
        user_messages = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            status=ChatMessage.Status.FINAL,
        ).order_by("-created_at")
        
        if not user_messages.exists():
            return
        
        # Get the most recent user message
        latest_user_msg = user_messages.first()
        
        # Check if there's already an assistant message for this user message
        # (created after the user message)
        assistant_exists = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT,
            created_at__gte=latest_user_msg.created_at,
        ).exists()
        
        # If no assistant message exists, trigger streaming
        if not assistant_exists:
            # Use user's stored preferred model (same as normal flow)
            user = self.scope.get("user")
            model = _get_user_chat_model(user) if user else CHAT_DEFAULT_MODEL
            model = model or CHAT_DEFAULT_MODEL
            self.chat_start_stream({
                "content": latest_user_msg.content,
                "user_id": user.id,
                "model": model,
            })

    # ------------------------------------------------------------------
    # Events from HTTP views (Option A flow)
    # ------------------------------------------------------------------
    def chat_start_stream(self, event):
        """
        Handle a streaming request triggered from a regular HTTP view.

        Expected event structure:
        {
            "type": "chat.start_stream",
            "content": "User message text...",
            "user_id": <int>,
        }
        """
        # Authorization: verify the event's user_id matches the connected user
        event_user_id = event.get("user_id")
        connected_user = self.scope.get("user")
        if not connected_user or event_user_id != connected_user.id:
            # Reject unauthorized stream request
            return

        text = (event.get("content") or "").strip()
        if not text:
            return

        model = event.get("model") or CHAT_DEFAULT_MODEL
        service = ChatService()
        
        # Find the user message that was already created (by the view)
        # It should be the most recent user message with this exact content
        user_message_obj = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            content=text,
            status=ChatMessage.Status.FINAL,
        ).order_by("-created_at").first()
        
        # Check if this is the first message in the thread (for title generation)
        # Count all user messages - if there's only 1 (the current one), it's the first
        user_message_count = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.USER
        ).count()
        is_first_message = user_message_count == 1

        for event_type, event_obj in service.stream_reply(
            thread=self.thread,
            user=connected_user,
            user_message=text,
            user_message_obj=user_message_obj,
            model=model,
        ):
            payload = self._serialize_event(event_type, event_obj)
            self.send_json(payload)
            
            # After streaming completes, generate title if this is the first message
            if event_type == "final" and is_first_message:
                # Trigger async title generation (non-blocking) using threading
                # This runs in background without blocking the WebSocket
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Triggering title generation for thread {self.thread.id} (first message)")
                
                def generate_title_background():
                    try:
                        title = service.generate_thread_title(
                            thread=self.thread,
                            user=connected_user,
                            user_message=text,
                        )
                        if title:
                            logger.info(f"Title generated for thread {self.thread.id}: {title}")
                            # Notify all connected clients via group_send
                            from channels.layers import get_channel_layer
                            channel_layer = get_channel_layer()
                            async_to_sync(channel_layer.group_send)(
                                self.group_name,
                                {
                                    "type": "thread.title.updated",
                                    "title": title,
                                },
                            )
                        else:
                            logger.warning(f"Title generation returned None for thread {self.thread.id}")
                    except Exception as e:
                        # Log error but don't fail - title generation is optional
                        logger.warning(f"Failed to generate title for thread {self.thread.id}: {e}", exc_info=True)
                
                # Run in background thread
                thread = threading.Thread(target=generate_title_background, daemon=True)
                thread.start()
            elif event_type == "final":
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Skipping title generation for thread {self.thread.id} (not first message, count={user_message_count})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def thread_title_updated(self, event):
        """
        Handle thread title update event.
        Send the new title to the WebSocket client so frontend can update the sidebar.
        """
        self.send_json({
            "event_type": "thread.title.updated",
            "title": event.get("title", ""),
        })

    def _serialize_event(self, event_type: str, event: object) -> dict:
        """
        Convert raw LLM stream events into a minimal JSON payload suitable
        for the frontend. We intentionally keep this thin for now.
        """
        # Text delta events
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            return {
                "event_type": event_type,
                "delta": delta,
            }

        # Final event from LLMService.call_llm_stream
        if event_type == "final":
            data = event or {}
            call_log = data.get("call_log")
            call_log_id = getattr(call_log, "id", None) if call_log else None
            return {
                "event_type": event_type,
                "call_log_id": str(call_log_id) if call_log_id is not None else None,
            }

        # Generic fallback
        return {
            "event_type": event_type,
        }

    # ------------------------------------------------------------------
    # Heartbeat / Ping-Pong
    # ------------------------------------------------------------------
    def _start_heartbeat(self):
        """Start sending periodic ping messages to detect dead connections."""
        def heartbeat_tick():
            if not self._heartbeat_active:
                return
            
            # Check if we've received a pong recently
            time_since_pong = time.time() - self.last_pong_received
            if time_since_pong > HEARTBEAT_TIMEOUT:
                # No pong received within timeout, close connection
                self.close(code=4005)  # Heartbeat timeout
                return
            
            # Send ping
            try:
                self.send_json({
                    "event_type": "ping",
                    "timestamp": time.time(),
                })
            except Exception:
                # Connection likely closed, stop heartbeat
                self._heartbeat_active = False
                return
            
            # Schedule next heartbeat
            if self._heartbeat_active:
                self.heartbeat_timer = threading.Timer(HEARTBEAT_INTERVAL, heartbeat_tick)
                self.heartbeat_timer.daemon = True
                self.heartbeat_timer.start()
        
        # Start the first heartbeat after the interval
        self.heartbeat_timer = threading.Timer(HEARTBEAT_INTERVAL, heartbeat_tick)
        self.heartbeat_timer.daemon = True
        self.heartbeat_timer.start()

    def _stop_heartbeat(self):
        """Stop the heartbeat timer."""
        if hasattr(self, "_heartbeat_active"):
            self._heartbeat_active = False
        if hasattr(self, "heartbeat_timer") and self.heartbeat_timer:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None

    def receive_json(self, content, **kwargs):
        """
        Handle incoming JSON messages from the client.
        Used for pong responses to heartbeat pings.
        """
        event_type = content.get("event_type")
        
        if event_type == "pong":
            # Update last pong received timestamp
            import time
            self.last_pong_received = time.time()
            # Optionally send a confirmation
            self.send_json({
                "event_type": "pong_ack",
            })

