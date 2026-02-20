from __future__ import annotations

import json
import logging
import re
from typing import Generator, Tuple
from django.utils import timezone

from django.contrib.auth import get_user_model

from llm_service.conf import get_default_model
from llm_service.services import LLMService
from .models import ChatMessage, ChatThread
from .system_instructions import assemble_system_instruction

logger = logging.getLogger(__name__)


User = get_user_model()


class ChatService:
    """
    High-level chat orchestration service.

    Handles streaming LLM responses and persists ChatMessage instances to the database.
    """

    def __init__(self) -> None:
        self.llm_service = LLMService()

        # Minimal JSON schema for a single "message" field. This keeps
        # compatibility with the existing LLMService API, which expects a
        # structured schema, while we still treat the response as plain text.
        self._json_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
        self._schema_name = "chat_message"

        # JSON schema for title generation
        self._title_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short, descriptive title (max 50 characters) for this chat conversation",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        }
        self._title_schema_name = "chat_title"

    def generate_thread_title(self, *, thread: ChatThread, user: User, user_message: str) -> str | None:
        """
        Generate a title for a chat thread based on the first user message.
        
        Uses gpt-5-nano to create a short, descriptive title.
        Returns None if generation fails (thread keeps default title).
        """
        if thread.user_id != user.id:
            raise PermissionError("User does not own this thread.")
        
        # Only generate title if thread still has default title
        if thread.title and thread.title != "New chat":
            return None
        
        try:
            call_log = self.llm_service.call_llm(
                model="openai/gpt-5-nano",
                reasoning_effort="low",
                system_instructions="Respond with valid JSON only, using a single key 'title'. Generate a short, descriptive title (max 30 characters) for this chat conversation based on the user's first message. The title should be concise and capture the main topic or intent.",
                user_prompt=user_message,
                tools=None,
                json_schema=self._title_schema,
                schema_name=self._title_schema_name,
                user=user,
            )
            
            if call_log and call_log.succeeded and call_log.parsed_json:
                title = call_log.parsed_json.get("title", "").strip()
                if title:
                    # Limit to 255 chars (model field max_length)
                    title = title[:255]
                    thread.title = title
                    thread.save(update_fields=["title"])
                    return title
        except Exception as e:
            # Log error but don't fail - title generation is optional
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to generate thread title for thread {thread.id}: {e}")
        
        return None

    def stream_reply(
        self,
        *,
        thread: ChatThread,
        user: User,
        user_message: str,
        user_message_obj: ChatMessage | None = None,
        model: str | None = None,
    ) -> Generator[Tuple[str, object], None, None]:
        """
        Stream a reply for a given thread and user message.

        Creates and persists ChatMessage instances:
        - User message is created immediately (or uses provided user_message_obj)
        - Assistant message is created with "streaming" status
        - Assistant message content is updated as deltas arrive
        - Assistant message is marked "final" or "error" when done
        - LLMCallLog is attached to the assistant message
        
        Args:
            thread: The chat thread
            user: The user making the request
            user_message: The message text (for LLM call)
            user_message_obj: Optional pre-created ChatMessage instance (if created in view)
            model: Optional model id (e.g. moonshot/kimi-k2.5). If None, uses LLM default.
        """
        if thread.user_id != user.id:
            raise PermissionError("User does not own this thread.")

        # Use provided user message or create a new one
        if user_message_obj:
            user_msg = user_message_obj
        else:
            # Create user message
            user_msg = ChatMessage.objects.create(
                thread=thread,
                role=ChatMessage.Role.USER,
                status=ChatMessage.Status.FINAL,
                content=user_message,
            )
            
            # Update thread's last_message_at
            thread.last_message_at = timezone.now()
            thread.save(update_fields=["last_message_at"])

        # Create assistant message with streaming status
        assistant_msg = ChatMessage.objects.create(
            thread=thread,
            role=ChatMessage.Role.ASSISTANT,
            status=ChatMessage.Status.STREAMING,
            content="",
        )

        accumulated_json = ""  # Raw JSON accumulation
        accumulated_message = ""  # Extracted message content
        call_log = None

        # Assemble the complete system instruction (includes chat history)
        system_instructions = assemble_system_instruction(
            thread=thread,
            exclude_message=user_msg,
            target_tokens=20000
        )

        def extract_message_from_json(json_str: str) -> str:
            """
            Extract the message value from JSON, handling incomplete JSON during streaming.
            Returns the extracted message content, or empty string if extraction fails.
            """
            if not json_str.strip().startswith("{"):
                return ""
            
            # Try to parse complete JSON first
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "message" in parsed:
                    return str(parsed["message"])
            except (json.JSONDecodeError, ValueError):
                pass
            
            # JSON is incomplete, extract message value incrementally
            # Find position after "message": "
            message_key_pos = json_str.find('"message"')
            if message_key_pos == -1:
                return ""
            
            after_key = json_str[message_key_pos + 9:]  # 9 = len('"message"')
            colon_pos = after_key.find(':')
            if colon_pos == -1:
                return ""
            
            after_colon = after_key[colon_pos + 1:].strip()
            if not after_colon.startswith('"'):
                return ""
            
            # Extract value between quotes, handling escapes
            # Start after the opening quote
            extracted = ""
            i = 1  # Skip opening quote
            in_escape = False
            
            while i < len(after_colon):
                char = after_colon[i]
                
                if in_escape:
                    # Handle escape sequences
                    if char == 'n':
                        extracted += '\n'
                    elif char == '"':
                        extracted += '"'
                    elif char == '\\':
                        extracted += '\\'
                    elif char == 't':
                        extracted += '\t'
                    elif char == 'r':
                        extracted += '\r'
                    elif char == 'u' and i + 4 < len(after_colon):
                        # Handle \uXXXX unicode escapes
                        hex_str = after_colon[i + 1:i + 5]
                        try:
                            extracted += chr(int(hex_str, 16))
                            i += 4  # Skip the 4 hex digits
                        except ValueError:
                            extracted += char
                    else:
                        extracted += char
                    in_escape = False
                elif char == '\\':
                    in_escape = True
                elif char == '"':
                    # Found closing quote - end of value (complete JSON)
                    break
                else:
                    # Regular character - add to extracted content
                    # This handles incomplete JSON where we haven't hit the closing quote yet
                    extracted += char
                
                i += 1
            
            # Return extracted content (even if incomplete, it's better than showing JSON)
            return extracted

        try:
            # Stream events from LLMService
            resolved_model = model or get_default_model()
            for event_type, event in self.llm_service.call_llm_stream(
                model=resolved_model,
                reasoning_effort="low",
                system_instructions=system_instructions,
                user_prompt=user_message,
                tools=None,
                json_schema=self._json_schema,
                schema_name=self._schema_name,
                user=user,
            ):
                # Handle text delta events
                if event_type == "response.output_text.delta":
                    raw_delta = getattr(event, "delta", "")
                    
                    # Accumulate raw content (may be JSON or plain text)
                    accumulated_json += raw_delta
                    
                    # Check if content looks like JSON (starts with {)
                    trimmed_content = accumulated_json.strip()
                    is_json_like = trimmed_content.startswith("{")
                    
                    if is_json_like:
                        # Try to extract message content from accumulated JSON
                        new_message_content = extract_message_from_json(accumulated_json)
                        # If extraction returns empty and we haven't extracted anything yet,
                        # don't yield - wait for more content
                        if not new_message_content and not accumulated_message:
                            continue
                    else:
                        # Plain text - use directly
                        new_message_content = accumulated_json
                    
                    # Calculate the delta of the message content
                    message_delta = new_message_content[len(accumulated_message):]
                    
                    # Update accumulated message content
                    accumulated_message = new_message_content
                    
                    # Update assistant message content in DB (store extracted message, not JSON)
                    assistant_msg.content = accumulated_message
                    assistant_msg.save(update_fields=["content"])
                    
                    # Create a new event object with the extracted message delta
                    # The consumer uses getattr(event, "delta", ""), so we just need a delta attribute
                    class MessageDeltaEvent:
                        def __init__(self, message_delta):
                            self.delta = message_delta
                    
                    modified_event = MessageDeltaEvent(message_delta)
                    
                    # Only yield if we have a meaningful delta
                    if message_delta:
                        yield event_type, modified_event

                # Handle final event
                elif event_type == "final":
                    data = event or {}
                    call_log = data.get("call_log")
                    response = data.get("response", {})
                    
                    # Extract the message from parsed JSON if available (final fallback)
                    if call_log and hasattr(call_log, "parsed_json") and call_log.parsed_json:
                        parsed_message = call_log.parsed_json.get("message", "")
                        if parsed_message:
                            accumulated_message = parsed_message
                    
                    # Ensure we have the final message content
                    if not accumulated_message and accumulated_json:
                        # Try to extract from JSON, or use as plain text
                        trimmed_content = accumulated_json.strip()
                        if trimmed_content.startswith("{") and '"message"' in accumulated_json:
                            accumulated_message = extract_message_from_json(accumulated_json)
                        else:
                            # Plain text - use directly
                            accumulated_message = accumulated_json
                    
                    # Update assistant message: mark as final, set content, attach call_log
                    assistant_msg.status = ChatMessage.Status.FINAL
                    assistant_msg.content = accumulated_message
                    if call_log:
                        assistant_msg.llm_call_log = call_log
                    assistant_msg.save(update_fields=["status", "content", "llm_call_log"])

                    # Update thread's last_message_at
                    thread.last_message_at = timezone.now()
                    thread.save(update_fields=["last_message_at"])

                    # Yield the final event
                    yield event_type, event

                # Yield other events as-is
                else:
                    yield event_type, event

        except Exception as e:
            # Mark assistant message as error
            assistant_msg.status = ChatMessage.Status.ERROR
            assistant_msg.error = str(e)
            # Ensure content is always a string
            assistant_msg.content = str(accumulated_message) if accumulated_message else ""
            assistant_msg.save(update_fields=["status", "error", "content"])
            raise

