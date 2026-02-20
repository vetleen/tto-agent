"""
System instructions for LLM chat interactions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ChatThread

from .models import ChatMessage


def assemble_chat_history(
    thread: "ChatThread", 
    target_tokens: int = 20000,
    exclude_message: ChatMessage | None = None
) -> str:
    """
    Assemble chat history from a thread, including messages up to the target token limit.
    
    Messages are included from most recent backwards until the token limit is reached.
    If adding a message would exceed the limit, it's excluded to stay under the threshold.
    
    Uses the stored token_count field on ChatMessage for efficiency.
    
    Args:
        thread: The ChatThread to get messages from
        target_tokens: Maximum number of tokens to include (default: 20000)
        exclude_message: Optional ChatMessage to exclude from history (e.g., current user message)
    
    Returns:
        Formatted string with chat history in the format:
        User - date:
        [message]
        Assistant - date:
        [message]
        ...
    """
    # Get all messages ordered by creation time (oldest first)
    messages_queryset = ChatMessage.objects.filter(thread=thread)
    
    # Exclude the specified message if provided
    if exclude_message:
        messages_queryset = messages_queryset.exclude(id=exclude_message.id)
    
    messages = list(
        messages_queryset
        .order_by("created_at", "id")
        .select_related("thread")
    )
    
    if not messages:
        return ""
    
    # Estimate tokens for formatting overhead (role name + date + newlines)
    # Rough estimate: ~10 tokens per message for formatting
    FORMATTING_TOKENS_PER_MESSAGE = 10
    
    # Build history from most recent backwards, tracking token count
    total_tokens = 0
    included_messages = []
    
    # Start from the most recent message and work backwards
    for msg in reversed(messages):
        # Use stored token_count for message content
        # Add formatting overhead estimate
        msg_tokens = msg.token_count + FORMATTING_TOKENS_PER_MESSAGE
        
        # Check if adding this message would exceed the limit
        if total_tokens + msg_tokens > target_tokens:
            # Stop here - don't include this message to stay under limit
            break
        
        # Add this message
        included_messages.append(msg)
        total_tokens += msg_tokens
    
    # Reverse to get chronological order (oldest first)
    included_messages.reverse()
    
    # Format the final history string
    history_lines = []
    for msg in included_messages:
        role_name = "User" if msg.role == ChatMessage.Role.USER else "Assistant"
        date_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        history_lines.append(f"{role_name} - {date_str}:")
        history_lines.append(msg.content)
        history_lines.append("")  # Empty line between messages
    
    return "\n".join(history_lines)


def assemble_system_instruction(
    thread: "ChatThread",
    exclude_message: ChatMessage | None = None,
    target_tokens: int = 20000,
    **kwargs
) -> str:
    """
    Assemble the complete system instruction for LLM chat interactions.
    
    Args:
        thread: The ChatThread to get chat history from
        exclude_message: Optional ChatMessage to exclude from history (e.g., current user message)
        target_tokens: Maximum number of tokens for chat history (default: 20000)
        **kwargs: Additional variables for the system instruction template
    
    Returns:
        Complete system instruction string
    """
    # Assemble chat history
    chat_history = assemble_chat_history(thread, target_tokens=target_tokens, exclude_message=exclude_message)
    
    # Base system instruction template
    system_instruction = """## Role
You are a helpful AI assistant engaged in a conversation with a user.

## Chat History

{chat_history}

## Instructions

Continue the conversation naturally, responding to the user's most recent message. Be helpful, accurate, and concise. If the chat history is empty, this is the start of a new conversation.

## Response format

You must respond with valid JSON only. Use a single key "message" whose value is your reply text (you may use Markdown inside that string).

## Formatting

You should use Markdown formatting inside your 'message'to improve readability. Use appropriate formatting like:
- **Bold** for emphasis
- *Italic* for subtle emphasis
- `Code blocks` for code snippets or technical terms
- Lists (bulleted or numbered) when presenting multiple items
- Headers when organizing longer responses
- Links when referencing external resources

Format your responses to be clear and well-structured, making them easier to read and understand."""

    # Format the system instruction with provided variables
    formatted_instruction = system_instruction.format(
        chat_history=chat_history if chat_history else "(No previous messages)",
        **kwargs
    )
    
    return formatted_instruction
