from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from accounts.models import UserSettings
from llm_service.conf import get_allowed_models

from .constants import CHAT_DEFAULT_MODEL, CHAT_KEY_MODELS


def _get_user_chat_model(user) -> str | None:
    """Return the user's stored chat model if allowed, else app default (if allowed), else first allowed."""
    allowed = set(get_allowed_models())
    if not allowed:
        return None
    try:
        stored = (user.settings.chat_model or "").strip()
    except UserSettings.DoesNotExist:
        stored = ""
    if stored and stored in allowed:
        return stored
    if CHAT_DEFAULT_MODEL in allowed:
        return CHAT_DEFAULT_MODEL
    return list(allowed)[0]


@login_required
def chat_view(request, thread_id=None, **kwargs):
    from .models import ChatThread, ChatMessage
    
    # Get all threads for the user, sorted by last_message_at (most recent first)
    threads = list(ChatThread.objects.filter(user=request.user, is_archived=False).order_by("-last_message_at", "-created_at"))
    
    # Determine active thread from URL path parameter
    # thread_id comes from URL pattern, but handle both cases
    thread_id = thread_id or kwargs.get('thread_id')
    active_thread = None
    
    if thread_id:
        try:
            active_thread = ChatThread.objects.get(id=thread_id, user=request.user)
        except ChatThread.DoesNotExist:
            # Requested thread doesn't exist - show empty chat
            active_thread = None

    if request.method == "POST":
        # HTTP POST creates the user message, then triggers
        # the WebSocket consumer via Channels group_send.
        message_text = (request.POST.get("message") or "").strip()
        if message_text:
            # If no active thread, create a new one
            thread_was_created = False
            if not active_thread:
                active_thread = ChatThread.objects.create(
                    user=request.user,
                    title="New chat",
                )
                threads.append(active_thread)
                thread_was_created = True
            
            # Create the user message in the database BEFORE sending group_send
            # This ensures that _check_and_process_pending_messages() can find it
            # even if the WebSocket connects after the group_send event is sent
            from llm_chat.models import ChatMessage
            from django.utils import timezone
            
            user_message = ChatMessage.objects.create(
                thread=active_thread,
                role=ChatMessage.Role.USER,
                status=ChatMessage.Status.FINAL,
                content=message_text,
            )
            
            # Update thread's last_message_at
            active_thread.last_message_at = timezone.now()
            active_thread.save(update_fields=["last_message_at"])
            
            channel_layer = get_channel_layer()
            group_name = f"thread_{active_thread.id}"
            
            # Model: use form value if present and allowed (avoids one-message lag after dropdown change);
            # otherwise use stored preference. Save form value to settings so reload stays in sync.
            allowed = set(get_allowed_models())
            form_model = (request.POST.get("model") or "").strip()
            if form_model and form_model in allowed:
                model = form_model
                settings_obj, _ = UserSettings.objects.get_or_create(user=request.user)
                settings_obj.chat_model = model
                settings_obj.save(update_fields=["chat_model"])
            else:
                model = _get_user_chat_model(request.user)
            # Note: If WebSocket isn't connected yet, the consumer will check for
            # pending messages when it connects (see _check_and_process_pending_messages)
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    "type": "chat.start_stream",
                    "content": message_text,
                    "user_id": request.user.id,
                    "model": model,
                },
            )

        # For AJAX requests, return the new thread ID if one was created
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            if active_thread:
                # Return thread ID so frontend can update URL
                return JsonResponse({"thread_id": str(active_thread.id)}, status=201)
            from django.http import HttpResponse
            return HttpResponse(status=204)

    # Fetch messages for the active thread (if exists)
    # Default: show most recent 50 messages
    MESSAGE_LIMIT = 50
    before_cursor = request.GET.get("before")  # Format: "timestamp,uuid"
    
    if active_thread:
        messages_queryset = ChatMessage.objects.filter(thread=active_thread)
        
        if before_cursor:
            # Parse cursor for pagination
            try:
                cursor_timestamp, cursor_id = before_cursor.split(",", 1)
                cursor_timestamp = timezone.datetime.fromisoformat(cursor_timestamp.replace("Z", "+00:00"))
                cursor_id = cursor_id
                # Get messages before this cursor
                messages_queryset = messages_queryset.filter(
                    created_at__lt=cursor_timestamp
                ) | messages_queryset.filter(
                    created_at=cursor_timestamp,
                    id__lt=cursor_id
                )
            except (ValueError, AttributeError):
                # Invalid cursor, just use default
                pass
        
        # Order by created_at, id (ascending for pagination, then reverse for display)
        messages_queryset = messages_queryset.order_by("-created_at", "-id")[:MESSAGE_LIMIT]
        messages = list(messages_queryset)
        messages.reverse()  # Reverse to show oldest first
        
        # Check if there are older messages
        if messages:
            oldest_message = messages[0]
            has_older = ChatMessage.objects.filter(
                thread=active_thread,
                created_at__lt=oldest_message.created_at
            ).exists() or ChatMessage.objects.filter(
                thread=active_thread,
                created_at=oldest_message.created_at,
                id__lt=oldest_message.id
            ).exists()
            # Create cursor for "load more" link: "timestamp,uuid"
            older_cursor = f"{oldest_message.created_at.isoformat()},{oldest_message.id}"
        else:
            has_older = False
            older_cursor = None
        
        # Check if there's a streaming message
        has_streaming = ChatMessage.objects.filter(
            thread=active_thread,
            status=ChatMessage.Status.STREAMING
        ).exists()
    else:
        # No active thread - empty chat
        messages = []
        has_older = False
        older_cursor = None
        has_streaming = False
    
    allowed = set(get_allowed_models())
    chat_key_models = [{"value": m[0], "label": m[1]} for m in CHAT_KEY_MODELS if m[0] in allowed]
    chat_default_model = _get_user_chat_model(request.user) or (
        chat_key_models[0]["value"] if chat_key_models else None
    )
    context = {
        "threads": threads,
        "active_thread": active_thread,
        "messages": messages,
        "has_older": has_older,
        "older_cursor": older_cursor,
        "has_streaming": has_streaming,
        "chat_key_models": chat_key_models,
        "chat_default_model": chat_default_model,
    }
    return render(request, "llm_chat/chat.html", context)


@login_required
def chat_messages_json(request, thread_id):
    """
    AJAX endpoint to fetch messages for a thread as JSON.
    Used for loading messages when switching threads without page reload.
    """
    from .models import ChatThread, ChatMessage
    
    # Get thread and verify ownership
    thread = get_object_or_404(ChatThread, id=thread_id, user=request.user)
    
    # Fetch messages (same logic as chat_view)
    MESSAGE_LIMIT = 50
    before_cursor = request.GET.get("before")
    
    messages_queryset = ChatMessage.objects.filter(thread=thread)
    
    if before_cursor:
        try:
            cursor_timestamp, cursor_id = before_cursor.split(",", 1)
            cursor_timestamp = timezone.datetime.fromisoformat(cursor_timestamp.replace("Z", "+00:00"))
            messages_queryset = messages_queryset.filter(
                created_at__lt=cursor_timestamp
            ) | messages_queryset.filter(
                created_at=cursor_timestamp,
                id__lt=cursor_id
            )
        except (ValueError, AttributeError):
            pass
    
    messages_queryset = messages_queryset.order_by("-created_at", "-id")[:MESSAGE_LIMIT]
    messages = list(messages_queryset)
    messages.reverse()
    
    # Check if there are older messages
    if messages:
        oldest_message = messages[0]
        has_older = ChatMessage.objects.filter(
            thread=thread,
            created_at__lt=oldest_message.created_at
        ).exists() or ChatMessage.objects.filter(
            thread=thread,
            created_at=oldest_message.created_at,
            id__lt=oldest_message.id
        ).exists()
        older_cursor = f"{oldest_message.created_at.isoformat()},{oldest_message.id}"
    else:
        has_older = False
        older_cursor = None
    
    # Serialize messages
    messages_data = []
    for msg in messages:
        messages_data.append({
            "id": str(msg.id),
            "role": msg.role,
            "status": msg.status,
            "content": msg.content,
            "error": msg.error,
            "created_at": msg.created_at.isoformat(),
        })
    
    return JsonResponse({
        "messages": messages_data,
        "has_older": has_older,
        "older_cursor": older_cursor,
        "thread_id": str(thread.id),
        "thread_title": thread.title or "Untitled chat",
    })


def _get_preferred_model_response(user):
    """Return JSON-serialisable dict with current preferred model for user."""
    model = _get_user_chat_model(user)
    return {"model": model or ""}


@login_required
def chat_preferred_model_update(request):
    """GET: return current preferred model. POST: save preferred model (body or form: model=<id>)."""
    if request.method == "GET":
        return JsonResponse(_get_preferred_model_response(request.user))
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    model = (request.POST.get("model") or request.body.decode("utf-8").strip() or "").strip()
    if not model:
        return JsonResponse({"error": "Missing model"}, status=400)
    allowed = set(get_allowed_models())
    if model not in allowed:
        return JsonResponse({"error": "Model not allowed"}, status=400)
    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    settings.chat_model = model
    settings.save(update_fields=["chat_model"])
    return JsonResponse({"model": settings.chat_model})
