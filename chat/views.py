import io
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods, require_POST

from chat.models import ChatCanvas, ChatThread
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentTag

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET"])
def chat_home(request):
    """Main chat page. Loads a thread via ?thread=<uuid> if provided."""
    thread = None
    chat_messages = []

    if request.GET.get("thread"):
        thread_id = request.GET["thread"]
        thread = ChatThread.objects.select_related("skill").filter(
            id=thread_id, created_by=request.user
        ).first()
        if thread:
            # Auto-restore archived threads when opened
            if thread.is_archived:
                thread.is_archived = False
                thread.save(update_fields=["is_archived"])
            chat_messages = list(thread.messages.order_by("created_at")[:100])

    threads = list(
        ChatThread.objects.filter(created_by=request.user, is_archived=False)
        .order_by("-updated_at")
    )
    archived_threads = list(
        ChatThread.objects.filter(created_by=request.user, is_archived=True)
        .order_by("-updated_at")
    )

    # User's data rooms for the attach modal
    data_rooms = list(
        DataRoom.objects.filter(created_by=request.user, is_archived=False)
        .order_by("-updated_at")
        .values("pk", "uuid", "name")
    )

    # If thread selected, get its attached data rooms
    thread_data_rooms = []
    thread_skill = None
    if thread:
        thread_data_rooms = list(
            thread.data_rooms.values("pk", "uuid", "name")
        )
        if thread.skill_id and thread.skill and thread.skill.is_active:
            thread_skill = {"id": str(thread.skill_id), "name": thread.skill.name}

    # Resolve preferences (model choices, allowed skills, etc.)
    from core.preferences import get_preferences
    from llm.display import get_display_name, supports_thinking

    prefs = get_preferences(request.user)

    # Skills for the skill modal — use prefs.allowed_skills which respects
    # org admin enable/disable settings, rather than raw get_available_skills()
    skills_json = json.dumps([
        {"id": s["id"], "name": s["name"], "description": s.get("description", "")}
        for s in prefs.allowed_skills
    ])
    model_choices = [
        {
            "id": m,
            "display_name": get_display_name(m),
            "supports_thinking": supports_thinking(m),
        }
        for m in prefs.allowed_models
    ]

    return render(
        request,
        "chat/chat.html",
        {
            "thread": thread,
            "threads": threads,
            "archived_threads": archived_threads,
            "messages": chat_messages,
            "data_rooms": data_rooms,
            "thread_data_rooms": thread_data_rooms,
            "skills_json": skills_json,
            "thread_skill": thread_skill,
            "model_choices_json": json.dumps(model_choices),
            "default_model": prefs.top_model,
            "default_model_display": get_display_name(prefs.top_model),
        },
    )


@login_required
@require_POST
def thread_delete(request, thread_id):
    thread = get_object_or_404(
        ChatThread, id=thread_id, created_by=request.user
    )
    thread.delete()
    return JsonResponse({"ok": True})


@login_required
@require_POST
def thread_archive(request, thread_id):
    thread = get_object_or_404(
        ChatThread, id=thread_id, created_by=request.user
    )
    thread.is_archived = not thread.is_archived
    thread.save(update_fields=["is_archived"])
    return JsonResponse({"ok": True, "is_archived": thread.is_archived})


@login_required
@require_http_methods(["GET"])
async def canvas_export(request, thread_id, canvas_id=None):
    """Export the canvas as a .docx file."""
    import asyncio

    from asgiref.sync import sync_to_async

    thread = await sync_to_async(get_object_or_404)(ChatThread, id=thread_id, created_by=request.user)
    if canvas_id:
        canvas = await sync_to_async(get_object_or_404)(ChatCanvas, pk=canvas_id, thread=thread)
    else:
        canvas = await sync_to_async(get_object_or_404)(ChatCanvas, pk=thread.active_canvas_id, thread=thread)

    import markdown as md
    from html2docx import html2docx

    from chat.services import replace_email_with_html, replace_mermaid_with_images

    # Run the slow mermaid rendering in a thread so the event loop stays
    # free and WebSocket connections are not disrupted.
    content = await asyncio.to_thread(replace_mermaid_with_images, canvas.content)
    content = replace_email_with_html(content)
    html_content = md.markdown(content, extensions=["tables", "fenced_code"])
    full_html = f"<html><body>{html_content}</body></html>"
    buf = html2docx(full_html, title=canvas.title)

    safe_title = "".join(c for c in canvas.title if c.isalnum() or c in " _-").strip() or "document"
    filename = f"{safe_title}.docx"
    return FileResponse(
        io.BytesIO(buf.getvalue()),
        as_attachment=True,
        filename=filename,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@login_required
@require_POST
def canvas_import(request, thread_id, canvas_id=None):
    """Import a .docx file and convert to markdown canvas content."""
    thread = get_object_or_404(ChatThread, id=thread_id, created_by=request.user)

    uploaded = request.FILES.get("file")
    if not uploaded:
        return HttpResponseBadRequest("No file uploaded.")

    from chat.services import import_docx_to_canvas, set_active_canvas

    title, content, truncated = import_docx_to_canvas(uploaded, request.user)

    if canvas_id:
        canvas = get_object_or_404(ChatCanvas, pk=canvas_id, thread=thread)
        canvas.title = title
        canvas.content = content
        canvas.save(update_fields=["title", "content", "updated_at"])
    else:
        from django.db import IntegrityError
        try:
            canvas = ChatCanvas.objects.get(thread=thread, title=title)
            canvas.content = content
            canvas.save(update_fields=["content", "updated_at"])
        except ChatCanvas.DoesNotExist:
            canvas = ChatCanvas.objects.create(
                thread=thread, title=title, content=content,
            )

    from chat.services import create_canvas_checkpoint
    cp = create_canvas_checkpoint(canvas, source="import", description="Imported from .docx")
    canvas.accepted_checkpoint = cp
    canvas.save(update_fields=["accepted_checkpoint"])
    set_active_canvas(thread.pk, canvas)

    generated_title = None
    if not thread.title:
        from chat.services import generate_canvas_title

        generated_title = generate_canvas_title(title, content, request.user)
        if generated_title:
            ChatThread.objects.filter(pk=thread.pk).update(title=generated_title)

    resp = {"title": title, "content": content}
    if generated_title:
        resp["thread_title"] = generated_title
    if truncated:
        resp["truncated"] = True
    return JsonResponse(resp)


@login_required
@require_POST
def thread_create(request):
    """Create a new empty thread and return its ID."""
    thread = ChatThread.objects.create(created_by=request.user)
    return JsonResponse({"thread_id": str(thread.id)})


@login_required
@require_POST
def canvas_save_to_data_room(request, thread_id, canvas_id=None):
    """Save canvas content as a markdown document in a data room."""
    thread = get_object_or_404(ChatThread, id=thread_id, created_by=request.user)
    if canvas_id:
        canvas = get_object_or_404(ChatCanvas, pk=canvas_id, thread=thread)
    else:
        canvas = get_object_or_404(ChatCanvas, pk=thread.active_canvas_id, thread=thread)

    body = json.loads(request.body)
    data_room_id = body.get("data_room_id")
    if not data_room_id:
        return JsonResponse({"error": "data_room_id required"}, status=400)

    data_room = get_object_or_404(DataRoom, pk=data_room_id)

    from documents.views import _user_can_access_data_room

    if not _user_can_access_data_room(request.user, data_room):
        return JsonResponse({"error": "Access denied"}, status=403)

    title = canvas.title or "Untitled document"
    content = canvas.content or ""

    from django.core.files.base import ContentFile

    safe_title = (
        "".join(c for c in title if c.isalnum() or c in " _-").strip() or "document"
    )
    filename = f"{safe_title}.md"
    file_bytes = content.encode("utf-8")
    file_content = ContentFile(file_bytes, name=filename)

    doc = DataRoomDocument.objects.create(
        data_room=data_room,
        uploaded_by=request.user,
        original_file=file_content,
        original_filename=filename,
        mime_type="text/markdown",
        size_bytes=len(file_bytes),
        status=DataRoomDocument.Status.UPLOADED,
    )
    DataRoomDocumentTag.objects.create(
        document=doc, key="source", value="canvas_export"
    )

    try:
        from documents.tasks import process_document_task

        process_document_task.delay(doc.id)
    except ImportError:
        try:
            from documents.services.process_document import process_document

            process_document(doc.id)
        except Exception as exc:
            logger.exception(
                "canvas_save_to_data_room: failed to process document_id=%s (sync fallback)",
                doc.id,
            )
            doc.status = DataRoomDocument.Status.FAILED
            doc.processing_error = str(exc)[:2000]
            doc.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as exc:
        logger.exception(
            "canvas_save_to_data_room: failed to enqueue processing for document_id=%s",
            doc.id,
        )
        doc.status = DataRoomDocument.Status.FAILED
        doc.processing_error = str(exc)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])

    return JsonResponse(
        {
            "ok": True,
            "document_id": doc.id,
            "filename": filename,
            "data_room_name": data_room.name,
        }
    )


@login_required
@require_http_methods(["GET"])
def skills_for_user(request):
    """JSON API returning the user's available skills."""
    from core.preferences import get_preferences

    prefs = get_preferences(request.user)
    return JsonResponse({
        "skills": [
            {"id": s["id"], "name": s["name"], "description": s.get("description", "")}
            for s in prefs.allowed_skills
        ]
    })


@login_required
@require_http_methods(["GET"])
def data_rooms_for_user(request):
    """JSON API returning the user's data rooms for the attach dropdown."""
    rooms = list(
        DataRoom.objects.filter(created_by=request.user, is_archived=False)
        .order_by("-updated_at")
        .values("pk", "uuid", "name")
    )
    # Convert UUIDs to strings for JSON serialization
    for r in rooms:
        r["uuid"] = str(r["uuid"])
    return JsonResponse({"data_rooms": rooms})
