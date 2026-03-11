import io
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods, require_POST

from chat.models import ChatCanvas, ChatThread
from documents.models import DataRoom

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET"])
def chat_home(request):
    """Main chat page. Loads a thread via ?thread=<uuid> if provided."""
    thread = None
    chat_messages = []

    if request.GET.get("thread"):
        thread_id = request.GET["thread"]
        thread = ChatThread.objects.filter(
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
    if thread:
        thread_data_rooms = list(
            thread.data_rooms.values("pk", "uuid", "name")
        )

    # Model choices for the model selector dropdown
    from core.preferences import get_preferences
    from llm.display import get_display_name, supports_thinking

    prefs = get_preferences(request.user)
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
            "model_choices_json": json.dumps(model_choices),
            "default_model": prefs.primary_model,
            "default_model_display": get_display_name(prefs.primary_model),
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
def canvas_export(request, thread_id):
    """Export the canvas as a .docx file."""
    thread = get_object_or_404(ChatThread, id=thread_id, created_by=request.user)
    canvas = get_object_or_404(ChatCanvas, thread=thread)

    import markdown as md
    from html2docx import html2docx

    html_content = md.markdown(canvas.content, extensions=["tables", "fenced_code"])
    # html2docx expects a full HTML document or just body content
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
def canvas_import(request, thread_id):
    """Import a .docx file and convert to markdown canvas content."""
    thread = get_object_or_404(ChatThread, id=thread_id, created_by=request.user)

    uploaded = request.FILES.get("file")
    if not uploaded:
        return HttpResponseBadRequest("No file uploaded.")

    from chat.services import import_docx_to_canvas

    title, content, truncated = import_docx_to_canvas(uploaded, request.user)

    canvas, _ = ChatCanvas.objects.update_or_create(
        thread=thread,
        defaults={"title": title, "content": content},
    )

    from chat.services import create_canvas_checkpoint
    cp = create_canvas_checkpoint(canvas, source="import", description="Imported from .docx")
    canvas.accepted_checkpoint = cp
    canvas.save(update_fields=["accepted_checkpoint"])

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
