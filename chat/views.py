import io
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods, require_POST

from chat.models import ChatAttachment, ChatCanvas, ChatThread, ThreadTask
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentTag

logger = logging.getLogger(__name__)


def _get_accessible_data_rooms(user):
    """Return data rooms the user owns, each annotated with its active
    (non-archived) document count for display in the attach modal."""
    from django.db.models import Count, Q

    return list(
        DataRoom.objects.filter(created_by=user, is_archived=False)
        .annotate(
            document_count=Count("documents", filter=Q(documents__is_archived=False))
        )
        .order_by("-updated_at")
        .values("pk", "uuid", "name", "document_count")
    )


def _group_threads_by_time(threads):
    """Bucket threads (already ordered by ``-updated_at``) into Today / Yesterday /
    Earlier by their ``updated_at`` in the active local timezone.

    Returns a list of ``{"label", "threads"}`` dicts, omitting empty groups. The
    sidebar always presents the Today group first; JS recreates it when a new or
    just-touched thread needs to land there.
    """
    from datetime import timedelta

    from django.utils import timezone

    today = timezone.localdate()
    yesterday = today - timedelta(days=1)
    today_items, yesterday_items, earlier_items = [], [], []
    for t in threads:
        dt = t.updated_at
        if timezone.is_aware(dt):
            dt = timezone.localtime(dt)
        d = dt.date()
        if d >= today:
            today_items.append(t)
        elif d == yesterday:
            yesterday_items.append(t)
        else:
            earlier_items.append(t)
    groups = []
    if today_items:
        groups.append({"label": "Today", "threads": today_items})
    if yesterday_items:
        groups.append({"label": "Yesterday", "threads": yesterday_items})
    if earlier_items:
        groups.append({"label": "Earlier", "threads": earlier_items})
    return groups


@login_required
@require_http_methods(["GET"])
def chat_home(request):
    """Main chat page. Loads a thread via ?thread=<uuid> if provided."""
    thread = None
    chat_messages = []
    thread_loop_id = None

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
            # If this thread is a loop, mark it seen (clears the unread badge) and
            # expose its id so /loop in this thread opens the edit modal.
            from django.utils import timezone

            from chat.models import Loop

            loop = Loop.objects.filter(thread=thread).first()
            if loop:
                loop.last_seen_at = timezone.now()
                loop.save(update_fields=["last_seen_at"])
                thread_loop_id = str(loop.id)
            # Visible messages, plus hidden assistant tool-loop messages that
            # carry narration or thinking — those render as collapsed
            # "Thought further" blocks (matching the live-streaming UI).
            from django.db.models import Q

            chat_messages = list(
                thread.messages.filter(
                    Q(is_hidden_from_user=False)
                    | Q(is_hidden_from_user=True, role="assistant", is_redacted=False)
                ).order_by("created_at")[:100]
            )
            filtered = []
            for m in chat_messages:
                if m.is_hidden_from_user:
                    has_narration = bool(m.content.strip()) or bool(
                        (m.metadata or {}).get("thinking")
                    )
                    if not has_narration:
                        continue
                    m.is_intermediate = True
                filtered.append(m)
            chat_messages = filtered

            # Annotate user messages with attachment filenames for rendering
            msg_ids = [m.pk for m in chat_messages if m.role == "user"]
            if msg_ids:
                from collections import defaultdict
                att_qs = ChatAttachment.objects.filter(
                    message_id__in=msg_ids
                ).values_list("message_id", "original_filename", "content_type")
                att_map = defaultdict(list)
                for mid, fname, ctype in att_qs:
                    att_map[mid].append((fname, ctype))
                for m in chat_messages:
                    m.attachment_names = att_map.get(m.pk, [])

    all_threads = ChatThread.objects.filter(created_by=request.user).order_by("-updated_at")
    threads = [t for t in all_threads if not t.is_archived]
    archived_threads = [t for t in all_threads if t.is_archived]

    # Annotate threads that are loops so the sidebar can show a loop icon + an
    # unread dot when a scheduled run produced a result the user hasn't opened.
    from chat.models import Loop

    loop_map = {
        loop.thread_id: loop
        for loop in Loop.objects.filter(thread__in=threads)
    }
    for t in threads:
        loop = loop_map.get(t.id)
        t.is_loop = loop is not None
        t.loop_unread = loop.is_unread if loop else False
        t.loop_id = str(loop.id) if loop else ""
        t.loop_paused = (loop.status == Loop.Status.PAUSED) if loop else False

    thread_groups = _group_threads_by_time(threads)

    # User's data rooms (owned + shared) for the attach modal
    data_rooms = _get_accessible_data_rooms(request.user)

    # If thread selected, get its tasks
    thread_tasks_json = "[]"
    if thread:
        thread_tasks = list(
            ThreadTask.objects.filter(thread=thread)
            .order_by("order", "created_at")
            .values("id", "title", "status")
        )
        if thread_tasks:
            # Convert UUIDs to strings for JSON serialization
            for t in thread_tasks:
                t["id"] = str(t["id"])
            thread_tasks_json = json.dumps(thread_tasks)

    # Compute thread cost
    thread_cost_usd = 0.0
    if thread:
        from django.db.models import Sum

        from llm.models import LLMCallLog

        result = LLMCallLog.objects.filter(
            conversation_id=str(thread.id),
        ).aggregate(total=Sum("cost_usd"))
        thread_cost_usd = float(result["total"]) if result["total"] is not None else 0.0

    # Active sub-agents for status bar
    active_subagent_count = 0
    if thread:
        from chat.models import SubAgentRun

        active_subagent_count = SubAgentRun.objects.filter(
            thread_id=thread.id,
            status__in=[SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING],
        ).count()

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
    from django.conf import settings as django_settings

    from core.preferences import get_preferences
    from llm.display import get_display_name, get_thinking_levels, supports_thinking, supports_vision

    prefs = get_preferences(request.user)

    # Skills for the skill modal — use prefs.allowed_skills which respects
    # org admin enable/disable settings, rather than raw get_available_skills()
    skills_json = json.dumps([
        {
            "id": s["id"],
            "name": s["name"],
            "emoji": s.get("emoji", ""),
            "description": s.get("description", ""),
        }
        for s in prefs.allowed_skills
    ])
    model_choices = [
        {
            "id": m,
            "display_name": get_display_name(m),
            "supports_thinking": supports_thinking(m),
            "supports_vision": supports_vision(m),
            "thinking_levels": get_thinking_levels(m),
        }
        for m in prefs.allowed_models
    ]

    pending_initial_turn = bool(
        thread and (thread.metadata or {}).get("pending_initial_turn")
    )

    # Model picker default: the loaded thread's effective model (honoring its
    # stored choice, with tier fallback), or the user's preferred chat model for
    # a new thread.
    from core.preferences import resolve_thread_model

    preferred_chat_model = prefs.feature_models.get("chat", prefs.top_model)
    effective_model = (
        resolve_thread_model(thread.model, prefs) if thread else preferred_chat_model
    )

    return render(
        request,
        "chat/chat.html",
        {
            "thread": thread,
            "thread_loop_id": thread_loop_id,
            "threads": threads,
            "thread_groups": thread_groups,
            "archived_threads": archived_threads,
            "messages": chat_messages,
            "data_rooms": data_rooms,
            "thread_data_rooms": thread_data_rooms,
            "skills_json": skills_json,
            "thread_skill": thread_skill,
            "model_choices_json": json.dumps(model_choices),
            "default_model": effective_model,
            "default_model_display": get_display_name(effective_model),
            "preferred_chat_model": preferred_chat_model,
            "preferred_chat_model_display": get_display_name(preferred_chat_model),
            "thread_tasks_json": thread_tasks_json,
            "thread_cost_usd": thread_cost_usd,
            "pending_initial_turn": pending_initial_turn,
            "allow_agent_attach_skills": prefs.allow_agent_attach_skills,
            "active_subagent_count": active_subagent_count,
            "assistant_name": django_settings.ASSISTANT_NAME,
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
@require_POST
def thread_rename(request, thread_id):
    thread = get_object_or_404(
        ChatThread, id=thread_id, created_by=request.user
    )
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    title = body.get("title", "").strip()[:255]
    if not title:
        return JsonResponse({"error": "Please enter a title."}, status=400)
    thread.title = title
    thread.save(update_fields=["title"])
    return JsonResponse({"ok": True, "title": thread.title})


@login_required
@require_POST
def thread_emoji(request, thread_id):
    thread = get_object_or_404(
        ChatThread, id=thread_id, created_by=request.user
    )
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    emoji = body.get("emoji", "")[:8]
    thread.emoji = emoji
    thread.save(update_fields=["emoji"])
    return JsonResponse({"ok": True, "emoji": thread.emoji})


@login_required
@require_http_methods(["POST"])
async def canvas_export(request, thread_id, canvas_id=None):
    """Export the canvas as a .docx file.

    The client sends JSON ``{"content": "..."}`` with mermaid blocks already
    rendered as ``<img>`` tags (rendered in the browser), so the server never
    runs a headless browser over untrusted canvas content.
    """
    from asgiref.sync import sync_to_async

    thread = await sync_to_async(get_object_or_404)(ChatThread, id=thread_id, created_by=request.user)
    if canvas_id:
        canvas = await sync_to_async(get_object_or_404)(ChatCanvas, pk=canvas_id, thread=thread)
    else:
        canvas = await sync_to_async(get_object_or_404)(ChatCanvas, pk=thread.active_canvas_id, thread=thread)

    import markdown as md
    from html2docx import html2docx

    from chat.services import replace_email_with_html

    # Client already replaced mermaid blocks with <img> tags.
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return HttpResponseBadRequest("Invalid JSON")
    content = body.get("content", canvas.content)

    content = replace_email_with_html(content)
    html_content = md.markdown(content, extensions=["tables", "fenced_code"])
    full_html = f"<html><body>{html_content}</body></html>"
    buf = html2docx(full_html, title=canvas.title)

    # html2docx sets gridCol widths to span the full page but leaves
    # cell widths at 0 and table width at auto. With autofit Word
    # auto-sizes from content, making tables narrower than the page.
    # Fix: set table width to 100%, copy gridCol widths into cells,
    # and switch to fixed layout so Word honours the widths exactly.
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn

    doc = DocxDocument(buf)
    for table in doc.tables:
        table.autofit = False
        tblW = table._tbl.tblPr.find(qn("w:tblW"))
        if tblW is not None:
            tblW.set(qn("w:type"), "pct")
            tblW.set(qn("w:w"), "5000")  # 5000 = 100%
        grid_cols = table._tbl.tblGrid.findall(qn("w:gridCol"))
        col_widths = [int(gc.get(qn("w:w"))) for gc in grid_cols]
        for row in table.rows:
            for i, cell in enumerate(row.cells):
                if i < len(col_widths):
                    tcW = cell._tc.tcPr.find(qn("w:tcW"))
                    if tcW is not None:
                        tcW.set(qn("w:w"), str(col_widths[i]))
                        tcW.set(qn("w:type"), "dxa")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe_title = "".join(c for c in canvas.title if c.isalnum() or c in " _-").strip() or "document"
    filename = f"{safe_title}.docx"
    return FileResponse(
        buf,
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

    from chat.services import MAX_ATTACHMENT_SIZE, SUPPORTED_DOCX_TYPES, import_docx_to_canvas, set_active_canvas

    # Validate file type
    ct = uploaded.content_type or ""
    is_docx = ct in SUPPORTED_DOCX_TYPES or (uploaded.name and uploaded.name.lower().endswith(".docx"))
    if not is_docx:
        return JsonResponse({"error": "Only .docx files are supported for import."}, status=400)

    # Validate file size
    if uploaded.size > MAX_ATTACHMENT_SIZE:
        return JsonResponse(
            {"error": f"That file is too large (max {MAX_ATTACHMENT_SIZE // (1024 * 1024)} MB)."},
            status=400,
        )

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
            try:
                canvas = ChatCanvas.objects.create(
                    thread=thread, title=title, content=content,
                )
            except IntegrityError:
                canvas = ChatCanvas.objects.get(thread=thread, title=title)
                canvas.content = content
                canvas.save(update_fields=["content", "updated_at"])

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

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    data_room_id = body.get("data_room_id")
    if not data_room_id:
        return JsonResponse({"error": "data_room_id required"}, status=400)

    data_room = get_object_or_404(DataRoom, pk=data_room_id)

    from documents.views import _user_can_modify_data_room

    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Access denied"}, status=403)

    from chat.services import save_canvas_to_data_room as save_canvas_to_data_room_service

    doc = save_canvas_to_data_room_service(canvas, data_room, request.user)

    return JsonResponse(
        {
            "ok": True,
            "document_id": doc.id,
            "filename": doc.original_filename,
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
@require_POST
def upload_attachments(request, thread_id):
    """Upload file attachments for a chat message.

    Security note: the stored ``content_type`` is client-asserted (the browser's
    reported MIME type, only validated against an allow-list below). It is trusted
    solely for routing the file to the LLM (image vs. pdf vs. text block) and is
    never echoed back to a browser. Attachments are not served back as downloads;
    if a serve-back endpoint is ever added it MUST send ``Content-Disposition:
    attachment`` + ``X-Content-Type-Options: nosniff`` and must NOT reflect this
    stored value as the response ``Content-Type`` (it could be a mislabeled file).
    """
    from chat.services import SUPPORTED_ATTACHMENT_TYPES, SUPPORTED_DOCX_TYPES, max_size_for_content_type

    thread = get_object_or_404(ChatThread, id=thread_id, created_by=request.user)

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    results = []
    for f in files:
        ct = f.content_type
        # Browsers sometimes report .docx as application/octet-stream
        if ct not in SUPPORTED_ATTACHMENT_TYPES:
            if f.name and f.name.lower().endswith(".docx"):
                ct = next(iter(SUPPORTED_DOCX_TYPES))
                f.content_type = ct
            else:
                return JsonResponse(
                    {"error": f"Unsupported file type: {ct}"},
                    status=400,
                )
        max_size = max_size_for_content_type(ct)
        if f.size > max_size:
            return JsonResponse(
                {"error": f"{f.name} is too large (max {max_size // (1024 * 1024)} MB)."},
                status=400,
            )

    for f in files:
        att = ChatAttachment.objects.create(
            thread=thread,
            uploaded_by=request.user,
            file=f,
            original_filename=f.name[:255],
            content_type=f.content_type,
            size_bytes=f.size,
        )
        results.append({
            "id": str(att.id),
            "filename": att.original_filename,
            "content_type": att.content_type,
            "size_bytes": att.size_bytes,
        })

    return JsonResponse({"attachments": results})


@login_required
@require_http_methods(["GET"])
def data_rooms_for_user(request):
    """JSON API returning the user's data rooms for the attach dropdown."""
    rooms = _get_accessible_data_rooms(request.user)
    for r in rooms:
        r["uuid"] = str(r["uuid"])
    return JsonResponse({"data_rooms": rooms})


# Sidebar search: caps keep the response small and bound the icontains scan.
SEARCH_TITLE_LIMIT = 10
SEARCH_CONTENT_LIMIT = 10
SEARCH_CONTENT_MIN_QUERY = 3
SEARCH_CONTENT_SCAN_CAP = 500


def _search_snippet(content, query, before=45, after=65):
    """Short window of ``content`` around the first case-insensitive match of
    ``query``, whitespace-collapsed, with ellipses marking truncation."""
    idx = content.lower().find(query.lower())
    if idx == -1:
        idx = 0
    start = max(0, idx - before)
    end = min(len(content), idx + len(query) + after)
    snippet = " ".join(content[start:end].split())
    if start > 0:
        snippet = "…" + snippet
    if end < len(content):
        snippet = snippet + "…"
    return snippet


@login_required
@require_http_methods(["GET"])
def search_threads(request):
    """JSON API for the sidebar chat search.

    Returns one merged list: threads whose title matches first, then threads
    with a matching user/assistant message (deduped, with a snippet of the
    most recent matching message). Archived threads are included and flagged.
    """
    from chat.models import ChatMessage

    query = request.GET.get("q", "").strip()
    if not query:
        return JsonResponse({"results": []})

    results = []

    title_threads = list(
        ChatThread.objects.filter(created_by=request.user, title__icontains=query)
        .order_by("-updated_at")[:SEARCH_TITLE_LIMIT]
    )
    for t in title_threads:
        results.append({
            "id": str(t.id),
            "title": str(t),
            "emoji": t.emoji,
            "is_archived": t.is_archived,
            "snippet": None,
        })

    # Content matches: skip very short queries (they match everything) and
    # threads already found by title. Hidden messages aren't shown in the UI,
    # so they shouldn't be searchable either.
    if len(query) >= SEARCH_CONTENT_MIN_QUERY:
        title_ids = {t.id for t in title_threads}
        matched = {}  # thread_id -> snippet of most recent matching message
        message_rows = (
            ChatMessage.objects.filter(
                thread__created_by=request.user,
                role__in=[ChatMessage.Role.USER, ChatMessage.Role.ASSISTANT],
                is_hidden_from_user=False,
                content__icontains=query,
            )
            .exclude(thread_id__in=title_ids)
            .order_by("-created_at")
            .values_list("thread_id", "content")[:SEARCH_CONTENT_SCAN_CAP]
        )
        for thread_id, content in message_rows:
            if thread_id in matched:
                continue
            matched[thread_id] = _search_snippet(content, query)
            if len(matched) >= SEARCH_CONTENT_LIMIT:
                break

        if matched:
            content_threads = ChatThread.objects.filter(id__in=matched)
            threads_by_id = {t.id: t for t in content_threads}
            # Preserve message-recency order from the scan above.
            for thread_id, snippet in matched.items():
                t = threads_by_id.get(thread_id)
                if t is None:
                    continue
                results.append({
                    "id": str(t.id),
                    "title": str(t),
                    "emoji": t.emoji,
                    "is_archived": t.is_archived,
                    "snippet": snippet,
                })

    return JsonResponse({"results": results})


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

# Loop payload validation, resource linking, and create/edit/pause/resume
# orchestration live in ``chat/loop_service.py`` (shared with the agent tools in
# ``chat/tool_loops.py``). The views below are thin request wrappers around them.


@login_required
@require_http_methods(["GET"])
def loops_list(request):
    """Loops page: active + paused loops with unread indicators and a create modal."""
    import json as _json

    from django.conf import settings as django_settings
    from django.db.models import F

    from chat.loop_service import _loop_form_json
    from chat.models import Loop
    from core.preferences import get_preferences
    from llm.display import get_display_name

    prefs = get_preferences(request.user)

    base = Loop.objects.filter(created_by=request.user).select_related("thread")
    order = [F("last_result_at").desc(nulls_last=True), "-created_at"]
    active_loops = list(base.filter(status=Loop.Status.ACTIVE).order_by(*order))
    paused_loops = list(base.filter(status=Loop.Status.PAUSED).order_by(*order))

    data_rooms = _get_accessible_data_rooms(request.user)
    for r in data_rooms:
        r["uuid"] = str(r["uuid"])
    skills = [
        {"id": str(s["id"]), "name": s["name"], "emoji": s.get("emoji", "")}
        for s in prefs.allowed_skills
    ]
    model_choices = [{"id": m, "display": get_display_name(m)} for m in prefs.allowed_models]
    preferred_chat_model = prefs.feature_models.get("chat", prefs.top_model)
    loops_json = {
        str(loop.id): _loop_form_json(loop, prefs)
        for loop in (active_loops + paused_loops)
    }

    return render(request, "chat/loops_list.html", {
        "active_loops": active_loops,
        "paused_loops": paused_loops,
        "data_rooms": data_rooms,
        "skills": skills,
        "data_rooms_json": _json.dumps([{"id": r["pk"], "name": r["name"]} for r in data_rooms]),
        "skills_json": _json.dumps(skills),
        "model_choices_json": _json.dumps(model_choices),
        "preferred_chat_model": preferred_chat_model,
        "preferred_chat_model_display": get_display_name(preferred_chat_model),
        "loops_json": _json.dumps(loops_json),
        "assistant_name": django_settings.ASSISTANT_NAME,
        # /loop prefill / deep-links
        "open_create": request.GET.get("new") == "1",
        "open_edit_id": request.GET.get("edit", "") if request.GET.get("edit") in loops_json else "",
        "prefill_prompt": request.GET.get("prompt", ""),
        "prefill_interval": request.GET.get("interval", ""),
    })


@login_required
@require_POST
def loop_create(request):
    """Create a loop (and its backing thread) from the modal payload."""
    from django.utils import timezone

    from chat.loop_service import create_loop

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    loop, was_reduced, errors = create_loop(
        user=request.user, body=body,
        now=timezone.now(), tz_name=timezone.get_current_timezone_name(),
    )
    if errors:
        return JsonResponse({"error": " ".join(errors)}, status=400)

    return JsonResponse({
        "ok": True,
        "loop_id": str(loop.id),
        "thread_id": str(loop.thread_id),
        "was_reduced": was_reduced,
        "max_runs": loop.max_runs,
        "notice": (
            f"Run count reduced to {loop.max_runs} so the last run stays within a year."
            if was_reduced else ""
        ),
    })


@login_required
@require_POST
def loop_edit(request, loop_id):
    """Edit an existing loop. Recomputes next_run going forward."""
    from django.utils import timezone

    from chat.loop_service import update_loop
    from chat.models import Loop

    loop = get_object_or_404(Loop.objects.select_related("thread"), id=loop_id, created_by=request.user)
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    loop, was_reduced, errors = update_loop(
        loop=loop, user=request.user, body=body,
        now=timezone.now(), tz_name=timezone.get_current_timezone_name(),
    )
    if errors:
        return JsonResponse({"error": " ".join(errors)}, status=400)

    return JsonResponse({
        "ok": True,
        "loop_id": str(loop.id),
        "was_reduced": was_reduced,
        "max_runs": loop.max_runs,
        "notice": (
            f"Run count reduced to {loop.max_runs} so the last run stays within a year."
            if was_reduced else ""
        ),
    })


@login_required
@require_POST
def loop_pause(request, loop_id):
    from chat.loop_service import pause_loop
    from chat.models import Loop

    loop = get_object_or_404(Loop, id=loop_id, created_by=request.user)
    pause_loop(loop)
    return JsonResponse({"ok": True})


@login_required
@require_POST
def loop_resume(request, loop_id):
    """Resume a paused loop: fire on the next tick, then continue the cadence."""
    from django.utils import timezone

    from chat.loop_service import resume_loop
    from chat.models import Loop

    loop = get_object_or_404(Loop, id=loop_id, created_by=request.user)
    resume_loop(loop, timezone.now())
    return JsonResponse({"ok": True})
