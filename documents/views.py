import json
import logging
import ntpath
import os

from django.conf import settings
from django.db import IntegrityError
from django.db.models import Count, Prefetch, Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods, require_POST

from .models import DataRoom, DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentTag
from .pii_labels import CRIMINAL_TOOLTIP, PILL_LABEL, SPECIAL_TOOLTIP, summarize_pii_keys


logger = logging.getLogger(__name__)


def _relative_upload_date(value):
    """Format a datetime as 'today at HH.mm', 'yesterday', 'x days ago', etc."""
    if value is None:
        return ""
    now = timezone.now()
    if timezone.is_naive(value):
        value = timezone.make_aware(value)
    value = timezone.localtime(value)
    now = timezone.localtime(now)
    today = now.date()
    upload_date = value.date()
    delta = today - upload_date
    if delta.days == 0:
        return f"Today at {value.strftime('%H:%M')}"
    if delta.days == 1:
        return "Yesterday"
    if delta.days <= 30:
        return f"{delta.days} days ago"
    months = delta.days // 30
    if months <= 11:
        return "1 month ago" if months == 1 else f"{months} months ago"
    years = delta.days // 365
    return "1 year ago" if years == 1 else f"{years} years ago"


def _parse_json_body(request):
    """Parse JSON request body. Returns (data, None) on success or (None, error response)."""
    try:
        return json.loads(request.body), None
    except (json.JSONDecodeError, ValueError):
        return None, JsonResponse({"error": "Invalid JSON"}, status=400)


def _annotate_relative_dates(docs):
    """Add relative_upload_display to each document in a list."""
    for doc in docs:
        doc.relative_upload_display = _relative_upload_date(doc.uploaded_at)
    return docs


def _user_can_access_data_room(user, data_room: DataRoom) -> bool:
    return data_room.created_by_id == user.id


def _user_can_modify_data_room(user, data_room: DataRoom) -> bool:
    return data_room.created_by_id == user.id


@login_required
@require_http_methods(["GET", "POST"])
def data_room_list(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if name:
            base_slug = slugify(name) or "data-room"
            n = 0
            data_room = None
            while True:
                slug = base_slug if n == 0 else f"{base_slug}-{n}"
                try:
                    description = (request.POST.get("description") or "").strip()[:1000]
                    data_room = DataRoom.objects.create(
                        name=name, slug=slug, created_by=request.user,
                        description=description,
                    )
                    break
                except IntegrityError:
                    n += 1
                    if n > 50:
                        messages.error(request, "Could not create data room right now. Please try again.")
                        break
            if data_room:
                return redirect("data_room_documents", data_room_id=data_room.uuid)
        return redirect("data_room_list")
    active_doc = Q(documents__is_archived=False)
    all_rooms = (
        DataRoom.objects.filter(created_by=request.user)
        .annotate(
            document_count=Count("documents", filter=active_doc),
            processing_count=Count(
                "documents",
                filter=active_doc & Q(documents__status__in=["uploaded", "processing"]),
            ),
        )
        .order_by("-updated_at")
    )
    data_rooms = [r for r in all_rooms if not r.is_archived]
    archived_data_rooms = [r for r in all_rooms if r.is_archived]
    return render(request, "documents/data_room_list.html", {
        "data_rooms": data_rooms,
        "archived_data_rooms": archived_data_rooms,
    })


@login_required
@require_POST
def data_room_delete(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    if request.POST.get("delete_threads") == "true":
        doc_ids = list(data_room.documents.values_list("pk", flat=True))
        if doc_ids:
            _delete_threads_for_documents(doc_ids, data_room)
    data_room.delete()
    messages.success(request, "Data room deleted.")
    return redirect("data_room_list")


@login_required
@require_POST
def data_room_delete_check(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)

    from chat.models import ThreadChunkUsage

    thread_qs = (
        ThreadChunkUsage.objects.filter(document__data_room=data_room)
        .values("thread_id", "thread__title")
        .distinct()
    )
    threads = [
        {"id": str(row["thread_id"]), "title": row["thread__title"] or "Untitled"}
        for row in thread_qs
    ]
    return JsonResponse({
        "affected_threads": threads,
        "affected_thread_count": len(threads),
    })


@login_required
@require_http_methods(["GET", "POST"])
def data_room_rename(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    if request.method != "POST":
        return redirect("data_room_list")
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Data room name cannot be empty.")
        return redirect("data_room_list")
    if len(name) > 255:
        name = name[:255]
    data_room.name = name
    data_room.save(update_fields=["name", "updated_at"])
    messages.success(request, "Data room renamed.")
    return redirect("data_room_list")


@login_required
@require_POST
def data_room_archive(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    data_room.is_archived = not data_room.is_archived
    data_room.save(update_fields=["is_archived", "updated_at"])
    label = "archived" if data_room.is_archived else "restored"
    messages.success(request, f"Data room {label}.")
    return redirect("data_room_list")


@login_required
@require_http_methods(["GET"])
def data_room_documents(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        return redirect("data_room_list")
    all_docs = list(
        data_room.documents.order_by("-uploaded_at").prefetch_related(
            Prefetch(
                "tags",
                queryset=DataRoomDocumentTag.objects.filter(key__startswith="pii_"),
                to_attr="pii_tags",
            )
        )
    )
    for doc in all_docs:
        doc.pii_summary = summarize_pii_keys([t.key for t in doc.pii_tags])
    documents = _annotate_relative_dates([d for d in all_docs if not d.is_archived])
    archived_documents = _annotate_relative_dates([d for d in all_docs if d.is_archived])
    return render(
        request,
        "documents/data_room_documents.html",
        {
            "data_room": data_room,
            "documents": documents,
            "archived_documents": archived_documents,
            "pii_pill_label": PILL_LABEL,
            "pii_special_tooltip": SPECIAL_TOOLTIP,
            "pii_criminal_tooltip": CRIMINAL_TOOLTIP,
        },
    )


def _safe_original_filename(filename: str, max_length: int = 255) -> str:
    """Normalize and cap client-provided file names for safe persistence/display."""
    raw = (filename or "").strip()
    if not raw:
        return "document"
    # Handle both Unix and Windows style paths that may be sent by clients.
    name = os.path.basename(ntpath.basename(raw)).strip()
    if not name:
        return "document"
    if len(name) <= max_length:
        return name
    base, ext = os.path.splitext(name)
    if not ext:
        return name[:max_length]
    reserved = len(ext)
    if reserved >= max_length:
        return name[:max_length]
    return f"{base[: max_length - reserved]}{ext}"


def _allowed_extension(filename: str) -> bool:
    ext = (filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return ext in getattr(settings, "DOCUMENT_ALLOWED_EXTENSIONS", {"pdf", "txt", "md", "html"})


def _allowed_mime(mime_type: str) -> bool:
    allowed_mime_types = getattr(settings, "DOCUMENT_ALLOWED_MIME_TYPES", None)
    # Empty/undefined allowlist means MIME checking is disabled.
    if not allowed_mime_types:
        return True
    return mime_type in allowed_mime_types


@login_required
@require_POST
def document_upload(request, data_room_id):
    is_ajax = "application/json" in request.headers.get("Accept", "")
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        if is_ajax:
            return JsonResponse({"status": "error", "error": "Permission denied."}, status=403)
        return redirect("data_room_list")
    files = request.FILES.getlist("file")
    if not files:
        if is_ajax:
            return JsonResponse({"status": "error", "error": "No file selected."}, status=400)
        messages.error(request, "No file selected. Please choose a file to upload.")
        return redirect("data_room_documents", data_room_id=data_room.uuid)

    from llm.transcription_registry import AUDIO_EXTENSIONS

    max_size = getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 10_000_000)
    audio_max_size = getattr(settings, "AUDIO_UPLOAD_MAX_SIZE_BYTES", 25_000_000)
    errors = []
    created_docs = []

    for file_obj in files:
        safe_filename = _safe_original_filename(file_obj.name, max_length=75)
        file_ext = (safe_filename.rsplit(".", 1)[-1].lower()) if "." in safe_filename else ""
        is_audio = file_ext in AUDIO_EXTENSIONS

        if file_obj.size <= 0:
            errors.append(f"{safe_filename}: file is empty.")
            continue
        if is_audio and file_obj.size > audio_max_size:
            errors.append(f"{safe_filename}: audio file is too large (max {audio_max_size / 1_000_000:.0f} MB).")
            continue
        if file_obj.size > max_size:
            errors.append(f"{safe_filename}: file is too large (max {max_size / 1_000_000:.0f} MB).")
            continue
        if not _allowed_extension(safe_filename):
            errors.append(f"{safe_filename}: unsupported file type.")
            continue
        mime = getattr(file_obj, "content_type", "") or ""
        if mime and not _allowed_mime(mime):
            errors.append(f"{safe_filename}: unsupported file type.")
            continue
        if is_audio:
            from core.preferences import get_preferences
            prefs = get_preferences(request.user)
            if not prefs.allowed_transcription_models:
                errors.append(f"{safe_filename}: audio transcription is not enabled for your organization.")
                continue
        stored_filename = _safe_original_filename(file_obj.name, max_length=180)
        file_obj.name = stored_filename
        doc = DataRoomDocument.objects.create(
            data_room=data_room,
            uploaded_by=request.user,
            original_file=file_obj,
            original_filename=safe_filename,
            mime_type=mime,
            size_bytes=file_obj.size,
            status=DataRoomDocument.Status.UPLOADED,
        )
        created_docs.append(doc)

    if created_docs:
        DataRoomDocumentTag.objects.bulk_create(
            [DataRoomDocumentTag(document=doc, key="source", value="user_uploaded") for doc in created_docs]
        )

    for doc in created_docs:
        try:
            try:
                from documents.tasks import process_document_task

                process_document_task.delay(doc.id)
            except Exception:
                from documents.services.process_document import process_document

                process_document(doc.id)
        except Exception as exc:
            logger.exception("document_upload: failed to enqueue processing for document_id=%s", doc.id)
            doc.status = DataRoomDocument.Status.FAILED
            doc.processing_error = str(exc)[:2000]
            doc.save(update_fields=["status", "processing_error", "updated_at"])
            errors.append(f"{doc.original_filename}: processing could not be started.")

    if is_ajax:
        if created_docs:
            doc = created_docs[0]
            return JsonResponse({
                "status": "ok",
                "document": {"id": doc.id, "filename": doc.original_filename, "status": doc.status},
                "errors": errors,
            })
        return JsonResponse({"status": "error", "error": errors[0] if errors else "Upload failed."}, status=400)

    if created_docs:
        count = len(created_docs)
        messages.success(request, f"{count} file{'s' if count != 1 else ''} uploaded.")
    for err in errors:
        messages.error(request, err)
    return redirect("data_room_documents", data_room_id=data_room.uuid)


@login_required
@require_POST
def document_delete(request, data_room_id, document_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    if request.POST.get("delete_threads") == "true":
        _delete_threads_for_documents([doc.pk], data_room)
    doc.delete()
    messages.success(request, "Document deleted.")
    return redirect("data_room_documents", data_room_id=data_room.uuid)


def _delete_threads_for_documents(doc_ids: list[int], data_room) -> int:
    from chat.models import ChatThread, ThreadChunkUsage

    thread_ids = list(
        ThreadChunkUsage.objects.filter(
            document_id__in=doc_ids,
            document__data_room=data_room,
        )
        .values_list("thread_id", flat=True)
        .distinct()
    )
    if thread_ids:
        deleted, _ = ChatThread.objects.filter(pk__in=thread_ids).delete()
        return deleted
    return 0


@login_required
@require_POST
def document_delete_check(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    body, err = _parse_json_body(request)
    if err:
        return err
    doc_ids = body.get("document_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return JsonResponse({"error": "document_ids must be a non-empty list"}, status=400)

    from chat.models import ThreadChunkUsage

    thread_qs = (
        ThreadChunkUsage.objects.filter(
            document_id__in=doc_ids,
            document__data_room=data_room,
        )
        .values("thread_id", "thread__title")
        .distinct()
    )
    threads = [
        {"id": str(row["thread_id"]), "title": row["thread__title"] or "Untitled"}
        for row in thread_qs
    ]
    return JsonResponse({
        "affected_threads": threads,
        "affected_thread_count": len(threads),
    })


@login_required
@require_http_methods(["POST"])
def document_rename(request, data_room_id, document_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Document name cannot be empty.")
        return redirect("data_room_documents", data_room_id=data_room.uuid)
    doc.original_filename = _safe_original_filename(name, max_length=75)
    doc.save(update_fields=["original_filename", "updated_at"])
    messages.success(request, "Document renamed.")
    return redirect("data_room_documents", data_room_id=data_room.uuid)


@login_required
@require_POST
def document_archive(request, data_room_id, document_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return redirect("data_room_list")
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    doc.is_archived = not doc.is_archived
    doc.save(update_fields=["is_archived", "updated_at"])
    label = "archived" if doc.is_archived else "restored"
    messages.success(request, f"Document {label}.")
    return redirect("data_room_documents", data_room_id=data_room.uuid)


@login_required
@require_http_methods(["GET"])
def document_chunks(request, data_room_id, document_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    chunks = []
    for c in doc.chunks.filter(is_quarantined=False).order_by("chunk_index"):
        chunks.append({
            "id": c.id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
            "source_page_start": c.source_page_start,
            "source_page_end": c.source_page_end,
            "source_offset_start": c.source_offset_start,
            "source_offset_end": c.source_offset_end,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })
    return JsonResponse({"chunks": chunks})


@login_required
@require_POST
def document_bulk_delete(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    body, err = _parse_json_body(request)
    if err:
        return err
    doc_ids = body.get("document_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return JsonResponse({"error": "document_ids must be a non-empty list"}, status=400)
    if body.get("delete_threads"):
        _delete_threads_for_documents(doc_ids, data_room)
    deleted, _ = DataRoomDocument.objects.filter(pk__in=doc_ids, data_room=data_room).delete()
    return JsonResponse({"deleted": deleted})


@login_required
@require_POST
def document_bulk_archive(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    body, err = _parse_json_body(request)
    if err:
        return err
    doc_ids = body.get("document_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return JsonResponse({"error": "document_ids must be a non-empty list"}, status=400)
    action = body.get("action")
    if action not in ("archive", "restore"):
        return JsonResponse({"error": "action must be 'archive' or 'restore'"}, status=400)
    is_archived = action == "archive"
    updated = DataRoomDocument.objects.filter(pk__in=doc_ids, data_room=data_room).update(
        is_archived=is_archived, updated_at=timezone.now()
    )
    return JsonResponse({"updated": updated})


@login_required
@require_http_methods(["GET"])
def document_status(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    statuses = {
        str(pk): status
        for pk, status in data_room.documents.filter(is_archived=False).values_list("id", "status")
    }
    return JsonResponse({"statuses": statuses})


@login_required
@require_POST
def data_room_generate_description(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    try:
        from documents.services.data_room_description import generate_data_room_description
        description = generate_data_room_description(data_room.pk, user_id=request.user.pk)
        return JsonResponse({"description": description})
    except Exception:
        logger.exception("data_room_generate_description failed for %s", data_room_id)
        return JsonResponse({"error": f"{settings.ASSISTANT_NAME} couldn't generate a description right now. Please try again."}, status=500)


@login_required
@require_POST
def data_room_update_description(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    body, err = _parse_json_body(request)
    if err:
        return err
    description = (body.get("description") or "").strip()[:1000]
    data_room.description = description
    data_room.save(update_fields=["description", "updated_at"])
    return JsonResponse({"status": "ok"})
