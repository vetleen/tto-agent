from __future__ import annotations

import logging
import ntpath
import os
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods, require_POST

from .models import Meeting, MeetingArtifact, MeetingAttachment, MeetingDataRoom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(filename: str, max_length: int = 255) -> str:
    """Normalize and cap client-provided file names. Mirrors documents/views.py."""
    raw = (filename or "").strip()
    if not raw:
        return "file"
    name = os.path.basename(ntpath.basename(raw)).strip()
    if not name:
        return "file"
    if len(name) <= max_length:
        return name
    base, ext = os.path.splitext(name)
    if not ext:
        return name[:max_length]
    reserved = len(ext)
    if reserved >= max_length:
        return name[:max_length]
    return f"{base[: max_length - reserved]}{ext}"


def _user_can_access_meeting(user, meeting: Meeting) -> bool:
    return meeting.created_by_id == user.id


def _user_can_modify_meeting(user, meeting: Meeting) -> bool:
    return meeting.created_by_id == user.id


def _format_relative(value):
    """Lightweight 'today/yesterday/N days ago' formatter."""
    if value is None:
        return ""
    now = timezone.localtime(timezone.now())
    value = timezone.localtime(value)
    delta = now.date() - value.date()
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


# ---------------------------------------------------------------------------
# List + create + CRUD
# ---------------------------------------------------------------------------


@login_required
@require_http_methods(["GET"])
def meeting_list(request):
    all_meetings = Meeting.objects.filter(created_by=request.user).order_by("-updated_at")
    active = [m for m in all_meetings if not m.is_archived]
    archived = [m for m in all_meetings if m.is_archived]
    for m in active:
        m.relative_updated = _format_relative(m.updated_at)
    for m in archived:
        m.relative_updated = _format_relative(m.updated_at)
    return render(
        request,
        "meetings/meeting_list.html",
        {"meetings": active, "archived_meetings": archived},
    )


@login_required
@require_POST
def meeting_create(request):
    """Create a meeting with a default name and redirect to its detail page.

    The list page exposes two buttons (plain "Create meeting" and "Start
    transcription"). Both call this endpoint; the latter sets ``transcribe=1``
    so the detail page auto-starts live transcription on load.
    """
    default_name = f"{timezone.localtime(timezone.now()).strftime('%y%m%d')} - New meeting"
    base_slug = slugify(default_name) or "meeting"
    n = 0
    meeting = None
    while True:
        slug = base_slug if n == 0 else f"{base_slug}-{n}"
        try:
            with transaction.atomic():
                meeting = Meeting.objects.create(
                    name=default_name[:255],
                    slug=slug,
                    created_by=request.user,
                )
            break
        except IntegrityError:
            n += 1
            if n > 50:
                messages.error(request, "Could not create meeting right now. Please try again.")
                return redirect("meeting_list")
    target = reverse("meeting_detail", kwargs={"meeting_uuid": meeting.uuid})
    if (request.POST.get("transcribe") or "").strip() in ("1", "true", "yes"):
        target = f"{target}?transcribe=1"
    return redirect(target)


@login_required
@require_http_methods(["GET"])
def meeting_detail(request, meeting_uuid):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_access_meeting(request.user, meeting):
        return redirect("meeting_list")

    segments = list(meeting.segments.all().order_by("segment_index"))
    artifacts = list(meeting.artifacts.all().order_by("-created_at"))
    attachments = list(meeting.attachments.all().order_by("-uploaded_at"))

    from documents.models import DataRoom

    user_data_rooms = list(
        DataRoom.objects.filter(created_by=request.user, is_archived=False).order_by("name")
    )

    # Build the transcription model picker choices: every model the user is
    # allowed to use (per cascading prefs), with display names from the
    # transcription registry. Default = the meeting's saved model if any,
    # else the user's resolved transcription default.
    from llm.transcription_registry import get_transcription_model_info

    try:
        from core.preferences import get_preferences
        prefs = get_preferences(request.user)
        allowed_models = list(prefs.allowed_transcription_models or [])
        prefs_default = prefs.transcription_model or ""
    except Exception:
        allowed_models = []
        prefs_default = ""

    transcription_model_choices = []
    for model_id in allowed_models:
        info = get_transcription_model_info(model_id)
        transcription_model_choices.append({
            "id": model_id,
            "display_name": info.display_name if info else model_id,
        })

    selected_model = meeting.transcription_model or prefs_default or (
        allowed_models[0] if allowed_models else ""
    )
    selected_display = next(
        (c["display_name"] for c in transcription_model_choices if c["id"] == selected_model),
        selected_model,
    )

    return render(
        request,
        "meetings/meeting_detail.html",
        {
            "meeting": meeting,
            "segments": segments,
            "artifacts": artifacts,
            "attachments": attachments,
            "user_data_rooms": user_data_rooms,
            "auto_stop_default_seconds": getattr(settings, "MEETING_AUTO_STOP_DEFAULT_SECONDS", 3600),
            "auto_stop_max_seconds": getattr(settings, "MEETING_AUTO_STOP_MAX_SECONDS", 14400),
            "transcription_model_choices": transcription_model_choices,
            "transcription_model_selected": selected_model,
            "transcription_model_selected_display": selected_display,
        },
    )


@login_required
@require_POST
def meeting_rename(request, meeting_uuid):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Meeting name cannot be empty.")
        return redirect("meeting_list")
    meeting.name = name[:255]
    meeting.save(update_fields=["name", "updated_at"])
    messages.success(request, "Meeting renamed.")
    return redirect("meeting_list")


@login_required
@require_POST
def meeting_archive(request, meeting_uuid):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    meeting.is_archived = not meeting.is_archived
    meeting.save(update_fields=["is_archived", "updated_at"])
    label = "archived" if meeting.is_archived else "restored"
    messages.success(request, f"Meeting {label}.")
    return redirect("meeting_list")


@login_required
@require_POST
def meeting_delete(request, meeting_uuid):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    meeting.delete()
    messages.success(request, "Meeting deleted.")
    return redirect("meeting_list")


@login_required
@require_POST
def meeting_update_metadata(request, meeting_uuid):
    """Update editable metadata fields (name, agenda, participants, description, transcription_model)."""
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return JsonResponse({"error": "Forbidden"}, status=403)
    fields = []
    for field in ("name", "agenda", "participants", "description"):
        if field in request.POST:
            value = (request.POST.get(field) or "").strip()
            if field == "name":
                if not value:
                    return JsonResponse({"error": "Name cannot be empty."}, status=400)
                value = value[:255]
            setattr(meeting, field, value)
            fields.append(field)
    if "transcription_model" in request.POST:
        new_model = (request.POST.get("transcription_model") or "").strip()
        # Validate against the user's allowed transcription models so we can't
        # be talked into using something the org has disabled.
        try:
            from core.preferences import get_preferences
            allowed = list(get_preferences(request.user).allowed_transcription_models or [])
        except Exception:
            allowed = []
        if new_model and new_model not in allowed:
            return JsonResponse(
                {"error": f"Model '{new_model}' is not in your allowed transcription models."},
                status=400,
            )
        meeting.transcription_model = new_model
        fields.append("transcription_model")
    if not fields:
        return JsonResponse({"error": "No editable fields supplied"}, status=400)
    fields.append("updated_at")
    meeting.save(update_fields=fields)
    return JsonResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Linking to data rooms
# ---------------------------------------------------------------------------


@login_required
@require_POST
def meeting_link_data_room(request, meeting_uuid):
    from documents.models import DataRoom

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    data_room_id = request.POST.get("data_room_id")
    if not data_room_id:
        messages.error(request, "No data room selected.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    try:
        data_room = DataRoom.objects.get(uuid=data_room_id)
    except (DataRoom.DoesNotExist, ValueError):
        messages.error(request, "Data room not found.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    if data_room.created_by_id != request.user.id:
        messages.error(request, "You can only link data rooms you own.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    MeetingDataRoom.objects.get_or_create(meeting=meeting, data_room=data_room)
    meeting.save(update_fields=["updated_at"])
    messages.success(request, f"Linked data room: {data_room.name}.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


@login_required
@require_POST
def meeting_unlink_data_room(request, meeting_uuid, data_room_uuid):
    from documents.models import DataRoom

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    try:
        data_room = DataRoom.objects.get(uuid=data_room_uuid)
    except (DataRoom.DoesNotExist, ValueError):
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    MeetingDataRoom.objects.filter(meeting=meeting, data_room=data_room).delete()
    meeting.save(update_fields=["updated_at"])
    messages.success(request, "Data room unlinked.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


# ---------------------------------------------------------------------------
# Transcript & audio uploads
# ---------------------------------------------------------------------------


@login_required
@require_POST
def meeting_upload(request, meeting_uuid):
    """Single upload endpoint that routes to audio or transcript handlers based on extension."""
    from llm.transcription_registry import AUDIO_EXTENSIONS

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    file_obj = request.FILES.get("file")
    if not file_obj:
        messages.error(request, "Please choose a file to upload.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    safe_name = _safe_filename(file_obj.name, max_length=255)
    ext = (safe_name.rsplit(".", 1)[-1].lower()) if "." in safe_name else ""
    transcript_exts = getattr(settings, "MEETING_TRANSCRIPT_ALLOWED_EXTENSIONS", {"txt", "md"})

    if ext in AUDIO_EXTENSIONS:
        # Re-route to the audio handler with the file under the expected key.
        request.FILES["audio"] = file_obj
        return meeting_upload_audio(request, meeting_uuid)
    if ext in transcript_exts:
        request.FILES["transcript"] = file_obj
        return meeting_upload_transcript(request, meeting_uuid)
    messages.error(
        request,
        f"Unsupported file type. Upload audio ({sorted(AUDIO_EXTENSIONS)}) "
        f"or text transcript ({sorted(transcript_exts)}).",
    )
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


@login_required
@require_POST
def meeting_save_to_data_room(request, meeting_uuid):
    """Save the meeting transcript (or most recent artifact) to a data room as a Document."""
    from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentTag

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")

    data_room_id = (request.POST.get("data_room_id") or "").strip()
    if not data_room_id:
        messages.error(request, "Pick a data room to save to.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    try:
        data_room = DataRoom.objects.get(uuid=data_room_id)
    except (DataRoom.DoesNotExist, ValueError):
        messages.error(request, "Data room not found.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    if data_room.created_by_id != request.user.id:
        messages.error(request, "You can only save to data rooms you own.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    # Prefer the most recent artifact (e.g. minutes generated by Wilfred). If
    # none exists yet, fall back to saving the raw transcript so the button
    # always does *something* useful.
    latest_artifact = meeting.artifacts.order_by("-created_at").first()
    if latest_artifact and (latest_artifact.content_md or "").strip():
        body = latest_artifact.content_md
        kind_label = latest_artifact.get_kind_display().lower()
        filename = f"{meeting.slug}-{kind_label}.md"
        mime = "text/markdown"
    elif (meeting.transcript or "").strip():
        body = meeting.transcript
        filename = f"{meeting.slug}-transcript.md"
        mime = "text/markdown"
    else:
        messages.error(request, "Nothing to save yet — record or upload a transcript first.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    from django.core.files.base import ContentFile

    payload = body.encode("utf-8")
    safe_filename = _safe_filename(filename, max_length=180)
    file_obj = ContentFile(payload, name=safe_filename)
    doc = DataRoomDocument.objects.create(
        data_room=data_room,
        uploaded_by=request.user,
        original_file=file_obj,
        original_filename=safe_filename,
        mime_type=mime,
        size_bytes=len(payload),
        status=DataRoomDocument.Status.UPLOADED,
    )
    DataRoomDocumentTag.objects.create(document=doc, key="source", value="meeting_export")
    DataRoomDocumentTag.objects.create(document=doc, key="meeting_uuid", value=str(meeting.uuid))
    try:
        from documents.tasks import process_document_task
        process_document_task.delay(doc.id)
    except Exception:
        try:
            from documents.services.process_document import process_document
            process_document(doc.id)
        except Exception as exc:
            logger.exception("meeting_save_to_data_room: processing failed for doc %s", doc.id)
            doc.status = DataRoomDocument.Status.FAILED
            doc.processing_error = str(exc)[:2000]
            doc.save(update_fields=["status", "processing_error", "updated_at"])
            messages.error(request, "Could not start processing the saved document.")
            return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    messages.success(request, f"Saved to data room: {data_room.name}.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


@login_required
@require_POST
def meeting_upload_transcript(request, meeting_uuid):
    """Upload a pre-existing transcript text file (.txt or .md)."""
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    file_obj = request.FILES.get("transcript")
    if not file_obj:
        messages.error(request, "Please choose a transcript file to upload.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    safe_name = _safe_filename(file_obj.name, max_length=255)
    ext = (safe_name.rsplit(".", 1)[-1].lower()) if "." in safe_name else ""
    allowed = getattr(settings, "MEETING_TRANSCRIPT_ALLOWED_EXTENSIONS", {"txt", "md"})
    if ext not in allowed:
        messages.error(request, f"Unsupported transcript format. Use one of: {sorted(allowed)}.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    max_bytes = getattr(settings, "MEETING_TRANSCRIPT_UPLOAD_MAX_BYTES", 2_000_000)
    if file_obj.size > max_bytes:
        messages.error(request, f"Transcript is too large (max {max_bytes // 1024} KB).")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    try:
        raw = file_obj.read()
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        messages.error(request, "Transcript must be UTF-8 encoded text.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    meeting.transcript = text
    meeting.transcript_source = Meeting.TranscriptSource.TEXT_UPLOAD
    meeting.transcription_model = ""
    meeting.transcription_error = ""
    meeting.status = Meeting.Status.READY
    meeting.save(update_fields=[
        "transcript", "transcript_source", "transcription_model",
        "transcription_error", "status", "updated_at",
    ])
    messages.success(request, "Transcript uploaded.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


@login_required
@require_POST
def meeting_upload_audio(request, meeting_uuid):
    """Upload an audio file to be transcribed by the existing TranscriptionService."""
    from llm.transcription_registry import AUDIO_EXTENSIONS

    from .tasks import transcribe_uploaded_audio_task

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    file_obj = request.FILES.get("audio")
    if not file_obj:
        messages.error(request, "Please choose an audio file to upload.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    safe_name = _safe_filename(file_obj.name, max_length=180)
    ext = (safe_name.rsplit(".", 1)[-1].lower()) if "." in safe_name else ""
    if ext not in AUDIO_EXTENSIONS:
        messages.error(request, f"Unsupported audio format. Use one of: {sorted(AUDIO_EXTENSIONS)}.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    max_bytes = getattr(settings, "MEETING_AUDIO_UPLOAD_MAX_BYTES", 200 * 1024 * 1024)
    if file_obj.size > max_bytes:
        messages.error(request, f"Audio file is too large (max {max_bytes // (1024 * 1024)} MB).")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    # Pick a transcription model from user prefs, falling back to the project default.
    try:
        from core.preferences import get_preferences
        prefs = get_preferences(request.user)
        allowed_models = list(getattr(prefs, "allowed_transcription_models", None) or [])
    except Exception:
        allowed_models = []
    model_id = (
        allowed_models[0]
        if allowed_models
        else getattr(settings, "TRANSCRIPTION_DEFAULT_MODEL", "openai/gpt-4o-mini-transcribe")
    )

    # Persist the audio to a temp location under MEETING_CHUNK_TEMP_DIR so the
    # task can read it. We delete the file after transcription succeeds OR fails.
    from pathlib import Path

    temp_dir = Path(getattr(settings, "MEETING_CHUNK_TEMP_DIR", "")) / str(meeting.uuid)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload-{int(timezone.now().timestamp())}.{ext}"
    with open(temp_path, "wb") as f:
        for chunk in file_obj.chunks():
            f.write(chunk)

    meeting.status = Meeting.Status.LIVE_TRANSCRIBING  # treat upload as in-progress until task finishes
    meeting.transcript_source = Meeting.TranscriptSource.AUDIO_UPLOAD
    meeting.started_at = meeting.started_at or timezone.now()
    meeting.transcription_error = ""
    meeting.save(update_fields=[
        "status", "transcript_source", "started_at", "transcription_error", "updated_at",
    ])

    try:
        transcribe_uploaded_audio_task.delay(
            meeting_id=meeting.id,
            temp_path=str(temp_path),
            model_id=model_id,
            user_id=request.user.id,
        )
    except Exception as exc:
        logger.exception("meeting_upload_audio: failed to enqueue task for meeting %s", meeting.uuid)
        meeting.status = Meeting.Status.FAILED
        meeting.transcription_error = str(exc)[:1000]
        meeting.save(update_fields=["status", "transcription_error", "updated_at"])
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        messages.error(request, "Could not start audio transcription. Please try again.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)

    messages.success(request, "Audio uploaded — transcribing in the background.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


# ---------------------------------------------------------------------------
# Meeting attachments (slides, agenda PDFs, etc.)
# ---------------------------------------------------------------------------


@login_required
@require_POST
def meeting_upload_attachment(request, meeting_uuid):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    file_obj = request.FILES.get("file")
    if not file_obj:
        messages.error(request, "Please choose a file to attach.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    safe_name = _safe_filename(file_obj.name, max_length=255)
    MeetingAttachment.objects.create(
        meeting=meeting,
        uploaded_by=request.user,
        file=file_obj,
        original_filename=safe_name,
        content_type=getattr(file_obj, "content_type", "") or "",
        size_bytes=file_obj.size or 0,
    )
    meeting.save(update_fields=["updated_at"])
    messages.success(request, f"Attached {safe_name}.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


@login_required
@require_POST
def meeting_delete_attachment(request, meeting_uuid, attachment_id):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    attachment = get_object_or_404(MeetingAttachment, pk=attachment_id, meeting=meeting)
    attachment.delete()
    messages.success(request, "Attachment removed.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


# ---------------------------------------------------------------------------
# Artifact deletion (Wilfred-saved minutes)
# ---------------------------------------------------------------------------


@login_required
@require_POST
def meeting_delete_artifact(request, meeting_uuid, artifact_id):
    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    artifact = get_object_or_404(MeetingArtifact, pk=artifact_id, meeting=meeting)
    artifact.delete()
    messages.success(request, "Artifact deleted.")
    return redirect("meeting_detail", meeting_uuid=meeting.uuid)


# ---------------------------------------------------------------------------
# "Create meeting minutes with Wilfred" — redirects into a chat thread
# ---------------------------------------------------------------------------


@login_required
@require_POST
def meeting_create_minutes_thread(request, meeting_uuid):
    from .services.minutes import create_minutes_thread

    meeting = get_object_or_404(Meeting, uuid=meeting_uuid)
    if not _user_can_modify_meeting(request.user, meeting):
        return redirect("meeting_list")
    if not (meeting.transcript or "").strip():
        messages.error(request, "This meeting has no transcript yet.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    thread, err = create_minutes_thread(request.user, meeting)
    if err or thread is None:
        messages.error(request, err or "Could not start a chat session.")
        return redirect("meeting_detail", meeting_uuid=meeting.uuid)
    target = f"{reverse('chat_home')}?{urlencode({'thread': str(thread.id)})}"
    return redirect(target)
