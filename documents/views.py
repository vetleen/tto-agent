import json
import logging
import re

from django.conf import settings
from django.db import IntegrityError
from django.db.models import Count, Prefetch, Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.http import require_http_methods, require_POST
from csp.constants import SELF
from csp.decorators import csp_replace
from django_ratelimit.decorators import ratelimit

from core.files import safe_filename, sha256_of_upload
from core.http import parse_json_object
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
    """Parse a JSON object request body. Returns (dict, None) or (None, 400)."""
    return parse_json_object(request)


def _parse_document_ids(body):
    """Validate body['document_ids'] as a non-empty list of ints.

    Returns (ids, None) on success or (None, error response) — non-integer ids
    would otherwise blow up as a 500 inside the queryset filter.
    """
    doc_ids = body.get("document_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return None, JsonResponse({"error": "document_ids must be a non-empty list"}, status=400)
    try:
        return [int(x) for x in doc_ids], None
    except (TypeError, ValueError):
        return None, JsonResponse({"error": "document_ids must contain only integers"}, status=400)


def _annotate_relative_dates(docs):
    """Add relative_upload_display to each document in a list."""
    for doc in docs:
        doc.relative_upload_display = _relative_upload_date(doc.uploaded_at)
    return docs


def _user_can_access_data_room(user, data_room: DataRoom) -> bool:
    # Same ownership rule as documents.access.accessible_data_rooms (the queryset
    # form); keep the two in sync if access ever broadens (e.g. shared rooms).
    return data_room.created_by_id == user.id


def _user_can_modify_data_room(user, data_room: DataRoom) -> bool:
    return data_room.created_by_id == user.id


def _document_file_kind(doc):
    """Classify a document by file kind (image / pdf / text / …) from its stored
    mime type, falling back to the original filename's extension."""
    from core.file_types import kind_for_extension, kind_for_mime

    kind = kind_for_mime(doc.mime_type or "")
    if kind is None and doc.original_filename and "." in doc.original_filename:
        kind = kind_for_extension(doc.original_filename.rsplit(".", 1)[-1].lower())
    return kind


def _content_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _inline_edit_max_chars() -> int:
    """Save-mode switch for the in-browser edit path: up to this many chars the
    save re-indexes inline (instant verdict); above it the version is queued."""
    return getattr(settings, "DOCUMENT_INLINE_EDIT_MAX_CHARS", 150_000)


def _browser_edit_max_chars() -> int:
    """What the editor + a multipart POST can carry; above it the modal is read-only."""
    return getattr(settings, "DOCUMENT_BROWSER_EDIT_MAX_CHARS", 1_500_000)


def _queued_payload(data_room, doc, version) -> dict:
    """``document_save`` response for a version handed to the async pipeline. The
    modal keeps polling ``verdict_url`` (``document_version_verdict``) until the
    verdict is in, then shows it exactly like an inline save."""
    from django.urls import reverse

    return {
        "ok": True,
        "verdict": "queued",
        "unchanged": False,
        "version_id": version.id,
        "verdict_url": reverse(
            "document_version_verdict",
            kwargs={
                "data_room_id": data_room.uuid,
                "document_id": doc.id,
                "version_id": version.id,
            },
        ),
        "is_quarantined": bool(doc.is_quarantined),
    }


def _requeue_after_sync_failure(doc, version_id: int) -> bool:
    """Hand a version whose INLINE scan failed to the async pipeline instead.

    The inline path (``scan_version_synchronously``) is eager: a transient
    classifier / PII-LLM failure marks the version SCAN_FAILED and stops, where the
    async pipeline would have retried (Celery retry ladders + the dispatch-retry
    marker). So reset the row to UPLOADED and re-join the dispatch gate; the re-run
    is idempotent (``process_document_version`` replaces the chunks and vectors).

    UPLOADED specifically: the dispatcher only claims UPLOADED/PROCESSING, and
    ``process_document_version`` skips a *fresh* PROCESSING row. A FAILED version
    is NOT re-queued — extraction/empty-content errors are deterministic. Returns
    True when the version was queued; False degrades to the ``scan_failed`` verdict.
    """
    from documents.models import DataRoomDocumentVersion
    from documents.services.versioning import _enqueue_processing

    Status = DataRoomDocument.Status
    try:
        reset = DataRoomDocumentVersion.objects.filter(
            pk=version_id,
            status__in=(
                Status.UPLOADED, Status.PROCESSING, Status.SCANNING, Status.SCAN_FAILED,
            ),
        ).update(status=Status.UPLOADED, processing_error=None, dispatched_at=None)
        if not reset:
            return False
        # An eager scan_failed is mirrored onto a document with no live version
        # (same guard as process_document._mirror_doc_status) — mirror back so the
        # row shows "processing" rather than "scan failed" while the worker runs.
        if doc.active_searchable_version_id in (None, version_id):
            DataRoomDocument.objects.filter(pk=doc.pk).update(
                status=Status.UPLOADED, processing_error=None,
            )
        _enqueue_processing(version_id)
    except Exception:
        logger.exception(
            "document_save: could not re-queue version %s after an inline scan failure",
            version_id,
        )
        return False
    logger.warning(
        "document_save: inline scan failed for version %s; re-queued through the dispatch gate",
        version_id,
    )
    return True


@login_required
@require_http_methods(["GET", "POST"])
def data_room_list(request):
    if request.method == "POST":
        # Truncate to the model field limits — longer values are a DB error (500).
        name = (request.POST.get("name") or "").strip()[:255]
        if name:
            # Slug capped at 90 to leave room for the -N collision suffix (max_length=100).
            base_slug = (slugify(name) or "data-room")[:90]
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
                filter=active_doc & Q(documents__status__in=["uploaded", "processing", "scanning"]),
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
    # current_version feeds display_status (the "queued" mapping) — fetch it in
    # the same query or the template would trigger one extra query per row.
    all_docs = list(
        data_room.documents.select_related("current_version").order_by("-uploaded_at")
    )
    # PII tags live on versions now; summarise from each document's active (or
    # current) version for the per-document badge.
    from collections import defaultdict

    version_ids = [d.active_searchable_version_id or d.current_version_id for d in all_docs]
    version_ids = [v for v in version_ids if v]
    pii_by_version: dict[int, list[str]] = defaultdict(list)
    if version_ids:
        for vid, key in DataRoomDocumentTag.objects.filter(
            version_id__in=version_ids, key__startswith="pii_",
        ).values_list("version_id", "key"):
            pii_by_version[vid].append(key)
    for doc in all_docs:
        vid = doc.active_searchable_version_id or doc.current_version_id
        doc.pii_summary = summarize_pii_keys(pii_by_version.get(vid, []))
        # File kind drives which viewer/editor the modal opens (text = editable).
        doc.file_kind = _document_file_kind(doc)
    documents = _annotate_relative_dates([d for d in all_docs if not d.is_archived])
    archived_documents = _annotate_relative_dates([d for d in all_docs if d.is_archived])

    # Build the file-picker accept list from the unified capability table so it
    # always matches what document_upload actually accepts. Images are only
    # offered when the org has a vision-capable model (the same gate the upload
    # view enforces) — otherwise the picker would surface photos that get
    # rejected. Including image/* makes iOS Safari offer the photo library
    # (and convert HEIC -> JPEG) instead of defaulting to the camera/video flow.
    from core.file_types import DATA_ROOM_KINDS, KIND_IMAGE, accept_attr, allowed_extensions
    from core.preferences import feature_is_available

    images_enabled = feature_is_available(request.user, "document_image_description")
    upload_kinds = set(DATA_ROOM_KINDS)
    if not images_enabled:
        upload_kinds.discard(KIND_IMAGE)

    return render(
        request,
        "documents/data_room_documents.html",
        {
            "data_room": data_room,
            "documents": documents,
            "archived_documents": archived_documents,
            "can_modify": _user_can_modify_data_room(request.user, data_room),
            "pii_pill_label": PILL_LABEL,
            "pii_special_tooltip": SPECIAL_TOOLTIP,
            "pii_criminal_tooltip": CRIMINAL_TOOLTIP,
            "upload_accept": accept_attr(upload_kinds),
            "upload_images_enabled": images_enabled,
            # Client-side pre-upload validation list — derived from the same
            # table as the picker and the server check so they cannot drift.
            "upload_extensions_json": json.dumps(sorted(allowed_extensions(upload_kinds))),
            # Upload limits, rendered into the help text and the client-side
            # pre-checks straight from settings (same defaults the upload view
            # enforces) so they always reflect the real, env-configurable caps.
            "upload_max_size_bytes": getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 50_000_000),
            "upload_max_size_mb": getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 50_000_000) // 1_000_000,
            "upload_in_flight_cap": getattr(settings, "DOCUMENT_MAX_IN_FLIGHT_PER_USER", 100),
        },
    )


def _safe_original_filename(filename: str, max_length: int = 255) -> str:
    """Normalize and cap client-provided file names for safe persistence/display.

    Thin wrapper over the shared ``core.files.safe_filename`` (with a
    document-flavoured fallback) so the documents and meetings apps can't drift.
    """
    return safe_filename(filename, fallback="document", max_length=max_length)


def _allowed_extension(filename: str) -> bool:
    ext = (filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return ext in getattr(settings, "DOCUMENT_ALLOWED_EXTENSIONS", {"pdf", "txt", "md", "html"})


def _allowed_mime(mime_type: str) -> bool:
    allowed_mime_types = getattr(settings, "DOCUMENT_ALLOWED_MIME_TYPES", None)
    # Empty/undefined allowlist means MIME checking is disabled.
    if not allowed_mime_types:
        return True
    return mime_type in allowed_mime_types


# Browsers send these for any type they don't recognize — they carry no signal,
# so they always pass the extension cross-check.
_GENERIC_MIME_TYPES = {"", "application/octet-stream"}


def _live_documents_qs(data_room):
    """Documents in *data_room* that count as "already uploaded".

    Archived documents don't count — re-dropping a file you archived is a
    deliberate way to bring it back. Neither do failed ones: re-uploading is the
    normal way to retry a document whose processing broke. Shared by the upload
    view and the pre-check endpoint so the two can't drift.
    """
    return DataRoomDocument.objects.filter(
        data_room=data_room, is_archived=False
    ).exclude(status=DataRoomDocument.Status.FAILED)


def _duplicate_in_data_room(data_room, sha: str):
    """Existing live document in *data_room* with identical bytes, or None."""
    if not sha:
        return None
    return _live_documents_qs(data_room).filter(content_sha256=sha).order_by("id").first()


def _mime_matches_extension(ext: str, mime_type: str) -> bool:
    """Cross-check the browser-supplied MIME type against the file extension.

    A mapped extension must carry one of its expected MIME types (or a generic
    one); unmapped extensions fall back to the global allowlist.
    """
    if mime_type in _GENERIC_MIME_TYPES:
        return True
    allowed_for_ext = getattr(settings, "DOCUMENT_EXTENSION_MIME_MAP", {}).get(ext)
    if allowed_for_ext is None:
        return _allowed_mime(mime_type)
    return mime_type in allowed_for_ext


@login_required
@require_POST
@ratelimit(key="user", rate="600/h", method="POST", block=True)
def document_upload(request, data_room_id):
    is_ajax = "application/json" in request.headers.get("Accept", "")

    # Reject oversized requests from the Content-Length header BEFORE accessing
    # request.FILES — once the body is parsed, Django has already spooled the
    # whole thing to disk and the per-file size checks come too late.
    try:
        content_length = int(request.META.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        content_length = 0
    max_request_bytes = getattr(settings, "DOCUMENT_UPLOAD_REQUEST_MAX_BYTES", 60_000_000)
    if content_length > max_request_bytes:
        msg = f"Upload too large (max {max_request_bytes / 1_000_000:.0f} MB per request)."
        if is_ajax:
            return JsonResponse({"status": "error", "error": msg}, status=413)
        messages.error(request, msg)
        return redirect("data_room_documents", data_room_id=data_room_id)

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

    from core.file_types import is_image_extension
    from llm.transcription_registry import AUDIO_EXTENSIONS

    max_size = getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 50_000_000)
    audio_max_size = getattr(settings, "AUDIO_UPLOAD_MAX_SIZE_BYTES", 50_000_000)
    errors = []
    created_docs = []
    # Files not uploaded because their exact bytes are already in this data room
    # (or appeared twice in this same request). Reported separately from errors —
    # a skip is a success, not a failure.
    skipped = []
    seen_shas = {}  # sha256 -> filename accepted earlier in this same request

    # In-flight cap: how many more this user may start right now, across all their
    # data rooms. "In flight" = non-terminal (uploaded/processing/scanning or
    # auto-retrying), which is what still consumes worker + Redis capacity. The
    # client mirrors this best-effort, but the server is authoritative (a non-JS
    # <form multiple> submit or the API bypasses the client entirely).
    from documents.services.pii_scan import SCAN_DISPATCH_RETRY_MESSAGE

    Status = DataRoomDocument.Status
    in_flight_cap = getattr(settings, "DOCUMENT_MAX_IN_FLIGHT_PER_USER", 100)
    in_flight = DataRoomDocument.objects.filter(uploaded_by=request.user).filter(
        Q(status__in=[Status.UPLOADED, Status.PROCESSING, Status.SCANNING])
        | Q(status=Status.SCAN_FAILED, processing_error=SCAN_DISPATCH_RETRY_MESSAGE)
    ).count()
    remaining_slots = max(0, in_flight_cap - in_flight)
    cap_hit = False
    IN_FLIGHT_CAP_MESSAGE = "You're uploading too many files — wait for some to finish before adding more."

    for file_obj in files:
        safe_filename = _safe_original_filename(file_obj.name, max_length=75)
        file_ext = (safe_filename.rsplit(".", 1)[-1].lower()) if "." in safe_filename else ""
        is_audio = file_ext in AUDIO_EXTENSIONS
        is_image = is_image_extension(file_ext)

        if file_obj.size <= 0:
            errors.append(f"{safe_filename}: file is empty.")
            continue
        # Audio gets its own cap; the generic document cap must not also apply,
        # or AUDIO_UPLOAD_MAX_SIZE_BYTES could never exceed the document cap.
        if is_audio and file_obj.size > audio_max_size:
            errors.append(f"{safe_filename}: audio file is too large (max {audio_max_size / 1_000_000:.0f} MB).")
            continue
        if not is_audio and file_obj.size > max_size:
            errors.append(f"{safe_filename}: file is too large (max {max_size / 1_000_000:.0f} MB).")
            continue
        if not _allowed_extension(safe_filename):
            errors.append(f"{safe_filename}: unsupported file type.")
            continue
        mime = getattr(file_obj, "content_type", "") or ""
        if not _mime_matches_extension(file_ext, mime):
            errors.append(f"{safe_filename}: file content doesn't match its extension.")
            continue
        if is_audio:
            from core.preferences import get_preferences
            prefs = get_preferences(request.user)
            if not prefs.allowed_transcription_models:
                errors.append(f"{safe_filename}: audio transcription is not enabled for your organization.")
                continue
        if is_image:
            from core.preferences import feature_is_available
            if not feature_is_available(request.user, "document_image_description"):
                errors.append(f"{safe_filename}: image uploads require a vision-capable model, which isn't enabled for your organization.")
                continue
        # Content-identity dedupe, before the cap check (a skipped file consumes
        # no slot) and before create (so the bytes never reach storage and no
        # processing task is enqueued). The client pre-checks hashes to avoid
        # sending duplicates at all, but this is what actually enforces it — a
        # plain <form> post never calls that endpoint.
        sha = sha256_of_upload(file_obj)
        if sha:
            if sha in seen_shas:
                skipped.append({"filename": safe_filename, "duplicate_of": seen_shas[sha]})
                continue
            dup = _duplicate_in_data_room(data_room, sha)
            if dup is not None:
                skipped.append({"filename": safe_filename, "duplicate_of": dup.display_name})
                continue

        # Fill the remaining in-flight slots, then stop (fill-slots overflow). One
        # generic message covers the rest — a per-file line for a large overflow
        # would just spam the error list. Only files that pass every other check
        # count against the cap; invalid files got their own error above.
        if len(created_docs) >= remaining_slots:
            cap_hit = True
            errors.append(IN_FLIGHT_CAP_MESSAGE)
            break
        stored_filename = _safe_original_filename(file_obj.name, max_length=180)
        file_obj.name = stored_filename
        doc = DataRoomDocument.objects.create(
            data_room=data_room,
            uploaded_by=request.user,
            original_file=file_obj,
            original_filename=safe_filename,
            mime_type=mime,
            size_bytes=file_obj.size,
            content_sha256=sha,
            status=DataRoomDocument.Status.UPLOADED,
        )
        created_docs.append(doc)
        if sha:
            seen_shas[sha] = safe_filename

    # Provenance is recorded by the v0 version's origin=uploaded (set when
    # process_document creates it); no separate "source" tag is needed.

    # Create v0 eagerly and put it in the dispatch queue (the document dispatch
    # gate hands at most DOCUMENT_WORKER_SLOTS versions to the worker at once —
    # see documents.services.dispatch). A broker blip no longer fails the
    # document: the row simply stays queued and the beat backstop dispatches it.
    # Only a genuine DB failure writing the queue row marks the document FAILED.
    from documents.services.dispatch import mark_version_queued, safe_dispatch
    from documents.services.process_document import ensure_initial_version

    queued_any = False
    for doc in created_docs:
        try:
            version = ensure_initial_version(doc)
            mark_version_queued(version.id)
            queued_any = True
        except Exception as exc:
            logger.exception("document_upload: failed to queue processing for document_id=%s", doc.id)
            doc.status = DataRoomDocument.Status.FAILED
            doc.processing_error = str(exc)[:2000]
            doc.save(update_fields=["status", "processing_error", "updated_at"])
            errors.append(f"{doc.original_filename}: processing could not be started.")
    if queued_any:
        safe_dispatch("document_upload")

    if is_ajax:
        if created_docs:
            doc = created_docs[0]
            return JsonResponse({
                "status": "ok",
                "document": {"id": doc.id, "filename": doc.original_filename, "status": doc.status},
                "skipped": skipped,
                "errors": errors,
            })
        if skipped:
            # Distinct from the error branch below: nothing was created, but
            # nothing went wrong either — the file is already in the room.
            return JsonResponse({"status": "skipped", "skipped": skipped, "errors": errors})
        if cap_hit:
            # Distinct from the 600/h rate-limit 429 (via `code`) so the client can
            # show the in-flight message and stop its sequential upload loop.
            return JsonResponse(
                {"status": "error", "code": "in_flight_cap", "error": IN_FLIGHT_CAP_MESSAGE},
                status=429,
            )
        return JsonResponse({"status": "error", "error": errors[0] if errors else "Upload failed."}, status=400)

    if created_docs:
        count = len(created_docs)
        messages.success(request, f"{count} file{'s' if count != 1 else ''} uploaded.")
    if skipped:
        count = len(skipped)
        messages.info(
            request,
            f"{count} file{'s were' if count != 1 else ' was'} already in this data room "
            f"and {'were' if count != 1 else 'was'} skipped.",
        )
    for err in errors:
        messages.error(request, err)
    return redirect("data_room_documents", data_room_id=data_room.uuid)


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_DUPLICATE_CHECK_MAX_HASHES = 200


@login_required
@require_POST
@ratelimit(key="user", rate="600/h", method="POST", block=True)
def document_duplicate_check(request, data_room_id):
    """Report which of the client's SHA-256 hashes are already in this data room.

    Lets the browser skip sending bytes it knows are duplicates — the win is
    avoiding the upload entirely on a large drop. Advisory only: document_upload
    re-hashes every file and re-checks, which is what actually enforces the skip
    (a plain <form> post never calls this endpoint).
    """
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    body, err = _parse_json_body(request)
    if err:
        return err

    hashes = body.get("hashes")
    if not isinstance(hashes, list) or not hashes:
        return JsonResponse({"error": "hashes must be a non-empty list"}, status=400)
    if len(hashes) > _DUPLICATE_CHECK_MAX_HASHES:
        return JsonResponse(
            {"error": f"hashes must contain at most {_DUPLICATE_CHECK_MAX_HASHES} entries"},
            status=400,
        )
    cleaned = []
    for h in hashes:
        if not isinstance(h, str) or not _SHA256_HEX_RE.match(h.strip().lower()):
            return JsonResponse({"error": "hashes must be hex-encoded SHA-256 digests"}, status=400)
        cleaned.append(h.strip().lower())

    rows = (
        _live_documents_qs(data_room)
        .filter(content_sha256__in=cleaned)
        .order_by("id")
        .values_list("content_sha256", "name", "original_filename")
    )
    duplicates = {}
    for sha, name, original_filename in rows:
        # First writer wins, matching _duplicate_in_data_room's order_by("id").
        duplicates.setdefault(sha, name or original_filename)
    return JsonResponse({"duplicates": duplicates})


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
    doc_ids, err = _parse_document_ids(body)
    if err:
        return err

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
    # Sets the mutable display name; original_filename (provenance) is preserved.
    from documents.services.versioning import rename_document
    rename_document(doc, name)
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
@require_POST
@ratelimit(key="user", rate="30/m", method="POST", block=True)
def document_rescan(request, data_room_id, document_id):
    """Re-run the PII scan for a document stuck in SCANNING or marked SCAN_FAILED."""
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    from documents.models import DataRoomDocumentVersion

    version = doc.current_version
    if version is None or version.status not in (
        DataRoomDocument.Status.SCANNING, DataRoomDocument.Status.SCAN_FAILED,
    ):
        return JsonResponse({"error": "This document is not waiting on a scan."}, status=409)
    # A manual retry is an explicit human override, so it restores the full
    # automatic-recovery budget. requeue_count bounds *automatic* re-dispatch of a
    # poison document; it is not a quota on deliberate user action. Without the
    # reset, a document that had already exhausted its retries would get a single
    # sweeper tick after the click before being marked terminal again.
    version.status = DataRoomDocument.Status.SCANNING
    version.processing_error = None
    version.requeue_count = 0
    version.save(update_fields=["status", "processing_error", "requeue_count", "updated_at"])
    # Mirror onto the document only when this version is the live/fresh one.
    if doc.active_searchable_version_id in (None, version.id):
        doc.status = DataRoomDocument.Status.SCANNING
        doc.processing_error = None
        doc.save(update_fields=["status", "processing_error", "updated_at"])
    try:
        # Re-run the full gate: the guardrail chunk scan first, which hands off to
        # finalize (the sole releaser). Dispatching finalize directly would release
        # the version with unscanned chunks, reopening the guardrail gap.
        from guardrails.tasks import scan_document_version

        scan_document_version.delay(version.id)
    except Exception:
        # A broker blip at dispatch is transient — mirror process_document_version
        # and leave the auto-retry marker so requeue_stale_documents re-dispatches
        # once the broker is reachable again. Writing the terminal
        # SCAN_FAILED_MESSAGE here would strand the document: the sweeper only
        # picks up versions carrying SCAN_DISPATCH_RETRY_MESSAGE, so a manual
        # retry during an outage would *disable* the very recovery it's meant to
        # trigger. requeue_count is deliberately not spent (same reasoning as the
        # sweeper's own re-dispatch): a busy broker is not the document's fault.
        from documents.services.pii_scan import (
            SCAN_DISPATCH_RETRY_MESSAGE,
            SCAN_RETRYING_STATUS,
        )

        logger.exception("document_rescan: failed to enqueue scan for version_id=%s", version.id)
        version.status = DataRoomDocument.Status.SCAN_FAILED
        version.processing_error = SCAN_DISPATCH_RETRY_MESSAGE
        version.save(update_fields=["status", "processing_error", "updated_at"])
        if doc.active_searchable_version_id in (None, version.id):
            doc.status = DataRoomDocument.Status.SCAN_FAILED
            doc.processing_error = SCAN_DISPATCH_RETRY_MESSAGE
            doc.save(update_fields=["status", "processing_error", "updated_at"])
        # 503, not 500: the scan is queued and will be retried, and it separates a
        # transient upstream outage from a genuine server error in metrics.
        return JsonResponse(
            {
                "error": SCAN_DISPATCH_RETRY_MESSAGE,
                "status": SCAN_RETRYING_STATUS,
            },
            status=503,
        )
    return JsonResponse({"status": "ok", "document": {"id": doc.id, "status": DataRoomDocument.Status.SCANNING}})


@login_required
@require_http_methods(["GET"])
def document_chunks(request, data_room_id, document_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    chunks = []
    # Only expose the released (scanned, non-quarantined) version. Do NOT fall
    # back to current_version: it advances immediately on upload, so a version
    # still being scanned would otherwise leak un-screened chunks. A never-scanned
    # doc therefore returns [] until its first version reaches READY.
    version = doc.active_searchable_version
    version_chunks = version.chunks.filter(is_quarantined=False).order_by("chunk_index") if version else []
    for c in version_chunks:
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
    doc_ids, err = _parse_document_ids(body)
    if err:
        return err
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
    doc_ids, err = _parse_document_ids(body)
    if err:
        return err
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
# The PDF preview is a same-origin <iframe> of this URL (the CSP's object-src
# 'none' rules out <embed>). The framed response must therefore allow same-origin
# framing: site-wide it's X-Frame-Options DENY + frame-ancestors 'none'.
@xframe_options_sameorigin
@csp_replace({"frame-ancestors": [SELF]})
def document_file(request, data_room_id, document_id):
    """Stream a document's native file so a modal can preview/download it.

    Images and PDFs are served inline (for <img>/<iframe>); everything else — and
    ``?download=1`` — is forced to download. Bytes stream through this view (no
    presigned URL). Resolves the document's newest native file, matching the
    ``[[file:]]`` download the agent tools offer.
    """
    from django.http import FileResponse, Http404

    from chat.assets import latest_native_file
    from core.file_types import KIND_IMAGE, KIND_PDF, kind_for_mime

    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        raise Http404
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    source, filename, ct = latest_native_file(doc)
    if source is None:
        raise Http404
    displayable = kind_for_mime(ct) in (KIND_IMAGE, KIND_PDF)
    inline = displayable and request.GET.get("download") != "1"
    try:
        # as_attachment/filename → Django builds the Content-Disposition header
        # (RFC 6266): quotes/backslashes escaped, non-ASCII names (æøå) emitted
        # as filename*=utf-8''… instead of a MIME-encoded header the browser
        # would render as garbage.
        resp = FileResponse(
            source.open("rb"),
            content_type=ct if inline else "application/octet-stream",
            as_attachment=not inline,
            filename=filename or f"document-{doc.pk}",
        )
    except Exception as exc:  # noqa: BLE001 — a row can outlive its blob
        logger.warning(
            "document %s native blob unreadable (%s)", doc.pk, type(exc).__name__
        )
        raise Http404
    resp["X-Content-Type-Options"] = "nosniff"
    return resp


@login_required
@require_http_methods(["GET"])
def document_edit_source(request, data_room_id, document_id):
    """Return the editable working markdown for a TEXT document (Edit mode).

    Reads the working version directly via ``open_working_version`` so it works
    even for a quarantined doc the user is remediating. Text docs only.
    """
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    from core.file_types import KIND_TEXT

    if _document_file_kind(doc) != KIND_TEXT:
        return JsonResponse({"ok": False, "error": "not_editable"}, status=400)
    from documents.services.versioning import open_working_version

    content, _version, warning = open_working_version(doc)
    max_chars = _browser_edit_max_chars()
    if len(content or "") > max_chars:
        # Beyond what the editor + a form POST can carry (see
        # DOCUMENT_BROWSER_EDIT_MAX_CHARS). The modal falls back to read-only.
        return JsonResponse({
            "ok": True,
            "editable": False,
            "reason": "too_large",
            "max_chars": max_chars,
            "is_quarantined": bool(doc.is_quarantined),
        })
    return JsonResponse({
        "ok": True,
        "editable": True,
        "content": content,
        "sha256": _content_sha(content),
        "warning": warning,
        "is_quarantined": bool(doc.is_quarantined),
        # Above this a save is queued for the worker instead of re-indexed inline;
        # the modal tells the user to expect a wait.
        "inline_max_chars": _inline_edit_max_chars(),
    })


@login_required
@require_POST
def document_save(request, data_room_id, document_id):
    """Save edited markdown for a TEXT document and re-index it.

    If the submitted content is unchanged from the working version this is a
    no-op (no new version, no re-scan) — so re-opening and saving a quarantined
    doc without edits does not re-run the scan.

    A changed save creates a new version and then, by size:

    - up to ``DOCUMENT_INLINE_EDIT_MAX_CHARS``: runs the chunk→embed→scan pipeline
      INLINE and returns the verdict (a clean edit of a quarantined doc thereby
      clears the quarantine). If that inline scan fails transiently (or raises)
      the same version is re-queued through the async gate instead of failing
      (``_requeue_after_sync_failure``);
    - above it: the version joins the document dispatch gate like an upload and
      the response is ``verdict: "queued"`` with a ``verdict_url`` the modal polls
      (``document_version_verdict``).
    """
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    from core.file_types import KIND_TEXT

    if _document_file_kind(doc) != KIND_TEXT:
        return JsonResponse({"ok": False, "error": "not_editable"}, status=400)
    new_content = request.POST.get("content")
    if new_content is None:
        return JsonResponse({"ok": False, "error": "content_required"}, status=400)
    browser_max = _browser_edit_max_chars()
    if len(new_content) > browser_max:
        return JsonResponse(
            {"ok": False, "error": "too_large", "max_chars": browser_max}, status=400
        )

    from documents.models import DataRoomDocumentVersion
    from documents.services.sync_scan import scan_version_synchronously
    from documents.services.versioning import create_version, open_working_version

    current, _version, _warning = open_working_version(doc)
    if new_content.strip() == (current or "").strip():
        return JsonResponse({
            "ok": True,
            "unchanged": True,
            "verdict": "quarantined" if doc.is_quarantined else "clean",
            "is_quarantined": bool(doc.is_quarantined),
        })

    if len(new_content) > _inline_edit_max_chars():
        # Too big to re-index inside the router timeout: queue it like an upload
        # (mark_version_queued + safe_dispatch via _enqueue_processing).
        version = create_version(
            doc,
            content=new_content,
            origin=DataRoomDocumentVersion.Origin.USER_EDITED,
            created_by=request.user,
            enqueue=True,
        )
        return JsonResponse(_queued_payload(data_room, doc, version))

    version = create_version(
        doc,
        content=new_content,
        origin=DataRoomDocumentVersion.Origin.USER_EDITED,
        created_by=request.user,
        enqueue=False,
    )
    try:
        verdict = scan_version_synchronously(version.id)
    except Exception:
        logger.exception("document_save: inline scan raised for version %s", version.id)
        verdict = None
    if verdict is None or verdict.status == "scan_failed":
        if _requeue_after_sync_failure(doc, version.id):
            return JsonResponse(_queued_payload(data_room, doc, version))
        if verdict is None:
            doc.refresh_from_db()
            return JsonResponse({
                "ok": False,
                "verdict": "scan_failed",
                "reason": "The safety scan could not complete. Try saving again.",
                "reviewer_finding": "",
                "unchanged": False,
                "is_quarantined": bool(doc.is_quarantined),
            })
    doc.refresh_from_db()
    payload = verdict.to_http_json()
    payload["unchanged"] = False
    payload["is_quarantined"] = bool(doc.is_quarantined)
    return JsonResponse(payload)


@login_required
@require_http_methods(["GET"])
def document_version_verdict(request, data_room_id, document_id, version_id):
    """Poll target for a queued edit (``document_save`` → ``verdict: "queued"``).

    ``pending: true`` (with the row's presentation status) while the async pipeline
    is still working — including a SCAN_FAILED that carries the dispatch-retry
    marker, which the worker retries on its own. Otherwise the same verdict shape
    an inline save returns, so the modal handles both identically.
    """
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_modify_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(DataRoomDocument, pk=document_id, data_room=data_room)
    from documents.models import DataRoomDocumentVersion
    from documents.services.pii_scan import SCAN_DISPATCH_RETRY_MESSAGE
    from documents.services.sync_scan import verdict_for_version

    version = get_object_or_404(DataRoomDocumentVersion, pk=version_id, document=doc)
    Status = DataRoomDocument.Status
    pending = version.status in (Status.UPLOADED, Status.PROCESSING, Status.SCANNING) or (
        version.status == Status.SCAN_FAILED
        and version.processing_error == SCAN_DISPATCH_RETRY_MESSAGE
    )
    if pending:
        return JsonResponse({
            "ok": True,
            "pending": True,
            "status": DataRoomDocument.presentation_status(
                version.status,
                version.processing_error,
                waiting=bool(version.queued_at and not version.dispatched_at),
            ),
        })
    payload = verdict_for_version(version.id).to_http_json()
    payload["pending"] = False
    payload["unchanged"] = False
    payload["is_quarantined"] = bool(doc.is_quarantined)
    return JsonResponse(payload)


@login_required
@require_http_methods(["GET"])
def document_status(request, data_room_id):
    data_room = get_object_or_404(DataRoom, uuid=data_room_id)
    if not _user_can_access_data_room(request.user, data_room):
        return JsonResponse({"error": "Forbidden"}, status=403)
    statuses = {
        str(pk): DataRoomDocument.presentation_status(
            status, err, waiting=bool(q_at and not d_at),
        )
        for pk, status, err, q_at, d_at in data_room.documents.filter(
            is_archived=False,
        ).values_list(
            "id", "status", "processing_error",
            "current_version__queued_at", "current_version__dispatched_at",
        )
    }
    return JsonResponse({"statuses": statuses})


@login_required
@require_POST
@ratelimit(key="user", rate="10/m", method="POST", block=True)
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
