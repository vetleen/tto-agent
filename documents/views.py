import logging
import ntpath
import os

from django.conf import settings
from django.db import IntegrityError
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods, require_POST

from .models import Project, ProjectDocument, ProjectDocumentChunk


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


def _user_owns_project(user, project: Project) -> bool:
    return project.created_by_id == user.id


@login_required
@require_http_methods(["GET", "POST"])
def project_list(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if name:
            base_slug = slugify(name) or "project"
            n = 0
            project = None
            while True:
                slug = base_slug if n == 0 else f"{base_slug}-{n}"
                try:
                    project = Project.objects.create(name=name, slug=slug, created_by=request.user)
                    break
                except IntegrityError:
                    # Handle concurrent creates picking the same slug.
                    n += 1
                    if n > 50:
                        messages.error(request, "Could not create project right now. Please try again.")
                        break
            if project:
                return redirect("project_chat", project_id=project.uuid)
        return redirect("project_list")
    projects = Project.objects.filter(created_by=request.user, is_archived=False).order_by("-updated_at")
    archived_projects = Project.objects.filter(created_by=request.user, is_archived=True).order_by("-updated_at")
    return render(request, "documents/project_list.html", {
        "projects": projects,
        "archived_projects": archived_projects,
    })


@login_required
@require_POST
def project_delete(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    project.delete()
    messages.success(request, "Project deleted.")
    return redirect("project_list")


@login_required
@require_http_methods(["GET", "POST"])
def project_rename(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    if request.method != "POST":
        return redirect("project_list")
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Project name cannot be empty.")
        return redirect("project_list")
    if len(name) > 255:
        name = name[:255]
    project.name = name
    project.save(update_fields=["name", "updated_at"])
    messages.success(request, "Project renamed.")
    return redirect("project_list")


@login_required
@require_POST
def project_archive(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    project.is_archived = not project.is_archived
    project.save(update_fields=["is_archived", "updated_at"])
    label = "archived" if project.is_archived else "restored"
    messages.success(request, f"Project {label}.")
    return redirect("project_list")


@login_required
@require_http_methods(["GET"])
def project_detail_redirect(request, project_id):
    """Redirect /projects/<uuid>/ to /projects/<uuid>/chat/."""
    return redirect("project_chat", project_id=project_id)


@login_required
@require_http_methods(["GET"])
def project_chat(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")

    from chat.models import ChatThread

    threads = list(
        ChatThread.objects.filter(project=project, created_by=request.user)
        .order_by("-updated_at")
    )

    thread = None
    chat_messages = []

    if request.GET.get("thread"):
        # Load a specific thread by UUID
        thread_id = request.GET["thread"]
        thread = ChatThread.objects.filter(
            id=thread_id, project=project, created_by=request.user
        ).first()
        if thread:
            chat_messages = list(thread.messages.order_by("created_at")[:100])

    return render(
        request,
        "documents/project_chat.html",
        {
            "project": project,
            "active_tab": "chat",
            "thread": thread,
            "threads": threads,
            "messages": chat_messages,
        },
    )


@login_required
@require_http_methods(["GET"])
def project_documents(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    documents = list(
        project.documents.exclude(status=ProjectDocument.Status.FAILED)
        .filter(is_archived=False).order_by("-uploaded_at")
    )
    for doc in documents:
        doc.relative_upload_display = _relative_upload_date(doc.uploaded_at)
    archived_documents = list(
        project.documents.exclude(status=ProjectDocument.Status.FAILED)
        .filter(is_archived=True).order_by("-uploaded_at")
    )
    for doc in archived_documents:
        doc.relative_upload_display = _relative_upload_date(doc.uploaded_at)
    return render(
        request,
        "documents/project_documents.html",
        {
            "project": project,
            "documents": documents,
            "archived_documents": archived_documents,
            "active_tab": "documents",
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
def document_upload(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    files = request.FILES.getlist("file")
    if not files:
        messages.error(request, "No file selected. Please choose a file to upload.")
        return redirect("project_documents", project_id=project.uuid)

    max_size = getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 10_000_000)
    errors = []
    created_docs = []

    for file_obj in files:
        safe_filename = _safe_original_filename(file_obj.name, max_length=75)
        if file_obj.size <= 0:
            errors.append(f"{safe_filename}: file is empty.")
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
        stored_filename = _safe_original_filename(file_obj.name, max_length=180)
        file_obj.name = stored_filename
        doc = ProjectDocument.objects.create(
            project=project,
            uploaded_by=request.user,
            original_file=file_obj,
            original_filename=safe_filename,
            mime_type=mime,
            size_bytes=file_obj.size,
            status=ProjectDocument.Status.UPLOADED,
        )
        created_docs.append(doc)

    for doc in created_docs:
        try:
            from documents.tasks import process_document_task

            process_document_task.delay(doc.id)
        except ImportError:
            from documents.services.process_document import process_document

            process_document(doc.id)
        except Exception as exc:
            logger.exception("document_upload: failed to enqueue processing for document_id=%s", doc.id)
            doc.status = ProjectDocument.Status.FAILED
            doc.processing_error = str(exc)[:2000]
            doc.save(update_fields=["status", "processing_error", "updated_at"])
            errors.append(f"{doc.original_filename}: processing could not be started.")

    if created_docs:
        count = len(created_docs)
        messages.success(request, f"{count} file{'s' if count != 1 else ''} uploaded successfully.")
    for err in errors:
        messages.error(request, err)
    return redirect("project_documents", project_id=project.uuid)


@login_required
@require_POST
def document_delete(request, project_id, document_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    doc = get_object_or_404(ProjectDocument, pk=document_id, project=project)
    doc.delete()
    messages.success(request, "Document deleted.")
    return redirect("project_documents", project_id=project.uuid)


@login_required
@require_http_methods(["POST"])
def document_rename(request, project_id, document_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    doc = get_object_or_404(ProjectDocument, pk=document_id, project=project)
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Document name cannot be empty.")
        return redirect("project_documents", project_id=project.uuid)
    doc.original_filename = _safe_original_filename(name, max_length=75)
    doc.save(update_fields=["original_filename", "updated_at"])
    messages.success(request, "Document renamed.")
    return redirect("project_documents", project_id=project.uuid)


@login_required
@require_POST
def document_archive(request, project_id, document_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    doc = get_object_or_404(ProjectDocument, pk=document_id, project=project)
    doc.is_archived = not doc.is_archived
    doc.save(update_fields=["is_archived", "updated_at"])
    label = "archived" if doc.is_archived else "restored"
    messages.success(request, f"Document {label}.")
    return redirect("project_documents", project_id=project.uuid)


@login_required
@require_http_methods(["GET"])
def document_chunks(request, project_id, document_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return JsonResponse({"error": "Forbidden"}, status=403)
    doc = get_object_or_404(ProjectDocument, pk=document_id, project=project)
    chunks = []
    for c in doc.chunks.order_by("chunk_index"):
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
