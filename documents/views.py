from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods, require_POST

from .models import Project, ProjectDocument, ProjectDocumentChunk


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
            slug = base_slug
            n = 0
            while Project.objects.filter(slug=slug).exists():
                n += 1
                slug = f"{base_slug}-{n}"
            Project.objects.create(name=name, slug=slug, created_by=request.user)
        return redirect("project_list")
    projects = Project.objects.filter(created_by=request.user).order_by("-updated_at")
    return render(request, "documents/project_list.html", {"projects": projects})


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
@require_http_methods(["GET"])
def project_detail(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    documents = list(project.documents.all().order_by("-uploaded_at"))
    for doc in documents:
        doc.relative_upload_display = _relative_upload_date(doc.uploaded_at)
    return render(
        request,
        "documents/project_detail.html",
        {"project": project, "documents": documents},
    )


def _allowed_extension(filename: str) -> bool:
    ext = (filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return ext in getattr(settings, "DOCUMENT_ALLOWED_EXTENSIONS", {"pdf", "txt", "md", "html"})


def _allowed_mime(mime_type: str) -> bool:
    return mime_type in getattr(settings, "DOCUMENT_ALLOWED_MIME_TYPES", frozenset())


@login_required
@require_POST
def document_upload(request, project_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    file_obj = request.FILES.get("file")
    if not file_obj:
        messages.error(request, "No file selected. Please choose a file to upload.")
        return redirect("project_detail", project_id=project.uuid)
    max_size = getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 10_000_000)
    if file_obj.size > max_size:
        messages.error(request, "File is too large. Maximum size is 10 MB.")
        return redirect("project_detail", project_id=project.uuid)
    if not _allowed_extension(file_obj.name):
        messages.error(request, "Unsupported file type. Allowed: PDF, TXT, MD, HTML.")
        return redirect("project_detail", project_id=project.uuid)
    mime = getattr(file_obj, "content_type", "") or ""
    if mime and not _allowed_mime(mime):
        messages.error(request, "Unsupported file type. Allowed: PDF, TXT, MD, HTML.")
        return redirect("project_detail", project_id=project.uuid)
    doc = ProjectDocument.objects.create(
        project=project,
        uploaded_by=request.user,
        original_file=file_obj,
        original_filename=file_obj.name,
        mime_type=mime,
        size_bytes=file_obj.size,
        status=ProjectDocument.Status.UPLOADED,
    )
    try:
        from documents.tasks import process_document_task
        process_document_task.delay(doc.id)
    except ImportError:
        from documents.services.process_document import process_document
        process_document(doc.id)
    return redirect("project_detail", project_id=project.uuid)


@login_required
@require_POST
def document_delete(request, project_id, document_id):
    project = get_object_or_404(Project, uuid=project_id)
    if not _user_owns_project(request.user, project):
        return redirect("project_list")
    doc = get_object_or_404(ProjectDocument, pk=document_id, project=project)
    doc.delete()
    messages.success(request, "Document deleted.")
    return redirect("project_detail", project_id=project.uuid)


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
