import logging

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django_ratelimit.decorators import ratelimit

from .models import Feedback
from .tasks import notify_admin_feedback_task
from .validation import (
    clean_feedback_url,
    reencode_screenshot,
    sanitize_console_errors,
)

logger = logging.getLogger(__name__)

MAX_SCREENSHOT_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TEXT_LENGTH = 5000
# ~5 MB screenshot + text + console errors, with headroom for multipart framing.
FEEDBACK_UPLOAD_REQUEST_MAX_BYTES = 7_000_000


@login_required
@require_POST
@ratelimit(key="user", rate="10/h", method="POST", block=True)
def submit_feedback(request):
    # Reject oversized requests from the Content-Length header BEFORE touching
    # request.POST/request.FILES — once the body is parsed, Django has already
    # spooled the whole thing to disk and the per-field checks come too late.
    try:
        content_length = int(request.META.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        content_length = 0
    max_request_bytes = getattr(
        settings, "FEEDBACK_UPLOAD_REQUEST_MAX_BYTES", FEEDBACK_UPLOAD_REQUEST_MAX_BYTES
    )
    if content_length > max_request_bytes:
        return JsonResponse({"error": "Request too large."}, status=413)

    text = (request.POST.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "Feedback text is required."}, status=400)
    if len(text) > MAX_TEXT_LENGTH:
        return JsonResponse({"error": "That's a bit long — please keep your feedback under 5,000 characters."}, status=400)

    url = clean_feedback_url(request.POST.get("url"))
    user_agent = (request.POST.get("user_agent") or "")[:1000]
    viewport = (request.POST.get("viewport") or "")[:50]

    console_errors = sanitize_console_errors(request.POST.get("console_errors", ""))

    feedback = Feedback(
        user=request.user,
        url=url,
        user_agent=user_agent,
        viewport=viewport,
        text=text,
        console_errors=console_errors,
    )

    # The screenshot is captured automatically on the client, so a too-large or
    # unsupported image isn't the user's fault. Drop it quietly and still save the
    # feedback rather than blocking the submission over it. We re-encode rather
    # than trusting the client's content_type/filename — that validates the bytes
    # are a real image and strips any embedded payload.
    screenshot_file = request.FILES.get("screenshot")
    if screenshot_file:
        result = None
        if screenshot_file.size <= MAX_SCREENSHOT_SIZE:
            result = reencode_screenshot(screenshot_file)
        if result:
            name, content = result
            feedback.screenshot.save(name, content, save=False)
        else:
            logger.info(
                "Dropping feedback screenshot from user %d (size=%d, type=%s)",
                request.user.pk,
                screenshot_file.size,
                screenshot_file.content_type,
            )

    feedback.save()
    logger.info("Feedback #%d submitted by user %d", feedback.pk, request.user.pk)

    # Notify off the request path — a slow SMTP call must not stall the user,
    # and a broker outage must not fail the submission.
    try:
        notify_admin_feedback_task.delay(feedback.pk)
    except Exception:
        logger.exception(
            "Failed to enqueue admin notification for feedback #%d", feedback.pk
        )

    return JsonResponse({"ok": True})
