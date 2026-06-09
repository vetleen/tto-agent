import json
import logging

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import Feedback

logger = logging.getLogger(__name__)

MAX_SCREENSHOT_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TEXT_LENGTH = 5000
MAX_CONSOLE_ERRORS = 50


@login_required
@require_POST
def submit_feedback(request):
    text = (request.POST.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "Feedback text is required."}, status=400)
    if len(text) > MAX_TEXT_LENGTH:
        return JsonResponse({"error": "That's a bit long — please keep your feedback under 5,000 characters."}, status=400)

    url = (request.POST.get("url") or "")[:2000]
    user_agent = (request.POST.get("user_agent") or "")[:1000]
    viewport = (request.POST.get("viewport") or "")[:50]

    console_errors_raw = request.POST.get("console_errors", "[]")
    try:
        console_errors = json.loads(console_errors_raw)
        if not isinstance(console_errors, list):
            console_errors = []
        console_errors = console_errors[:MAX_CONSOLE_ERRORS]
    except (json.JSONDecodeError, ValueError):
        console_errors = []

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
    # feedback rather than blocking the submission over it.
    screenshot_file = request.FILES.get("screenshot")
    if screenshot_file:
        valid_type = screenshot_file.content_type in ("image/jpeg", "image/png", "image/webp")
        if screenshot_file.size <= MAX_SCREENSHOT_SIZE and valid_type:
            feedback.screenshot = screenshot_file
        else:
            logger.info(
                "Dropping feedback screenshot from user %d (size=%d, type=%s)",
                request.user.pk,
                screenshot_file.size,
                screenshot_file.content_type,
            )

    feedback.save()
    logger.info("Feedback #%d submitted by user %d", feedback.pk, request.user.pk)

    _notify_admin(feedback)

    return JsonResponse({"ok": True})


def _notify_admin(feedback):
    """Send email notification to ADMINS if email is enabled."""
    if not getattr(settings, "EMAIL_SENDING_ENABLED", False):
        return
    admins = getattr(settings, "ADMINS", [])
    if not admins:
        return
    try:
        from django.core.mail import send_mail

        subject = f"New feedback #{feedback.pk} from {feedback.user}"
        body = (
            f"User: {feedback.user}\n"
            f"URL: {feedback.url}\n"
            f"Text: {feedback.text}\n"
            f"Time: {feedback.created_at}\n"
        )
        send_mail(
            subject=subject,
            message=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email for _, email in admins],
            fail_silently=True,
        )
    except Exception:
        logger.exception("Failed to send feedback notification email")
