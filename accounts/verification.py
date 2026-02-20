"""
Email verification: token creation, sending, verification, and resend rate limit.
"""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils import timezone

from .models import EmailVerificationToken, User

if TYPE_CHECKING:
    from django.http import HttpRequest

VERIFICATION_WINDOW_HOURS = 24
VERIFICATION_TOKEN_VALID_HOURS = 24


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def create_token(user: User) -> EmailVerificationToken:
    """Create a new verification token for the user. Old tokens are not deleted."""
    return EmailVerificationToken.objects.create(
        user=user,
        token=_generate_token(),
    )


def get_verification_link(request: HttpRequest, token: str) -> str:
    """Build absolute verification URL using the request."""
    from django.urls import reverse

    path = reverse("accounts:verify_email", kwargs={"token": token})
    return request.build_absolute_uri(path)


def send_verification_email(
    request: HttpRequest, user: User, *, is_resend: bool = False
) -> EmailVerificationToken:
    """
    Create a token, send verification email, and update user's resend tracking.
    At signup use is_resend=False (resend_count not incremented). On resend use is_resend=True.
    """
    token_obj = create_token(user)
    link = get_verification_link(request, token_obj.token)

    subject = render_to_string(
        "registration/email_verification_subject.txt",
        {"user": user},
    ).strip()
    body = render_to_string(
        "registration/email_verification_body.txt",
        {"user": user, "link": link},
    )

    send_mail(
        subject=subject,
        message=body,
        from_email=settings.DEFAULT_FROM_EMAIL or None,
        recipient_list=[user.email],
        fail_silently=False,
    )

    now = timezone.now()
    user.last_verification_email_sent_at = now
    if user.verification_resend_window_start is None:
        user.verification_resend_window_start = now
    if is_resend:
        user.verification_resend_count += 1
    update_fields = [
        "last_verification_email_sent_at",
        "verification_resend_window_start",
    ]
    if is_resend:
        update_fields.append("verification_resend_count")
    user.save(update_fields=update_fields)
    return token_obj


def verify_token(token: str) -> tuple[User | None, str | None]:
    """
    Validate token and mark user verified. Returns (user, None) on success,
    (None, error_message) on failure. Deletes the token on success.
    """
    try:
        token_obj = EmailVerificationToken.objects.select_related("user").get(
            token=token
        )
    except EmailVerificationToken.DoesNotExist:
        return None, "invalid"

    user = token_obj.user
    timeout_seconds = getattr(
        settings, "EMAIL_VERIFICATION_TIMEOUT", 86400
    )
    if (timezone.now() - token_obj.created_at).total_seconds() > timeout_seconds:
        token_obj.delete()
        return None, "expired"

    user.email_verified = True
    user.save(update_fields=["email_verified"])
    token_obj.delete()
    return user, None


def can_resend_verification(user: User) -> tuple[bool, int | None]:
    """
    Check if a verification email can be sent. Returns (allowed, wait_seconds).
    If allowed is True, wait_seconds is None. If allowed is False, wait_seconds
    is how long to wait (can be 0 if e.g. window reset).
    """
    now = timezone.now()
    window_start = user.verification_resend_window_start
    last_sent = user.last_verification_email_sent_at

    # Reset window after 24 hours
    if window_start is not None:
        if (now - window_start).total_seconds() >= VERIFICATION_WINDOW_HOURS * 3600:
            user.verification_resend_count = 0
            user.verification_resend_window_start = now
            user.save(update_fields=["verification_resend_count", "verification_resend_window_start"])
            # After reset, first resend is allowed immediately (1 min was the initial wait from signup)
            return True, None

    if last_sent is None:
        return True, None

    wait_minutes = 2 ** user.verification_resend_count
    next_allowed = last_sent + timedelta(minutes=wait_minutes)
    if now >= next_allowed:
        return True, None
    wait_seconds = int((next_allowed - now).total_seconds())
    return False, wait_seconds
