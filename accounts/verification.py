"""
Email verification: token creation, sending, verification, and resend rate limit.
"""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.mail import send_mail
from django.db import transaction
from django.template.loader import render_to_string
from django.utils import timezone

from .models import EmailVerificationToken, User

if TYPE_CHECKING:
    from django.http import HttpRequest

VERIFICATION_WINDOW_HOURS = 24


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def create_token(user: User) -> EmailVerificationToken:
    """Create a new verification token for the user. Old tokens are deleted first."""
    EmailVerificationToken.objects.filter(user=user).delete()
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
) -> EmailVerificationToken | None:
    """
    Create a token, update the user's resend tracking, then send the email.
    At signup use is_resend=False (resend_count not incremented). On resend use
    is_resend=True.

    Token creation and bookkeeping run atomically under a row lock, with the
    rate limit re-checked inside it, so two concurrent resends can't both pass
    can_resend_verification(); the loser returns None without sending. The
    email goes out only after the counter is committed — a crash or mail
    failure can cost the user a resend slot but never grants a free one
    (deliberate fail-closed).
    """
    with transaction.atomic():
        # Fresh locked row; the caller's instance may be stale.
        locked_user = User.objects.select_for_update().get(pk=user.pk)
        if is_resend:
            allowed, _wait = can_resend_verification(locked_user)
            if not allowed:
                return None

        token_obj = create_token(locked_user)

        now = timezone.now()
        locked_user.last_verification_email_sent_at = now
        if locked_user.verification_resend_window_start is None:
            locked_user.verification_resend_window_start = now
        # Reset the rate-limit window if 24 hours have passed
        if is_resend and _is_window_expired(locked_user):
            locked_user.verification_resend_count = 0
            locked_user.verification_resend_window_start = now
        if is_resend:
            locked_user.verification_resend_count += 1
        locked_user.save(update_fields=[
            "last_verification_email_sent_at",
            "verification_resend_window_start",
            "verification_resend_count",
        ])

    # Keep the caller's instance in sync with what was persisted.
    user.last_verification_email_sent_at = locked_user.last_verification_email_sent_at
    user.verification_resend_window_start = locked_user.verification_resend_window_start
    user.verification_resend_count = locked_user.verification_resend_count

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
    return token_obj


def verify_token(token: str) -> tuple[User | None, str | None]:
    """
    Validate token and mark user verified. Returns (user, None) on success,
    (None, error_message) on failure. Deletes the token on success.

    Runs atomically with the token row locked so a concurrent verify (or a
    resend deleting/replacing the token mid-flight) can't double-fire.
    """
    with transaction.atomic():
        try:
            # FOR UPDATE locks the joined user row too on Postgres (intended).
            token_obj = (
                EmailVerificationToken.objects.select_for_update()
                .select_related("user")
                .get(token=token)
            )
        except EmailVerificationToken.DoesNotExist:
            return None, "invalid"

        user = token_obj.user
        # Deactivated accounts must never be re-verified — verify_email() logs the
        # returned user in directly, bypassing the auth form's is_active gate. The
        # token is deliberately kept (still time-limited) so a legitimate
        # reactivation within the window doesn't strand the user.
        if not user.is_active:
            return None, "invalid"

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


def _is_window_expired(user: User) -> bool:
    """Check if the 24-hour resend window has expired (read-only)."""
    window_start = user.verification_resend_window_start
    if window_start is None:
        return False
    return (timezone.now() - window_start).total_seconds() >= VERIFICATION_WINDOW_HOURS * 3600


def can_resend_verification(user: User) -> tuple[bool, int | None]:
    """
    Check if a verification email can be sent. Returns (allowed, wait_seconds).
    If allowed is True, wait_seconds is None. If allowed is False, wait_seconds
    is how long to wait. This is a read-only check — no database writes.
    """
    now = timezone.now()
    last_sent = user.last_verification_email_sent_at

    # Window expired — allow immediately (reset happens in send_verification_email)
    if _is_window_expired(user):
        return True, None

    if last_sent is None:
        return True, None

    wait_minutes = 2 ** user.verification_resend_count
    next_allowed = last_sent + timedelta(minutes=wait_minutes)
    if now >= next_allowed:
        return True, None
    wait_seconds = int((next_allowed - now).total_seconds())
    return False, wait_seconds
