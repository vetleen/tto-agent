from __future__ import annotations

from datetime import timedelta

RETENTION_PERIODS = {
    "chat.ChatThread": timedelta(days=365),
    "documents.DataRoom": timedelta(days=365),
    "meetings.Meeting": timedelta(days=90),
    "guardrails.GuardrailEvent": timedelta(days=180),
    "feedback.Feedback": timedelta(days=90),
    "accounts.EmailVerificationToken": timedelta(days=1),
}
