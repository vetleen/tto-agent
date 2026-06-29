"""Single source of truth for which data rooms a user may access.

Today access is purely ownership (``created_by``). Keeping the rule in one place
means a future change (e.g. organization-shared rooms) updates every call site —
``documents.views._user_can_access_data_room`` and ``chat.loop_service`` both
defer to this — instead of each re-deriving it.
"""

from __future__ import annotations

from django.db.models import QuerySet

from documents.models import DataRoom


def accessible_data_rooms(user) -> QuerySet:
    """Return the queryset of data rooms ``user`` may access."""
    return DataRoom.objects.filter(created_by=user)
