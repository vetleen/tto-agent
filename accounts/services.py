"""Concurrency-safe writers for the JSON ``preferences`` fields.

Every settings endpoint used to do load -> mutate dict -> save, so two
concurrent POSTs (the settings UI fires one fetch() per toggle) could
interleave and silently drop one write. These helpers serialize writers of
the same row with ``select_for_update`` so each mutation is applied to the
latest committed state.

Scope: they fix *lost updates* only. Validation that happens before the
helper is called (e.g. "is this model in the allowed list") still reads a
snapshot — that validate-then-write gap is unchanged and accepted; the
stored value is itself re-validated wherever it is consumed.

Note: ``select_for_update`` is a silent no-op on SQLite (the test backend),
so real locking exists only on Postgres. Tests exercise the code path but
cannot catch locking regressions.
"""
from __future__ import annotations

from typing import Callable

from django.db import transaction

from .models import Organization, UserSettings


def update_user_preferences(user, mutate: Callable[[dict], None]) -> dict:
    """Atomically apply ``mutate`` to ``user``'s UserSettings.preferences.

    ``mutate`` receives the preferences dict and modifies it in place.
    Returns the saved preferences dict.
    """
    with transaction.atomic():
        # PK is user_id, so a concurrent create resolves inside get_or_create
        # (IntegrityError retry); the lock serializes the mutate+save.
        obj, _ = UserSettings.objects.select_for_update().get_or_create(user=user)
        prefs = obj.preferences or {}
        mutate(prefs)
        obj.preferences = prefs
        obj.save(update_fields=["preferences"])
    return prefs


def update_org_preferences(org_id: int, mutate: Callable[[dict], None]) -> dict:
    """Atomically apply ``mutate`` to the Organization's preferences.

    ``mutate`` receives the preferences dict and modifies it in place.
    Returns the saved preferences dict.
    """
    with transaction.atomic():
        org = Organization.objects.select_for_update().get(pk=org_id)
        prefs = org.preferences or {}
        mutate(prefs)
        org.preferences = prefs
        org.save(update_fields=["preferences"])
    return prefs
