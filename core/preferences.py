"""Cascading preferences resolver: System -> Organization -> User."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.conf import settings as django_settings


@dataclass
class ResolvedPreferences:
    primary_model: str
    mid_model: str
    cheap_model: str
    allowed_models: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    theme: str = "light"


def get_preferences(user) -> ResolvedPreferences:
    """Resolve cascading preferences for a user: System -> Org -> User.

    Returns ResolvedPreferences with the effective model for each tier,
    the allowed models list, and the allowed tools list.
    """
    from llm.service.policies import get_allowed_models
    from llm.tools.registry import get_tool_registry

    # --- System level ---
    system_allowed = get_allowed_models()
    system_primary = getattr(django_settings, "LLM_DEFAULT_MODEL", "") or ""
    system_mid = getattr(django_settings, "LLM_DEFAULT_MID_MODEL", "") or ""
    system_cheap = getattr(django_settings, "LLM_DEFAULT_CHEAP_MODEL", "") or ""

    all_tools = list(get_tool_registry().list_tools().keys())

    # --- Organization level ---
    org_prefs = _get_org_preferences(user)
    org_allowed = org_prefs.get("allowed_models") if org_prefs else None
    org_models = org_prefs.get("models", {}) if org_prefs else {}
    org_tools = org_prefs.get("tools", {}) if org_prefs else {}

    # Effective allowed models: org restricts to a subset of system
    if org_allowed and isinstance(org_allowed, list):
        effective_allowed = [m for m in org_allowed if m in system_allowed]
    else:
        effective_allowed = list(system_allowed)

    # --- User level ---
    user_prefs = _get_user_preferences(user)
    user_models = user_prefs.get("models", {}) if user_prefs else {}
    user_theme = user_prefs.get("theme", "light") if user_prefs else "light"

    # Resolve each model tier with cascade
    primary_model = _resolve_tier(
        user_choice=user_models.get("primary"),
        org_default=org_models.get("primary"),
        system_default=system_primary,
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
    )
    mid_model = _resolve_tier(
        user_choice=user_models.get("mid"),
        org_default=org_models.get("mid"),
        system_default=system_mid,
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
    )
    cheap_model = _resolve_tier(
        user_choice=user_models.get("cheap"),
        org_default=org_models.get("cheap"),
        system_default=system_cheap,
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
    )

    # Resolve tools: start with all, filter out org-disabled
    allowed_tools = [
        t for t in all_tools
        if org_tools.get(t, True) is not False
    ]

    return ResolvedPreferences(
        primary_model=primary_model,
        mid_model=mid_model,
        cheap_model=cheap_model,
        allowed_models=effective_allowed,
        allowed_tools=allowed_tools,
        theme=user_theme,
    )


def _resolve_tier(
    user_choice: Optional[str],
    org_default: Optional[str],
    system_default: str,
    effective_allowed: list[str],
    system_allowed: list[str],
) -> str:
    """Resolve a single model tier using the cascade.

    1. User's choice if set AND in effective allowed list
    2. Org's default if set AND in system allowed list
    3. System env var default
    4. First model in effective allowed list
    """
    if user_choice and user_choice in effective_allowed:
        return user_choice

    if org_default and org_default in system_allowed:
        return org_default

    if system_default and system_default in system_allowed:
        return system_default

    if effective_allowed:
        return effective_allowed[0]

    return system_default or ""


def _get_org_preferences(user) -> Optional[dict]:
    """Get the organization preferences for a user, or None if no org."""
    from accounts.models import Membership

    membership = (
        Membership.objects.filter(user=user)
        .select_related("org")
        .first()
    )
    if membership and membership.org:
        return membership.org.preferences or {}
    return None


def _get_user_preferences(user) -> Optional[dict]:
    """Get the user preferences dict from UserSettings."""
    from accounts.models import UserSettings

    try:
        us = UserSettings.objects.get(user=user)
        return us.preferences or {}
    except UserSettings.DoesNotExist:
        return None
