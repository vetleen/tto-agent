"""Cascading preferences resolver: System -> Organization -> User."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from django.conf import settings as django_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 200_000
MIN_CONTEXT_TOKENS = 10_000
# Upper bound for org/user max_context_tokens settings — generous headroom
# above today's largest real context windows, low enough that a typo'd value
# can't poison downstream token math.
MAX_CONTEXT_TOKENS = 2_000_000

# Live transcription path preference — cascades system → org → user → meeting.
# "chunked" (default) uses the HTTP /v1/audio/transcriptions batching path;
# "realtime" opens an OpenAI Realtime session for sub-second latency;
# "realtime_with_fallback" tries realtime and falls back to chunked on connect
# failure. Invalid values fall back to "chunked".
LIVE_TRANSCRIPTION_MODES = ("chunked", "realtime", "realtime_with_fallback")


@dataclass
class ResolvedPreferences:
    top_model: str
    mid_model: str
    cheap_model: str
    allowed_models: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_skills: list[dict] = field(default_factory=list)
    theme: str = "light"
    parallel_subagents: bool = True
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    transcription_model: str = ""
    allowed_transcription_models: list[str] = field(default_factory=list)
    # Capability-specific defaults. ``transcription_model_live`` is the one
    # used when starting a live recording; ``transcription_model_upload`` is
    # used for uploaded audio files. The generic ``transcription_model``
    # above is the fallback when neither is set — kept for backwards compat.
    transcription_model_live: str = ""
    transcription_model_upload: str = ""
    live_transcription_mode: str = "chunked"
    allow_agent_attach_skills: bool = True
    feature_models: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# Per-feature model catalog: feature_key -> (default_slot, min_tier, scope).
# default_slot: which tier to use by default ("primary", "mid", "cheap").
# min_tier: minimum model tier allowed ("cheap", "mid", "standard").
# scope: "user" = user can override, "org" = org admin can override.
FEATURE_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "chat": ("primary", "standard", "user"),
    "thread_title": ("cheap", "cheap", "user"),
    "thread_emoji": ("cheap", "cheap", "user"),
    "canvas_title": ("cheap", "cheap", "user"),
    "image_description": ("cheap", "cheap", "user"),
    "message_summary": ("mid", "mid", "org"),
    "guardrails_classifier": ("cheap", "cheap", "org"),
    "guardrails_reviewer": ("primary", "standard", "org"),
    "document_description": ("mid", "mid", "org"),
    "skill_emoji": ("cheap", "cheap", "org"),
    "guardrail_chunk_scan": ("cheap", "cheap", "org"),
    "pii_scan": ("mid", "mid", "org"),
}

_SLOT_TO_ATTR = {"primary": "top_model", "mid": "mid_model", "cheap": "cheap_model"}


def _get_system_model_defaults() -> dict[str, str]:
    """Return system-level default model for each tier from Django settings."""
    return {
        "primary": getattr(django_settings, "LLM_DEFAULT_MODEL", "") or "",
        "mid": getattr(django_settings, "LLM_DEFAULT_MID_MODEL", "") or "",
        "cheap": getattr(django_settings, "LLM_DEFAULT_CHEAP_MODEL", "") or "",
    }


def get_preferences(user) -> ResolvedPreferences:
    """Resolve cascading preferences for a user: System -> Org -> User.

    Returns ResolvedPreferences with the effective model for each tier,
    the allowed models list, and the allowed tools list.
    """
    from llm.service.policies import get_allowed_models
    from llm.tools.registry import get_tool_registry

    # --- System level ---
    system_allowed = get_allowed_models()
    sys_models = _get_system_model_defaults()

    registry = get_tool_registry()
    all_tools_dict = registry.list_tools()
    chat_tools = [n for n, t in all_tools_dict.items() if getattr(t, "section", "chat") == "chat"]

    # --- Organization level ---
    org_prefs = _get_org_preferences(user)
    org_allowed = org_prefs.get("allowed_models")
    org_models = org_prefs.get("models", {})
    org_tools = org_prefs.get("tools", {})

    # Effective allowed models: org restricts to a subset of system
    if org_allowed and isinstance(org_allowed, list):
        effective_allowed = [m for m in org_allowed if m in system_allowed]
    else:
        effective_allowed = list(system_allowed)

    # --- User level ---
    user_prefs = _get_user_preferences(user)
    user_models = user_prefs.get("models", {})
    user_theme = user_prefs.get("theme", "light")
    allow_agent_attach_skills = bool(user_prefs.get("allow_agent_attach_skills", True))

    # Resolve each model tier with cascade
    from llm.model_registry import get_models_for_slot

    top_model = _resolve_tier(
        user_choice=user_models.get("primary"),
        org_default=org_models.get("primary"),
        system_default=sys_models["primary"],
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
        slot="primary",
    )
    mid_model = _resolve_tier(
        user_choice=user_models.get("mid"),
        org_default=org_models.get("mid"),
        system_default=sys_models["mid"],
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
        slot="mid",
    )
    cheap_model = _resolve_tier(
        user_choice=user_models.get("cheap"),
        org_default=org_models.get("cheap"),
        system_default=sys_models["cheap"],
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
        slot="cheap",
    )

    # Fallback: if no model available for a tier, fall up to the next tier
    warnings: list[str] = []
    if not cheap_model and mid_model:
        cheap_model = mid_model
        warnings.append("No cheap-tier models in your organization's allowed list. Cheap-tier features will use a more expensive model.")
    if not mid_model and top_model:
        mid_model = top_model
        warnings.append("No mid-tier models in your organization's allowed list. Mid-tier features will use the primary model.")
    if not get_models_for_slot("primary", effective_allowed):
        warnings.append("No standard-tier models in your organization's allowed list. Chat and primary features may not work correctly.")

    # --- Transcription model cascade ---
    system_transcription_allowed = list(getattr(django_settings, "TRANSCRIPTION_ALLOWED_MODELS", []))
    system_transcription_default = getattr(django_settings, "TRANSCRIPTION_DEFAULT_MODEL", "") or ""

    org_transcription_allowed = org_prefs.get("allowed_transcription_models") if org_prefs else None
    org_transcription_models = org_prefs.get("transcription_models", {}) if org_prefs else {}
    user_transcription_models = user_prefs.get("transcription_models", {}) if user_prefs else {}

    # Org restricts to subset of system; explicitly empty list = disabled
    if org_transcription_allowed is not None and isinstance(org_transcription_allowed, list):
        effective_transcription_allowed = [m for m in org_transcription_allowed if m in system_transcription_allowed]
    else:
        # None/missing = inherit all system models
        effective_transcription_allowed = list(system_transcription_allowed)

    if effective_transcription_allowed:
        transcription_model = _resolve_tier(
            user_choice=user_transcription_models.get("default"),
            org_default=org_transcription_models.get("default"),
            system_default=system_transcription_default,
            effective_allowed=effective_transcription_allowed,
            system_allowed=system_transcription_allowed,
        )
    else:
        # Empty allowed list = transcription disabled
        transcription_model = ""

    # Capability-specific defaults. The pool of viable models for each path
    # is the effective allow-list filtered by the relevant capability flag.
    # Diarize models, whisper-1, or any model without ``supports_live_streaming``
    # are dropped from the live pool so a user with "openai/gpt-4o-transcribe-
    # diarize" as their default can still start a live recording — the server
    # picks the first live-capable model from the allow-list instead.
    from llm.transcription_registry import get_transcription_model_info

    def _filter_by_capability(models: list[str], flag: str) -> list[str]:
        filtered = []
        for mid in models:
            info = get_transcription_model_info(mid)
            if info is None:
                continue
            if getattr(info, flag, False):
                filtered.append(mid)
        return filtered

    live_capable = _filter_by_capability(effective_transcription_allowed, "supports_live_streaming")
    # "upload" means anything callable via the batch transcription endpoint —
    # that's every known transcription model in the registry (including
    # diarize and whisper-1), so we don't filter here.
    upload_capable = list(effective_transcription_allowed)

    # Per-capability system defaults. The env vars let ops pin a specific
    # model per path without touching the generic TRANSCRIPTION_DEFAULT_MODEL
    # (which stays around as the ultimate fallback).
    system_default_live = (
        getattr(django_settings, "TRANSCRIPTION_DEFAULT_MODEL_LIVE", "")
        or system_transcription_default
    )
    system_default_upload = (
        getattr(django_settings, "TRANSCRIPTION_DEFAULT_MODEL_UPLOAD", "")
        or system_transcription_default
    )

    def _resolve_for_pool(pool: list[str], key: str, system_default: str) -> str:
        if not pool:
            return ""
        return _resolve_tier(
            user_choice=user_transcription_models.get(key),
            org_default=org_transcription_models.get(key),
            system_default=system_default,
            effective_allowed=pool,
            system_allowed=system_transcription_allowed,
        )

    transcription_model_live = _resolve_for_pool(live_capable, "live", system_default_live)
    transcription_model_upload = _resolve_for_pool(upload_capable, "upload", system_default_upload)

    # --- Live transcription mode (user override only; system default is fixed) ---
    # Every org gets realtime-with-fallback — realtime when it works, legacy
    # chunked otherwise. Individual users can opt themselves into a specific
    # path via the settings page (rarely needed, exposed mainly for debugging).
    # There is no org-level override because no org has asked for one and
    # the complexity isn't worth a hypothetical need.
    user_live_mode = user_prefs.get("live_transcription_mode")
    resolved_live_mode = user_live_mode or "realtime_with_fallback"
    if resolved_live_mode not in LIVE_TRANSCRIPTION_MODES:
        resolved_live_mode = "realtime_with_fallback"

    # Resolve tools: base chat-section tools filtered by org toggles
    base_allowed = [t for t in chat_tools if org_tools.get(t, True) is not False]

    # Resolve allowed skills — skill-section tools are gated by the skill's
    # tool_names field, so they only appear in a chat when the active skill
    # declares them.  They are further filtered by org per-skill tool toggles
    # and by the org-level tool toggles.
    from agent_skills.services import get_available_skills

    org_skills_prefs = org_prefs.get("skills", {})
    user_skills = get_available_skills(user)
    allowed_skills = []

    for skill in user_skills:
        skill_pref = org_skills_prefs.get(skill.slug, {})
        if skill_pref.get("enabled", skill.level != "system") is False:
            continue
        # Filter tool_names through org per-skill tool settings AND org tool toggles
        tool_toggles = skill_pref.get("tools", {})
        filtered_tools = [
            t for t in (skill.tool_names or [])
            if tool_toggles.get(t, True) is not False
            and org_tools.get(t, True) is not False
        ]
        allowed_skills.append({
            "id": str(skill.id),
            "slug": skill.slug,
            "name": skill.name,
            "emoji": skill.emoji,
            "description": skill.description,
            "tool_names": filtered_tools,
        })

    allowed_tools = base_allowed

    # Resolve subagent settings
    org_subagent_prefs = org_prefs.get("subagents", {})
    parallel_subagents = org_subagent_prefs.get("parallel", True)

    # Resolve max context tokens: org sets limit, user can lower it
    org_max_ctx = org_prefs.get("max_context_tokens")
    org_max_ctx = org_max_ctx if isinstance(org_max_ctx, int) else DEFAULT_MAX_CONTEXT_TOKENS
    user_max_ctx = user_prefs.get("max_context_tokens")
    if isinstance(user_max_ctx, int):
        max_context_tokens = min(user_max_ctx, org_max_ctx)
    else:
        max_context_tokens = org_max_ctx
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)

    # Resolve per-feature model overrides
    from llm.model_registry import TIER_ORDER, get_model_tier

    slot_model = {"primary": top_model, "mid": mid_model, "cheap": cheap_model}
    user_feature_prefs = user_prefs.get("feature_models") or {}
    org_feature_prefs = org_prefs.get("feature_models") or {}

    feature_models: dict[str, str] = {}
    for fkey, (default_slot, min_tier, scope) in FEATURE_DEFAULTS.items():
        override = None
        if scope == "user":
            override = user_feature_prefs.get(fkey)
        if not override:
            override = org_feature_prefs.get(fkey)

        if override and override in effective_allowed:
            tier = get_model_tier(override)
            if tier and TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(min_tier, 0):
                feature_models[fkey] = override
                continue

        feature_models[fkey] = slot_model.get(default_slot, top_model)

    return ResolvedPreferences(
        top_model=top_model,
        mid_model=mid_model,
        cheap_model=cheap_model,
        allowed_models=effective_allowed,
        allowed_tools=allowed_tools,
        allowed_skills=allowed_skills,
        theme=user_theme,
        parallel_subagents=parallel_subagents,
        max_context_tokens=max_context_tokens,
        transcription_model=transcription_model,
        allowed_transcription_models=effective_transcription_allowed,
        transcription_model_live=transcription_model_live,
        transcription_model_upload=transcription_model_upload,
        live_transcription_mode=resolved_live_mode,
        allow_agent_attach_skills=allow_agent_attach_skills,
        feature_models=feature_models,
        warnings=warnings,
    )


def _resolve_tier(
    user_choice: Optional[str],
    org_default: Optional[str],
    system_default: str,
    effective_allowed: list[str],
    system_allowed: list[str],
    slot: str | None = None,
) -> str:
    """Resolve a single model tier using the cascade.

    1. User's choice if set AND in effective allowed list AND valid for slot
    2. Org's default if set AND in effective allowed list AND valid for slot
    3. System env var default if in effective allowed list AND valid for slot
    4. First model in effective allowed list that is valid for slot
    5. Empty string (no valid model available)
    """
    from llm.model_registry import is_model_valid_for_slot

    def _valid(model_id: str | None) -> bool:
        if not model_id or model_id not in effective_allowed:
            return False
        if slot and not is_model_valid_for_slot(model_id, slot):
            return False
        return True

    if user_choice and user_choice in effective_allowed and not _valid(user_choice):
        logger.info("Tier constraint: user choice %s skipped for slot %s", user_choice, slot)

    if _valid(user_choice):
        return user_choice

    if _valid(org_default):
        return org_default

    if _valid(system_default):
        return system_default

    for m in effective_allowed:
        if slot is None or is_model_valid_for_slot(m, slot):
            return m

    # No valid model in the effective allowed list. Do NOT fall back to the
    # system default here — that would bypass org-level restrictions when the
    # org's allowed_models list has no overlap with the system list.
    return ""


def get_tier_defaults(user) -> dict[str, str]:
    """Return the resolved default model for each tier, *excluding* the user's own choices.

    This is what the user sees as "Default" — i.e. what would apply if they pick no model.
    Cascade: org default → system default → first allowed.
    """
    from llm.service.policies import get_allowed_models

    system_allowed = get_allowed_models()
    sys_models = _get_system_model_defaults()

    org_prefs = _get_org_preferences(user)
    org_allowed = org_prefs.get("allowed_models")
    org_models = org_prefs.get("models", {})

    if org_allowed and isinstance(org_allowed, list):
        effective_allowed = [m for m in org_allowed if m in system_allowed]
    else:
        effective_allowed = list(system_allowed)

    return {
        "primary": _resolve_tier(None, org_models.get("primary"), sys_models["primary"], effective_allowed, system_allowed, slot="primary"),
        "mid": _resolve_tier(None, org_models.get("mid"), sys_models["mid"], effective_allowed, system_allowed, slot="mid"),
        "cheap": _resolve_tier(None, org_models.get("cheap"), sys_models["cheap"], effective_allowed, system_allowed, slot="cheap"),
    }


def get_system_defaults() -> dict[str, str]:
    """Return the system-level default model for each tier (env vars / first allowed)."""
    from llm.service.policies import get_allowed_models

    system_allowed = get_allowed_models()
    sys_models = _get_system_model_defaults()

    return {
        "primary": _resolve_tier(None, None, sys_models["primary"], system_allowed, system_allowed, slot="primary"),
        "mid": _resolve_tier(None, None, sys_models["mid"], system_allowed, system_allowed, slot="mid"),
        "cheap": _resolve_tier(None, None, sys_models["cheap"], system_allowed, system_allowed, slot="cheap"),
    }


def _get_org_preferences(user) -> dict:
    """Get the organization preferences for a user, or empty dict if no org."""
    from accounts.models import get_membership

    membership = get_membership(user)
    if membership and membership.org:
        return membership.org.preferences or {}
    return {}


def _get_user_preferences(user) -> dict:
    """Get the user preferences dict from UserSettings, or empty dict if missing."""
    from accounts.models import UserSettings

    try:
        us = UserSettings.objects.get(user=user)
        return us.preferences or {}
    except UserSettings.DoesNotExist:
        return {}


def resolve_org_feature_model(org_id: int | None, feature_key: str) -> str:
    """Resolve a feature model from org preferences, falling back to system env vars.

    Used by org-scoped features (guardrails, document processing) that run in
    Celery tasks or system contexts where only the org ID is available.
    """
    from accounts.models import Organization
    from llm.model_registry import TIER_ORDER, get_model_tier
    from llm.service.policies import get_allowed_models

    feature_def = FEATURE_DEFAULTS.get(feature_key)
    if not feature_def:
        return getattr(django_settings, "LLM_DEFAULT_MODEL", "") or ""

    default_slot, min_tier, _scope = feature_def
    sys_models = _get_system_model_defaults()
    system_allowed = get_allowed_models()

    # System-level default for this feature's tier
    sys_default = sys_models.get(default_slot, sys_models.get("primary", ""))

    if not org_id:
        return sys_default if sys_default in system_allowed else (system_allowed[0] if system_allowed else "")

    try:
        org = Organization.objects.get(pk=org_id)
    except Organization.DoesNotExist:
        return sys_default if sys_default in system_allowed else (system_allowed[0] if system_allowed else "")

    org_prefs = org.preferences or {}
    org_allowed = org_prefs.get("allowed_models")
    if org_allowed and isinstance(org_allowed, list):
        effective_allowed = [m for m in org_allowed if m in system_allowed]
    else:
        effective_allowed = list(system_allowed)

    # Check org-level feature override
    override = (org_prefs.get("feature_models") or {}).get(feature_key)
    if override and override in effective_allowed:
        tier = get_model_tier(override)
        if tier and TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(min_tier, 0):
            return override

    # Fall back to org's tier default → system tier default
    org_models = org_prefs.get("models", {})
    return _resolve_tier(
        user_choice=None,
        org_default=org_models.get(default_slot),
        system_default=sys_default,
        effective_allowed=effective_allowed,
        system_allowed=system_allowed,
        slot=default_slot,
    )
