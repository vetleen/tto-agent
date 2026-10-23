"""Context builders shared by the user and org settings pages."""
from __future__ import annotations


def partition_transcription_models(allowed: list[str]) -> tuple[list[str], list[str]]:
    """Split an allow-list into (live_capable, upload_capable) pools.

    Each dropdown only offers options that work in its context. Unknown
    registry entries are dropped silently (same filtering the meeting
    picker does).
    """
    from llm.transcription_registry import get_transcription_model_info

    live_capable = [
        mid for mid in allowed
        if (info := get_transcription_model_info(mid)) and info.supports_live_streaming
    ]
    upload_capable = [
        mid for mid in allowed
        if get_transcription_model_info(mid) is not None
    ]
    return live_capable, upload_capable


def build_feature_rows(
    scope: str,
    current_overrides: dict,
    eligible_allowed: list[str],
    meta: dict[str, tuple[str, str]],
    resolved: dict | None = None,
) -> list[dict]:
    """Build the per-feature model rows for a settings page.

    ``scope`` selects which FEATURE_DEFAULTS entries appear ("user" or
    "org"); ``current_overrides`` is the stored feature_models dict for
    that scope; ``eligible_allowed`` bounds the model choices; ``meta``
    maps feature key -> (label, description). ``resolved`` (user page
    only) adds the effective value after the full cascade.
    """
    from core.preferences import FEATURE_DEFAULTS, feature_model_pool

    rows = []
    for fkey, fdef in FEATURE_DEFAULTS.items():
        if fdef.scope != scope:
            continue
        # Any allowed model may serve an ungated feature; a required modality
        # or capability narrows it. No capable model in the allow-list →
        # feature unavailable; omit the row entirely (matches
        # core.preferences.feature_is_available()).
        eligible = feature_model_pool(fdef, eligible_allowed)
        if fdef.gated and not eligible:
            continue
        label, desc = meta.get(fkey, (fkey.replace("_", " ").title(), ""))
        row = {
            "key": fkey,
            "label": label,
            "desc": desc,
            "default_slot": fdef.default_slot,
            "current": current_overrides.get(fkey) or "",
            "eligible_models": eligible,
            "required_modality": fdef.required_modality,
            "required_capability": fdef.required_capability,
        }
        if resolved is not None:
            row["resolved"] = resolved.get(fkey, "")
        rows.append(row)
    return rows


def build_tier_rows(defaults: dict[str, str], allowed: list[str]) -> list[dict]:
    """Build the primary/mid/cheap tier rows for a settings page.

    Every allowed model is offered for every tier.
    """
    descriptions = {
        "primary": ("Primary model", "Used for important tasks like chat and writing."),
        "mid": ("Mid model", "Used for tasks that don't need the best model, like text summarization or tagging."),
        "cheap": ("Cheap model", "Used for very simple tasks, like yes/no questions."),
    }
    return [
        {
            "key": key,
            "label": label,
            "desc": desc,
            "default_model": defaults[key],
            "slot_models": list(allowed),
        }
        for key, (label, desc) in descriptions.items()
    ]
