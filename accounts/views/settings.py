import json
import math

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST
from django_ratelimit.decorators import ratelimit

from django.conf import settings as django_settings

from accounts.models import UserSettings
from accounts.services import update_org_preferences, update_user_preferences
from accounts.views._builders import (
    build_feature_rows,
    build_tier_rows,
    partition_transcription_models,
)
from accounts.views._helpers import org_admin_required


def _parse_json_body(request):
    """Parse JSON request body. Returns (data, None) on success or (None, error response)."""
    try:
        return json.loads(request.body), None
    except (json.JSONDecodeError, ValueError):
        return None, JsonResponse({"error": "Invalid JSON"}, status=400)


@login_required
@require_POST
def theme_update(request):
    theme_value = (request.POST.get("theme") or "").strip().lower()
    if theme_value not in (UserSettings.Theme.LIGHT, UserSettings.Theme.DARK):
        return JsonResponse({"error": "Invalid theme"}, status=400)
    # Dual-write: keep the legacy CharField and preferences["theme"] in sync.
    # Inline (not update_user_preferences) because it writes both fields; same
    # locking discipline as the helper.
    with transaction.atomic():
        settings, _ = UserSettings.objects.select_for_update().get_or_create(user=request.user)
        settings.theme = theme_value
        prefs = settings.preferences or {}
        prefs["theme"] = theme_value
        settings.preferences = prefs
        settings.save(update_fields=["theme", "preferences"])
    return JsonResponse({"theme": theme_value})


# ---- User Settings Page ----


@login_required
@require_GET
def settings_page(request):
    from core.preferences import get_preferences, get_tier_defaults

    prefs = get_preferences(request.user)
    user_settings, _ = UserSettings.objects.get_or_create(user=request.user)
    user_models = (user_settings.preferences or {}).get("models", {})
    tier_defaults = get_tier_defaults(request.user)

    from core.preferences import DEFAULT_MAX_CONTEXT_TOKENS, _get_org_preferences

    org_prefs = _get_org_preferences(request.user)
    org_max_context = org_prefs.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
    if not isinstance(org_max_context, int):
        org_max_context = DEFAULT_MAX_CONTEXT_TOKENS
    user_max_context = (user_settings.preferences or {}).get("max_context_tokens")

    user_transcription_prefs = (user_settings.preferences or {}).get("transcription_models", {})
    user_transcription_model = user_transcription_prefs.get("default")
    user_transcription_model_live = user_transcription_prefs.get("live")
    user_transcription_model_upload = user_transcription_prefs.get("upload")
    user_live_transcription_mode = (user_settings.preferences or {}).get("live_transcription_mode", "")

    from llm.transcription_registry import get_all_transcription_models
    all_models = get_all_transcription_models()
    transcription_model_display = {mid: info.display_name for mid, info in all_models.items()}

    live_capable_models, upload_capable_models = partition_transcription_models(
        prefs.allowed_transcription_models
    )

    user_image_model = (user_settings.preferences or {}).get("image_models", {}).get("default")
    from llm.image_generation_registry import get_image_generation_models
    image_model_display = {mid: info.display_name for mid, info in get_image_generation_models().items()}

    user_feature_models = (user_settings.preferences or {}).get("feature_models", {})
    _USER_FEATURE_META = {
        "chat": ("Chat", "The primary model used for conversations."),
        "thread_title": ("Thread title", "Generates a short title for new chat threads."),
        "thread_emoji": ("Thread emoji", "Picks an emoji when you use the /tag command on a chat thread."),
        "canvas_title": ("Canvas title", "Generates a title when a new canvas is created."),
        "image_description": ("Image description", f"Describes images pasted or uploaded in chat so {django_settings.ASSISTANT_NAME} can understand them."),
    }
    user_features = build_feature_rows(
        "user",
        user_feature_models,
        prefs.allowed_models,
        _USER_FEATURE_META,
        resolved=prefs.feature_models,
    )

    return render(request, "accounts/settings.html", {
        "resolved": prefs,
        "user_models": json.dumps(user_models),
        "allowed_models": prefs.allowed_models,
        "org_max_context_tokens": org_max_context,
        "user_max_context_tokens": user_max_context,
        "allowed_transcription_models": prefs.allowed_transcription_models,
        "live_capable_transcription_models": live_capable_models,
        "upload_capable_transcription_models": upload_capable_models,
        "user_transcription_model": user_transcription_model or "",
        "user_transcription_model_live": user_transcription_model_live or "",
        "user_transcription_model_upload": user_transcription_model_upload or "",
        "resolved_transcription_model": prefs.transcription_model,
        "resolved_transcription_model_live": prefs.transcription_model_live,
        "resolved_transcription_model_upload": prefs.transcription_model_upload,
        "user_live_transcription_mode": user_live_transcription_mode,
        "resolved_live_transcription_mode": prefs.live_transcription_mode,
        "transcription_model_display": transcription_model_display,
        "allowed_image_models": prefs.allowed_image_models,
        "user_image_model": user_image_model or "",
        "resolved_image_model": prefs.image_model,
        "image_model_display": image_model_display,
        "allow_agent_attach_skills": prefs.allow_agent_attach_skills,
        "assistant_name": django_settings.ASSISTANT_NAME,
        "preference_warnings": prefs.warnings,
        "user_features": user_features,
        "tiers": build_tier_rows(tier_defaults, prefs.allowed_models),
    })


@login_required
@require_POST
def preferences_models_update(request):
    """Update user's preferred model for a tier."""
    from core.preferences import get_preferences

    data, err = _parse_json_body(request)
    if err:
        return err

    tier = data.get("tier", "").strip()
    model = data.get("model", "").strip() or None

    if tier not in ("primary", "mid", "cheap"):
        return JsonResponse({"error": "Invalid tier"}, status=400)

    # Validate model is in the user's allowed list and correct tier
    if model:
        from llm.model_registry import is_model_valid_for_slot

        prefs = get_preferences(request.user)
        if model not in prefs.allowed_models:
            return JsonResponse({"error": "Model not allowed"}, status=400)
        if not is_model_valid_for_slot(model, tier):
            return JsonResponse({"error": f"This model cannot be used as a {tier} model."}, status=400)

    def mutate(prefs):
        models = prefs.get("models", {})
        models[tier] = model
        prefs["models"] = models

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "tier": tier, "model": model})


@login_required
@require_POST
def preferences_transcription_model_update(request):
    """Update a user's preferred transcription model.

    Accepts ``kind`` in the POST body: ``default`` (generic fallback),
    ``live`` (used when starting a live meeting — must be a streaming-
    capable model), or ``upload`` (used for uploaded audio files).
    Missing / unknown ``kind`` is treated as ``default`` for backwards
    compatibility with older clients.
    """
    from core.preferences import get_preferences
    from llm.transcription_registry import get_transcription_model_info

    data, err = _parse_json_body(request)
    if err:
        return err

    model = (data.get("model") or "").strip() or None
    kind = (data.get("kind") or "default").strip().lower()
    if kind not in ("default", "live", "upload"):
        return JsonResponse({"error": "Invalid kind"}, status=400)

    if model:
        prefs = get_preferences(request.user)
        if model not in prefs.allowed_transcription_models:
            return JsonResponse({"error": "Model not allowed"}, status=400)
        if kind == "live":
            info = get_transcription_model_info(model)
            if info is None or not info.supports_live_streaming:
                return JsonResponse(
                    {"error": "This model cannot be used for live transcription."},
                    status=400,
                )

    def mutate(prefs):
        transcription_models = prefs.get("transcription_models", {})
        transcription_models[kind] = model
        prefs["transcription_models"] = transcription_models

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "model": model, "kind": kind})


@login_required
@require_POST
def preferences_image_model_update(request):
    """Update a user's preferred image generation model (must be org-allowed)."""
    from core.preferences import get_preferences

    data, err = _parse_json_body(request)
    if err:
        return err

    model = (data.get("model") or "").strip() or None
    if model:
        prefs = get_preferences(request.user)
        if model not in prefs.allowed_image_models:
            return JsonResponse({"error": "Model not allowed"}, status=400)

    def mutate(prefs):
        image_models = prefs.get("image_models", {})
        image_models["default"] = model
        prefs["image_models"] = image_models

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "model": model})


@login_required
@require_POST
def preferences_live_transcription_mode_update(request):
    """Update the user's live transcription mode preference.

    Values: ``chunked``, ``realtime``, ``realtime_with_fallback``. An
    empty string / missing value clears the user's override and lets the
    org or system default take effect.
    """
    from core.preferences import LIVE_TRANSCRIPTION_MODES

    data, err = _parse_json_body(request)
    if err:
        return err

    mode = (data.get("mode") or "").strip().lower() or None
    if mode and mode not in LIVE_TRANSCRIPTION_MODES:
        return JsonResponse(
            {"error": f"Invalid mode. Choose from {list(LIVE_TRANSCRIPTION_MODES)}."},
            status=400,
        )

    def mutate(prefs):
        if mode is None:
            prefs.pop("live_transcription_mode", None)
        else:
            prefs["live_transcription_mode"] = mode

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "mode": mode})


@login_required
@require_POST
def preferences_agent_attach_skills_update(request):
    """Toggle whether the assistant may autonomously attach skills to a thread."""
    data, err = _parse_json_body(request)
    if err:
        return err
    enabled = bool(data.get("enabled", True))

    def mutate(prefs):
        prefs["allow_agent_attach_skills"] = enabled

    update_user_preferences(request.user, mutate)
    return JsonResponse({"ok": True, "enabled": enabled})


@login_required
@require_POST
def preferences_feature_model_update(request):
    """Update user's preferred model for a specific feature."""
    from core.preferences import FEATURE_DEFAULTS, get_preferences
    from llm.model_registry import TIER_ORDER, get_model_tier

    data, err = _parse_json_body(request)
    if err:
        return err

    feature = data.get("feature", "").strip()
    model = data.get("model", "").strip() or None

    if feature not in FEATURE_DEFAULTS:
        return JsonResponse({"error": "Unknown feature"}, status=400)

    _fdef = FEATURE_DEFAULTS[feature]
    min_tier, scope = _fdef.min_tier, _fdef.scope
    if scope != "user":
        return JsonResponse({"error": "This feature is not user-configurable"}, status=400)

    if model:
        prefs = get_preferences(request.user)
        if model not in prefs.allowed_models:
            return JsonResponse({"error": "Model not allowed"}, status=400)
        tier = get_model_tier(model)
        if tier and TIER_ORDER.get(tier, 0) < TIER_ORDER.get(min_tier, 0):
            return JsonResponse({"error": f"Model tier too low for this feature (minimum: {min_tier})"}, status=400)

    def mutate(prefs):
        feature_models = prefs.get("feature_models", {})
        feature_models[feature] = model
        prefs["feature_models"] = feature_models

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "feature": feature, "model": model})


# ---- Organization Settings Page ----


@login_required
@require_GET
@org_admin_required
def org_settings_page(request):
    from agent_skills.models import AgentSkill
    from core.preferences import get_system_defaults
    from llm.service.policies import get_allowed_models
    from llm.tools.registry import get_tool_registry

    membership = request.org_membership

    org = membership.org
    org_prefs = org.preferences or {}
    org_allowed = org_prefs.get("allowed_models") or []
    org_models = org_prefs.get("models", {})
    org_tools = org_prefs.get("tools", {})
    org_skills_prefs = org_prefs.get("skills", {})

    SECTION_META = {
        "chat": {
            "label": f"{django_settings.ASSISTANT_NAME} Chat",
            "description": "Tools available during chat conversations.",
        },
        "document_processing": {
            "label": "Document Processing",
            "description": "Tools used during automated document processing.",
        },
    }
    TOOL_NOTES = {}

    system_models = get_allowed_models()
    from llm.model_registry import get_model_info
    system_models_data = []
    for mid in system_models:
        info = get_model_info(mid)
        price = ""
        if info and info.input_price is not None and info.output_price is not None:
            price = f"${info.input_price} / ${info.output_price}"
        system_models_data.append({"id": mid, "price": price})
    system_defaults = get_system_defaults()
    all_tools = get_tool_registry().list_tools()

    # Group tools by section (skip "skills" — managed in Skills section)
    tool_sections = {}
    for name, tool in sorted(all_tools.items()):
        section_key = getattr(tool, "section", "chat")
        if section_key == "skills":
            continue
        if section_key not in tool_sections:
            meta = SECTION_META.get(section_key, {"label": section_key.replace("_", " ").title(), "description": ""})
            tool_sections[section_key] = {
                "label": meta["label"],
                "description": meta["description"],
                "tools": [],
            }
        tool_sections[section_key]["tools"].append({
            "name": name,
            "description": tool.description,
            "enabled": org_tools.get(name, True) is not False,
            "note": TOOL_NOTES.get(name, ""),
        })

    # Build skills data for the settings page
    visible_skills = list(
        AgentSkill.objects.filter(level="system", is_active=True)
    ) + list(
        AgentSkill.objects.filter(level="org", organization=org, is_active=True)
    )
    skills_data = []
    for skill in sorted(visible_skills, key=lambda s: s.name):
        sp = org_skills_prefs.get(skill.slug, {})
        tool_toggles = sp.get("tools", {})
        skills_data.append({
            "slug": skill.slug,
            "name": skill.name,
            "emoji": skill.emoji,
            "description": skill.description,
            "tool_names": skill.tool_names or [],
            "enabled": sp.get("enabled", skill.level != "system") is not False,
            "tools": {t: tool_toggles.get(t, True) is not False for t in (skill.tool_names or [])},
        })

    org_subagent_prefs = org_prefs.get("subagents", {})
    parallel_subagents = org_subagent_prefs.get("parallel", True)
    pii_scan_enabled = org_prefs.get("pii_scan_enabled", True)
    pii_quarantine_enabled = org_prefs.get("pii_quarantine_enabled", True)

    from core.preferences import DEFAULT_MAX_CONTEXT_TOKENS

    org_max_context_tokens = org_prefs.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
    if not isinstance(org_max_context_tokens, int):
        org_max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS

    from llm.transcription_registry import get_all_transcription_models

    system_transcription_models = list(getattr(django_settings, "TRANSCRIPTION_ALLOWED_MODELS", []))
    org_allowed_transcription = org_prefs.get("allowed_transcription_models")
    org_transcription_models = org_prefs.get("transcription_models", {})
    transcription_model_display = {
        mid: info.display_name for mid, info in get_all_transcription_models().items()
    }

    effective_allowed = (
        org_allowed_transcription
        if isinstance(org_allowed_transcription, list)
        else system_transcription_models
    )
    live_capable_transcription_models, upload_capable_transcription_models = (
        partition_transcription_models(effective_allowed)
    )
    system_transcription_default_live = getattr(django_settings, "TRANSCRIPTION_DEFAULT_MODEL_LIVE", "") or ""
    system_transcription_default_upload = getattr(django_settings, "TRANSCRIPTION_DEFAULT_MODEL_UPLOAD", "") or ""

    from llm.image_generation_registry import get_image_generation_models
    system_image_models = list(getattr(django_settings, "IMAGE_ALLOWED_MODELS", []))
    org_allowed_image = org_prefs.get("allowed_image_models")
    org_image_models = org_prefs.get("image_models", {})
    effective_image_allowed = (
        [m for m in org_allowed_image if m in system_image_models]
        if isinstance(org_allowed_image, list)
        else list(system_image_models)
    )
    image_model_display = {
        mid: info.display_name for mid, info in get_image_generation_models().items()
    }

    effective_org_allowed = [m for m in org_allowed if m in system_models] if org_allowed else list(system_models)

    org_feature_models = org_prefs.get("feature_models", {})
    _ORG_FEATURE_META = {
        "message_summary": ("Message summary", "When conversations get long, this model summarizes the chat history to stay within the context window."),
        "guardrails_classifier": ("Guardrails classifier", "Screens every user message and profile description for adversarial or policy-violating content. A cheap, fast model is ideal here since it only flags content for further review."),
        "guardrails_reviewer": ("Guardrails reviewer", "Reviews content flagged by the classifier and decides whether action is needed (warn, block message, ban user, etc.). A stronger model is recommended since it actually makes the final decision."),
        "document_description": ("Document description", f"Generates a short description of uploaded documents to help {django_settings.ASSISTANT_NAME} judge relevance."),
        "document_image_description": ("Image description", "Describes images uploaded to data rooms so they become searchable. Requires a vision-capable model; if none is allowed, image uploads are disabled."),
        "skill_emoji": ("Skill emoji", "Picks an emoji for newly created skills."),
        "guardrail_chunk_scan": ("Chunk scan", "Scans document chunks for hidden adversarial content during file processing. Runs on every chunk, so a cheap, fast model keeps costs low."),
        "pii_scan": ("PII classification", "Classifies documents by GDPR personal data categories during processing. Uses a mid-tier model for accuracy."),
    }
    org_features = build_feature_rows(
        "org", org_feature_models, effective_org_allowed, _ORG_FEATURE_META
    )

    from core.styles import FONT_CHOICES, get_org_styles

    org_styles = get_org_styles(org)

    return render(request, "accounts/org_settings.html", {
        "org": org,
        "styles": org_styles,
        "styles_json": json.dumps(org_styles),
        "font_choices": FONT_CHOICES,
        "monthly_budget_per_user": org_prefs.get("monthly_budget_per_user", 0),
        "monthly_budget_org": org_prefs.get("monthly_budget_org", 0),
        "system_models": system_models,
        "system_models_data": system_models_data,
        "org_allowed": org_allowed,
        "org_models": org_models,
        "org_models_json": json.dumps(org_models),
        "org_tools_json": json.dumps(org_tools),
        "tool_sections": tool_sections,
        "skills_data": skills_data,
        "skills_data_json": json.dumps(skills_data),
        "parallel_subagents": parallel_subagents,
        "pii_scan_enabled": pii_scan_enabled,
        "pii_quarantine_enabled": pii_quarantine_enabled,
        "org_max_context_tokens": org_max_context_tokens,
        "system_transcription_models": system_transcription_models,
        "org_allowed_transcription": org_allowed_transcription,
        "org_transcription_default": org_transcription_models.get("default", ""),
        "org_transcription_default_live": org_transcription_models.get("live", ""),
        "org_transcription_default_upload": org_transcription_models.get("upload", ""),
        "live_capable_transcription_models": live_capable_transcription_models,
        "upload_capable_transcription_models": upload_capable_transcription_models,
        "system_transcription_default_live": system_transcription_default_live,
        "system_transcription_default_upload": system_transcription_default_upload,
        "transcription_model_display": transcription_model_display,
        "system_image_models": system_image_models,
        "org_allowed_image": org_allowed_image,
        "org_image_default": org_image_models.get("default", ""),
        "image_model_display": image_model_display,
        "org_features": org_features,
        "tiers": build_tier_rows(system_defaults, effective_org_allowed),
    })


@login_required
@require_POST
@org_admin_required
def org_allowed_models_update(request):
    """Set org's allowed_models list."""
    from llm.service.policies import get_allowed_models

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    models = data.get("allowed_models", [])
    if not isinstance(models, list):
        return JsonResponse({"error": "allowed_models must be a list"}, status=400)

    # Validate all are in system allowlist
    system_models = get_allowed_models()
    invalid = [m for m in models if m not in system_models]
    if invalid:
        return JsonResponse({"error": f"These models aren't available: {', '.join(invalid)}."}, status=400)

    def mutate(prefs):
        prefs["allowed_models"] = models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "allowed_models": models})


@login_required
@require_POST
@org_admin_required
def org_styles_update(request):
    """Set the org's document export styles (fonts/colours)."""
    from core.styles import validate_styles

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    clean, error = validate_styles(data)
    if error:
        return JsonResponse({"error": error}, status=400)

    def mutate(prefs):
        prefs["styles"] = clean

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "styles": clean})


@login_required
@require_POST
@org_admin_required
def org_allowed_transcription_models_update(request):
    """Set org's allowed transcription models list."""
    from django.conf import settings as django_settings

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    models = data.get("allowed_transcription_models")
    if not isinstance(models, list):
        return JsonResponse({"error": "allowed_transcription_models must be a list"}, status=400)

    system_models = list(getattr(django_settings, "TRANSCRIPTION_ALLOWED_MODELS", []))
    invalid = [m for m in models if m not in system_models]
    if invalid:
        return JsonResponse({"error": f"These models aren't available: {', '.join(invalid)}."}, status=400)

    def mutate(prefs):
        prefs["allowed_transcription_models"] = models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "allowed_transcription_models": models})


@login_required
@require_POST
@org_admin_required
def org_transcription_model_update(request):
    """Set an organization-level default transcription model.

    Accepts ``kind`` in the POST body: ``default`` (generic), ``live``
    (must be a streaming-capable model), or ``upload`` (any registered
    transcription model, including diarize). Unknown / missing ``kind``
    is treated as ``default``.
    """
    from llm.transcription_registry import get_transcription_model_info

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    model = (data.get("model") or "").strip() or None
    kind = (data.get("kind") or "default").strip().lower()
    if kind not in ("default", "live", "upload"):
        return JsonResponse({"error": "Invalid kind"}, status=400)

    if model:
        org_allowed = (membership.org.preferences or {}).get("allowed_transcription_models")
        system_models = list(getattr(django_settings, "TRANSCRIPTION_ALLOWED_MODELS", []))
        effective = [m for m in org_allowed if m in system_models] if isinstance(org_allowed, list) else system_models
        if model not in effective:
            return JsonResponse({"error": "Model not in allowed list"}, status=400)
        if kind == "live":
            info = get_transcription_model_info(model)
            if info is None or not info.supports_live_streaming:
                return JsonResponse(
                    {"error": "This model cannot be used for live transcription."},
                    status=400,
                )

    def mutate(prefs):
        transcription_models = prefs.get("transcription_models", {})
        transcription_models[kind] = model
        prefs["transcription_models"] = transcription_models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "model": model, "kind": kind})


@login_required
@require_POST
@org_admin_required
def org_allowed_image_models_update(request):
    """Set org's allowed image generation models list."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    models = data.get("allowed_image_models")
    if not isinstance(models, list):
        return JsonResponse({"error": "allowed_image_models must be a list"}, status=400)

    system_models = list(getattr(django_settings, "IMAGE_ALLOWED_MODELS", []))
    invalid = [m for m in models if m not in system_models]
    if invalid:
        return JsonResponse({"error": f"These models aren't available: {', '.join(invalid)}."}, status=400)

    def mutate(prefs):
        prefs["allowed_image_models"] = models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "allowed_image_models": models})


@login_required
@require_POST
@org_admin_required
def org_image_model_update(request):
    """Set the organization-level default image generation model."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    model = (data.get("model") or "").strip() or None
    if model:
        org_allowed = (membership.org.preferences or {}).get("allowed_image_models")
        system_models = list(getattr(django_settings, "IMAGE_ALLOWED_MODELS", []))
        effective = [m for m in org_allowed if m in system_models] if isinstance(org_allowed, list) else system_models
        if model not in effective:
            return JsonResponse({"error": "Model not in allowed list"}, status=400)

    def mutate(prefs):
        image_models = prefs.get("image_models", {})
        image_models["default"] = model
        prefs["image_models"] = image_models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "model": model})


@login_required
@require_POST
@org_admin_required
def org_models_update(request):
    """Set org's default model for a tier."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    tier = data.get("tier", "").strip()
    model = data.get("model", "").strip() or None

    if tier not in ("primary", "mid", "cheap"):
        return JsonResponse({"error": "Invalid tier"}, status=400)

    org_allowed = (membership.org.preferences or {}).get("allowed_models") or []

    if model and org_allowed and model not in org_allowed:
        return JsonResponse({"error": "Model not in org allowed list"}, status=400)

    if model:
        from llm.model_registry import is_model_valid_for_slot

        if not is_model_valid_for_slot(model, tier):
            return JsonResponse({"error": f"This model cannot be used as a {tier} model."}, status=400)

    def mutate(prefs):
        models = prefs.get("models", {})
        models[tier] = model
        prefs["models"] = models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "tier": tier, "model": model})


@login_required
@require_POST
@org_admin_required
def org_tools_update(request):
    """Toggle a tool on/off for the org."""
    from llm.tools.registry import get_tool_registry

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    tool_name = data.get("name", "").strip()
    enabled = data.get("enabled", True)

    if not tool_name:
        return JsonResponse({"error": "Tool name required"}, status=400)

    # Reject unknown names so typos and stale clients can't pile junk keys
    # into org.preferences["tools"].
    if tool_name not in get_tool_registry().list_tools():
        return JsonResponse({"error": "Unknown tool"}, status=400)

    def mutate(prefs):
        tools = prefs.get("tools", {})
        tools[tool_name] = bool(enabled)
        prefs["tools"] = tools

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "name": tool_name, "enabled": bool(enabled)})


@login_required
@require_POST
@org_admin_required
def org_subagents_update(request):
    """Update org's sub-agent settings (e.g. parallel toggle)."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    parallel = data.get("parallel", True)

    def mutate(prefs):
        subagents = prefs.get("subagents", {})
        subagents["parallel"] = bool(parallel)
        prefs["subagents"] = subagents

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "parallel": bool(parallel)})


@login_required
@require_POST
@org_admin_required
def org_pii_scan_toggle_update(request):
    """Toggle PII classification for document processing."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    enabled = data.get("enabled", True)

    def mutate(prefs):
        prefs["pii_scan_enabled"] = bool(enabled)

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "enabled": bool(enabled)})


@login_required
@require_POST
@org_admin_required
def org_pii_quarantine_toggle_update(request):
    """Toggle automatic quarantine of documents containing GDPR Art. 9/10 data."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    enabled = data.get("enabled", True)

    def mutate(prefs):
        prefs["pii_quarantine_enabled"] = bool(enabled)

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "enabled": bool(enabled)})


@login_required
@require_POST
@org_admin_required
def org_max_context_update(request):
    """Set org's max context tokens limit."""
    from core.preferences import MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    value = data.get("max_context_tokens")
    if value is None:
        # Clear (reset to default)
        update_org_preferences(
            membership.org_id, lambda prefs: prefs.pop("max_context_tokens", None)
        )
        return JsonResponse({"ok": True, "max_context_tokens": None})

    if not isinstance(value, int) or isinstance(value, bool):
        return JsonResponse({"error": "max_context_tokens must be an integer"}, status=400)

    if value < MIN_CONTEXT_TOKENS:
        return JsonResponse({"error": f"max_context_tokens must be at least {MIN_CONTEXT_TOKENS:,}"}, status=400)

    if value > MAX_CONTEXT_TOKENS:
        return JsonResponse({"error": f"max_context_tokens must be at most {MAX_CONTEXT_TOKENS:,}"}, status=400)

    def mutate(prefs):
        prefs["max_context_tokens"] = value

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "max_context_tokens": value})


@login_required
@require_POST
@org_admin_required
def org_budget_update(request):
    """Update org's monthly budget settings."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    updates = {}
    for key in ("monthly_budget_per_user", "monthly_budget_org"):
        if key in data:
            try:
                val = float(data[key])
            except (TypeError, ValueError):
                return JsonResponse({"error": f"Invalid value for {key}"}, status=400)
            # json.loads accepts the NaN/Infinity literals and NaN < 0 is False,
            # so non-finite values would slip past the negative check (and then
            # fail the Postgres jsonb write or poison budget comparisons).
            if not math.isfinite(val):
                return JsonResponse({"error": f"Invalid value for {key}"}, status=400)
            if val < 0:
                return JsonResponse({"error": "Budget cannot be negative"}, status=400)
            updates[key] = val

    update_org_preferences(membership.org_id, lambda prefs: prefs.update(updates))

    return JsonResponse({"ok": True})


@login_required
@require_POST
def preferences_max_context_update(request):
    """Update user's max context tokens preference."""
    from core.preferences import (
        DEFAULT_MAX_CONTEXT_TOKENS,
        MAX_CONTEXT_TOKENS,
        MIN_CONTEXT_TOKENS,
        _get_org_preferences,
    )

    data, err = _parse_json_body(request)
    if err:
        return err

    value = data.get("max_context_tokens")
    if value is None:
        # Clear user override
        update_user_preferences(
            request.user, lambda prefs: prefs.pop("max_context_tokens", None)
        )
        return JsonResponse({"ok": True, "max_context_tokens": None})

    if not isinstance(value, int) or isinstance(value, bool):
        return JsonResponse({"error": "max_context_tokens must be an integer"}, status=400)

    if value < MIN_CONTEXT_TOKENS:
        return JsonResponse({"error": f"max_context_tokens must be at least {MIN_CONTEXT_TOKENS:,}"}, status=400)

    if value > MAX_CONTEXT_TOKENS:
        return JsonResponse({"error": f"max_context_tokens must be at most {MAX_CONTEXT_TOKENS:,}"}, status=400)

    # Check against org limit
    org_prefs = _get_org_preferences(request.user)
    org_limit = org_prefs.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
    if not isinstance(org_limit, int):
        org_limit = DEFAULT_MAX_CONTEXT_TOKENS
    if value > org_limit:
        return JsonResponse({"error": f"Cannot exceed organization limit of {org_limit:,} tokens"}, status=400)

    def mutate(prefs):
        prefs["max_context_tokens"] = value

    update_user_preferences(request.user, mutate)

    return JsonResponse({"ok": True, "max_context_tokens": value})


@login_required
@require_POST
@org_admin_required
def org_skills_update(request):
    """Toggle a skill or a per-skill tool on/off for the org."""
    from django.db.models import Q

    from agent_skills.models import AgentSkill

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    slug = data.get("slug", "").strip()
    if not slug:
        return JsonResponse({"error": "Skill slug required"}, status=400)

    # Same visibility set org_settings_page renders (system + this org,
    # active) — deliberately broader than get_available_skills so admins can
    # re-enable skills they previously disabled. Rejecting everything else
    # keeps typos and stale clients from piling junk keys into
    # org.preferences["skills"].
    slug_visible = AgentSkill.objects.filter(
        Q(level="system") | Q(level="org", organization=membership.org),
        slug=slug,
        is_active=True,
    ).exists()
    if not slug_visible:
        return JsonResponse({"error": "Unknown skill"}, status=400)

    tool_name = data.get("tool", "").strip() if "tool" in data else None
    enabled = data.get("enabled", True)

    def mutate(prefs):
        skills = prefs.get("skills", {})
        if slug not in skills:
            skills[slug] = {}
        if tool_name:
            # Per-tool toggle within a skill
            tool_toggles = skills[slug].get("tools", {})
            tool_toggles[tool_name] = bool(enabled)
            skills[slug]["tools"] = tool_toggles
        else:
            # Skill-level toggle
            skills[slug]["enabled"] = bool(enabled)
        prefs["skills"] = skills

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "slug": slug, "enabled": bool(enabled)})


@login_required
@require_POST
@org_admin_required
def org_feature_model_update(request):
    """Set org's preferred model for a specific feature."""
    from core.preferences import FEATURE_DEFAULTS
    from llm.model_registry import TIER_ORDER, get_model_tier
    from llm.service.policies import get_allowed_models

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    feature = data.get("feature", "").strip()
    model = data.get("model", "").strip() or None

    if feature not in FEATURE_DEFAULTS:
        return JsonResponse({"error": "Unknown feature"}, status=400)

    _fdef = FEATURE_DEFAULTS[feature]
    min_tier, scope = _fdef.min_tier, _fdef.scope
    if scope != "org":
        return JsonResponse({"error": "This feature is not org-configurable"}, status=400)

    if model:
        org_allowed = (membership.org.preferences or {}).get("allowed_models") or []
        system_models = get_allowed_models()
        effective = [m for m in org_allowed if m in system_models] if org_allowed else list(system_models)
        if model not in effective:
            return JsonResponse({"error": "Model not in allowed list"}, status=400)
        tier = get_model_tier(model)
        if tier and TIER_ORDER.get(tier, 0) < TIER_ORDER.get(min_tier, 0):
            return JsonResponse({"error": f"Model tier too low for this feature (minimum: {min_tier})"}, status=400)

    def mutate(prefs):
        feature_models = prefs.get("feature_models", {})
        feature_models[feature] = model
        prefs["feature_models"] = feature_models

    update_org_preferences(membership.org_id, mutate)

    return JsonResponse({"ok": True, "feature": feature, "model": model})


# ---- User Profile ----

MAX_DESCRIPTION_LENGTH = 5000
MAX_NAME_LENGTH = 150
MAX_ORG_NAME_LENGTH = 255


@login_required
@require_GET
def profile_page(request):
    # Superseded by the "My Agent" page; keep the old URL working.
    return redirect("accounts:agent")


# The endpoints below each fire a synchronous LLM classifier call per request, so
# they carry a per-user throttle (cost-amplification guard). login_required stays
# first so anonymous requests never reach the user-pk rate key.
@login_required
@require_POST
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def profile_update(request):
    import logging

    from guardrails.classifier import classify_description_sync

    logger = logging.getLogger(__name__)

    data, err = _parse_json_body(request)
    if err:
        return err

    update_fields = []
    user = request.user

    for field in ("first_name", "last_name", "title"):
        if field in data:
            val = str(data[field]).strip()
            if len(val) > MAX_NAME_LENGTH:
                return JsonResponse(
                    {"error": f"{field.replace('_', ' ').capitalize()} must be {MAX_NAME_LENGTH} characters or fewer."},
                    status=400,
                )
            setattr(user, field, val)
            update_fields.append(field)

    if "description" in data:
        desc = str(data["description"]).strip()
        if len(desc) > MAX_DESCRIPTION_LENGTH:
            return JsonResponse(
                {"error": f"Description must be {MAX_DESCRIPTION_LENGTH} characters or fewer."},
                status=400,
            )
        if desc:
            try:
                result = classify_description_sync(desc, user.pk)
                if result.is_suspicious:
                    logger.warning("Profile description blocked for user %s: %s", user.pk, result.reasoning)
                    return JsonResponse(
                        {"error": "Description could not be saved. Please revise and try again."},
                        status=400,
                    )
            except Exception:
                logger.exception("Description classifier failed for user %s", user.pk)
                return JsonResponse(
                    {"error": "Unable to verify description right now. Please try again later."},
                    status=503,
                )
        user.description = desc
        update_fields.append("description")

    if update_fields:
        user.save(update_fields=update_fields)

    return JsonResponse({"ok": True})


# ---- Profile picture ----

# ~8 MB image + multipart framing headroom. Rejected from Content-Length before
# Django spools the body to disk.
PROFILE_PICTURE_REQUEST_MAX_BYTES = 12_000_000


def _delete_profile_picture_files(user):
    """Delete both stored avatar files from storage, clearing the fields in memory.

    ``FieldFile.delete(save=False)`` removes the file and empties the field on
    the instance; missing files are ignored. The caller persists the change.
    """
    for field in (user.profile_picture, user.profile_picture_original):
        if field:
            field.delete(save=False)


@login_required
@require_POST
@ratelimit(key="user", rate="30/h", method="POST", block=True)
def profile_picture_update(request):
    from accounts.avatars import (
        InvalidProfilePicture,
        max_upload_bytes,
        process_profile_picture,
    )

    # Reject oversized requests from Content-Length BEFORE request.FILES spools
    # the whole multipart body to disk.
    try:
        content_length = int(request.META.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        content_length = 0
    max_request_bytes = getattr(
        django_settings, "PROFILE_PICTURE_REQUEST_MAX_BYTES", PROFILE_PICTURE_REQUEST_MAX_BYTES
    )
    if content_length > max_request_bytes:
        return JsonResponse({"error": "Image is too large."}, status=413)

    upload = request.FILES.get("picture")
    if not upload:
        return JsonResponse({"error": "No image was provided."}, status=400)
    if upload.size > max_upload_bytes():
        return JsonResponse({"error": "Image is too large (max 8 MB)."}, status=400)

    try:
        ext, original, resized = process_profile_picture(upload)
    except InvalidProfilePicture:
        return JsonResponse(
            {"error": "That file couldn't be read as an image. Use a JPEG, PNG, or WebP."},
            status=400,
        )

    user = request.user
    # Drop any previous files first so a replacement doesn't leave orphans.
    _delete_profile_picture_files(user)
    user.profile_picture_original.save(f"user_{user.pk}.{ext}", original, save=False)
    user.profile_picture.save(f"user_{user.pk}.{ext}", resized, save=False)
    user.save(update_fields=["profile_picture", "profile_picture_original"])

    return JsonResponse({"ok": True, "url": user.profile_picture.url})


@login_required
@require_POST
@ratelimit(key="user", rate="30/h", method="POST", block=True)
def profile_picture_delete(request):
    user = request.user
    if user.profile_picture or user.profile_picture_original:
        _delete_profile_picture_files(user)
        user.save(update_fields=["profile_picture", "profile_picture_original"])
    return JsonResponse({"ok": True})


# ---- Organization Description ----


@login_required
@require_POST
@org_admin_required
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def org_description_update(request):
    import logging

    from guardrails.classifier import classify_description_sync

    logger = logging.getLogger(__name__)

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    desc = str(data.get("description", "")).strip()
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return JsonResponse(
            {"error": f"Description must be {MAX_DESCRIPTION_LENGTH} characters or fewer."},
            status=400,
        )

    if desc:
        try:
            result = classify_description_sync(desc, request.user.pk, membership.org_id)
            if result.is_suspicious:
                logger.warning(
                    "Org description blocked for org %s by user %s: %s",
                    membership.org_id, request.user.pk, result.reasoning,
                )
                return JsonResponse(
                    {"error": "Description could not be saved. Please revise and try again."},
                    status=400,
                )
        except Exception:
            logger.exception("Description classifier failed for org %s", membership.org_id)
            return JsonResponse(
                {"error": "Unable to verify description right now. Please try again later."},
                status=503,
            )

    org = membership.org
    org.description = desc
    org.save(update_fields=["description"])
    return JsonResponse({"ok": True})


# ---- My Agent (SOUL / USER / ORG identity) ----


@login_required
@require_GET
def agent_page(request):
    from accounts.agent_customization import resolve_agent_customization
    from accounts.models import get_user_org

    cust = resolve_agent_customization(request.user)
    org = get_user_org(request.user)
    # Raw (stored) org description so the editor doesn't auto-save the injected
    # boilerplate back as a real value when the admin merely opens the page.
    org_description_raw = org.description if org else ""
    return render(
        request,
        "accounts/agent.html",
        {"cust": cust, "org_description_raw": org_description_raw},
    )


@login_required
@require_POST
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def soul_update(request):
    """Save the user's personal SOUL override (gated by the org's allow_user_soul)."""
    import logging

    from accounts.agent_customization import MAX_SOUL_LENGTH, org_allows_user_soul
    from accounts.models import get_user_org
    from guardrails.classifier import classify_soul_sync

    logger = logging.getLogger(__name__)

    org = get_user_org(request.user)
    if not org_allows_user_soul(org):
        return HttpResponseForbidden("Personal SOUL editing is disabled by your organization.")

    data, err = _parse_json_body(request)
    if err:
        return err

    soul = str(data.get("soul", "")).strip()
    if len(soul) > MAX_SOUL_LENGTH:
        return JsonResponse(
            {"error": f"SOUL must be {MAX_SOUL_LENGTH} characters or fewer."},
            status=400,
        )

    if soul:
        try:
            result = classify_soul_sync(soul, request.user.pk, org.pk if org else None)
            if result.is_suspicious:
                logger.warning("Personal SOUL blocked for user %s: %s", request.user.pk, result.reasoning)
                return JsonResponse(
                    {"error": "SOUL could not be saved. Please revise and try again."},
                    status=400,
                )
        except Exception:
            logger.exception("SOUL classifier failed for user %s", request.user.pk)
            return JsonResponse(
                {"error": "Unable to verify SOUL right now. Please try again later."},
                status=503,
            )

    request.user.soul = soul
    request.user.save(update_fields=["soul"])
    return JsonResponse({"ok": True})


@login_required
@require_POST
def soul_reset(request):
    """Clear the user's personal SOUL; effective value falls back to org, then system."""
    from accounts.agent_customization import org_allows_user_soul, resolve_agent_customization
    from accounts.models import get_user_org

    if not org_allows_user_soul(get_user_org(request.user)):
        return HttpResponseForbidden("Personal SOUL editing is disabled by your organization.")

    request.user.soul = ""
    request.user.save(update_fields=["soul"])
    cust = resolve_agent_customization(request.user)
    return JsonResponse({"ok": True, "soul": cust.soul})


@login_required
@require_POST
@org_admin_required
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def org_soul_update(request):
    """Admin: set the org-wide SOUL baseline."""
    import logging

    from accounts.agent_customization import MAX_SOUL_LENGTH
    from guardrails.classifier import classify_soul_sync

    logger = logging.getLogger(__name__)

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    soul = str(data.get("soul", "")).strip()
    if len(soul) > MAX_SOUL_LENGTH:
        return JsonResponse(
            {"error": f"SOUL must be {MAX_SOUL_LENGTH} characters or fewer."},
            status=400,
        )

    if soul:
        try:
            result = classify_soul_sync(soul, request.user.pk, membership.org_id)
            if result.is_suspicious:
                logger.warning(
                    "Org SOUL blocked for org %s by user %s: %s",
                    membership.org_id, request.user.pk, result.reasoning,
                )
                return JsonResponse(
                    {"error": "SOUL could not be saved. Please revise and try again."},
                    status=400,
                )
        except Exception:
            logger.exception("SOUL classifier failed for org %s", membership.org_id)
            return JsonResponse(
                {"error": "Unable to verify SOUL right now. Please try again later."},
                status=503,
            )

    org = membership.org
    org.soul = soul
    org.save(update_fields=["soul"])
    return JsonResponse({"ok": True})


@login_required
@require_POST
@org_admin_required
def org_soul_reset(request):
    """Admin: clear the org-wide SOUL; effective value falls back to the system default."""
    from accounts.agent_customization import DEFAULT_SOUL

    membership = request.org_membership

    org = membership.org
    org.soul = ""
    org.save(update_fields=["soul"])
    return JsonResponse({"ok": True, "soul": DEFAULT_SOUL})


@login_required
@require_POST
@org_admin_required
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def org_name_update(request):
    """Admin: rename the organization (leaves the slug untouched)."""
    import logging

    from guardrails.classifier import classify_description_sync

    logger = logging.getLogger(__name__)

    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    # Collapse whitespace/newlines — the name reaches the assistant's identity line.
    name = " ".join(str(data.get("name", "")).split())
    if not name:
        return JsonResponse({"error": "Organization name is required."}, status=400)
    if len(name) > MAX_ORG_NAME_LENGTH:
        return JsonResponse(
            {"error": f"Organization name must be {MAX_ORG_NAME_LENGTH} characters or fewer."},
            status=400,
        )

    try:
        result = classify_description_sync(name, request.user.pk, membership.org_id)
        if result.is_suspicious:
            logger.warning(
                "Org name blocked for org %s by user %s: %s",
                membership.org_id, request.user.pk, result.reasoning,
            )
            return JsonResponse(
                {"error": "Name could not be saved. Please revise and try again."},
                status=400,
            )
    except Exception:
        logger.exception("Name classifier failed for org %s", membership.org_id)
        return JsonResponse(
            {"error": "Unable to verify name right now. Please try again later."},
            status=503,
        )

    org = membership.org
    org.name = name
    org.save(update_fields=["name"])
    return JsonResponse({"ok": True})


@login_required
@require_POST
@org_admin_required
def org_allow_user_soul_update(request):
    """Admin: toggle whether members may set a personal SOUL override."""
    membership = request.org_membership

    data, err = _parse_json_body(request)
    if err:
        return err

    allow = bool(data.get("allow_user_soul", False))

    def mutate(prefs):
        prefs["allow_user_soul"] = allow

    update_org_preferences(membership.org_id, mutate)
    return JsonResponse({"ok": True, "allow_user_soul": allow})
