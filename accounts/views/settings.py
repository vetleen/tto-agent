import json

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from accounts.models import Membership, Organization, UserSettings


@login_required
@require_POST
def theme_update(request):
    theme_value = (request.POST.get("theme") or "").strip().lower()
    if theme_value not in (UserSettings.Theme.LIGHT, UserSettings.Theme.DARK):
        return JsonResponse({"error": "Invalid theme"}, status=400)
    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    settings.theme = theme_value
    # Dual-write: keep preferences["theme"] in sync during transition
    prefs = settings.preferences or {}
    prefs["theme"] = theme_value
    settings.preferences = prefs
    settings.save()
    return JsonResponse({"theme": settings.theme})


# ---- User Settings Page ----


@login_required
@require_GET
def settings_page(request):
    from core.preferences import get_preferences, get_tier_defaults

    prefs = get_preferences(request.user)
    user_settings, _ = UserSettings.objects.get_or_create(user=request.user)
    user_models = (user_settings.preferences or {}).get("models", {})
    tier_defaults = get_tier_defaults(request.user)

    return render(request, "accounts/settings.html", {
        "resolved": prefs,
        "user_models": user_models,
        "allowed_models": prefs.allowed_models,
        "tiers": [
            {"key": "primary", "label": "Primary model", "desc": "Used for important tasks like chat and writing.", "default_model": tier_defaults["primary"]},
            {"key": "mid", "label": "Mid model", "desc": "Used for tasks that don't need the best model, like text summarization or tagging.", "default_model": tier_defaults["mid"]},
            {"key": "cheap", "label": "Cheap model", "desc": "Used for very simple tasks, like yes/no questions.", "default_model": tier_defaults["cheap"]},
        ],
    })


@login_required
@require_POST
def preferences_models_update(request):
    """Update user's preferred model for a tier."""
    from core.preferences import get_preferences

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tier = data.get("tier", "").strip()
    model = data.get("model", "").strip() or None

    if tier not in ("primary", "mid", "cheap"):
        return JsonResponse({"error": "Invalid tier"}, status=400)

    # Validate model is in the user's allowed list
    if model:
        prefs = get_preferences(request.user)
        if model not in prefs.allowed_models:
            return JsonResponse({"error": "Model not allowed"}, status=400)

    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    prefs_dict = settings.preferences or {}
    models = prefs_dict.get("models", {})
    models[tier] = model
    prefs_dict["models"] = models
    settings.preferences = prefs_dict
    settings.save()

    return JsonResponse({"ok": True, "tier": tier, "model": model})


# ---- Organization Settings Page ----


def _get_admin_membership(user):
    """Return the user's admin membership or None."""
    return (
        Membership.objects.filter(user=user, role=Membership.Role.ADMIN)
        .select_related("org")
        .first()
    )


@login_required
@require_GET
def org_settings_page(request):
    from agent_skills.models import AgentSkill
    from core.preferences import get_system_defaults
    from llm.service.policies import get_allowed_models
    from llm.tools.registry import get_tool_registry

    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    org = membership.org
    org_prefs = org.preferences or {}
    org_allowed = org_prefs.get("allowed_models") or []
    org_models = org_prefs.get("models", {})
    org_tools = org_prefs.get("tools", {})
    org_skills_prefs = org_prefs.get("skills", {})

    SECTION_META = {
        "chat": {
            "label": "Wilfred Chat",
            "description": "Tools available during chat conversations.",
        },
        "document_processing": {
            "label": "Document Processing",
            "description": "Tools used during automated document processing.",
        },
    }
    TOOL_NOTES = {}

    system_models = get_allowed_models()
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
            "description": skill.description,
            "tool_names": skill.tool_names or [],
            "enabled": sp.get("enabled", True) is not False,
            "tools": {t: tool_toggles.get(t, True) is not False for t in (skill.tool_names or [])},
        })

    org_subagent_prefs = org_prefs.get("subagents", {})
    parallel_subagents = org_subagent_prefs.get("parallel", True)

    return render(request, "accounts/org_settings.html", {
        "org": org,
        "system_models": system_models,
        "org_allowed": org_allowed,
        "org_models": org_models,
        "org_models_json": json.dumps(org_models),
        "org_tools_json": json.dumps(org_tools),
        "tool_sections": tool_sections,
        "skills_data": skills_data,
        "skills_data_json": json.dumps(skills_data),
        "parallel_subagents": parallel_subagents,
        "tiers": [
            {"key": "primary", "label": "Primary model", "desc": "Used for important tasks like chat and writing.", "default_model": system_defaults["primary"]},
            {"key": "mid", "label": "Mid model", "desc": "Used for tasks that don't need the best model, like text summarization or tagging.", "default_model": system_defaults["mid"]},
            {"key": "cheap", "label": "Cheap model", "desc": "Used for very simple tasks, like yes/no questions.", "default_model": system_defaults["cheap"]},
        ],
    })


@login_required
@require_POST
def org_allowed_models_update(request):
    """Set org's allowed_models list."""
    from llm.service.policies import get_allowed_models

    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    models = data.get("allowed_models", [])
    if not isinstance(models, list):
        return JsonResponse({"error": "allowed_models must be a list"}, status=400)

    # Validate all are in system allowlist
    system_models = get_allowed_models()
    invalid = [m for m in models if m not in system_models]
    if invalid:
        return JsonResponse({"error": f"Models not in system allowlist: {invalid}"}, status=400)

    org = membership.org
    prefs = org.preferences or {}
    prefs["allowed_models"] = models
    org.preferences = prefs
    org.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "allowed_models": models})


@login_required
@require_POST
def org_models_update(request):
    """Set org's default model for a tier."""
    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tier = data.get("tier", "").strip()
    model = data.get("model", "").strip() or None

    if tier not in ("primary", "mid", "cheap"):
        return JsonResponse({"error": "Invalid tier"}, status=400)

    org = membership.org
    prefs = org.preferences or {}
    org_allowed = prefs.get("allowed_models") or []

    if model and org_allowed and model not in org_allowed:
        return JsonResponse({"error": "Model not in org allowed list"}, status=400)

    models = prefs.get("models", {})
    models[tier] = model
    prefs["models"] = models
    org.preferences = prefs
    org.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "tier": tier, "model": model})


@login_required
@require_POST
def org_tools_update(request):
    """Toggle a tool on/off for the org."""
    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tool_name = data.get("name", "").strip()
    enabled = data.get("enabled", True)

    if not tool_name:
        return JsonResponse({"error": "Tool name required"}, status=400)

    org = membership.org
    prefs = org.preferences or {}
    tools = prefs.get("tools", {})
    tools[tool_name] = bool(enabled)
    prefs["tools"] = tools
    org.preferences = prefs
    org.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "name": tool_name, "enabled": bool(enabled)})


@login_required
@require_POST
def org_subagents_update(request):
    """Update org's sub-agent settings (e.g. parallel toggle)."""
    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    parallel = data.get("parallel", True)

    org = membership.org
    prefs = org.preferences or {}
    subagents = prefs.get("subagents", {})
    subagents["parallel"] = bool(parallel)
    prefs["subagents"] = subagents
    org.preferences = prefs
    org.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "parallel": bool(parallel)})


@login_required
@require_POST
def org_skills_update(request):
    """Toggle a skill or a per-skill tool on/off for the org."""
    membership = _get_admin_membership(request.user)
    if not membership:
        return HttpResponseForbidden("Admin access required.")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    slug = data.get("slug", "").strip()
    if not slug:
        return JsonResponse({"error": "Skill slug required"}, status=400)

    org = membership.org
    prefs = org.preferences or {}
    skills = prefs.get("skills", {})

    if slug not in skills:
        skills[slug] = {}

    tool_name = data.get("tool", "").strip() if "tool" in data else None
    enabled = data.get("enabled", True)

    if tool_name:
        # Per-tool toggle within a skill
        tool_toggles = skills[slug].get("tools", {})
        tool_toggles[tool_name] = bool(enabled)
        skills[slug]["tools"] = tool_toggles
    else:
        # Skill-level toggle
        skills[slug]["enabled"] = bool(enabled)

    prefs["skills"] = skills
    org.preferences = prefs
    org.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "slug": slug, "enabled": bool(enabled)})
