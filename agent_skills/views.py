"""Views for the Skills management UI."""

from __future__ import annotations

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from accounts.models import Membership
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    get_accessible_skills,
    get_user_skill_prefs,
    shadowing_default,
    can_edit_skill,
    create_org_skill,
    create_user_skill,
    fork_skill,
    get_skill_for_user,
    promote_skill_to_org,
    set_user_skill_selection,
)

logger = logging.getLogger(__name__)


# ----- Helpers -----------------------------------------------------------


def _user_org(user):
    """Return the user's organization (first membership) or None."""
    membership = Membership.objects.filter(user=user).select_related("org").first()
    return membership.org if membership else None


def _is_org_admin(user, organization) -> bool:
    if organization is None:
        return False
    return Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()


def _relative_date(value) -> str:
    """Render a datetime as 'today', 'yesterday', 'N days ago', etc."""
    if value is None:
        return ""
    now = timezone.now()
    if timezone.is_naive(value):
        value = timezone.make_aware(value)
    delta_days = (timezone.localdate(now) - timezone.localdate(value)).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 30:
        return f"{delta_days} days ago"
    months = delta_days // 30
    if months <= 11:
        return "1 month ago" if months == 1 else f"{months} months ago"
    years = delta_days // 365
    return "1 year ago" if years == 1 else f"{years} years ago"


_LEVEL_DISPLAY = {"system": "built-in", "org": "organization", "user": "your"}


def _annotate_skills(user, skills: list[AgentSkill]) -> list[dict]:
    """Compute per-row metadata for the list template.

    Returns dicts with: skill, is_enabled, can_edit, has_conflict,
    conflict_label, relative_date.
    """
    user_skill_prefs = get_user_skill_prefs(user)

    # Group all accessible skills by slug for conflict detection.
    accessible = get_accessible_skills(user)
    by_slug: dict[str, list[AgentSkill]] = {}
    for s in accessible:
        by_slug.setdefault(s.slug, []).append(s)

    # For each slug, decide which skill is currently active for the user.
    active_by_slug: dict[str, AgentSkill | None] = {}
    for slug, candidates in by_slug.items():
        pref = user_skill_prefs.get(slug)
        if isinstance(pref, dict):
            sel_id = pref.get("selected_skill_id")
            if sel_id is None:
                active_by_slug[slug] = None
                continue
            chosen = next((c for c in candidates if str(c.id) == str(sel_id)), None)
            active_by_slug[slug] = chosen or shadowing_default(candidates)
        else:
            active_by_slug[slug] = shadowing_default(candidates)

    rows: list[dict] = []
    for skill in skills:
        candidates = by_slug.get(skill.slug, [skill])
        active = active_by_slug.get(skill.slug)
        is_enabled = active is not None and active.pk == skill.pk
        has_conflict = len(candidates) > 1

        conflict_label = ""
        if has_conflict:
            if is_enabled:
                others = [c for c in candidates if c.pk != skill.pk]
                other_levels = ", ".join(
                    _LEVEL_DISPLAY.get(o.level, o.level) for o in others
                )
                conflict_label = f"Replaces the {other_levels} version of this skill"
            elif active is not None:
                conflict_label = (
                    f"Replaced by the {_LEVEL_DISPLAY.get(active.level, active.level)}"
                    f" version: {active.name}"
                )
            else:
                conflict_label = "You have disabled this skill name"

        rows.append({
            "skill": skill,
            "is_enabled": is_enabled,
            "can_edit": can_edit_skill(user, skill),
            "has_conflict": has_conflict,
            "conflict_label": conflict_label,
            "relative_date": _relative_date(skill.updated_at),
        })
    return rows


def _available_skill_tools() -> list[dict]:
    """Return [{name, description}] for tools registered with section='skills'."""
    from llm.tools.registry import get_tool_registry

    registry = get_tool_registry()
    return sorted(
        [
            {
                "name": tool.name,
                "description": tool.description or "",
            }
            for tool in registry.list_tools_by_section("skills").values()
        ],
        key=lambda t: t["name"],
    )


# ----- List + create -----------------------------------------------------


@login_required
@require_http_methods(["GET"])
def skills_list(request):
    org = _user_org(request.user)
    is_org_admin = _is_org_admin(request.user, org)

    accessible = get_accessible_skills(request.user)
    user_skills = sorted(
        [s for s in accessible if s.level == "user"], key=lambda s: s.name
    )
    org_skills = sorted(
        [s for s in accessible if s.level == "org"], key=lambda s: s.name
    )
    system_skills = sorted(
        [s for s in accessible if s.level == "system"], key=lambda s: s.name
    )

    return render(
        request,
        "agent_skills/skills_list.html",
        {
            "user_rows": _annotate_skills(request.user, user_skills),
            "org_rows": _annotate_skills(request.user, org_skills),
            "system_rows": _annotate_skills(request.user, system_skills),
            "user_org": org,
            "is_org_admin": is_org_admin,
        },
    )


@login_required
@require_POST
def skills_create(request):
    name = (request.POST.get("name") or "").strip() or "Untitled skill"
    skill = create_user_skill(request.user, name[:255])
    messages.success(request, f"Created skill '{skill.name}'.")
    return redirect("agent_skills_detail", skill_id=skill.id)


@login_required
@require_POST
def skills_create_org(request):
    org = _user_org(request.user)
    if not _is_org_admin(request.user, org):
        return HttpResponseForbidden("Org admin required.")
    name = (request.POST.get("name") or "").strip() or "Untitled skill"
    try:
        skill = create_org_skill(request.user, name[:255], org)
    except PermissionError:
        return HttpResponseForbidden("Org admin required.")
    messages.success(request, f"Created organization skill '{skill.name}'.")
    return redirect("agent_skills_detail", skill_id=skill.id)


# ----- Detail + save -----------------------------------------------------


@login_required
@require_http_methods(["GET"])
def skills_detail(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return redirect("agent_skills_list")

    editable = can_edit_skill(request.user, skill)
    org = _user_org(request.user)
    is_org_admin = _is_org_admin(request.user, org)

    templates = list(skill.templates.order_by("name"))

    # Colleague count for the org-skill save warning.
    colleague_count = 0
    if skill.level == "org" and skill.organization_id:
        colleague_count = Membership.objects.filter(
            org_id=skill.organization_id
        ).exclude(user=request.user).count()

    # Build slug → name map for system/org skills so JS can show override
    # warnings dynamically as the user edits the slug field.
    override_slug_map = {}
    if skill.level == "user":
        from django.db.models import Q

        q = Q(level="system")
        if org:
            q |= Q(level="org", organization=org)
        for s in AgentSkill.objects.filter(q, is_active=True).values("slug", "name"):
            override_slug_map[s["slug"]] = s["name"]

    return render(
        request,
        "agent_skills/skills_detail.html",
        {
            "skill": skill,
            "templates": templates,
            "templates_json": json.dumps([
                {"id": str(t.id), "name": t.name, "content": t.content}
                for t in templates
            ]),
            "tool_names_json": json.dumps(list(skill.tool_names or [])),
            "editable": editable,
            "available_tools": _available_skill_tools(),
            "is_org_admin": is_org_admin,
            "user_org": org,
            "colleague_count": colleague_count,
            "level_label": skill.get_level_display(),
            "override_slug_map_json": json.dumps(override_slug_map),
        },
    )


def _parse_templates_json(raw: str) -> list[dict]:
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    cleaned = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        cleaned.append({
            "id": entry.get("id") or None,
            "name": name[:255],
            "content": entry.get("content") or "",
        })
    return cleaned


def _parse_tool_names_json(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if isinstance(item, (str, int))]


def _filter_skill_tools(tool_names: list[str]) -> list[str]:
    """Drop chat-section tools — only skill-section tools belong on skills."""
    from llm.tools.registry import get_tool_registry

    registry = get_tool_registry()
    out = []
    for name in tool_names:
        tool = registry.get_tool(name)
        if tool is None:
            # Unknown tools may belong to another app — keep them.
            out.append(name)
            continue
        if getattr(tool, "section", "chat") == "skills":
            out.append(name)
    return out


class SkillFormValidationError(Exception):
    """Raised by ``_apply_skill_form`` when the submitted form is invalid.

    The view catches this and surfaces ``args[0]`` to the user via the
    Django messages framework.
    """


def _apply_skill_form(skill: AgentSkill, request) -> None:
    """Update fields, tools, and templates from POST data on ``skill``.

    Raises ``SkillFormValidationError`` if the submission is structurally
    invalid (e.g. two templates share the same name). Any other database
    error propagates — silently swallowing them previously hid a bug where
    every template on a freshly-copied skill was deleted.
    """
    name = (request.POST.get("name") or skill.name).strip() or skill.name
    emoji = (request.POST.get("emoji") or "").strip()[:16]
    description = request.POST.get("description") or ""
    instructions = request.POST.get("instructions") or ""
    tool_names = _filter_skill_tools(
        _parse_tool_names_json(request.POST.get("tool_names_json", ""))
    )

    # Handle slug updates for user-level skills.
    raw_slug = request.POST.get("slug", "").strip()
    if raw_slug and skill.level == "user":
        from django.utils.text import slugify

        new_slug = slugify(raw_slug)[:64]
        if new_slug and new_slug != skill.slug:
            conflict = AgentSkill.objects.filter(
                slug=new_slug, level="user", created_by=skill.created_by,
            ).exclude(pk=skill.pk).exists()
            if conflict:
                raise SkillFormValidationError(
                    f"You already have a skill with slug '{new_slug}'."
                )
            skill.slug = new_slug

    # Reconcile templates: incoming list is the source of truth.
    incoming = _parse_templates_json(request.POST.get("templates_json", ""))

    # Validate within-submission name uniqueness BEFORE writing anything.
    # The (skill, name) DB constraint would otherwise raise mid-loop and
    # leave the skill in a half-updated state.
    seen_names: set[str] = set()
    for entry in incoming:
        if entry["name"] in seen_names:
            raise SkillFormValidationError(
                f"Two templates share the name '{entry['name']}'. "
                "Template names must be unique within a skill."
            )
        seen_names.add(entry["name"])

    skill.name = name[:255]
    skill.emoji = emoji
    skill.description = description[:1024]
    skill.instructions = instructions
    skill.tool_names = tool_names
    skill.save(update_fields=[
        "slug", "name", "emoji", "description", "instructions", "tool_names", "updated_at",
    ])

    # Delete templates the user removed BEFORE updating kept ones, so a
    # rename like "remove B, rename A→B" doesn't trip the unique constraint.
    keep_existing_ids: set[str] = set()
    for entry in incoming:
        tmpl_id = entry["id"]
        if tmpl_id and skill.templates.filter(pk=tmpl_id).exists():
            keep_existing_ids.add(str(tmpl_id))
    skill.templates.exclude(pk__in=keep_existing_ids).delete()

    # Update kept templates and create new ones.
    for entry in incoming:
        tmpl_id = entry["id"]
        if tmpl_id and str(tmpl_id) in keep_existing_ids:
            tmpl = skill.templates.get(pk=tmpl_id)
            tmpl.name = entry["name"]
            tmpl.content = entry["content"]
            tmpl.save(update_fields=["name", "content", "updated_at"])
        else:
            SkillTemplate.objects.create(
                skill=skill, name=entry["name"], content=entry["content"],
            )


@login_required
@require_POST
def skills_save(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return redirect("agent_skills_list")

    action = request.POST.get("action", "save")

    if action == "save":
        if not can_edit_skill(request.user, skill):
            return HttpResponseForbidden("Cannot edit this skill.")
        try:
            _apply_skill_form(skill, request)
        except SkillFormValidationError as exc:
            messages.error(request, str(exc))
            return redirect("agent_skills_detail", skill_id=skill.id)
        messages.success(request, f"Saved '{skill.name}'.")
        return redirect("agent_skills_list")

    if action == "save_as_user":
        # Make a user copy first, then write the form data into it.
        # ``copy_templates=False`` because _apply_skill_form will recreate
        # the templates from the submitted form data — letting fork_skill
        # also copy them would trip the unique_template_per_skill constraint.
        copy = fork_skill(request.user, skill, copy_templates=False)
        try:
            _apply_skill_form(copy, request)
        except SkillFormValidationError as exc:
            copy.delete()
            messages.error(request, str(exc))
            return redirect("agent_skills_detail", skill_id=skill.id)
        messages.success(request, f"Saved as new copy '{copy.name}'.")
        return redirect("agent_skills_list")

    if action == "save_as_org":
        org = _user_org(request.user)
        if not _is_org_admin(request.user, org):
            return HttpResponseForbidden("Org admin required.")
        try:
            promoted = promote_skill_to_org(
                request.user, skill, org, copy_templates=False
            )
        except PermissionError:
            return HttpResponseForbidden("Org admin required.")
        try:
            _apply_skill_form(promoted, request)
        except SkillFormValidationError as exc:
            promoted.delete()
            messages.error(request, str(exc))
            return redirect("agent_skills_detail", skill_id=skill.id)
        messages.success(request, f"Saved as organization skill '{promoted.name}'.")
        return redirect("agent_skills_list")

    return redirect("agent_skills_detail", skill_id=skill.id)


# ----- Copy / promote / delete / toggle ---------------------------------


@login_required
@require_POST
def skills_copy(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return redirect("agent_skills_list")
    new_skill = fork_skill(request.user, skill)
    messages.success(request, f"Copied as '{new_skill.name}'.")
    return redirect("agent_skills_detail", skill_id=new_skill.id)


@login_required
@require_POST
def skills_promote(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return redirect("agent_skills_list")
    org = _user_org(request.user)
    if not _is_org_admin(request.user, org):
        return HttpResponseForbidden("Org admin required.")
    try:
        promoted = promote_skill_to_org(request.user, skill, org)
    except PermissionError:
        return HttpResponseForbidden("Org admin required.")
    messages.success(request, f"Promoted to organization skill '{promoted.name}'.")
    return redirect("agent_skills_detail", skill_id=promoted.id)


@login_required
@require_POST
def skills_delete(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return redirect("agent_skills_list")
    if not can_edit_skill(request.user, skill):
        return HttpResponseForbidden("Cannot delete this skill.")
    name = skill.name
    skill.delete()
    messages.success(request, f"Deleted '{name}'.")
    return redirect("agent_skills_list")


@login_required
@require_POST
def skills_edit_in_chat(request, skill_id):
    """Open the target skill in a fresh chat thread with Skill Creator attached.

    Creates a thread, pre-loads description + instructions into canvases,
    persists a hidden seed message that primes Wilfred to greet the user
    and reference the skill's current state, and redirects to the chat
    page where the consumer auto-fires the first assistant turn.
    """
    from urllib.parse import urlencode

    from django.urls import reverse

    from agent_skills.tools import load_skill_field_into_canvas
    from chat.models import ChatMessage, ChatThread

    target = get_skill_for_user(request.user, str(skill_id))
    if target is None:
        messages.error(request, "Skill not found.")
        return redirect("agent_skills_list")

    skill_creator = AgentSkill.objects.filter(
        slug="skill-creator", level="system", is_active=True,
    ).first()
    if skill_creator is None:
        logger.error("Skill Creator system skill not found — cannot start edit-in-chat session")
        messages.error(
            request,
            "Editing in chat is unavailable right now (Skill Creator skill is missing).",
        )
        return redirect("agent_skills_list")

    thread = ChatThread.objects.create(
        created_by=request.user,
        skill=skill_creator,
        title=f"Editing {target.name}",
        metadata={
            "source_skill_id": str(target.id),
            "pending_initial_turn": True,
        },
    )

    load_skill_field_into_canvas(thread.id, target, "description")
    # Load instructions LAST so it ends up as the active canvas (set_active_canvas
    # is called inside the helper and the latest call wins).
    load_skill_field_into_canvas(thread.id, target, "instructions")

    tier_label = _LEVEL_DISPLAY.get(target.level, target.level)
    tools_label = ", ".join(target.tool_names) if target.tool_names else "none"
    seed_content = (
        f"The user opened this thread to edit the **{target.name}** skill "
        f"({tier_label} tier, slug `{target.slug}`). I've pre-loaded its "
        f"description and instructions into canvases titled "
        f"\"{target.name} \u2014 description\" and \"{target.name} \u2014 instructions\" "
        f"— both are ready for editing. "
        f"Current name: \"{target.name}\". Current tools: {tools_label}. "
        f"The user can edit any of these via chat or the canvases. "
        f"Greet the user warmly and ask what they'd like to change about "
        f"the {target.name} skill. Don't list everything — just open the "
        f"conversation."
    )
    ChatMessage.objects.create(
        thread=thread,
        role="user",
        content=seed_content,
        is_hidden_from_user=True,
    )

    target_url = f"{reverse('chat_home')}?{urlencode({'thread': str(thread.id)})}"
    return redirect(target_url)


@login_required
@require_POST
def skills_toggle(request, skill_id):
    skill = get_skill_for_user(request.user, str(skill_id))
    if skill is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    enabled_raw = request.POST.get("enabled", "")
    enabled = enabled_raw in ("1", "true", "True", "on")

    result = set_user_skill_selection(request.user, skill, enabled)
    return JsonResponse({"ok": True, **result})
