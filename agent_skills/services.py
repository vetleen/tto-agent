"""Skill resolution and access control."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from django.utils.text import slugify

from agent_skills.models import AgentSkill, SkillTemplate

if TYPE_CHECKING:
    from django.contrib.auth import get_user_model

    User = get_user_model()


# Priority order for shadowing — higher level wins.
_LEVEL_ORDER = {"system": 0, "org": 1, "user": 2}


def _org_disabled_slugs(user) -> set[str]:
    """Return slugs the user's org has disabled (empty set if no membership).

    Hidden from the Skills overview, detail pages, and every other flow that
    resolves skills through ``get_accessible_skills`` / ``get_skill_for_user``.
    The org settings page queries ``AgentSkill`` directly and intentionally
    bypasses this so admins can still toggle disabled skills back on.
    """
    from accounts.models import Membership

    membership = Membership.objects.filter(user=user).select_related("org").first()
    if not membership or not membership.org:
        return set()
    org_skills = (membership.org.preferences or {}).get("skills") or {}
    return {
        slug for slug, pref in org_skills.items()
        if isinstance(pref, dict) and pref.get("enabled", True) is False
    }


def get_accessible_skills(user) -> list[AgentSkill]:
    """Return every active skill the user has access to (no shadowing)."""
    from django.db.models import Q

    from accounts.models import Membership

    membership = Membership.objects.filter(user=user).select_related("org").first()

    q = Q(level="system")
    if membership:
        q |= Q(level="org", organization=membership.org)
    q |= Q(level="user", created_by=user)

    disabled = _org_disabled_slugs(user)
    return [s for s in AgentSkill.objects.filter(q, is_active=True) if s.slug not in disabled]


def shadowing_default(candidates: list[AgentSkill]) -> AgentSkill:
    """Return the highest-priority candidate (system < org < user)."""
    return max(candidates, key=lambda s: _LEVEL_ORDER.get(s.level, 0))


def get_user_skill_prefs(user) -> dict:
    """Return the per-user skill preferences sub-dict from UserSettings."""
    from accounts.models import UserSettings

    try:
        us = UserSettings.objects.get(user=user)
    except UserSettings.DoesNotExist:
        return {}
    prefs = us.preferences or {}
    skills_prefs = prefs.get("skills", {})
    return skills_prefs if isinstance(skills_prefs, dict) else {}


def get_available_skills(user) -> list[AgentSkill]:
    """Return the effective skill list for a user.

    For each slug, the user's explicit selection in
    ``UserSettings.preferences["skills"][slug]`` wins. Absent rows fall back
    to shadowing defaults (system < org < user). A row whose
    ``selected_skill_id`` is ``None`` means the user has explicitly disabled
    that slug.
    """
    accessible = get_accessible_skills(user)
    by_slug: dict[str, list[AgentSkill]] = {}
    for skill in accessible:
        by_slug.setdefault(skill.slug, []).append(skill)

    user_skill_prefs = get_user_skill_prefs(user)

    result: list[AgentSkill] = []
    for slug, candidates in by_slug.items():
        pref = user_skill_prefs.get(slug)
        if pref is not None:
            sel_id = pref.get("selected_skill_id") if isinstance(pref, dict) else None
            if sel_id is None:
                continue  # explicitly disabled
            chosen = next((c for c in candidates if str(c.id) == str(sel_id)), None)
            if chosen is None:
                # Stale selection (skill was deleted or access was revoked).
                chosen = shadowing_default(candidates)
            result.append(chosen)
        else:
            result.append(shadowing_default(candidates))

    return sorted(result, key=lambda s: s.name)


def get_skill_for_user(user, skill_id: str) -> AgentSkill | None:
    """Return the skill if the user has access, else None.

    Access rules:
    - system: always accessible
    - org: user must be a member of that org
    - user: must be the creator
    """
    from accounts.models import Membership

    try:
        skill = AgentSkill.objects.get(pk=skill_id, is_active=True)
    except AgentSkill.DoesNotExist:
        return None

    if skill.slug in _org_disabled_slugs(user):
        return None

    if skill.level == "system":
        return skill

    if skill.level == "org":
        if skill.organization_id and Membership.objects.filter(
            user=user, org_id=skill.organization_id
        ).exists():
            return skill
        return None

    if skill.level == "user":
        if skill.created_by_id == user.pk:
            return skill
        return None

    return None


def can_edit_skill(user, skill: AgentSkill) -> bool:
    """Check if a user can edit a skill.

    - System skills: never editable
    - Org skills: editable by org admins
    - User skills: editable by the creator
    """
    from accounts.models import Membership

    if skill.level == "system":
        return False

    if skill.level == "org":
        return Membership.objects.filter(
            user=user, org_id=skill.organization_id, role=Membership.Role.ADMIN
        ).exists()

    if skill.level == "user":
        return skill.created_by_id == user.pk

    return False


def get_editable_skill_for_user(user, slug: str) -> AgentSkill | None:
    """Look up a skill by slug (with shadowing) and return it only if editable."""
    skills = get_available_skills(user)
    for skill in skills:
        if skill.slug == slug:
            if can_edit_skill(user, skill):
                return skill
            return None
    return None


def create_user_skill(user, name: str, slug: str | None = None) -> AgentSkill:
    """Create a user-level skill with auto-generated slug."""
    if not slug:
        slug = slugify(name)[:64]
    if not slug:
        slug = "skill"

    # Ensure uniqueness for this user
    base_slug = slug
    counter = 1
    while AgentSkill.objects.filter(slug=slug, level="user", created_by=user).exists():
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1

    return AgentSkill.objects.create(
        slug=slug,
        name=name,
        instructions="",
        description="",
        level="user",
        created_by=user,
    )


# Matches "Base name" or "Base name (3)"
_NAME_SUFFIX_RE = re.compile(r"^(?P<base>.*?)(?: \((?P<num>\d+)\))?$")


def _next_user_skill_name(user, source_name: str) -> str:
    """Find the next free copy name like 'Foo (1)', 'Foo (2)' for this user.

    Strips an existing trailing ``(N)`` from ``source_name`` so that copying
    "Foo (2)" produces "Foo (3)" rather than "Foo (2) (1)".
    """
    match = _NAME_SUFFIX_RE.match(source_name.strip())
    base = match.group("base") if match else source_name.strip()
    base = base.strip() or source_name.strip() or "Skill"

    pattern = re.compile(rf"^{re.escape(base)}(?: \((\d+)\))?$")
    used: set[int] = set()
    base_taken = False
    existing_names = AgentSkill.objects.filter(
        level="user", created_by=user
    ).values_list("name", flat=True)
    for existing in existing_names:
        m = pattern.match(existing)
        if not m:
            continue
        num = m.group(1)
        if num is None:
            base_taken = True
        else:
            used.add(int(num))

    if not base_taken:
        return base
    n = 1
    while n in used:
        n += 1
    return f"{base} ({n})"


def fork_skill(
    user, source_skill: AgentSkill, *, copy_templates: bool = True
) -> AgentSkill:
    """Fork a skill to a user-level copy, including templates.

    Parent semantics:
    - Copying a system or org skill: ``parent = source_skill``.
    - Copying a user skill: ``parent = source_skill.parent`` so iterate-and-
      delete chains stay flat (the user can throw away the intermediate
      version without losing the link to the original system/org source).

    The new skill's name receives a ``(1)``/``(2)`` suffix so the user can
    distinguish many copies. The slug uses the existing ``-1``/``-2``
    deduplication.

    ``copy_templates`` defaults to True for callers that want a complete
    standalone copy. The detail-page form action passes ``False`` because
    it then re-creates the templates from the submitted form data; copying
    them here would clash with the ``unique_template_per_skill`` constraint.
    """
    new_name = _next_user_skill_name(user, source_skill.name)

    base_slug = source_skill.slug
    slug = base_slug
    counter = 1
    while AgentSkill.objects.filter(slug=slug, level="user", created_by=user).exists():
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1

    parent = source_skill.parent if source_skill.level == "user" else source_skill

    new_skill = AgentSkill.objects.create(
        slug=slug,
        name=new_name,
        emoji=source_skill.emoji,
        description=source_skill.description,
        instructions=source_skill.instructions,
        tool_names=list(source_skill.tool_names or []),
        level="user",
        created_by=user,
        parent=parent,
    )

    if copy_templates:
        for tmpl in source_skill.templates.all():
            SkillTemplate.objects.create(
                skill=new_skill,
                name=tmpl.name,
                content=tmpl.content,
            )

    return new_skill


def set_user_skill_selection(user, skill: AgentSkill, enabled: bool) -> dict:
    """Toggle the user's enablement for the slug owned by ``skill``.

    Writes ``UserSettings.preferences["skills"][skill.slug]`` to either
    ``{"selected_skill_id": <skill.id>}`` (enabled) or
    ``{"selected_skill_id": None}`` (disabled). The mutual-exclusivity
    invariant is enforced naturally: only one ``selected_skill_id`` exists
    per slug.

    Returns a dict describing what was replaced (if anything), so the caller
    can craft a "Disabled the org version" toast.
    """
    from accounts.models import UserSettings

    us, _ = UserSettings.objects.get_or_create(user=user)
    prefs = dict(us.preferences or {})
    skills_prefs = dict(prefs.get("skills") or {})

    # Detect what was previously active for this slug — either an explicit
    # selection or, if none, the shadowing default among accessible skills.
    accessible = get_accessible_skills(user)
    candidates = [s for s in accessible if s.slug == skill.slug]
    previous: AgentSkill | None = None
    existing_pref = skills_prefs.get(skill.slug)
    if isinstance(existing_pref, dict):
        prev_id = existing_pref.get("selected_skill_id")
        if prev_id is not None:
            previous = next((c for c in candidates if str(c.id) == str(prev_id)), None)
    elif candidates:
        previous = shadowing_default(candidates)

    if enabled:
        skills_prefs[skill.slug] = {"selected_skill_id": str(skill.id)}
    else:
        skills_prefs[skill.slug] = {"selected_skill_id": None}

    prefs["skills"] = skills_prefs
    us.preferences = prefs
    us.save(update_fields=["preferences"])

    replaced = None
    if enabled and previous and previous.pk != skill.pk:
        replaced = {
            "id": str(previous.id),
            "name": previous.name,
            "level": previous.level,
        }

    return {
        "now_active": bool(enabled),
        "replaced": replaced,
    }


def create_org_skill(user, name: str, organization, slug: str | None = None) -> AgentSkill:
    """Create an org-level skill. Caller must be an admin of ``organization``.

    Raises ``PermissionError`` if the user is not an admin of the org.
    """
    from accounts.models import Membership

    is_admin = Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can create org skills.")

    if not slug:
        slug = slugify(name)[:64]
    if not slug:
        slug = "skill"

    base_slug = slug
    counter = 1
    while AgentSkill.objects.filter(
        slug=slug, level="org", organization=organization
    ).exists():
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1

    return AgentSkill.objects.create(
        slug=slug,
        name=name,
        instructions="",
        description="",
        level="org",
        organization=organization,
    )


def promote_skill_to_org(
    user, source_skill: AgentSkill, organization, *, copy_templates: bool = True
) -> AgentSkill:
    """Create an org-level copy of ``source_skill`` (or a no-op if already org).

    The caller must be an admin of ``organization``. Templates are copied
    by default. The new skill's ``parent`` points back to the source so the
    link is preserved.

    ``copy_templates`` defaults to True for callers that want a complete
    standalone copy. The detail-page form action passes ``False`` because
    it then re-creates the templates from the submitted form data; copying
    them here would clash with the ``unique_template_per_skill`` constraint.
    """
    from accounts.models import Membership

    is_admin = Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can promote skills.")

    if source_skill.level == "org" and source_skill.organization_id == organization.id:
        return source_skill

    base_slug = source_skill.slug
    slug = base_slug
    counter = 1
    while AgentSkill.objects.filter(
        slug=slug, level="org", organization=organization
    ).exists():
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1

    new_skill = AgentSkill.objects.create(
        slug=slug,
        name=source_skill.name,
        emoji=source_skill.emoji,
        description=source_skill.description,
        instructions=source_skill.instructions,
        tool_names=list(source_skill.tool_names or []),
        level="org",
        organization=organization,
        parent=source_skill,
    )

    if copy_templates:
        for tmpl in source_skill.templates.all():
            SkillTemplate.objects.create(
                skill=new_skill,
                name=tmpl.name,
                content=tmpl.content,
            )

    return new_skill
