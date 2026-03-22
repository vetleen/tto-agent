"""Skill resolution and access control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.text import slugify

from agent_skills.models import AgentSkill, SkillTemplate

if TYPE_CHECKING:
    from django.contrib.auth import get_user_model

    User = get_user_model()


def get_available_skills(user) -> list[AgentSkill]:
    """Return the effective skill list for a user with shadowing.

    Higher-priority levels shadow lower ones by slug:
    system < org < user.
    """
    from django.db.models import Q

    from accounts.models import Membership

    membership = Membership.objects.filter(user=user).select_related("org").first()

    # Build a single query for all accessible skills
    q = Q(level="system")
    if membership:
        q |= Q(level="org", organization=membership.org)
    q |= Q(level="user", created_by=user)

    # Iterate in priority order (system < org < user) so higher levels
    # shadow lower ones when we overwrite by slug.
    _LEVEL_ORDER = {"system": 0, "org": 1, "user": 2}
    all_skills = AgentSkill.objects.filter(q, is_active=True)

    skills_by_slug: dict[str, AgentSkill] = {}
    for skill in sorted(all_skills, key=lambda s: _LEVEL_ORDER.get(s.level, 0)):
        skills_by_slug[skill.slug] = skill

    return sorted(skills_by_slug.values(), key=lambda s: s.name)


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


def fork_skill(user, source_skill: AgentSkill) -> AgentSkill:
    """Fork a skill to a user-level copy, including templates."""
    slug = source_skill.slug
    base_slug = slug
    counter = 1
    while AgentSkill.objects.filter(slug=slug, level="user", created_by=user).exists():
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1

    new_skill = AgentSkill.objects.create(
        slug=slug,
        name=source_skill.name,
        description=source_skill.description,
        instructions=source_skill.instructions,
        tool_names=list(source_skill.tool_names or []),
        level="user",
        created_by=user,
        parent=source_skill,
    )

    # Copy templates
    for tmpl in source_skill.templates.all():
        SkillTemplate.objects.create(
            skill=new_skill,
            name=tmpl.name,
            content=tmpl.content,
        )

    return new_skill
