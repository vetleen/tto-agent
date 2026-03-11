"""Skill resolution and access control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_skills.models import AgentSkill

if TYPE_CHECKING:
    from django.contrib.auth import get_user_model

    User = get_user_model()


def get_available_skills(user) -> list[AgentSkill]:
    """Return the effective skill list for a user with shadowing.

    Higher-priority levels shadow lower ones by slug:
    system < org < user.
    """
    from accounts.models import Membership

    # 1. System skills
    skills_by_slug: dict[str, AgentSkill] = {}
    for skill in AgentSkill.objects.filter(level="system", is_active=True):
        skills_by_slug[skill.slug] = skill

    # 2. Org skills (shadow system)
    membership = Membership.objects.filter(user=user).select_related("org").first()
    if membership:
        for skill in AgentSkill.objects.filter(
            level="org", organization=membership.org, is_active=True
        ):
            skills_by_slug[skill.slug] = skill

    # 3. User skills (shadow org/system)
    for skill in AgentSkill.objects.filter(
        level="user", created_by=user, is_active=True
    ):
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
