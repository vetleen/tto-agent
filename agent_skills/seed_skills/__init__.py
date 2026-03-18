"""System skill definitions seeded on every migrate."""

from agent_skills.seed_skills.irl_project_assessor import IRL_PROJECT_ASSESSOR
from agent_skills.seed_skills.rcn_qualification_grant import RCN_QUALIFICATION_GRANT
from agent_skills.seed_skills.skill_creator import SKILL_CREATOR
from agent_skills.seed_skills.web_deep_researcher import WEB_DEEP_RESEARCHER
from agent_skills.seed_skills.written_assignment_writer import WRITTEN_ASSIGNMENT_WRITER

SYSTEM_SKILLS = [SKILL_CREATOR, WRITTEN_ASSIGNMENT_WRITER, RCN_QUALIFICATION_GRANT, WEB_DEEP_RESEARCHER, IRL_PROJECT_ASSESSOR]


def seed_system_skills():
    """Create or update system-level skills. Idempotent."""
    from agent_skills.models import AgentSkill, SkillTemplate

    for skill_data in SYSTEM_SKILLS:
        skill, _ = AgentSkill.objects.update_or_create(
            slug=skill_data["slug"],
            level="system",
            defaults={
                "name": skill_data["name"],
                "description": skill_data["description"],
                "instructions": skill_data["instructions"],
                "tool_names": skill_data["tool_names"],
            },
        )
        # Seed templates from optional "templates" dict
        templates = skill_data.get("templates", {})
        for tmpl_name, tmpl_content in templates.items():
            SkillTemplate.objects.update_or_create(
                skill=skill,
                name=tmpl_name,
                defaults={"content": tmpl_content},
            )
        # Remove stale seeded templates no longer in seed data.
        # Only clean up when a "templates" key is explicitly present —
        # existing skills without it should not have templates deleted.
        if templates:
            skill.templates.exclude(name__in=templates.keys()).delete()
