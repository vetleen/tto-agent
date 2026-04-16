"""System skill definitions seeded on every migrate."""

from agent_skills.seed_skills.idea_evaluation_designer import IDEA_EVALUATION_DESIGNER
from agent_skills.seed_skills.irl_project_assessor import IRL_PROJECT_ASSESSOR
from agent_skills.seed_skills.rcn_qualification_grant import RCN_QUALIFICATION_GRANT
from agent_skills.seed_skills.skill_creator import SKILL_CREATOR
from agent_skills.seed_skills.web_deep_researcher import WEB_DEEP_RESEARCHER
from agent_skills.seed_skills.written_assignment_writer import WRITTEN_ASSIGNMENT_WRITER

SYSTEM_SKILLS = [SKILL_CREATOR, WRITTEN_ASSIGNMENT_WRITER, RCN_QUALIFICATION_GRANT, WEB_DEEP_RESEARCHER, IRL_PROJECT_ASSESSOR, IDEA_EVALUATION_DESIGNER]

# Cross-app seed skill from the meetings app. Wrapped in try/except so
# the agent_skills app remains importable even if `meetings` is removed
# from INSTALLED_APPS in some stripped test config.
try:
    from meetings.seed_skills.meeting_summarizer import MEETING_SUMMARIZER
    SYSTEM_SKILLS.append(MEETING_SUMMARIZER)
except ImportError:  # pragma: no cover
    pass


def seed_system_skills():
    """Create or update system-level skills. Idempotent."""
    from agent_skills.models import AgentSkill, SkillTemplate

    for skill_data in SYSTEM_SKILLS:
        fields = {
            "name": skill_data["name"],
            "description": skill_data["description"],
            "instructions": skill_data["instructions"],
            "tool_names": skill_data["tool_names"],
        }
        try:
            skill = AgentSkill.objects.get(slug=skill_data["slug"], level="system")
            if any(getattr(skill, k) != v for k, v in fields.items()):
                for k, v in fields.items():
                    setattr(skill, k, v)
                skill.save()
        except AgentSkill.DoesNotExist:
            skill = AgentSkill.objects.create(slug=skill_data["slug"], level="system", **fields)

        # Seed templates from optional "templates" dict
        templates = skill_data.get("templates", {})
        for tmpl_name, tmpl_content in templates.items():
            try:
                tmpl = skill.templates.get(name=tmpl_name)
                if tmpl.content != tmpl_content:
                    tmpl.content = tmpl_content
                    tmpl.save()
            except SkillTemplate.DoesNotExist:
                skill.templates.create(name=tmpl_name, content=tmpl_content)
        # Remove stale seeded templates no longer in seed data.
        # Only clean up when a "templates" key is explicitly present —
        # existing skills without it should not have templates deleted.
        if templates:
            skill.templates.exclude(name__in=templates.keys()).delete()
