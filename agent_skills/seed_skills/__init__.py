"""System skill definitions seeded on every migrate."""

import logging
from pathlib import Path

from agent_skills.seed_skills.assistant_loop_tools import ASSISTANT_LOOP_TOOLS
from agent_skills.seed_skills.canvas_collaborator import CANVAS_COLLABORATOR
from agent_skills.seed_skills.data_room_tools import DATA_ROOM_TOOLS
from agent_skills.seed_skills.image_generator import IMAGE_GENERATOR
from agent_skills.seed_skills.patent_searcher import PATENT_SEARCHER
from agent_skills.seed_skills.skill_creator import SKILL_CREATOR
from agent_skills.seed_skills.slide_deck_collaborator import SLIDE_DECK_COLLABORATOR
from agent_skills.seed_skills.web_deep_researcher import WEB_DEEP_RESEARCHER
from agent_skills.seed_skills.web_research_tools import WEB_RESEARCH_TOOLS
from agent_skills.seed_skills.web_researcher import WEB_RESEARCHER

SYSTEM_SKILLS = [SKILL_CREATOR, WEB_DEEP_RESEARCHER, WEB_RESEARCHER, PATENT_SEARCHER, IMAGE_GENERATOR, ASSISTANT_LOOP_TOOLS, CANVAS_COLLABORATOR, DATA_ROOM_TOOLS, WEB_RESEARCH_TOOLS, SLIDE_DECK_COLLABORATOR]

# Cross-app seed skill from the meetings app. Wrapped in try/except so
# the agent_skills app remains importable even if `meetings` is removed
# from INSTALLED_APPS in some stripped test config.
try:
    from meetings.seed_skills.meeting_summarizer import MEETING_SUMMARIZER
    SYSTEM_SKILLS.append(MEETING_SUMMARIZER)
except ImportError:  # pragma: no cover
    pass


logger = logging.getLogger(__name__)

# Root of the convention-based seed-resource tree: any file under
# ``resources/<skill-slug>/`` is seeded as a SkillResource on that skill.
_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def _resource_dir_for_slug(slug: str):
    """Locate a skill's seed-resource folder, tolerating ``_``/``-`` spelling
    (slugs vary across seed skills; the folder may use either)."""
    for candidate in (slug, slug.replace("_", "-"), slug.replace("-", "_")):
        directory = _RESOURCES_DIR / candidate
        if directory.is_dir():
            return directory
    return None


def _seed_file_resources(skill, slug: str) -> None:
    """Seed every file in ``resources/<slug>/`` as a file-backed SkillResource.

    Convention-based: drop a file in the folder and it becomes a bundled resource
    named by its filename. Idempotent (byte-hash guarded in ``seed_file_resource``).
    The folder is authoritative for file-backed rows: a resource whose source file
    was removed is pruned. No-op when the folder is absent, so skills without one
    are never touched.
    """
    from agent_skills.resources import UnsupportedResourceType, seed_file_resource

    directory = _resource_dir_for_slug(slug)
    if directory is None:
        return

    files = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and not p.name.startswith(".")
    ]
    for path in files:
        try:
            seed_file_resource(skill, data=path.read_bytes(), filename=path.name)
        except UnsupportedResourceType:
            logger.warning("seed: unsupported resource file skipped: %s", path.name)
        except Exception:
            logger.exception("seed: failed to seed resource file %s", path.name)

    # Prune file-backed rows whose source file is gone. Keyed on the folder's
    # actual contents (not seed success) so a transient failure never deletes a
    # still-present file's row; scoped to original_filename!="" so seeded text
    # templates are never touched.
    folder_names = [p.name for p in files]
    skill.templates.exclude(original_filename="").exclude(
        name__in=folder_names
    ).delete()


def seed_system_skills():
    """Create or update system-level skills. Idempotent."""
    from agent_skills.models import AgentSkill, SkillTemplate

    for skill_data in SYSTEM_SKILLS:
        fields = {
            "name": skill_data["name"],
            "emoji": skill_data.get("emoji", ""),
            "description": skill_data["description"],
            "instructions": skill_data["instructions"],
            "tool_names": skill_data["tool_names"],
            # Default to the main agent; sub-agent specializations (and any
            # future shared seeds) opt in explicitly.
            "audience": skill_data.get("audience", AgentSkill.Audience.MAIN),
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
        # existing skills without it should not have templates deleted. Scoped to
        # text rows (original_filename="") so file resources seeded from the
        # resources/<slug>/ folder below are never pruned by the text cleanup.
        if templates:
            skill.templates.filter(original_filename="").exclude(
                name__in=templates.keys()
            ).delete()

        # Seed file resources (image/PDF/text) from an optional
        # resources/<slug>/ folder — convention-based, idempotent.
        _seed_file_resources(skill, skill_data["slug"])
