from django.apps import AppConfig


class AgentSkillsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agent_skills"
    verbose_name = "Agent Skills"

    def ready(self):
        import agent_skills.tools  # noqa: F401 — register tools on startup

        from django.db.models.signals import post_migrate

        post_migrate.connect(_seed_system_skills, sender=self)


def _seed_system_skills(sender, **kwargs):
    """Ensure system-level skills exist after every migrate."""
    from agent_skills.seed_skills import seed_system_skills

    seed_system_skills()
