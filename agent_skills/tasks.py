"""Celery tasks for skill-resource processing and the enable-gate scan.

Thin wrappers over the synchronous logic in ``agent_skills.resources`` so the
work can run off the request (uploads and the approval scan may make LLM calls).
Remember to restart the Celery worker after editing task code.
"""

from __future__ import annotations

import logging

from celery import shared_task

from .models import AgentSkill, SkillResource

logger = logging.getLogger(__name__)


def _get_user(user_id):
    from django.contrib.auth import get_user_model

    if not user_id:
        return None
    return get_user_model().objects.filter(pk=user_id).first()


@shared_task
def process_skill_resource_upload_task(resource_id, user_id=None):
    """Extract + guardrail/PII scan an uploaded resource off the request."""
    from . import resources as svc

    resource = (
        SkillResource.objects.select_related("skill").filter(pk=resource_id).first()
    )
    if resource is None:
        logger.warning("process_skill_resource_upload_task: resource %s gone", resource_id)
        return
    svc.process_upload(resource, _get_user(user_id))


@shared_task
def scan_and_approve_skill_task(skill_id, user_id=None):
    """Run the enable-gate scan for a skill off the request."""
    from . import resources as svc

    skill = AgentSkill.objects.filter(pk=skill_id).first()
    if skill is None:
        logger.warning("scan_and_approve_skill_task: skill %s gone", skill_id)
        return
    svc.scan_and_approve_skill(skill, _get_user(user_id))
