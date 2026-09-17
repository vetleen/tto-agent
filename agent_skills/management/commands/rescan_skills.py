"""Queue the safety scan for skills whose approval is missing or stale.

Rollout helper for the scan-on-save approval gate: a skill created or edited
before the gate re-queued scans on every write can be switched on in the UI yet
refused by the agent's ``chat_skill_attach``. This queues the scan for every
non-system skill that isn't currently approved (``--state`` narrows it), so
they become attachable without anyone toggling them by hand. Runs on the web
dyno; the scans themselves run on the worker.

Usage:
    python manage.py rescan_skills                  # every non-system skill not approved (blocked ones excluded)
    python manage.py rescan_skills --state blocked  # re-scan blocked skills (after their content was fixed)
    python manage.py rescan_skills --state all      # everything not approved, blocked included
    python manage.py rescan_skills --dry-run --limit 50
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

STATES = ("unapproved", "blocked", "all")


class Command(BaseCommand):
    help = "Queue the safety scan for non-system skills that aren't currently approved."

    def add_arguments(self, parser):
        parser.add_argument(
            "--state",
            choices=STATES,
            default="unapproved",
            help=(
                "unapproved (default): not approved and not blocked; "
                "blocked: only blocked skills; all: both."
            ),
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="List the skills that would be queued without changing anything.",
        )
        parser.add_argument(
            "--limit", type=int, default=0,
            help="Stop after this many skills (0 = no limit).",
        )

    def handle(self, *args, **options):
        from agent_skills.models import AgentSkill
        from agent_skills.resources import (
            _scan_user_for_skill,
            approval_info,
            request_skill_rescan,
        )

        state = options["state"]
        dry_run = options["dry_run"]
        limit = options["limit"] or 0

        candidates = (
            AgentSkill.objects.exclude(level=AgentSkill.Level.SYSTEM)
            .filter(is_active=True, deleted_at__isnull=True)
            .order_by("created_at")
        )
        queued = skipped = 0
        for skill in candidates.iterator():
            info = approval_info(skill)
            blocked = skill.scan_state == AgentSkill.ScanState.BLOCKED
            if info["approved"]:
                continue
            if state == "unapproved" and blocked:
                continue
            if state == "blocked" and not blocked:
                continue
            if limit and queued >= limit:
                break
            label = f"{skill.level}/{skill.slug} ({skill.pk}) scan_state={skill.scan_state}"
            if dry_run:
                self.stdout.write(f"  would queue: {label}")
                queued += 1
                continue
            result = request_skill_rescan(skill, _scan_user_for_skill(skill, None))
            if result["approved"]:
                self.stdout.write(f"  approved inline: {label}")
            elif result["dispatched"]:
                self.stdout.write(f"  queued: {label}")
            else:
                self.stdout.write(f"  pending (upload in flight): {label}")
                skipped += 1
            queued += 1

        if dry_run:
            self.stdout.write(self.style.WARNING(f"--dry-run: {queued} skill(s) would be queued."))
        else:
            self.stdout.write(self.style.SUCCESS(f"Processed {queued} skill(s)."))
