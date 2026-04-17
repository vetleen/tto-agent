"""Restart Heroku dynos via the Platform API.

Scheduled via Heroku Scheduler to defeat long-lived RSS bloat on the web
dyno. A full `heroku ps:restart` returns to a known-clean memory baseline
every night, which is a cheap and boring safety net while we chase the
real memory leaks in upload paths and multimodal attachment encoding.

Requires the ``HEROKU_API_KEY`` config var (a long-lived OAuth
authorization with ``write`` scope — created via
``heroku authorizations:create --description "wilfred scheduled restart"
--scope write``).

Usage:
    python manage.py restart_dynos                # restart all web dynos
    python manage.py restart_dynos --type worker  # restart worker dynos
    python manage.py restart_dynos --type all     # restart every dyno
"""
from __future__ import annotations

import logging
import os

import requests
from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

_API_BASE = "https://api.heroku.com"
_HEADERS = {
    "Accept": "application/vnd.heroku+json; version=3",
    "Content-Type": "application/json",
}


class Command(BaseCommand):
    help = "Restart Heroku dynos for this app. Defaults to 'web'."

    def add_arguments(self, parser):
        parser.add_argument(
            "--type",
            default="web",
            help=(
                "Dyno formation name to restart ('web', 'worker', 'all'). "
                "'all' restarts every dyno on the app (DELETE /dynos)."
            ),
        )
        parser.add_argument(
            "--app",
            default=None,
            help=(
                "Heroku app name. Defaults to $HEROKU_APP_NAME, which is "
                "auto-set when the runtime-dyno-metadata lab is enabled."
            ),
        )

    def handle(self, *args, **opts):
        api_key = os.environ.get("HEROKU_API_KEY", "").strip()
        if not api_key:
            raise CommandError(
                "HEROKU_API_KEY is not set. Create a token with "
                '`heroku authorizations:create --scope write --description '
                '"..."` and set it as a config var on the app.',
            )

        app = opts.get("app") or os.environ.get("HEROKU_APP_NAME", "").strip()
        if not app:
            raise CommandError(
                "App name unknown: pass --app or set HEROKU_APP_NAME "
                "(enable the runtime-dyno-metadata lab to have it auto-set).",
            )

        dyno_type = (opts.get("type") or "web").strip().lower()

        if dyno_type == "all":
            url = f"{_API_BASE}/apps/{app}/dynos"
        else:
            # The Platform API accepts a process type name or a specific dyno
            # name (e.g. 'web.1'). Passing the formation name restarts every
            # dyno of that type, which is what we want for the daily sweep.
            url = f"{_API_BASE}/apps/{app}/dynos/{dyno_type}"

        headers = {**_HEADERS, "Authorization": f"Bearer {api_key}"}
        logger.info("restart_dynos: DELETE %s", url)
        resp = requests.delete(url, headers=headers, timeout=15)
        if resp.status_code >= 400:
            # 404 means the formation name doesn't exist (e.g. asking for
            # 'worker' on a web-only app). Surface it clearly without leaking
            # the token.
            raise CommandError(
                f"Heroku API returned {resp.status_code}: {resp.text[:500]}",
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Restart requested: app={app} type={dyno_type} "
                f"status={resp.status_code}",
            ),
        )
