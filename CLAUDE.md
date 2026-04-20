# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wilfred (tto-agent) is an AI-powered assistant for technology transfer offices (TTO). Django 6 app with Tailwind CSS v4/Flowbite UI, WebSocket chat, document processing, and multi-provider LLM integration.

## Bash Command Style

- Never chain commands with `&&` or `;`. Run them as separate Bash calls instead.
- Always write commands on a single line. Never break a command across multiple lines — newlines trigger a shell safety warning.
- When grepping for strings that contain quote characters, use the `-e` flag to avoid consecutive quote characters at word boundaries: `grep -B 5 -A 10 -e '"accepted_content"' file.py`
- Avoid `sed` — it triggers a shell safety warning. Use `awk` instead: `awk 'NR==1326,NR==1336' file.py`
- The working directory persists between Bash tool calls. Don't redundantly `cd` to the repo root.

## Tests

**Important:** Always unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests so Django uses local SQLite (the `.env` may point to a remote Postgres where you lack `CREATE DATABASE` permission):

```bash
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test                    # All tests
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts           # Single app
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth  # Single module
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth.LoginTestCase.test_login  # Single test
```

- The full suite takes ~10 minutes. Only run tests relevant to your changes unless changes are cross-cutting.
- **`chat` is the slow app.** It alone accounts for most of the 10-minute total (heavy modules: `test_chat_end_to_end`, `test_consumer`, `test_subagent`). When verifying a cross-cutting change, prefer running only the *modules* that actually exercise the code path you touched, not the whole `chat` app. Reference points: `agent_skills` ≈ 2 min (120 tests), `core.tests.test_preferences` + `accounts.tests.test_settings_views` together ≈ 2.5 min (86 tests).
- Use a generous timeout (5–10 min) or run in background. Use `TaskOutput` to await background runs rather than polling.
- Run `python manage.py check` and `python manage.py makemigrations --dry-run` first — they're seconds and catch import/model errors before you spend minutes on tests.
- Tracebacks like "DB locked", "Failed to write LLM call log", or `WARNING django.request: Forbidden:` lines are **expected** from error-handling and permission tests. Check the final summary line (`OK` / `FAILED`), not the noise above it.
- When planning new features, always include good test coverage in the plan.
- Set `TEST_APIS=True` in `.env` for live LLM API tests.

## Heroku & Environments

Three-app pipeline `wilfred` (EU region). Each app has its own Postgres (`essential-0`); staging and production also have their own `heroku-redis:mini`.

| App | Stage | Purpose |
|-----|-------|---------|
| `wilfred-dev` | development | DB-only (no dynos). Hosts the shared dev Postgres that local `.env` points at. |
| `wilfred-staging` | staging | Auto-deploys on push to GitHub `main`. Web + worker dynos. |
| `wilfred-production` | production | Promoted from staging via `heroku pipelines:promote -a wilfred-staging` (same slug, no rebuild). Web + worker dynos. |

**Workflow:** push to `main` on GitHub → staging auto-builds → verify on `wilfred-staging` → promote to production. Never `git push heroku main` directly to production — it bypasses staging.

**Local dev shares the `wilfred-dev` Postgres** (its `DATABASE_URL` is in the local `.env`). Staging and production have isolated databases. Two consequences:

1. Local migrations and any destructive shell commands hit the shared dev DB. Be mindful — there is no local-only DB unless you unset `DATABASE_URL`.
2. Tests must unset `DATABASE_URL` and `PGVECTOR_CONNECTION` (see Tests below) — otherwise Django tries to `CREATE DATABASE` on the dev cluster where you lack permission.

Local Redis is always local (`REDIS_URL` defaults to `redis://127.0.0.1:6379/0`); no Heroku Redis is shared with dev.

For deploy details, config vars, rollback, and ops, see `RUNBOOK.md`.

## Architecture

### Django Apps

- **config/** — Project settings, root URL conf, ASGI/WSGI entry points.
- **accounts/** — Auth (login, signup, email verification, password reset), user settings, organizations.
- **documents/** — Data Rooms, file upload, Celery-based processing pipeline (extract → chunk → embed), hybrid retrieval (pgvector + Postgres FTS).
- **chat/** — WebSocket consumer for LLM chat with streaming, multi-canvas editing, sub-agent delegation. Users attach data rooms and skills to threads.
- **meetings/** — Meetings with live WebSocket transcription, audio/text upload, attachments, artifacts, and "minutes with Wilfred" chat threads. Has its own Celery tasks for audio transcription.
- **llm/** — Multi-provider LLM abstraction. Entry point: `get_llm_service()` in `llm/service/llm_service.py`. Also hosts the transcription service and registry.
- **agent_skills/** — Skill and template management (system, organization, user tiers). Skills customize assistant behavior and tool availability per thread.
- **guardrails/** — Adversarial content scanning for document chunks (heuristic pre-filter + LLM classifier).
- **feedback/** — In-app feedback submissions (text + screenshot + console errors) via `/api/feedback/submit/`.
- **core/** — Shared utilities (tokens, preferences), custom error pages, request-ID middleware.

## Chat Tool Labels

When adding a new tool to the chat system, you **must** add display labels in `templates/chat/chat.html` for both:
- **`tool_start`** — present-tense label while running (e.g., "Searching the web...")
- **`tool_end`** — past-tense label when done (e.g., "Searched the web")

Look for the `tool_start` and `tool_end` event handler blocks and add `else if` branches for the new tool name.

## Logging

- Use `logger = logging.getLogger(__name__)` in every module.
- **Never log** passwords, tokens, API keys, session cookies, emails, or raw request/response bodies. Use object IDs instead.
- Log levels: `DEBUG` local only, `INFO` business events, `WARNING` recoverable issues, `ERROR` failed requests/tasks, `CRITICAL` service health.
- Log exceptions with `logger.exception()` to capture stack traces.
- Sentry captures `WARNING+` as events and `INFO+` as breadcrumbs when `SENTRY_DSN` is set.
