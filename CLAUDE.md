# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wilfred (tto-agent) is an AI-powered assistant for technology transfer offices (TTO). Django 6 app with Tailwind CSS v4/Flowbite UI, WebSocket chat, document processing, and multi-provider LLM integration.

## Commands

### Development
```bash
python manage.py runserver 8000          # Django dev server
celery -A config worker -l info          # Celery worker (required for document processing)
daphne -b 127.0.0.1 -p 8000 config.asgi:application  # ASGI with WebSocket support
npm run build                            # Build Tailwind CSS
```

### Tests

**Important:** The `.env` file may contain a `DATABASE_URL` pointing to a remote Postgres database where the user lacks `CREATE DATABASE` permission. Always unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests so Django uses local SQLite:

```bash
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test                    # All tests
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts           # Single app
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth  # Single module
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth.LoginTestCase.test_login  # Single test
```

**Create tests:** When planning new features always consider what should be tested, and ensure the plan includes creating good test coverage for the new feature.

**Notes:**
- The full test suite takes ~10 minutes. As a general rule, only run the tests relevant to your changes (single app, module, or test class). Only run the full suite when changes are cross-cutting or when explicitly asked.
- Use a generous timeout (e.g., 5–10 min) or run in background.
- Tracebacks in test output (e.g., "DB locked", "Failed to write LLM call log") are **expected** — they come from tests that verify error-handling paths, not from actual failures. Check the final summary line for pass/fail counts.

Set `TEST_APIS=True` in `.env` for live LLM API tests.

## Architecture

### Django Apps

- **config/** — Project settings, root URL conf, ASGI/WSGI entry points.
- **accounts/** — Auth (login, signup, email verification, password reset), user settings.
- **documents/** — Data Rooms, file upload, Celery-based processing pipeline (extract → chunk → embed).
- **chat/** — WebSocket consumer for LLM chat with streaming. Users attach data rooms to threads.
- **llm/** — Multi-provider LLM abstraction. Entry point: `get_llm_service()` in `llm/service/llm_service.py`.
- **core/** — Shared utilities (tokens, preferences).

## Chat Tool Labels

When adding a new tool to the chat system, you **must** add corresponding display labels in `templates/chat/chat.html` for both:
- **`tool_start`** — a present-tense "...ing" label shown while the tool runs (e.g., "Searching the web...")
- **`tool_end`** — a past-tense completion label shown when the tool finishes (e.g., "Searched the web")

Look for the `tool_start` and `tool_end` event handler blocks and add `else if` branches for the new tool name.

## Logging

- Use `logger = logging.getLogger(__name__)` in every module. The LOGGING config in `settings.py` routes by app name (`chat`, `documents`, `llm`, `core`, `accounts`).
- Set `LOG_LEVEL` env var to control app log verbosity (default: `INFO`). Django framework logs stay at `WARNING`.
- Every HTTP log line includes a `[request_id]` from Heroku's `X-Request-ID` header (or auto-generated UUID).
- **Never log** passwords, tokens, API keys, session cookies, emails, or raw request/response bodies. Use object IDs instead.
- Log levels: `DEBUG` for local troubleshooting only, `INFO` for normal business events, `WARNING` for unexpected but recoverable issues, `ERROR` for failed requests/tasks, `CRITICAL` for service health issues.
- Log exceptions with `logger.exception()` to capture stack traces.

## Working Directory

The working directory persists between Bash tool calls. If already in the repo root, don't redundantly prefix commands with `cd <path> &&` — just run them directly.
