# Runbook

Operational reference for Wilfred (tto-agent). For dev setup see README.md, for coding conventions see CLAUDE.md.

## Infrastructure

| Component | Role | Config |
|-----------|------|--------|
| **Daphne** | ASGI server (HTTP + WebSocket) | `config/asgi.py` |
| **Postgres** | Primary database + pgvector embeddings | `DATABASE_URL` |
| **Redis** | Celery broker (db 0), Channels/WebSocket (db 0), Django cache (db 1) | `REDIS_URL` |
| **Celery** | Async task processing (document pipeline, sub-agents, guardrails) | `config/celery.py` |
| **Sentry** | Error tracking + performance monitoring | `SENTRY_DSN` |

## Deployment (Heroku)

**Buildpacks:** Node.js (first, builds Tailwind CSS) → Python.

**Procfile:**
```
web:     daphne -b 0.0.0.0 -p $PORT config.asgi:application
worker:  celery -A config worker -l info --concurrency=10
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
```

**Deploy flow:** `git push heroku main` → release phase runs migrations + collectstatic → web/worker dynos restart.

**Required config vars:** `DJANGO_SECRET_KEY`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `DJANGO_ALLOWED_HOSTS`, at least one LLM API key (`OPENAI_API_KEY`), `LLM_DEFAULT_MODEL`, `LLM_ALLOWED_MODELS`.

**Auto-provisioned by Heroku add-ons:** `DATABASE_URL` (Heroku Postgres), `REDIS_URL` (Heroku Redis).

### Rollback

```bash
heroku releases                          # list releases
heroku rollback v42                      # roll back to specific release
heroku run python manage.py showmigrations  # check if migration rollback needed
```

If the release included a migration, rolling back the code without reversing the migration is usually safe (Django migrations are additive — nullable columns, new tables). Only reverse a migration if it removed columns or tables that the rolled-back code needs.

## Celery Tasks

| Task | App | Retries | Time Limit | Purpose |
|------|-----|---------|------------|---------|
| `process_document_task` | documents | 5 | 600s hard / 540s soft | Extract → chunk → embed uploaded documents |
| `run_subagent_task` | chat | 3 | 600s hard / 540s soft | Execute sub-agent runs |
| `scan_document_chunks` | guardrails | 3 | 300s hard / 270s soft | Adversarial content scanning (heuristic + LLM) |

All tasks use exponential backoff on retry.

### Stuck/failed tasks

```bash
# Check active/reserved tasks
heroku run celery -A config inspect active
heroku run celery -A config inspect reserved

# Purge all pending tasks (destructive — use only if queue is jammed)
heroku run celery -A config purge
```

A document stuck in PROCESSING usually means the Celery task failed after all retries. Check Sentry for the error, fix the cause, then re-trigger:

```python
# Django shell
from documents.tasks import process_document_task
process_document_task.delay(document_id)
```

## Management Commands

```bash
# Backfill full-text search vectors for existing document chunks
python manage.py backfill_search_vectors
python manage.py backfill_search_vectors --batch-size 200

# Generate LLM descriptions for documents missing them
python manage.py backfill_descriptions
python manage.py backfill_descriptions --doc-ids 30 33
```

## Database

**Production:** Postgres via `DATABASE_URL` (Heroku add-on). Connection settings: `conn_max_age=0` (close after each request), `conn_health_checks=True`.

**Local dev:** SQLite when `DATABASE_URL` is unset.

**pgvector:** Used for embedding storage and semantic search. Requires Postgres with the pgvector extension. Configured via `PGVECTOR_CONNECTION` (falls back to `DATABASE_URL`).

### Migrations

```bash
heroku run python manage.py showmigrations  # check status
heroku run python manage.py migrate          # apply (also runs automatically on deploy via release phase)
```

### Backups

```bash
heroku pg:backups:capture                   # manual backup
heroku pg:backups:schedules                 # view schedule
heroku pg:backups:restore b001              # restore specific backup
```

## Monitoring

### Sentry

Enabled when `SENTRY_DSN` is set. Graceful no-op otherwise.

- Auto-instruments: Django views, DB queries, template rendering, Celery tasks, Redis ops.
- `RequestIDMiddleware` (`core/middleware.py`) tags Sentry events with Heroku's `X-Request-ID` for log-to-error correlation.
- Celery tasks tagged with `celery_task_id` and `celery_task_name` via `task_prerun` signal.
- Sample rates (`SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_PROFILES_SAMPLE_RATE`) default to 1.0 (100%) for alpha. Lower when traffic grows.
- Custom error pages: `templates/errors/` (404, 403, 500) with bare fallback at `templates/500.html`.

### Logs

```bash
heroku logs --tail                          # all logs
heroku logs --tail --dyno=worker            # celery worker only
heroku logs --tail --source=app             # app logs only (no router/heroku)
```

Log format: `timestamp level logger [request_id] message`. Control verbosity with `LOG_LEVEL` env var (default: `INFO`). Django framework logs stay at `WARNING`.

Per-app loggers: `accounts`, `chat`, `documents`, `llm`, `core`, `guardrails`, `celery`.

## Redis

```bash
heroku redis:info                           # connection count, memory
heroku redis:cli                            # interactive shell
```

Redis is shared across three uses (Celery broker on db 0, Channels on db 0, Django cache on db 1). If Redis hits memory limits, Celery tasks and WebSocket connections will fail.

For Heroku Redis with TLS (`rediss://`), SSL cert verification is disabled in Channels config (Heroku uses self-signed certs).

## Environment Variables

See `.env.example` for the full list with comments. Key production variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `DJANGO_SECRET_KEY` | Yes | Cryptographic signing (sessions, CSRF) |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Yes | Comma-separated origins for CSRF (e.g., `https://app.herokuapp.com`) |
| `DATABASE_URL` | Auto | Postgres connection (set by Heroku add-on) |
| `REDIS_URL` | Auto | Redis connection (set by Heroku add-on) |
| `OPENAI_API_KEY` | Yes | Embeddings + OpenAI LLM provider |
| `LLM_DEFAULT_MODEL` | Yes | Primary model (e.g., `openai/gpt-5.2`) |
| `LLM_DEFAULT_MID_MODEL` | Yes | Mid-tier model for sub-agents |
| `LLM_DEFAULT_CHEAP_MODEL` | Yes | Fast model for descriptions, classification |
| `LLM_ALLOWED_MODELS` | Yes | Comma-separated `provider/model` allowlist |
| `ANTHROPIC_API_KEY` | No | Anthropic LLM provider |
| `GEMINI_API_KEY` | No | Google Gemini LLM provider |
| `MOONSHOT_API_KEY` | No | Moonshot/Kimi LLM provider |
| `BRAVE_SEARCH_API_KEY` | No | Web search tool in chat |
| `SENTRY_DSN` | No | Error tracking (disabled if unset) |
| `SENTRY_ENVIRONMENT` | No | Sentry environment tag (default: `production`) |
| `EMAIL_SENDING_ENABLED` | No | Enable email delivery (default: `false`) |
| `MAILGUN_API_KEY` | No | Mailgun email provider |
| `LOG_LEVEL` | No | App log verbosity (default: `INFO`) |
| `PGVECTOR_CONNECTION` | No | pgvector DB connection (falls back to `DATABASE_URL`) |
| `DOCUMENT_UPLOAD_MAX_SIZE_BYTES` | No | Max upload size (default: 50 MB) |

### Production security (automatic when `DEBUG=False`)

- `SESSION_COOKIE_SECURE=True` — HTTPS-only session cookies
- `CSRF_COOKIE_SECURE=True` — HTTPS-only CSRF tokens
- `SECURE_HSTS_SECONDS=3600` — HTTP Strict Transport Security
- `SECURE_SSL_REDIRECT=True` — HTTP → HTTPS redirect

## Common Issues

**WebSocket connections dropping:** Usually Redis. Check `heroku redis:info` for memory/connection limits. Heroku Redis hobby tier has a 20-connection limit.

**Document processing not starting:** Celery worker not running. Check `heroku ps` for worker dyno. Check `heroku logs --tail --dyno=worker` for errors.

**"DB locked" in tests:** Expected — tests use SQLite. Not a real issue.

**Rate limiting (429):** Login and signup views are rate-limited via `django_ratelimit`. Uses `X-Forwarded-For` on Heroku (not `REMOTE_ADDR`). See `accounts/views/auth.py`.

**Migrations fail on test DB:** Unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests (see CLAUDE.md).
