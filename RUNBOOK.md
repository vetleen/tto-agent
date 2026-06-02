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

**Buildpacks:** apt (first, installs ffmpeg) → Node.js (builds Tailwind CSS) → Python → pgbouncer (`heroku/heroku-buildpack-pgbouncer`, in-dyno connection pooler).

**Procfile:**
```
web:     bin/start-pgbouncer daphne -b 0.0.0.0 -p $PORT config.asgi:application
worker:  bin/start-pgbouncer celery -A config worker -l info --pool=threads --concurrency=${CELERY_WORKER_CONCURRENCY:-8} -B
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
```

**`bin/start-pgbouncer`** (from the pgbouncer buildpack) wraps the `web` and `worker`
processes so their DB connections go through an in-dyno PgBouncer; `release` stays direct
so migrations don't run through a transaction pooler. See the Database section.

**Worker pool & concurrency.** The worker runs the **threads** pool
(`--pool=threads`), Celery's `ThreadPoolExecutor` backend. It needs no monkeypatching
and no extra dependencies, and — unlike the gevent/eventlet pools — it supports embedded
beat (`-B`, below). Note the Celery 5 invocation order: `-A config` is a **global**
option *before* the `worker` subcommand (`celery -A config worker …`), not after it
(Celery 5 removed `-A` as a worker-subcommand option). Wilfred's tasks are largely
I/O-bound (they wait on LLM / transcription / embedding APIs), so threads overlap many
of them in one process at ~constant RAM while still preempting the partly CPU-bound work
(PDF parsing, rerank, tokenization). That shifts the binding constraint to **database
connections, not RAM**: Django holds one connection per thread, so `--concurrency=N` ≈
up to N connections — but the worker runs behind an in-dyno PgBouncer (see Database), so
those multiplex onto `PGBOUNCER_DEFAULT_POOL_SIZE` real connections rather than counting
1:1 against the cap. Concurrency is driven by the `CELERY_WORKER_CONCURRENCY` config var
(default **8**); with PgBouncer fronting Postgres it can be raised for throughput without
increasing real DB connections — no code change, just `heroku config:set`. Local Windows dev is unaffected: it uses the solo
pool (`config/celery.py:win32` override).

**`-B` embeds Celery beat in the worker.** Beat is the scheduler for periodic tasks
(`CELERY_BEAT_SCHEDULE` in `config/settings.py`, e.g. `expire_stale_subagent_runs`
every 120s) and **must run on exactly one process.** The threads pool supports embedding
it (the gevent/eventlet pools reject `-B`), and it is fine while there is a single worker
dyno. Before scaling the worker to 2+ dynos of the same type — or adding a second worker
process type — move beat to its own process (`beat: celery -A config beat -l info`) and
drop `-B` from the workers, or keep `-B` on exactly one process type that never scales
past one dyno. Otherwise every scheduled task fires once per worker dyno.

**Worker memory (R14) & the levers.** The threads pool is a single long-lived process
whose RSS **plateaus and never reclaims** (no `max-tasks-per-child` — that's prefork-only).
On the 512 MB Basic dyno the worker boots at **~250 MB** idle. Once a *real* workload runs —
a document upload (extract → chunk → embed → guardrails LLM scan over every chunk) and/or a
chat/subagent query — the lazily-imported LLM/ML stack (three provider SDKs, tiktoken,
embeddings, the guardrails classifier, pypdf, and FlashRank/onnxruntime *if* reranking) fully
materializes and the plateau climbs to **~500–510 MB**, tipping just over the cap → sustained
but **benign R14 (Memory quota exceeded)** (the process swaps a few MB; ~100% of quota, far
from the R15 / 300% OOM-kill — tasks complete normally). Measured on production 2026-06-02
with a 116-chunk doc + a multi-turn query: doc-processing + guardrails alone reached **~494 MB
*before* the query**, which then added only ~12 MB → ~506 RSS / ~516 total, sustained R14.
Per-component (measured, prod one-off dyno): the plateau is **runtime materialization, not
code imports** (library imports add only ~24 MB beyond Django boot). The big runtime slice is
**tiktoken's BPE encoding (~80 MB**, loaded on any LLM call that counts tokens), followed by
LLM client pools / embeddings / the guardrails classifier over chunks; **FlashRank is only
~45 MB** (the TinyBERT model + onnxruntime session — *not* the ~100–150 MB earlier estimates
assumed).

- **`MALLOC_ARENA_MAX=2`** (config var on staging + production) caps glibc malloc arenas to
  fight threads-pool fragmentation; it dropped the plateau from ~535–550 to ~500 MB. It is
  read once at process start, so a **worker restart** is required to pick it up.
- **`RERANK_ON_WORKER=false`** (the default) stops FlashRank reranking from running *on the
  worker* (verified: the worker process boots with `RERANK_ENABLED=False`), so the
  ~45 MB FlashRank/onnxruntime chunk (a ~3 MB TinyBERT-L-2 model + onnxruntime session) never
  loads there. **This does not, on its own, get
  a real doc+query workload under 512 MB** — production testing showed the plateau stays
  ~505 MB with rerank off, because the rest of the LLM/ML stack already sums to ~250 MB on top
  of the 250 MB idle. (The earlier "drops to ~360–400" estimate was an artifact of a trivial
  1-chunk test doc that under-loaded those other components.) Its real value is keeping
  FlashRank from stacking *another* ~45 MB on top, which under concurrency would push
  toward the R15 kill — so **keep it off** on the Basic worker. Reranking stays **on for
  main-thread chat**, which runs inline on the web dyno (not memory-constrained). Wired in the
  Procfile, which shadows the worker process's `RERANK_ENABLED` from this var
  (`RERANK_ENABLED=${RERANK_ON_WORKER:-false}`), leaving the app-wide `RERANK_ENABLED` (web)
  untouched. Enable `RERANK_ON_WORKER=true` only after a Standard-2X / 1 GB bump gives
  headroom; toggling requires a worker restart. Trade-off when off: subagent retrieval returns
  hybrid-search (RRF) order without the FlashRank re-ranking pass — slightly less precise
  top-k, but functionally complete; main-chat retrieval is unchanged.

**For durable headroom under real production load the lever is Standard-2X (1 GB), not the
free knobs above** — import-footprint trimming won't move the floor much because the worker is
already ~250 MB idle and the full stack loads on any heavy task. Consistent with the standing
decision: accept benign R14 on staging / low load, revisit Standard-2X before heavy production
load.

**Deploy flow:** push to `main` on GitHub → `wilfred-staging` auto-builds (release phase runs migrations + collectstatic, then web/worker dynos restart) → verify on staging → `heroku pipelines:promote -a wilfred-staging` ships the same slug to `wilfred-production`. Never `git push heroku main` directly to production — it bypasses staging. See CLAUDE.md > Heroku & Environments for the full pipeline.

**Required config vars:** `DJANGO_SECRET_KEY`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `DJANGO_ALLOWED_HOSTS`, at least one LLM API key (`OPENAI_API_KEY`), `LLM_DEFAULT_MODEL`, `LLM_ALLOWED_MODELS`.

**Auto-provisioned by Heroku add-ons:** `DATABASE_URL` (Heroku Postgres), `REDIS_URL` (Heroku Redis).

**Provisioning a fresh app:** `app.json` declares the stack, buildpacks, add-ons, formation, and config-var names, so a new app can be created from the manifest (Heroku "Deploy" button or `heroku` setup) instead of re-running the steps by hand. Buildpacks and config vars are **per-app** — `pipelines:promote` copies only the slug, not these — so each app (staging, production) needs them set once. The pgbouncer buildpack is what provides `bin/start-pgbouncer`; without it the wrapped Procfile lines fail.

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
| `scan_document_chunks` | guardrails | 3 | 600s hard / 570s soft | Adversarial content scanning (heuristic + LLM) |
| `transcribe_meeting_chunk_task` | meetings | default | 600s hard / 540s soft | Transcribe a live-meeting audio chunk |
| `transcribe_uploaded_audio_task` | meetings | default | 1800s hard / 1740s soft | Transcribe an uploaded audio file (may be long) |

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

# Enforce data-retention policies. Deletes expired records and redacts old
# LLM logs. Idempotent — safe to run repeatedly. Scheduled daily via Heroku Scheduler.
python manage.py enforce_retention
python manage.py enforce_retention --dry-run
python manage.py enforce_retention --target LLMCallLog   # single target
```

### Data-retention scheduling

`enforce_retention` must run at least daily in staging and production. It handles:

| Target | Retention | Action |
|--------|-----------|--------|
| ChatThread | 365 days after last activity | Delete (cascades messages, attachments, canvases) |
| DataRoom | 365 days after last activity | Delete (cascades documents, chunks) |
| Meeting | 90 days after last activity | Delete (cascades segments, attachments) |
| GuardrailEvent | 180 days | Delete |
| Feedback | 90 days | Delete (removes screenshot from storage) |
| EmailVerificationToken | 1 day | Delete |
| LLMCallLog | 90 days | Redact (preserves cost/usage analytics) |

Provisioned via the Heroku Scheduler add-on (one-off setup):

```bash
heroku addons:create scheduler:standard -a wilfred-staging
heroku addons:create scheduler:standard -a wilfred-production
heroku addons:open scheduler -a wilfred-staging     # then add the job in the UI
heroku addons:open scheduler -a wilfred-production  # then add the job in the UI
```

Job definition (same on both apps):

- Command: `python manage.py enforce_retention`
- Frequency: every day at 03:00 UTC
- Dyno size: Standard-1X

Verify after provisioning:

```bash
heroku run python manage.py enforce_retention --dry-run -a wilfred-staging
heroku logs --tail -a wilfred-staging
```

## Database

**Production:** Postgres via `DATABASE_URL` (Heroku add-on). `conn_max_age=0` (close each connection at the end of its request/task) and `disable_server_side_cursors=True` (required by PgBouncer transaction mode).

**Connection pooling (PgBouncer).** The `essential-0` plan caps the database at **20
connections**, shared across every web and worker dyno. To decouple app concurrency from
that cap, the `web` and `worker` processes run behind an **in-dyno PgBouncer**
(`heroku/heroku-buildpack-pgbouncer`, `transaction` pool mode) via the
`bin/start-pgbouncer` wrapper in the Procfile; the `release` dyno stays direct. Real
Postgres connections per dyno are capped by `PGBOUNCER_DEFAULT_POOL_SIZE`, so total real
connections ≈ `2 × pool_size`:

| DB plan | Conn limit | `PGBOUNCER_DEFAULT_POOL_SIZE` |
|---------|-----------|-------------------------------|
| essential-0 | 20 | 5 |
| essential-2 | 40 | 10 |
| standard-0 | 120 | 25 |

Config vars (set **per app** — staging and production separately; `pipelines:promote`
does not copy them): `PGBOUNCER_POOL_MODE=transaction`, `PGBOUNCER_DEFAULT_POOL_SIZE`
(per table), `PGBOUNCER_MAX_CLIENT_CONN=100`. With PgBouncer in front,
`CELERY_WORKER_CONCURRENCY` can be raised for throughput without increasing real DB
connections (the pool size is the real ceiling). Check live usage with `heroku pg:info`.

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

**Connection limits.** The `mini` plan caps Redis at **20 connections**, shared across
the broker, Channels (WebSockets), and the cache. The Celery broker pool is bounded by
`broker_pool_limit` (default 10); Channels and the cache draw from their own pools, so
raising worker concurrency increases simultaneous Redis usage but not 1:1 per task. With ~20 users holding live WebSocket connections this cap is a
likely early ceiling — watch `heroku redis:info` and move off `mini` if connections
saturate.

For Heroku Redis with TLS (`rediss://`), SSL cert verification is disabled in Channels config (Heroku uses self-signed certs).

## Environment Variables

See `.env.example` for the full list with comments. Key production variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `DJANGO_SECRET_KEY` | Yes | Cryptographic signing (sessions, CSRF) |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Yes | Comma-separated origins for CSRF (e.g., `https://app.herokuapp.com`) |
| `DATABASE_URL` | Auto | Postgres connection (set by Heroku add-on) |
| `REDIS_URL` | Auto | Redis connection (set by Heroku add-on) |
| `CELERY_WORKER_CONCURRENCY` | No | threads-pool worker thread count (default 8; ≈ max worker DB connections). Raise to ~16–20 on a 40-connection Postgres plan. |
| `RERANK_ON_WORKER` | No | Enable FlashRank rerank on the Celery worker (default `false`). Off keeps FlashRank/onnxruntime (~45 MB) from loading on the worker, but does **not** by itself get a real workload under the 512 MB cap (the LLM/ML stack dominates — see *Worker pool & concurrency*); enable only after a Standard-2X bump. Main-chat rerank is controlled separately by `RERANK_ENABLED`. Worker restart required to take effect. |
| `MALLOC_ARENA_MAX` | No | Caps glibc malloc arenas to reduce threads-pool memory fragmentation. Set to `2` on staging + production. Worker restart required to take effect. |
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
| `PGBOUNCER_POOL_MODE` | Heroku | In-dyno PgBouncer mode; keep `transaction` (see Database) |
| `PGBOUNCER_DEFAULT_POOL_SIZE` | Heroku | Real PG conns per dyno (5 = essential-0, 10 = essential-2) |
| `PGBOUNCER_MAX_CLIENT_CONN` | Heroku | Max app→local-pgbouncer conns (default 100) |
| `DOCUMENT_UPLOAD_MAX_SIZE_BYTES` | No | Max upload size (default: 50 MB) |

### Production security (automatic when `DEBUG=False`)

- `SESSION_COOKIE_SECURE=True` — HTTPS-only session cookies
- `CSRF_COOKIE_SECURE=True` — HTTPS-only CSRF tokens
- `SECURE_HSTS_SECONDS=3600` — HTTP Strict Transport Security
- `SECURE_SSL_REDIRECT=True` — HTTP → HTTPS redirect

## First Deploy Gotchas

**CSRF_TRUSTED_ORIGINS:** Set `DJANGO_CSRF_TRUSTED_ORIGINS` to your app URL (e.g., `https://myapp.herokuapp.com`) before the first deploy. Without it every form submission will 403.

**Secret key rotation:** The dev `SECRET_KEY` in `.env` must not be reused in production. Generate a fresh `DJANGO_SECRET_KEY` for Heroku config vars. Rotate any API keys that have appeared in the repo history.

**HSTS max-age:** Currently set to 1 hour (`SECURE_HSTS_SECONDS=3600`). Bump to a longer duration (e.g., 31536000 / 1 year) once HTTPS is confirmed stable.

## Common Issues

**WebSocket connections dropping:** Usually Redis. Check `heroku redis:info` for memory/connection limits. Heroku Redis hobby tier has a 20-connection limit.

**Document processing not starting:** Celery worker not running. Check `heroku ps` for worker dyno. Check `heroku logs --tail --dyno=worker` for errors.

**R14 (Memory quota exceeded) on the worker:** Expected and benign on the 512 MB Basic worker dyno under load — the threads pool plateaus and doesn't reclaim, so it swaps (far from the R15 / 300% kill). A real doc+query workload plateaus at ~505 MB RSS / ~516 total and fires **sustained** R14 even with `MALLOC_ARENA_MAX=2` and `RERANK_ON_WORKER=false` set — those lower the ceiling but don't clear it under real load (the shared LLM/ML import stack dominates; see *Worker pool & concurrency*). Tasks still complete normally. Only escalate to Standard-2X / 1 GB if it climbs toward R15 or causes dyno restarts.

**"DB locked" in tests:** Expected — tests use SQLite. Not a real issue.

**Rate limiting (429):** Login and signup views are rate-limited via `django_ratelimit`. Uses `X-Forwarded-For` on Heroku (not `REMOTE_ADDR`). See `accounts/views/auth.py`.

**Migrations fail on test DB:** Unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests (see CLAUDE.md).
