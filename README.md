# Wilfred (tto-agent)

**Wilfred** is an agentic system for technology transfer offices (TTO). It helps staff with routine workflows—intake, disclosure support, process guidance—via an AI-powered assistant, and provides a **documents** workspace: projects, PDF/text uploads, chunking, embeddings, and pgvector storage. Per-project **chat** (WebSocket) uses the LLM with optional tool use.

- **Stack:** Django 6, Tailwind CSS v4, Flowbite. Branded UI (“Wilfred”), dark/light theme.
- **Auth:** accounts app — login, signup, email verification, password reset, per-user theme (UserSettings).
- **Documents:** Projects (UUID), upload (PDF, TXT, MD, HTML), Celery processing, LangChain + tiktoken chunking, OpenAI embeddings, optional pgvector (Postgres).
- **Chat:** WebSocket assistant per project at `/projects/<uuid>/chat/`; uses llm app (multi-provider, tools).
- **LLM app:** Internal API — OpenAI, Anthropic, Gemini (and others via .env); pipeline registry, allowlist, optional live API tests.

## Setup

### Prerequisites

- **Python 3.12** (`.python-version`)
- **Node.js and npm** (Tailwind, Flowbite)
- **Redis** (Channels + Celery broker)
- **Postgres** (production/Heroku; optional locally). For vector search: Postgres + pgvector; set `PGVECTOR_CONNECTION` or `DATABASE_URL`.

### Installation

1. **Virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```

2. **Dependencies:**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

3. **Redis** must be running (e.g. `docker run -d -p 6379:6379 redis:latest` or local install).

4. **Database and static:**
   ```bash
   python manage.py migrate
   npm run build
   ```

5. **Environment:** Copy `.env.example` to `.env` and set required keys. See [Environment variables](#environment-variables).

### Running

```bash
# Terminal 1
python manage.py runserver 8000

# Terminal 2 (required for document processing)
celery -A config worker -l info
```

**ASGI (production/WebSockets):** `daphne -b 127.0.0.1 -p 8000 config.asgi:application`

### Tests

```bash
python manage.py test
# Or: python manage.py test accounts documents llm chat
```

Live LLM API tests: set `TEST_APIS=True` and add API keys to `.env`.

## Documents app

- **URLs:** `/projects/` (list), `/projects/<uuid>/` (detail), `/projects/<uuid>/chat/` (assistant), `/projects/<uuid>/documents/upload/`, delete/rename via UI, chunks at `.../documents/<id>/chunks/`.
- **Flow:** Upload → Celery `process_document_task` → extract text → chunk (tiktoken, max 1200 tokens, overlap 100) → save chunks → embed and pgvector when `PGVECTOR_CONNECTION` set.

## LLM app

```python
from llm import get_llm_service
from llm.types import ChatRequest, Message, RunContext

service = get_llm_service()
response = service.run("simple_chat", ChatRequest(messages=[Message(role="user", content="Hello")], stream=False, context=RunContext.create()))
```

Async: `await service.arun(...)` / `async for event in service.astream(...)`. Model from `LLM_ALLOWED_MODELS` and `DEFAULT_LLM_MODEL` (or `LLM_DEFAULT_MODEL`). See `llm/examples/channels_consumer_example.py` for WebSocket usage.

## Environment variables

Use `.env` (from `.env.example`). Key variables:

| Variable | Purpose |
|----------|---------|
| `DJANGO_SECRET_KEY` | Required when `DEBUG=False` |
| `REDIS_URL` | Redis for Channels and Celery (default `redis://127.0.0.1:6379/0`) |
| `DATABASE_URL` | Postgres; SQLite if unset |
| `PGVECTOR_CONNECTION` | Optional; defaults to `DATABASE_URL` for embeddings |
| `OPENAI_API_KEY` | Embeddings and OpenAI LLM provider |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Optional LLM providers |
| `LLM_ALLOWED_MODELS`, `DEFAULT_LLM_MODEL` / `LLM_DEFAULT_MODEL` | Model allowlist and default |
| `TEST_APIS` | Set `True` for live LLM API tests |
| `EMAIL_VERIFICATION_REQUIRED`, `DJANGO_EMAIL_BACKEND` | Auth and email |

See `.env.example` for more (e.g. Mailgun, Moonshot, chunk tuning, dev superuser).

## Heroku

- **Buildpacks:** Node then Python.
- **Add-ons:** Heroku Postgres, Heroku Redis.
- **Config:** `DJANGO_SECRET_KEY`, `DJANGO_CSRF_TRUSTED_ORIGINS`; add `worker: celery -A config worker -l info` to Procfile for document processing.
- **Release:** `migrate` and `collectstatic`; Tailwind built in Node build phase.

## Other

- **Admin:** `/admin/`. Auto superuser when `DJANGO_USER_NAME` and `DJANGO_PASSWORD` set (dev).
- **Auth URLs:** Login, signup, delete account, password change/reset, email verification under `/accounts/`.
- **Theme:** UserSettings stores light/dark; toggle in nav for logged-in users.
