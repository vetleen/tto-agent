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
- Tests can take 3+ minutes to run. Use a generous timeout (e.g., 5–10 min) or run in background.
- Tracebacks in test output (e.g., "DB locked", "Failed to write LLM call log") are **expected** — they come from tests that verify error-handling paths, not from actual failures. Check the final summary line for pass/fail counts.

Set `TEST_APIS=True` in `.env` for live LLM API tests.

### Setup
```bash
pip install -r requirements.txt
npm install
python manage.py migrate
```

## Architecture

### Django Apps

- **config/** — Project settings, root URL conf, ASGI/WSGI entry points. Settings loaded from `.env` via python-dotenv.
- **accounts/** — Auth (login, signup, email verification, password reset), user settings (theme). Views split into `views/auth.py` and `views/settings.py`.
- **documents/** — Data Rooms (UUID-keyed document collections), file upload (PDF/TXT/MD/HTML), Celery-based processing pipeline (extract → chunk via tiktoken → save → optionally embed with pgvector). Models: `DataRoom`, `DataRoomDocument`, `DataRoomDocumentChunk`.
- **chat/** — Standalone WebSocket consumer (`ChatConsumer`) for LLM chat with streaming. Decoupled from documents — users attach 0+ data rooms to any chat thread via M2M. Uses Django Channels + Redis. Consumer at `chat/consumers.py`, routing at `chat/routing.py`. Views at `chat/views.py`, URLs at `chat/urls.py`.
- **llm/** — Multi-provider LLM abstraction layer (see below).
- **core/** — Shared app (minimal).

### LLM App Architecture

The `llm` app is the internal LLM abstraction. Key concepts:

- **LLMService** (`llm/service/llm_service.py`) — Facade accessed via `get_llm_service()`. Methods: `run()`, `arun()`, `stream()`, `astream()`. Routes calls to pipelines by `pipeline_id`.
- **Pipelines** (`llm/pipelines/`) — Registered in `PipelineRegistry`. Each pipeline (e.g., `simple_chat`) defines how to process a `ChatRequest`. Extend `BasePipeline`.
- **Tools** (`llm/tools/`) — Registered in `ToolRegistry`. Implement `Tool` interface from `llm/tools/interfaces.py`. Built-in tools in `builtins.py`.
- **Providers** (`llm/core/providers/`) — LangChain-based adapters for OpenAI, Anthropic, Gemini. Provider resolution in `llm/core/registry.py`.
- **Types** (`llm/types/`) — Pydantic models: `ChatRequest`, `Message`, `RunContext`, `ChatResponse`, `StreamEvent`.
- **Policies** (`llm/service/policies.py`) — Model allowlist/default resolution from `LLM_ALLOWED_MODELS` / `DEFAULT_LLM_MODEL`.

Usage pattern:
```python
from llm import get_llm_service
from llm.types import ChatRequest, Message, RunContext
service = get_llm_service()
response = service.run("simple_chat", ChatRequest(
    messages=[Message(role="user", content="Hello")],
    stream=False,
    context=RunContext.create(),
))
```

### WebSocket Chat Flow

`chat/consumers.py` (`ChatConsumer`) connects at `ws/chat/` → authenticates user → on message: optionally attaches data rooms (M2M) → builds chat history with token-limited rolling summarization → calls `LLMService.astream()` → streams `StreamEvent`s back over WebSocket. Tools (`search_documents`, `read_document`) only available when data rooms are attached. Supports `chat.attach_data_room`, `chat.detach_data_room`, and `chat.load_thread` message types.

### Document Processing Flow

Upload → Celery task `process_document_task` (in `documents/tasks.py`) → text extraction → chunking (tiktoken, 1200 tokens max, 100 overlap) → save `DataRoomDocumentChunk` records → optional pgvector embedding when `PGVECTOR_CONNECTION` is set.

### URL Structure

- `/chat/` — Standalone chat interface (`chat.urls`). Thread via `?thread=<uuid>`.
- `/data-rooms/` — Data room management (`documents.urls`). Documents at `/<uuid>/documents/`.

### Infrastructure

- **ASGI**: Django Channels via Daphne. `config/asgi.py` sets up `ProtocolTypeRouter` with HTTP + WebSocket routing.
- **Celery**: Broker is Redis (`REDIS_URL`). Config in `config/celery.py`.
- **Database**: SQLite locally, Postgres in production (via `DATABASE_URL` + dj-database-url). pgvector optional.
- **CSS**: Tailwind v4 CLI, input at `static/src/input.css`, output at `static/src/output.css`. Flowbite for components.
- **Deployment**: Heroku (Node + Python buildpacks). See `Procfile`.

## Working Directory

The working directory persists between Bash tool calls. If already in the repo root, don't redundantly prefix commands with `cd <path> &&` — just run them directly.

## Key Environment Variables

Configure in `.env` (copy from `.env.example`):
- `DJANGO_DEBUG=True` for local dev
- `REDIS_URL` — Redis for Channels + Celery (default `redis://127.0.0.1:6379/0`)
- `DATABASE_URL` — Postgres connection (SQLite if unset)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` — LLM providers
- `LLM_ALLOWED_MODELS`, `DEFAULT_LLM_MODEL` — Model allowlist and default
