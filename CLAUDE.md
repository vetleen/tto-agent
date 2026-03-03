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
```bash
python manage.py test                    # All tests
python manage.py test accounts           # Single app
python manage.py test accounts.tests.test_auth  # Single module
python manage.py test accounts.tests.test_auth.LoginTestCase.test_login  # Single test
```
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
- **documents/** — Projects (UUID-keyed), file upload (PDF/TXT/MD/HTML), Celery-based processing pipeline (extract → chunk via tiktoken → save → optionally embed with pgvector).
- **chat/** — WebSocket consumer (`ProjectChatConsumer`) for per-project LLM chat with streaming. Uses Django Channels + Redis. Consumer at `chat/consumers.py`, routing at `chat/routing.py`.
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

`chat/consumers.py` (`ProjectChatConsumer`) → authenticates user → validates project ownership → on message: builds chat history with token-limited rolling summarization → calls `LLMService.astream()` → streams `StreamEvent`s back over WebSocket.

### Document Processing Flow

Upload → Celery task `process_document_task` (in `documents/tasks.py`) → text extraction → chunking (tiktoken, 1200 tokens max, 100 overlap) → save `DocumentChunk` records → optional pgvector embedding when `PGVECTOR_CONNECTION` is set.

### Infrastructure

- **ASGI**: Django Channels via Daphne. `config/asgi.py` sets up `ProtocolTypeRouter` with HTTP + WebSocket routing.
- **Celery**: Broker is Redis (`REDIS_URL`). Config in `config/celery.py`.
- **Database**: SQLite locally, Postgres in production (via `DATABASE_URL` + dj-database-url). pgvector optional.
- **CSS**: Tailwind v4 CLI, input at `static/src/input.css`, output at `static/src/output.css`. Flowbite for components.
- **Deployment**: Heroku (Node + Python buildpacks). See `Procfile`.

## Key Environment Variables

Configure in `.env` (copy from `.env.example`):
- `DJANGO_DEBUG=True` for local dev
- `REDIS_URL` — Redis for Channels + Celery (default `redis://127.0.0.1:6379/0`)
- `DATABASE_URL` — Postgres connection (SQLite if unset)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` — LLM providers
- `LLM_ALLOWED_MODELS`, `DEFAULT_LLM_MODEL` — Model allowlist and default
