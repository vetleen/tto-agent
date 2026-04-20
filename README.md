# Wilfred (tto-agent)

**Wilfred** is an agentic system for technology transfer offices (TTO). It helps staff with routine workflows—intake, disclosure support, process guidance—via an AI-powered assistant, and provides a **data rooms** workspace: file uploads, chunking, embeddings, and pgvector storage. Thread-based **chat** (WebSocket) uses the LLM with tool use, a canvas editor, sub-agent delegation, and a skill system.

- **Stack:** Django 6, Tailwind CSS v4, Flowbite. Branded UI ("Wilfred"), dark/light theme.
- **Auth:** accounts app — login, signup, email verification, password reset, per-user theme (UserSettings), organizations.
- **Documents:** Data Rooms (UUID), upload (PDF, TXT, MD, HTML, DOCX), Celery processing, LangChain + tiktoken chunking, OpenAI embeddings, hybrid retrieval (pgvector semantic + Postgres full-text search with RRF fusion).
- **Chat:** Thread-based WebSocket assistant at `/chat/`; threads attach one or more data rooms and an optional skill. Includes a multi-canvas editor and sub-agent delegation.
- **Meetings:** First-class meeting objects (`/meetings/`) with live WebSocket transcription, audio/transcript upload, attachments, artifacts, and "minutes with Wilfred" threads that can search linked data rooms.
- **LLM app:** Internal API — OpenAI, Anthropic, Gemini, Moonshot (and others via .env); pipeline registry, model allowlist, three-tier model defaults, transcription registry, optional live API tests.
- **Skills:** Agent skills system (system, organization, and user level) with templates, attached to chat threads to customize assistant behavior and available tools.
- **Feedback:** In-app feedback submission (text + screenshot + console errors) via `/api/feedback/submit/`.

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

3. **Redis** must be running.
   ``` bash
   sudo service redis-server start
   redis-cli ping
   ```

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

# Terminal 2 — CSS (Tailwind v4 + Flowbite): watches for changes and rebuilds
npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch

# Terminal 3 (required for document processing)
celery -A config worker -l info
```

**ASGI (production/WebSockets):** `daphne -b 127.0.0.1 -p 8000 config.asgi:application`

### Tests

```bash
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test
# Or: DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts documents llm chat
```

Unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests so Django uses local SQLite (the `.env` file may point to a remote Postgres where you lack `CREATE DATABASE` permission).

Live LLM API tests: set `TEST_APIS=True` and add API keys to `.env`.

## Architecture

### Django Apps

- **config/** — Project settings, root URL conf, ASGI/WSGI entry points.
- **accounts/** — Auth (login, signup, email verification, password reset), user settings, organizations.
- **documents/** — Data Rooms, file upload, Celery-based processing pipeline (extract → chunk → embed), hybrid retrieval.
- **chat/** — Thread-based WebSocket consumer for LLM chat with streaming, multi-canvas editing, and sub-agent delegation. Users attach data rooms and skills to threads.
- **meetings/** — Meetings with live transcription (WebSocket), audio/text upload transcription, attachments, artifacts, and minutes generation via chat threads.
- **llm/** — Multi-provider LLM abstraction. Entry point: `get_llm_service()` in `llm/service/llm_service.py`. Also hosts the transcription service and registry.
- **agent_skills/** — Skill and template management. Three-tier hierarchy: system, organization, user. Skills customize assistant behavior and tool availability per thread.
- **guardrails/** — Adversarial-content scanning of document chunks (heuristic pre-filter + LLM classifier).
- **feedback/** — User feedback submission (text, screenshot, console errors).
- **core/** — Shared utilities (tokens, preferences), custom error pages, request-ID middleware.

## Documents app

- **URLs:** `/data-rooms/` (list), `/data-rooms/<uuid>/documents/` (documents), `/data-rooms/<uuid>/documents/upload/`, delete/rename/archive via UI, chunks at `.../documents/<id>/chunks/`.
- **Flow:** Upload → Celery `process_document_task` → extract text → chunk (tiktoken, max 1200 tokens, overlap 100) → save chunks → populate tsvector `search_vector` + embed/pgvector when `PGVECTOR_CONNECTION` set → guardrails scan.
- **Retrieval:** Hybrid search combines pgvector semantic similarity with Postgres full-text search using Reciprocal Rank Fusion (RRF). Gracefully degrades to either backend if the other is unavailable. See `documents/services/retrieval.py`.

## Chat app

- **URLs:** `/chat/` (home, thread list and chat UI), `/chat/threads/<uuid>/delete/`, `/chat/threads/<uuid>/canvas/export/`, `/chat/threads/<uuid>/canvas/import/`.
- **Threads:** Each `ChatThread` can attach multiple data rooms (many-to-many via `ChatThreadDataRoom`) and any number of `AgentSkill`s.
- **Canvas:** Per-thread document editor (`ChatCanvas`) with multi-canvas support, checkpoint history, DOCX export/import, and save-to-data-room. Up to 3 canvases can be active at once; only active canvases are included in the LLM's context.
- **Sub-agents:** Delegate tasks to background or timeout-bounded sub-agent runs (`SubAgentRun`) with tiered model selection (cheap/mid/top).
- **Tools available in chat:**
  - `search_documents` — Hybrid (semantic + full-text) search on attached data rooms
  - `read_document` — Read full document content
  - `active_canvas` — Choose which canvases are active in context (max 3)
  - `write_canvas` / `edit_canvas` — Create/overwrite or find-replace on a canvas
  - `update_tasks` — Manage the thread's task list (`ThreadTask`)
  - `web_fetch` — Fetch and extract web page text
  - `brave_search` — Web search via Brave Search API
  - `create_subagent` — Delegate tasks to sub-agents (results returned via system prompt)
  - `save_meeting_minutes` — Persist canvas content as meeting minutes (meeting-attached threads)
  - Skill tools: `create_skill`, `edit_skill`, `delete_skill`, `attach_skills`, `view_template`, `load_template_to_canvas`, `save_canvas_to_skill_field`, `show_skill_field_in_canvas`, `list_all_tools`, `inspect_tool`

## Meetings app

- **URLs:** `/meetings/` (list), `/meetings/create/`, `/meetings/<uuid>/`, transcript upload/audio upload, live transcription WebSocket, "create minutes thread" that spawns a pre-attached chat thread.
- **Transcript sources:** live WebSocket streaming, uploaded audio (transcribed asynchronously via Celery), or direct text upload.
- **Data rooms:** Any meeting can link multiple data rooms via `MeetingDataRoom`; the chat tools then search those rooms from the minutes thread.
- **Transcription config:** `TRANSCRIPTION_ALLOWED_MODELS`, `TRANSCRIPTION_DEFAULT_MODEL`, optional per-path overrides `TRANSCRIPTION_DEFAULT_MODEL_LIVE` / `TRANSCRIPTION_DEFAULT_MODEL_UPLOAD`. Requires `ffmpeg` (installed via Heroku apt buildpack).

## LLM app

```python
from llm import get_llm_service
from llm.types import ChatRequest, Message, RunContext

service = get_llm_service()
request = ChatRequest(
    messages=[Message(role="user", content="Hello")],
    stream=False,
    context=RunContext.create(user_id=user.id),
)
response = service.run("simple_chat", request)
```

Streaming: `for event in service.stream("simple_chat", request): ...`

Async: `await service.arun(...)` / `async for event in service.astream(...)`. Model from `LLM_ALLOWED_MODELS` and `LLM_DEFAULT_MODEL`. See `llm/service/llm_service.py` docstring for full usage.

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
| `MOONSHOT_API_KEY` | Optional Moonshot/Kimi LLM provider |
| `BRAVE_SEARCH_API_KEY` | Web search via Brave Search API |
| `LLM_ALLOWED_MODELS` | Model allowlist (comma-separated `provider/model` pairs) |
| `LLM_DEFAULT_MODEL` | Default model for primary tasks |
| `LLM_DEFAULT_MID_MODEL` | Default model for mid-tier tasks (sub-agents) |
| `LLM_DEFAULT_CHEAP_MODEL` | Default model for cheap/fast tasks (sub-agents) |
| `LLM_REQUEST_TIMEOUT`, `LLM_MAX_RETRIES` | Request timeout (seconds) and retry count |
| `TRANSCRIPTION_ALLOWED_MODELS`, `TRANSCRIPTION_DEFAULT_MODEL` | Transcription model allowlist and default (meetings) |
| `TEST_APIS` | Set `True` for live LLM API tests |
| `EMAIL_VERIFICATION_REQUIRED`, `DJANGO_EMAIL_BACKEND` | Auth and email |
| `SENTRY_DSN`, `SENTRY_ENVIRONMENT` | Optional error tracking / performance monitoring |

See `.env.example` for more (e.g. Mailgun, chunk tuning, dev superuser).

## Heroku

- **Buildpacks:** Node then Python.
- **Add-ons:** Heroku Postgres, Heroku Redis.
- **Config:** `DJANGO_SECRET_KEY`, `DJANGO_CSRF_TRUSTED_ORIGINS`; Procfile includes `worker: celery -A config worker -l info --concurrency=10` for document processing.
- **Release:** `migrate` and `collectstatic`; Tailwind built in Node build phase.

## Other

- **Admin:** `/admin/`. Auto superuser when `DJANGO_USER_NAME` and `DJANGO_PASSWORD` set (dev).
- **Auth URLs:** Login, signup, password change/reset, email verification under `/accounts/`.
- **Theme:** UserSettings stores light/dark; toggle in nav for logged-in users.
