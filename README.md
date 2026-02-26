# Wilfred (tto-agent)

**Wilfred** is an agentic system for technology transfer offices (TTO). It helps staff with routine workflows—intake, disclosure support, process guidance—via an AI-powered assistant, and provides a **documents** workspace where users organize projects, upload PDFs and text files, and have them chunked, embedded, and stored for future retrieval.

- **Web app:** Django 6, Tailwind CSS v4, Flowbite. Branded UI (“Wilfred”) with custom font and dark/light theme.
- **Auth:** Accounts app with login, signup, email verification, password reset, and per-user theme (UserSettings).
- **Documents app:** Projects (UUID-based), document upload (PDF, TXT, MD, HTML), async processing via **Celery**, chunking (LangChain + tiktoken), OpenAI embeddings, and **pgvector** storage when Postgres is used.

## Setup

### Prerequisites

- **Python 3.12** (pinned in `.python-version`; used by Heroku buildpack)
- **Node.js and npm** (for Tailwind CSS and Flowbite)
- **Redis** (required for Django Channels and for Celery broker)
- **Postgres** (for production/Heroku; optional locally with SQLite). For vector search, use Postgres with pgvector and set `PGVECTOR_CONNECTION` (or `DATABASE_URL`).

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies: Django 6, Celery, channels/channels-redis/daphne, langchain/langchain-openai/langchain-community, pypdf, tiktoken, whitenoise, dj-database-url, psycopg2-binary, django-anymail (optional for production email).

3. **Install Node dependencies:**
   ```bash
   npm install
   ```
   Used for Tailwind CSS and Flowbite.

4. **Set up Redis** (required for Celery and Channels):
   - **Windows (WSL):** `sudo apt-get install redis-server` then `redis-server`
   - **Windows (native):** [Redis for Windows](https://github.com/microsoftarchive/redis/releases) or use WSL
   - **Linux/Mac:** `sudo apt-get install redis-server` or `brew install redis`, then start the service
   - **Docker:** `docker run -d -p 6379:6379 redis:latest`  
   Redis default port: 6379. Must be running before starting Django and Celery.

5. **Apply database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Build Tailwind CSS:**
   ```bash
   npm run build
   ```
   For development with auto-rebuild:
   ```bash
   npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch
   ```
   The app serves `static/src/output.css` directly (no Django Compressor for this file).

7. **Configure environment variables:**  
   Copy `.env.example` to `.env` and fill in values. See [Environment variables](#environment-variables) below.

### Running the app

**Django (and Celery worker for document processing):**

```bash
# Terminal 1: web server
python manage.py runserver 8000

# Terminal 2: Celery worker (required for document processing after upload)
celery -A config worker -l info
```

**In development you must run the Celery worker** in a second terminal (as above); otherwise uploaded documents will not be processed and will remain in "Processing".

**ASGI (e.g. production or WebSockets):**
```bash
daphne -b 127.0.0.1 -p 8000 config.asgi:application
```

Document processing (chunking, embedding, vector store) runs asynchronously via Celery; if the worker is not running, uploads will stay in “Processing” until the worker picks them up.

### Running tests

```bash
# All apps (accounts + documents)
python manage.py test

# Only accounts
python manage.py test accounts

# Only documents
python manage.py test documents
```

Run from the project root with the virtual environment activated. Use the project’s `.venv` so all dependencies (e.g. whitenoise) are available.

## Documents app

- **URLs:** Under `/projects/`. List at `/projects/`, project detail at `/projects/<uuid>/`, upload at `/projects/<uuid>/documents/upload/`, delete project/document via UI, chunks at `/projects/<uuid>/documents/<id>/chunks/`.
- **Models:** `Project` (UUID, name, slug, created_by), `ProjectDocument` (file, status, token_count, parser/chunking/embedding metadata), `ProjectDocumentChunk` (text, token_count, source fields).
- **Pipeline:** Upload → Celery task `process_document_task` → extract text (PyPDF/TextLoader), chunk (see below), persist `token_count` and chunks to DB, then embed and add to vector store (when `PGVECTOR_CONNECTION` is set).
- **Chunking** (`documents/services/chunking.py`): LangChain loaders (PDF, TXT, MD, HTML). Markdown-aware splitting when applicable; token-based with tiktoken (fallback estimate if tiktoken fails). Max chunk size 1200 tokens (configurable), overlap 100. If total tokens ≤ max, one chunk. Chunks under 200 tokens are merged into the smallest adjacent chunk; merged chunks can exceed the max. Document `token_count` is set after chunking and saved before the vector step so it persists even if embedding fails.
- **Vector store:** Optional. When `PGVECTOR_CONNECTION` (or `DATABASE_URL` for Postgres) is set, embeddings are stored in pgvector via LangChain PGVector and OpenAI embeddings. Idempotent per document: delete by document_id then add.

## Environment variables

Create a `.env` file in the project root (see `.env.example` for a template). `python-dotenv` loads it in `config/settings.py`.

| Variable | Purpose |
|----------|---------|
| `DJANGO_SECRET_KEY` | Required when `DEBUG=False`. Generate e.g. `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS` | Debug mode and allowed hosts |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | For production; set to app URL (e.g. `https://your-app.herokuapp.com`) |
| `REDIS_URL` | Redis URL (default `redis://127.0.0.1:6379/0`). Used by Channels and Celery broker |
| `CELERY_BROKER_URL` | Optional; defaults to `REDIS_URL` |
| `DATABASE_URL` | Postgres URL (Heroku sets this). SQLite used if not set |
| `PGVECTOR_CONNECTION` | Optional; defaults to `DATABASE_URL`. Postgres connection for pgvector; if set, embeddings are stored |
| `OPENAI_API_KEY` | Required for embeddings (and any OpenAI-based features) |
| `EMBEDDING_MODEL` | Optional; default `text-embedding-3-large` |
| `MAX_CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS` | Optional chunking tuning (defaults 1200, 100) |
| `DJANGO_USER_NAME`, `DJANGO_PASSWORD` | Dev-only; auto-create superuser on runserver |
| `DJANGO_EMAIL_BACKEND` | e.g. `django.core.mail.backends.console.EmailBackend` for local |
| `EMAIL_VERIFICATION_REQUIRED` | Set `False` to skip email verification (e.g. dev) |

See `.env.example` for LLM/API and Mailgun-related options.

## Heroku deployment

- **Buildpacks:** Node first (for `npm run build` → Tailwind), then Python.
- **Add-ons:** Heroku Postgres and Heroku Redis (sets `DATABASE_URL`, `REDIS_URL`).
- **Config vars:** Set `DJANGO_SECRET_KEY`, `DJANGO_CSRF_TRUSTED_ORIGINS`, and optionally `EMAIL_VERIFICATION_REQUIRED=False`. For document embeddings set `OPENAI_API_KEY` and optionally `PGVECTOR_CONNECTION` (or rely on `DATABASE_URL` for same DB).
- **Procfile:** `web: daphne -b 0.0.0.0 -p $PORT config.asgi:application`. Add a **worker** process for Celery: `worker: celery -A config worker -l info` so document processing runs.
- **Release:** `migrate` and `collectstatic`; Tailwind is built in the Node build phase (`npm run build`), not in release.
- **Gotchas:** CSRF 403 → set `DJANGO_CSRF_TRUSTED_ORIGINS`. Channel layer SSL → settings use `ssl_cert_reqs=ssl.CERT_NONE` for `rediss://`. Email verification → set `EMAIL_VERIFICATION_REQUIRED=False` or verify users in DB.

## UI and branding

- **Tailwind + Flowbite:** Source `static/src/input.css`, output `static/src/output.css`. Build: `npm run build`; dev watch: `npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch`. Base template links directly to `output.css`.
- **Wilfred:** Nav bar shows “Wilfred” with Delicious Handrawn font and a robot emoji (🤖). Dark mode is class-based (`.dark` on `<html>`); logged-in users’ theme is stored in `UserSettings`.
- **Documents UI:** Project list and detail (Flowbite/Tailwind), upload form, documents table with relative upload date, delete actions with confirmation.

## Admin and dev

- **Admin:** `/admin/`. Superuser can be auto-created on runserver when `DJANGO_USER_NAME` and `DJANGO_PASSWORD` are set.
- **Auth URLs:** Login `/accounts/login/`, signup `/accounts/signup/`, delete account `/accounts/delete/`, password change/reset, email verification (see accounts app and templates).

## Auth flows

- Login: `/accounts/login/`
- Signup: `/accounts/signup/`
- Delete account: `/accounts/delete/`
- Password change: `/accounts/password_change/`
- Password reset: `/accounts/password_reset/`
- Email verification: when `EMAIL_VERIFICATION_REQUIRED=True`, signup → “Check your email” → link verifies and logs in. URLs under `/accounts/verify-email/`. Set `EMAIL_VERIFICATION_REQUIRED=False` in dev to skip.

## Email and verification

- **Backend:** Set via `DJANGO_EMAIL_BACKEND` (e.g. console for dev, anymail for Mailgun in production).
- **Verification:** Token expiry configurable with `EMAIL_VERIFICATION_TIMEOUT`. Resend has rate limits. See accounts app for templates and flows.
- **Production:** Use django-anymail and set `EMAIL_SENDING_ENABLED`, `DEFAULT_FROM_EMAIL`, and provider keys (e.g. Mailgun). See `.env.example` comments.

## User settings and dark mode

- **UserSettings** (accounts): OneToOne to User, stores `theme` (light/dark). Toggle in nav for logged-in users; theme persisted via `POST /accounts/settings/theme/`.
- **Anonymous:** Theme from localStorage or system preference (no server persistence).
