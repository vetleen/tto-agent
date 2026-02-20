# tto-agent

## Setup

### Prerequisites
- Python 3.12 (pinned in `.python-version`; used by Heroku buildpack)
- Node.js and npm (for Tailwind CSS and Flowbite)
- Redis (required for Django Channels/WebSocket support)

### Installation Steps

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
   
   Key dependencies include:
   - Django 6.0.1
   - Django Channels (for WebSocket support)
   - channels-redis (Redis backend for Channels)
   - daphne (ASGI server)
   - django-anymail (optional; for production email via Mailgun etc.)
   - openai (OpenAI API client)
   - tiktoken (token counting)
   - whitenoise (static files), dj-database-url + psycopg2-binary (Postgres on Heroku)

3. **Install Node dependencies:**
   ```bash
   npm install
   ```
   
   Includes Tailwind CSS and Flowbite UI components.

4. **Set up Redis:**
   - **Windows (WSL)**: 
     ```bash
     # In WSL terminal:
     sudo apt-get update
     sudo apt-get install redis-server
     redis-server
     ```
   - **Windows (Native)**: Download from [Redis for Windows](https://github.com/microsoftarchive/redis/releases) or use WSL (recommended)
   - **Linux**: `sudo apt-get install redis-server` (Ubuntu/Debian) or `brew install redis` (Mac)
   - **Linux**: `sudo service redis start`
   - **Docker**: `docker run -d -p 6379:6379 redis:latest`
   
   Redis runs on port 6379 by default. Ensure Redis is running before starting the Django server.

5. **Apply database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Build Tailwind CSS (required for styling):**
   ```bash
   npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch
   ```
   
   Keep this running in a separate terminal while developing. The `--watch` flag automatically rebuilds CSS when you make changes.

7. **Configure environment variables:**
   Create a `.env` file in the project root (see [Environment variables](#environment-variables) section below).

### Running the Server

**For WebSocket support (required for chat application):**
```bash
daphne -b 127.0.0.1 -p 8000 config.asgi:application
```

**Alternative: Django 6.0 runserver (also supports WebSockets):**
```bash
python manage.py runserver 8000
```

Note: The chat application requires WebSocket support, so use one of the above methods. The `daphne` server is recommended for production-like environments.

### Heroku deployment

**Option: New app (clean slate)** — If you remade migrations or don't need existing Heroku data, creating a new app avoids DB/migration mismatches:

```bash
# Create app (pick a name or leave blank for a generated one)
heroku create your-app-name
# Or: heroku create

# Add-ons (Postgres + Redis). Heroku sets DATABASE_URL and REDIS_URL automatically.
heroku addons:create heroku-postgresql:essential-0 -a your-app-name
heroku addons:create heroku-redis:mini -a your-app-name

# Buildpacks: Node first (Tailwind), then Python
heroku buildpacks:add --index 1 heroku/nodejs -a your-app-name
heroku buildpacks:add heroku/python -a your-app-name

# Required config vars (replace your-app-name). Secret key: run
#   python -c "import secrets; print(secrets.token_urlsafe(50))"
# then set DJANGO_SECRET_KEY in Dashboard or: heroku config:set DJANGO_SECRET_KEY=<paste> -a your-app-name
heroku config:set DJANGO_CSRF_TRUSTED_ORIGINS=https://your-app-name.herokuapp.com -a your-app-name

# Optional for dev/staging: skip email verification
heroku config:set EMAIL_VERIFICATION_REQUIRED=False -a your-app-name

# Deploy (from repo root; ensure 'heroku' remote points to the new app)
git push heroku main
# Release phase runs migrate + collectstatic automatically.
```

Add any other vars (e.g. `OPENAI_API_KEY`, `LLM_ALLOWED_MODELS`) in Dashboard → Settings → Config Vars or via `heroku config:set` after the first deploy.

**Checklist before first deploy** (if not using the new-app flow above):

1. **Node.js buildpack** (required so Tailwind CSS is built during the build phase):
   ```bash
   heroku buildpacks:add --index 1 heroku/nodejs -a YOUR_APP_NAME
   ```
2. **Add-ons:** Attach Heroku Postgres and Heroku Redis. Heroku sets `DATABASE_URL` and `REDIS_URL` automatically.
3. **Config vars** (Settings → Reveal Config Vars). See [Heroku config vars](#heroku-config-vars) below.
4. **Python version:** Pinned in `.python-version` (Heroku’s Python buildpack uses this; `runtime.txt` is deprecated).

**How it works:**

- **Tailwind CSS** is built in the **Node build phase** (`npm run build` → `build:css`), so `static/src/output.css` is in the slug before the web dyno runs. The template links to it directly (no Django Compressor for that file).
- **Release phase:** `migrate` → `collectstatic` (no Tailwind step; that runs at build time).
- **Procfile:** `web: daphne -b 0.0.0.0 -p $PORT config.asgi:application` (ASGI for HTTP + WebSockets).

**Heroku config vars**

| Variable | Required | Notes |
|----------|----------|--------|
| `DJANGO_SECRET_KEY` | Yes (when `DEBUG=False`) | Generate: `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Yes for form POSTs | Your app URL, e.g. `https://your-app-xxxx.herokuapp.com` (login, signup, etc. will 403 without this) |
| `DATABASE_URL` | Set by Postgres add-on | — |
| `REDIS_URL` | Set by Redis add-on | Heroku Redis uses `rediss://` (TLS); settings skip cert verification for the channel layer |
| `DJANGO_ALLOWED_HOSTS` | Optional | Default includes `.herokuapp.com` |
| `EMAIL_VERIFICATION_REQUIRED` | Optional | Set `False` to allow login without verifying email (e.g. dev/staging) |
| **Production email (Mailgun)** | When using real email | `DJANGO_EMAIL_BACKEND=anymail.backends.mailgun.EmailBackend`, `EMAIL_SENDING_ENABLED=true`, `DEFAULT_FROM_EMAIL`, `MAILGUN_API_KEY`, `MAILGUN_SENDER_DOMAIN`. See [Email modes and django-anymail](#email-modes-and-django-anymail). |

**Gotchas we’ve fixed (for reference):**

- **CSS 404:** Tailwind’s `output.css` must exist in the slug. It is built in the **Node build phase** (`npm run build`), not in the release phase (release runs in a one-off dyno; files written there are not in the slug). The template uses a direct `<link>` to `output.css`; Django Compressor is not used for it (Heroku’s filesystem is read-only at runtime).
- **CSRF 403 on login/signup:** Set `DJANGO_CSRF_TRUSTED_ORIGINS` to your app’s HTTPS origin (e.g. `https://your-app.herokuapp.com`).
- **Chat / channel layer 500, SSL cert verify failed:** Heroku Redis uses TLS with a cert that fails default verification. Settings use `ssl_cert_reqs=ssl.CERT_NONE` for `rediss://` URLs so the channel layer can connect.
- **“Verify your email” after login:** The Heroku DB is separate from local. Either set the user’s `email_verified=True` via `heroku run python manage.py shell`, or set `EMAIL_VERIFICATION_REQUIRED=False` in Config Vars. To create a superuser: `heroku run python manage.py createsuperuser -a YOUR_APP_NAME`.
- **Viewing logs:** `heroku logs --tail` mixes addon (e.g. Redis) output. For web/release only: `heroku logs --tail --source app -a YOUR_APP_NAME`. Check release success: `heroku releases -a YOUR_APP_NAME`.

### Running Tests

```bash
# Account tests (auth, signup, password reset, email verification, email config)
python manage.py test accounts

# Chat application tests (comprehensive, includes WebSocket tests)
python manage.py test llm_chat

# LLM service tests (requires OPENAI_API_KEY)
TEST_APIS=True python manage.py test llm_service
```

## Admin & superuser (dev)
- Admin UI: `/admin/`
- A dev superuser is auto-created on runserver when `DJANGO_USER_NAME` and `DJANGO_PASSWORD` are set.

## Environment variables

Create a `.env` file in the project root:

```env
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
# Production (e.g. Heroku): set to app URL so CSRF works for form POSTs over HTTPS
# DJANGO_CSRF_TRUSTED_ORIGINS=https://your-app.herokuapp.com
DJANGO_PASSWORD_RESET_TIMEOUT=3600

# APIs
OPENAI_API_KEY=sk-...  # Required for LLM functionality
TEST_APIS=False  # Set to True to run LLM service tests

# Redis (for Django Channels/WebSocket support)
REDIS_URL=redis://127.0.0.1:6379/0  # Optional, defaults to localhost:6379

# Superuser (dev-only, auto-created on runserver)
DJANGO_USER_NAME=you@example.com
DJANGO_PASSWORD=your-password

# Email (local dev: console backend; no DEFAULT_FROM_EMAIL required)
DJANGO_EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
# Production/Staging (e.g. Mailgun): set backend, enable sending, and provider vars
# DJANGO_EMAIL_BACKEND=anymail.backends.mailgun.EmailBackend
# EMAIL_SENDING_ENABLED=true
# DEFAULT_FROM_EMAIL=noreply@yourdomain.com
# MAILGUN_API_KEY=
# MAILGUN_SENDER_DOMAIN=yourdomain.com
# MAILGUN_API_URL=   # optional (region-specific endpoint)

# Email verification (optional)
EMAIL_VERIFICATION_REQUIRED=True   # Set False to skip verification (e.g. dev)
EMAIL_VERIFICATION_TIMEOUT=86400   # Token validity in seconds (default: 24h)
```

`python-dotenv` loads `.env` in `config/settings.py`. On Heroku, set these in **Config Vars** (Settings → Reveal Config Vars); see [Heroku deployment](#heroku-deployment) for the required vars table and gotchas.

**Note:** The `REDIS_URL` is optional locally and defaults to `redis://127.0.0.1:6379/0`. Heroku Redis sets `REDIS_URL` (often `rediss://`); the app skips TLS cert verification for the channel layer so WebSockets/chat work.

### Email modes and django-anymail

The app uses Django's standard email API and supports any backend. For production we recommend [django-anymail](https://anymail.dev/) with a transactional provider. Switching providers is done by changing `DJANGO_EMAIL_BACKEND` and the corresponding ANYMAIL env vars; see [Anymail's provider list](https://anymail.dev/en/stable/esps/).

| Mode        | Backend                     | Sends externally? | Typical use   |
| ----------- | --------------------------- | ----------------- | ------------- |
| Local dev   | console / locmem / Mailpit   | No                | dev/test      |
| Staging     | Mailgun sandbox via anymail | Limited           | integration   |
| Production  | Mailgun (real domain)       | Yes               | real sends    |

- **Domain required for real sending.** A verified sender domain is needed for unrestricted delivery.
- **Sandbox (e.g. Mailgun):** Can send only to authorized recipients; see [Mailgun sandbox docs](https://documentation.mailgun.com/en/latest/api-sending.html#sandbox) for limits.
- **Mailgun example:** Set `DJANGO_EMAIL_BACKEND=anymail.backends.mailgun.EmailBackend`, `EMAIL_SENDING_ENABLED=true`, `DEFAULT_FROM_EMAIL`, `MAILGUN_API_KEY`, `MAILGUN_SENDER_DOMAIN`, and optionally `MAILGUN_API_URL` (for region-specific endpoint). Without a real domain you cannot fully test delivery, but the code path and settings can be validated (e.g. via provider dashboard or sandbox).

## Tailwind + Flowbite
- Input: `static/src/input.css`
- Output: `static/src/output.css`
- Build/watch:
```
npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch
```

Flowbite is installed via npm and loaded in `static/src/input.css` and `templates/_base.html`. Dark mode uses the class strategy: `@custom-variant dark` in `input.css` and a `.dark` class on `<html>`; templates use `dark:` variants for dark-mode styling.

## Django Compressor
Compressor is enabled in `config/settings.py` and used in `templates/_base.html` to load the CSS bundle.

## Auth flows
- Login: `/accounts/login/`
- Signup: `/accounts/signup/`
- Delete account: `/accounts/delete/`
- Password change: `/accounts/password_change/`
- Password reset: `/accounts/password_reset/`
- Email verification: `/accounts/verify-email/sent/`, `/accounts/verify-email/<token>/`, `/accounts/verify-email/resend/`, `/accounts/verify-required/` (see [Email verification](#email-verification)).

## Email verification

New signups must verify their email before they can log in (when `EMAIL_VERIFICATION_REQUIRED` is True).

- **Flow:** Sign up → "Check your email" page → user clicks link in email → verified and logged in.
- **Token expiry:** 24 hours (configurable via `EMAIL_VERIFICATION_TIMEOUT`).
- **Resend:** "Resend verification email" on the check-your-email and verify-required pages. Rate limit: 1 minute after signup, then doubles each time (1 → 2 → 4 → 8 minutes); resets after 24 hours.
- **URLs:** `/accounts/verify-email/sent/`, `/accounts/verify-email/<token>/`, `/accounts/verify-email/resend/`, `/accounts/verify-required/`.
- **Templates:** `registration/verify_email_sent.html`, `registration/verify_email_error.html`, `registration/verify_required.html`; email body/subject: `registration/email_verification_*.txt`.
- **Disable:** Set `EMAIL_VERIFICATION_REQUIRED=False` in `.env` to allow login without verification (e.g. for local dev).
- **Production:** Verification links are built with `request.build_absolute_uri()`, so they use the current host and work with HTTPS. Set `DEFAULT_FROM_EMAIL` (e.g. `noreply@yourdomain.com`) for a proper sender; the same `DJANGO_EMAIL_BACKEND` used for password reset is used for verification emails.
- **Admin:** User list shows `email_verified`; `EmailVerificationToken` is registered so you can see pending tokens and expiry.

## User settings & dark mode

**UserSettings** (accounts app): OneToOne to User, created automatically when a user is created (signal). Stores per-user preferences; currently `theme` with choices `light` or `dark` (default `light`). Access via `user.settings.theme`.

**Dark mode** (Flowbite class-based):
- Toggle in the top nav for **logged-in users only**; theme is saved in `UserSettings` and persists across sessions.
- Anonymous users get theme from `localStorage` or system preference (no persistence).
- Implemented via `@custom-variant dark` in `static/src/input.css`, inline script in `<head>` to avoid FOUC, and context processor `accounts.context_processors.theme` that passes `theme` into templates.
- **Theme update:** `POST /accounts/settings/theme/` with `theme=light` or `theme=dark` (login required); returns JSON `{"theme": "…"}`.

## LLM Service (LiteLLM)

The `llm_service` app provides a thin, provider-agnostic layer over [LiteLLM](https://docs.litellm.ai/). All calls go through `completion(**kwargs)` or `acompletion(**kwargs)` with policy (timeout, retry, optional guardrail hooks), and every call is logged to **LLMCallLog** after completion. Cost comes from LiteLLM when available, with an optional fallback in `llm_service/pricing.py` for allowed models.

### API

- **Sync**: `llm_service.client.completion(**kwargs)` — same kwargs as `litellm.completion()` (model, messages, stream, response_format, tools, etc.). Pass-through; add `metadata={}` and optional `user=` for attribution.
- **Async**: `llm_service.client.acompletion(**kwargs)` — same behaviour for async views/workers.
- **Backend**: `llm_service.client.get_client()` returns the configured client (LiteLLM by default). The client implements `BaseLLMClient` so you can swap the backend without changing callers.

### Allowed models

Only models in **LLM_ALLOWED_MODELS** (settings / env) are accepted; others raise `ValueError`. Set `LLM_ALLOWED_MODELS` to a comma-separated list in env, or override in Django settings. Default env: `openai/gpt-4o,openai/gpt-4o-mini`. Multiple providers can be in use at once (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`); set the corresponding API keys in env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

### Metadata

Pass `metadata={"feature": "...", "tenant_id": "...", "request_id": "...", "trace_id": "..."}` (and any other keys) into every call. It is stored as JSON on **LLMCallLog** for debugging and cost allocation. You can also pass `request_id=` as a top-level kwarg; it is indexed.

### Policy

- **Timeout**: `LLM_REQUEST_TIMEOUT` (default 60s).
- **Retries**: `LLM_MAX_RETRIES` (default 2) with exponential backoff on rate limit and transient errors.
- **Guardrails**: Optional pre-call and post-call hooks in settings (`LLM_PRE_CALL_HOOKS`, `LLM_POST_CALL_HOOKS`); receive `LLMRequest` / `LLMResult` and can block, sanitize, or validate.

### Logging

Every call writes one **LLMCallLog** when the call (or stream) ends. Fields include identity (id, created_at, duration_ms, user, metadata, request_id), request (model, is_stream, request_kwargs, prompt_preview), response (provider_response_id, response_model, response_preview), usage/cost (input/output/total tokens, cost_usd, cost_source), and errors (status, error_type, error_message, http_status, retry_count). Log write is bounded by `LLM_LOG_WRITE_TIMEOUT` (default 5s). If the full log fails (e.g. serialization), a minimal row (primitives only) is written so the process is not stuck.

### Basic usage

**Non-streaming:**
```python
from llm_service.client import completion

response = completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Say hello"}],
    user=request.user,  # optional, for attribution
    metadata={"feature": "greeting", "request_id": "req-123"},
)
# response is the raw LiteLLM response (e.g. response.choices[0].message.content)
```

**Streaming (chunks proxied verbatim; log written when stream ends or consumer stops):**
```python
for chunk in completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True):
    print(chunk.choices[0].delta.content, end="")
```

**Async:**
```python
from llm_service.client import acompletion

response = await acompletion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello"}])
```

### Env and settings

- **API keys**: Per provider, in env (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for each provider.
- **Settings**: `LLM_DEFAULT_MODEL`, `LLM_ALLOWED_MODELS`, `LLM_REQUEST_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_LOG_WRITE_TIMEOUT`. Optional: `LLM_PRE_CALL_HOOKS`, `LLM_POST_CALL_HOOKS`.

View logs in admin: `/admin/llm_service/llmcalllog/`.

## Chat Application (`llm_chat`)

The `llm_chat` app provides a real-time chat interface with an LLM assistant, featuring WebSocket-based streaming, markdown rendering, and persistent chat history.

### Features

- **Real-time streaming**: WebSocket-based message streaming using Django Channels
- **Markdown rendering**: Full markdown support with syntax highlighting, code blocks, lists, headers, links, etc.
- **Chat history**: Persistent chat threads with automatic title generation
- **Token-aware context**: Smart chat history truncation using token counts
- **Multi-tab support**: Chat updates sync across multiple browser tabs
- **Connection management**: Automatic reconnection with connection status indicators
- **Accessibility**: Proper ARIA handling and keyboard navigation support

### Architecture

**Models:**
- `ChatThread`: Represents a conversation thread with a user, title, and timestamps
- `ChatMessage`: Individual messages with role (user/assistant), content, status, and token counts

**Services:**
- `ChatService`: Orchestrates LLM streaming, message persistence, and thread title generation
- `assemble_system_instruction()`: Builds system prompts with dynamic chat history
- `assemble_chat_history()`: Retrieves and formats chat history up to token limits

**WebSocket Consumer:**
- `ChatConsumer`: Handles WebSocket connections, message streaming, and group broadcasting
- Routes: `/ws/chat/<thread_id>/`
- Events: `response.output_text.delta`, `final`, `thread.title.updated`, `response.error`

**Views:**
- `chat_view`: Main chat interface (GET/POST)
- `chat_messages_json`: API endpoint for AJAX message loading
- URLs: `/chat/` (new chat), `/chat/<uuid>/` (specific thread)

### Usage

**Access the chat:**
- Navigate to `/chat/` for a new chat
- Navigate to `/chat/<thread_id>/` for an existing thread
- Requires authentication (login required)

**Sending messages:**
- Type in the textarea and press Enter to send
- Use Shift+Enter for multi-line input
- Textarea auto-resizes up to ~5 lines

**Markdown support:**
The chat supports full markdown rendering including:
- Headers (`#`, `##`, `###`, etc.)
- **Bold** (`**text**`), *italic* (`*text*`), ~~strikethrough~~ (`~~text~~`)
- Code blocks (```python ... ```) and inline code (`` `code` ``)
- Lists (ordered and unordered)
- Blockquotes (`> quote`)
- Links (`[text](url)`)
- Tables
- Horizontal rules (`---`)
- Emoji shortcodes (`:sparkles:`, `:tada:`, etc.)

### System Instructions

The system automatically assembles prompts with:
- Dynamic chat history (up to 20,000 tokens by default)
- Formatting instructions encouraging markdown usage
- Context-aware conversation continuation

See `llm_chat/system_instructions.py` for customization.

### Testing

Run chat tests:
```bash
python manage.py test llm_chat
```

Test files:
- `llm_chat/tests/test_models.py` - Model tests
- `llm_chat/tests/test_services.py` - Service layer tests
- `llm_chat/tests/test_consumers.py` - WebSocket consumer tests
- `llm_chat/tests/test_views.py` - View tests
- `llm_chat/tests/test_connection.py` - WebSocket connection tests
- `llm_chat/tests/test_integration.py` - End-to-end integration tests
- `llm_chat/tests/test_markdown.py` - Markdown rendering tests

### Frontend Features

**Markdown Rendering:**
- Uses `marked.js` for parsing and `DOMPurify` for sanitization
- CSS styling scoped to `#chat-messages .markdown-content`
- Streaming markdown re-parses when complete blocks are detected
- Server-rendered messages are processed on page load

**WebSocket Management:**
- Automatic connection when thread is active
- Reconnection with exponential backoff (max 5 attempts)
- Connection status: green dot when connected (no label), spinner and "Connecting..." when connecting, text when reconnecting or disconnected
- Heartbeat ping/pong for connection health

**UI Features:**
- Auto-scrolling when messages arrive
- Sidebar navigation with thread history
- Dynamic thread ordering (most recent at top)
- Form disabling during streaming
- Rate limiting (1 second between messages)
