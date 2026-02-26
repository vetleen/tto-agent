# tto-agent

An **agentic system for technology transfer offices (TTO)** that handles routine workflows for TTO staff. The system assists employees with common tasks—such as intake, disclosure support, and process guidance—through an AI-powered assistant, reducing repetitive work and keeping workflows consistent.

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
   - Django Channels (for future WebSocket/real-time features)
   - channels-redis (Redis backend for Channels)
   - daphne (ASGI server)
   - django-anymail (optional; for production email via Mailgun etc.)
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

**Run the development server:**
```bash
python manage.py runserver 8000
```

**For ASGI (e.g. production or when adding WebSockets later):**
```bash
daphne -b 127.0.0.1 -p 8000 config.asgi:application
```

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

Add any other vars in Dashboard → Settings → Config Vars or via `heroku config:set` after the first deploy.

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
- **Channel layer 500, SSL cert verify failed:** Heroku Redis uses TLS with a cert that fails default verification. Settings use `ssl_cert_reqs=ssl.CERT_NONE` for `rediss://` URLs so the channel layer can connect.
- **“Verify your email” after login:** The Heroku DB is separate from local. Either set the user’s `email_verified=True` via `heroku run python manage.py shell`, or set `EMAIL_VERIFICATION_REQUIRED=False` in Config Vars. To create a superuser: `heroku run python manage.py createsuperuser -a YOUR_APP_NAME`.
- **Viewing logs:** `heroku logs --tail` mixes addon (e.g. Redis) output. For web/release only: `heroku logs --tail --source app -a YOUR_APP_NAME`. Check release success: `heroku releases -a YOUR_APP_NAME`.

### Running Tests

```bash
# Account tests (auth, signup, password reset, email verification, email config)
python manage.py test accounts
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

# Redis (for Django Channels; optional until real-time features are added)
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

**Note:** The `REDIS_URL` is optional locally and defaults to `redis://127.0.0.1:6379/0`. Heroku Redis sets `REDIS_URL` (often `rediss://`); the app skips TLS cert verification for the channel layer when used.

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

**UserSettings** (accounts app): OneToOne to User, created automatically when a user is created (signal). Stores per-user preferences: `theme` with choices `light` or `dark` (default `light`). Access via `user.settings.theme`.

**Dark mode** (Flowbite class-based):
- Toggle in the top nav for **logged-in users only**; theme is saved in `UserSettings` and persists across sessions.
- Anonymous users get theme from `localStorage` or system preference (no persistence).
- Implemented via `@custom-variant dark` in `static/src/input.css`, inline script in `<head>` to avoid FOUC, and context processor `accounts.context_processors.theme` that passes `theme` into templates.
- **Theme update:** `POST /accounts/settings/theme/` with `theme=light` or `theme=dark` (login required); returns JSON `{"theme": "…"}`.

