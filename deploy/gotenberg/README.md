# Document render service (Gotenberg on Heroku)

`wilfred-render` is a tiny, stateless Heroku app that turns Office files into PDFs for the
Wilfred worker. It runs the official [Gotenberg](https://gotenberg.dev) image (LibreOffice behind
an HTTP API) on the **container stack**, because LibreOffice never bootstraps under Heroku's apt
buildpack and moving the main app to Docker would break pipeline promotion.

- **One app, shared by staging and production.** Basic dyno (~$7/month), EU region, no addons,
  no database, no log drains.
- **Zero data retention by construction.** Gotenberg writes each request to a temp directory and
  deletes it when the response is sent; the dyno filesystem is ephemeral. The worker uploads files
  under opaque names (`<version_id>-<batch>.pptx`), so even the render app's error log never sees a
  document title.
- **Gate:** TLS + HTTP basic auth with a long random secret + an unguessable hostname. IP/host
  allowlisting is not available on Heroku's common runtime (dynamic outbound IPs, Gotenberg has no
  allowlist), so the secret is the whole gate — keep it long and rotate it if in doubt.
- **Consumer:** `documents/services/page_render.py` (the Celery task `render_document_pages`),
  configured on the *main* apps via `DOCUMENT_RENDER_SERVICE_URL` / `_USER` / `_PASSWORD`. An empty
  URL turns the feature off.

Files here: `Dockerfile` (image + flags, each explained inline), `heroku.yml` (build manifest).

> All commands below are PowerShell. `heroku config:set` silently stores **empty** values on the
> dev machine (CLI 11.x/win32-arm64) — always set config vars by piping a literal JSON string
> into `heroku api PATCH`, then verify with `heroku config:get`.

## Create the app (once)

```powershell
heroku apps:create wilfred-render --region eu --stack container
heroku git:remote -a wilfred-render -r heroku-render
heroku labs:enable log-runtime-metrics -a wilfred-render   # memory samples in the logs
```

Set the basic-auth credentials (generate a long random secret first, e.g. 32+ chars):

```powershell
$body = '{"GOTENBERG_API_BASIC_AUTH_USERNAME":"wilfred","GOTENBERG_API_BASIC_AUTH_PASSWORD":"<secret>"}'
$out = $body | heroku api PATCH /apps/wilfred-render/config-vars
heroku config:get GOTENBERG_API_BASIC_AUTH_USERNAME -a wilfred-render
```

## Deploy (build happens on Heroku; no local Docker needed)

The app is built from this directory only, pushed as a git subtree:

```powershell
git subtree push --prefix deploy/gotenberg heroku-render main
```

If `subtree push` refuses (history rewritten / diverged), split and force-push the split branch:

```powershell
git subtree split --prefix deploy/gotenberg -b gotenberg-deploy
git push heroku-render gotenberg-deploy:main --force
git branch -D gotenberg-deploy
```

Then scale (first deploy only; later deploys keep the formation):

```powershell
heroku ps:scale web=1:basic -a wilfred-render
```

Health check (unauthenticated `/health` reports only up/down; the convert route needs auth):

```powershell
curl.exe -s https://<render-app-host>/health
curl.exe -s -u wilfred:<secret> -o NUL -w "%{http_code}" -X POST https://<render-app-host>/forms/libreoffice/convert
```

Find `<render-app-host>` with `heroku info -a wilfred-render` (Heroku appends a random suffix).

## Point the main apps at it

Staging (fine to do without asking):

```powershell
$body = '{"DOCUMENT_RENDER_SERVICE_URL":"https://<render-app-host>","DOCUMENT_RENDER_SERVICE_USER":"wilfred","DOCUMENT_RENDER_SERVICE_PASSWORD":"<secret>"}'
$out = $body | heroku api PATCH /apps/wilfred-staging/config-vars
heroku ps:restart worker -a wilfred-staging
```

**Production — USER-GATED** (a config change restarts dynos under live users; see CLAUDE.md):

```powershell
$body = '{"DOCUMENT_RENDER_SERVICE_URL":"https://<render-app-host>","DOCUMENT_RENDER_SERVICE_USER":"wilfred","DOCUMENT_RENDER_SERVICE_PASSWORD":"<secret>"}'
$out = $body | heroku api PATCH /apps/wilfred-production/config-vars
heroku ps:restart worker -a wilfred-production
```

Tuning knobs on the main apps (all optional): `DOCUMENT_RENDER_BATCH_SIZE` (8),
`DOCUMENT_RENDER_MAX_SLIDES` (200), `DOCUMENT_RENDER_HTTP_TIMEOUT` (60),
`DOCUMENT_RENDER_VIEW_DEFAULT_SLIDES` (8), `DOCUMENT_RENDER_VIEW_MAX_SLIDES` (12).

## Rotate the secret

1. PATCH a new `GOTENBERG_API_BASIC_AUTH_PASSWORD` on `wilfred-render` (the dyno restarts).
2. PATCH `DOCUMENT_RENDER_SERVICE_PASSWORD` on `wilfred-staging`, then (USER-GATED) on
   `wilfred-production`, and restart each worker.
3. Renders that ran in between fail with 401 and stay `pending`; the stale sweeper re-dispatches
   them within ~30 minutes.

## Upgrade the image

Bump the tag in `Dockerfile` (check the [Gotenberg releases](https://github.com/gotenberg/gotenberg/releases)
for flag changes), commit, `git subtree push` as above, then re-run the health check and convert a
deck on staging. Gotenberg is internet-facing — upgrade on a schedule, not only when something breaks.

## Known limits (measured 2026-09-23/24, Basic dyno)

| Limit | Effect | Handling |
|-------|--------|----------|
| Heroku router 30 s cut (`H12`) | A conversion slower than ~25 s comes back as a 503 HTML page | Batches of 8 slides take 5–10 s; the worker treats H12 as "busy", retries, then falls back to single slides |
| `--libreoffice-max-queue-size=4` | Gotenberg answers **503** when four requests are already queued | The worker backs off and retries (Celery retry with 30 s → 600 s backoff) |
| 512 MB dyno memory | One batch of 8 slides peaks at ~350 MB; a 68-slide deck in one request hit 673 MB (R14) and timed out | Batches + `--libreoffice-restart-after=1`; move to Standard-2X only if `R14` shows up in `heroku logs -a wilfred-render` |
| Image is ~1 GB | Boot to healthy takes ~20 s | Nothing to do; `--libreoffice-auto-start` warms LibreOffice at boot |

## Watching it

```powershell
heroku logs --tail -a wilfred-render                       # errors + sample#memory_rss lines
heroku ps -a wilfred-render
```

On the main app the version row tells the story: `DataRoomDocumentVersion.page_render_state`
(`none` / `pending` / `partial` / `ready` / `skipped` / `failed`), `page_count`,
`page_render_attempts`, `page_render_error`.

## Retire the throwaway smoke-test app

`wilfred-render-smoke` (created 2026-09-23 for the feasibility test, scaled to 0) is not needed once
`wilfred-render` is live:

```powershell
heroku apps:destroy -a wilfred-render-smoke --confirm wilfred-render-smoke
```
