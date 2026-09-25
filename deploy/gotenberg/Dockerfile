# Wilfred document render service: Gotenberg (LibreOffice behind HTTP).
#
# The Celery worker POSTs pptx slide batches to /forms/libreoffice/convert and
# gets a PDF back (documents/services/page_render.py). Runs as its own Heroku
# container app (`wilfred-render`), shared by staging and production, with no
# addons and nothing persisted. See README.md for deploy/config/rotation.
#
# Pin the image tag deliberately — this is an internet-facing service; bump it
# on purpose and redeploy, never `latest`.
#
# Flags (all verified against https://gotenberg.dev/docs/configuration):
#   --api-port-from-env=PORT          Heroku hands the listen port over as $PORT.
#   --api-timeout=60s                 Per-request cap (Heroku's router cuts at 30 s anyway;
#                                     this bounds the LibreOffice run after a client hangs up).
#   --api-enable-basic-auth=true      Reads GOTENBERG_API_BASIC_AUTH_USERNAME / _PASSWORD.
#   --api-body-limit=32MB             Reject oversized multipart bodies before LibreOffice sees them.
#   --chromium-disable-routes=true    HTML/URL/Markdown routes off: no Chromium, smaller attack surface.
#   --libreoffice-restart-after=1     Fresh LibreOffice per conversion — it never frees memory
#                                     between jobs (measured 2026-09-24), so this keeps a Basic
#                                     dyno under 512 MB.
#   --libreoffice-max-queue-size=4    Answer 503 when busy instead of piling up; the worker retries.
#   --libreoffice-auto-start=true     Warm LibreOffice at boot so the first request pays no cold start.
#   --log-level=warn                  Access lines off; error lines still quote the (opaque) upload name.
FROM gotenberg/gotenberg:8

CMD ["gotenberg", \
  "--api-port-from-env=PORT", \
  "--api-timeout=60s", \
  "--api-enable-basic-auth=true", \
  "--api-body-limit=32MB", \
  "--chromium-disable-routes=true", \
  "--libreoffice-restart-after=1", \
  "--libreoffice-max-queue-size=4", \
  "--libreoffice-auto-start=true", \
  "--log-level=warn"]
