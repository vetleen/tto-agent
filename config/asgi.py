"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import asyncio
import os
import sys

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

# Windows Python 3.10+ defaults to SelectorEventLoop for asyncio, which does
# NOT implement ``subprocess_exec``. The realtime live-transcription path
# spawns ``ffmpeg`` via ``asyncio.create_subprocess_exec`` to decode WebM/Opus
# into PCM16 — that crashes with NotImplementedError on the selector loop.
# Force the Proactor policy before Daphne instantiates its loop so async
# subprocesses work during ``daphne`` / ``runserver`` on dev machines.
# Heroku (Linux) is unaffected — its default loop supports subprocesses.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Initialize Django ASGI application early to ensure Django is configured
# before importing any Django models or settings-dependent code
django_asgi_app = get_asgi_application()

from chat.routing import websocket_urlpatterns as chat_ws  # noqa: E402
from meetings.routing import websocket_urlpatterns as meetings_ws  # noqa: E402

websocket_urlpatterns = chat_ws + meetings_ws

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns),
        ),
    }
)
