#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import asyncio
import os
import sys

# Windows-only: force the Proactor asyncio policy before Django / Channels /
# Daphne import or instantiate their event loop. The Realtime live-transcription
# path uses ``asyncio.create_subprocess_exec`` to spawn ffmpeg, which the
# default SelectorEventLoop on Windows 3.10+ can't do (NotImplementedError).
# Setting this in asgi.py is too late — Daphne has already created its loop
# by then. Heroku (Linux) is unaffected.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
