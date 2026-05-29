"""Worker entrypoint that runs Celery under the gevent pool.

Used only by the Heroku ``worker`` process (see ``Procfile``). It monkeypatches
gevent and psycopg2 **before** Celery / Django / psycopg2 / httpx are imported,
which is the order gevent requires for all network I/O (OpenAI httpx, redis-py,
psycopg2) to cooperate with the event loop.

Kept separate from ``config/celery.py`` on purpose: the web (Daphne) process
imports ``config.celery`` via ``config/__init__.py`` but never imports this
module, so the web process is never monkeypatched and stays on asyncio.

The patching lives inside ``main()`` (not at import time) so the test suite can
import this module without monkeypatching the test runner.
"""

import sys


def _patch() -> None:
    """Apply gevent + psycopg2 monkeypatching. Must run before Celery imports."""
    from gevent import monkey

    monkey.patch_all()
    from psycogreen.gevent import patch_psycopg

    patch_psycopg()


def main() -> None:
    _patch()
    # Celery's click CLI inspects argv[0]; `python -m config.celery_gevent`
    # leaves the module path there, so rewrite it to look like the celery script.
    sys.argv[0] = "celery"
    from celery.__main__ import main as celery_main

    celery_main()


if __name__ == "__main__":
    main()
