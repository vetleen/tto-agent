"""Worker entrypoint that runs Celery under the gevent pool.

Used only by the Heroku ``worker`` process (see ``Procfile``). It monkeypatches
gevent and psycopg2 **before** Celery / Django / psycopg2 / httpx are imported,
which is the order gevent requires for all network I/O (OpenAI httpx, redis-py,
psycopg2) to cooperate with the event loop.

This lives at the repo root, *not* inside the ``config`` package, on purpose:
``python -m config.celery_gevent`` would first import the ``config`` package,
running ``config/__init__.py`` (which imports ``config.celery`` -> Celery +
Django + ssl) before any patching could happen, producing gevent's
"Monkey-patching ssl after ssl has already been imported" warning. As a
top-level module, importing it pulls in nothing but ``sys``, so ``patch_all()``
runs first and the heavy imports (triggered later by ``-A config``) land in an
already-patched process.

The web (Daphne) process never imports this module, so it is never
monkeypatched and stays on asyncio.

Patching lives inside ``main()`` (not at import time) so the test suite can
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
    # Celery's click CLI inspects argv[0]; `python -m celery_gevent` leaves the
    # module path there, so rewrite it to look like the celery script. The
    # remaining args come from the Procfile and must use Celery 5 syntax, with
    # `-A` as a global option *before* the `worker` subcommand.
    sys.argv[0] = "celery"
    from celery.__main__ import main as celery_main

    celery_main()


if __name__ == "__main__":
    main()
