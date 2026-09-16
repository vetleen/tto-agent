"""Thread ↔ skill attachment writes, serialized per thread.

Every writer that *replaces* a thread's skill set — the ``chat_skill_attach``
tool, the consumer's ``skills.set`` handler, loop setup — goes through
``replace_thread_skills`` so the replace runs under the thread's row lock.

Why: ``delete()`` + ``bulk_create()`` inside ``atomic()`` is atomic only
against itself. Two replaces on the same thread can run at the same instant
(the pipeline executes a round's tool calls in a thread pool, and the agent
tool can race the UI). Under READ COMMITTED the second transaction's DELETE
cannot see the first one's uncommitted INSERTs, so its own INSERT trips the
``(thread, skill)`` unique constraint (WILFRED-8P). With the lock the second
writer waits for the first to commit, then reads the committed rows and
replaces them — last writer wins, never an IntegrityError.
"""

from __future__ import annotations

import logging

from django.db import transaction

from chat.models import ChatThread, ChatThreadSkill

logger = logging.getLogger(__name__)


def lock_thread(thread_id) -> None:
    """Take the per-thread write lock. Must run inside ``transaction.atomic()``.

    ``SELECT … FOR UPDATE`` on the ``ChatThread`` row: a concurrent locker blocks
    until this transaction ends, and every statement it runs afterwards sees the
    committed skill rows. Callers that need to *read* the current set before
    deciding what to write (the attach tool's diff) call this first, so the read
    is consistent with the write. No-op on SQLite (tests), which has no row
    locks but also no concurrent writers.
    """
    list(
        ChatThread.objects.select_for_update()
        .filter(pk=thread_id)
        .values_list("pk", flat=True)
    )


def replace_thread_skills(thread, skill_ids) -> None:
    """Replace ``thread``'s attached skills with ``skill_ids``, in that order.

    Order is preserved by ``ChatThreadSkill``'s ``(attached_at, id)`` ordering,
    which the prompt renderer, tool union and template lookups rely on. Callers
    pass an already validated, de-duplicated list. Safe to call from inside an
    outer ``atomic()`` that already holds ``lock_thread`` (re-locking a row this
    transaction owns is immediate).
    """
    with transaction.atomic():
        lock_thread(thread.pk)
        ChatThreadSkill.objects.filter(thread=thread).delete()
        ChatThreadSkill.objects.bulk_create(
            [ChatThreadSkill(thread=thread, skill_id=sid) for sid in skill_ids]
        )
