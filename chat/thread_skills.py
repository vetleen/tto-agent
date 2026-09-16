"""Thread ↔ skill attachment writes, serialized per thread.

Every writer of a thread's skill set goes through this module so each write
runs under the thread's row lock:

- ``replace_thread_skills`` — the user's declarative set (UI pill/modal, the
  ``skills.set`` handler, loop setup). Survivors keep their ``attached_by``.
- ``add_thread_skills`` — the agent's ``chat_skill_attach`` (additive; rows are
  marked ``attached_by="agent"``).
- ``remove_thread_skills`` — the agent's ``chat_skill_detach`` (the tool only
  ever passes agent-attached rows; the protection check lives in the tool).

Why the lock: ``delete()`` + ``bulk_create()`` inside ``atomic()`` is atomic
only against itself. Two writes on the same thread can run at the same instant
(the pipeline executes a round's tool calls in a thread pool, and the agent
tool can race the UI). Under READ COMMITTED the second transaction's DELETE
cannot see the first one's uncommitted INSERTs, so its own INSERT trips the
``(thread, skill)`` unique constraint (WILFRED-8P). With the lock the second
writer waits for the first to commit, then reads the committed rows.
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
    deciding what to write (the attach/detach tools) call this first, so the
    read is consistent with the write. No-op on SQLite (tests), which has no row
    locks but also no concurrent writers.
    """
    list(
        ChatThread.objects.select_for_update()
        .filter(pk=thread_id)
        .values_list("pk", flat=True)
    )


def replace_thread_skills(thread, skill_ids, *, attached_by="user") -> None:
    """Replace ``thread``'s attached skills with ``skill_ids``, in that order.

    Order is preserved by ``ChatThreadSkill``'s ``(attached_at, id)`` ordering,
    which the prompt renderer, tool union and template lookups rely on. Callers
    pass an already validated, de-duplicated list.

    Rows that survive the replace KEEP their ``attached_by``: the browser
    re-sends the full skill list with every message and the consumer rewrites
    the rows whenever that list differs from the DB, so without this an
    agent-attached skill would silently become "user" (and stop being
    detachable by the agent). Ids not previously attached get ``attached_by``.
    """
    with transaction.atomic():
        lock_thread(thread.pk)
        origins = {
            str(sid): who
            for sid, who in ChatThreadSkill.objects.filter(thread=thread).values_list(
                "skill_id", "attached_by"
            )
        }
        ChatThreadSkill.objects.filter(thread=thread).delete()
        ChatThreadSkill.objects.bulk_create(
            [
                ChatThreadSkill(
                    thread=thread,
                    skill_id=sid,
                    attached_by=origins.get(str(sid), attached_by),
                )
                for sid in skill_ids
            ]
        )


def add_thread_skills(thread, skill_ids, *, attached_by="agent") -> list[str]:
    """Attach the ids in ``skill_ids`` that aren't attached yet; return the ids
    actually added (as ``str``), in the order given.

    Never deletes: existing rows keep their ``attached_at`` (and origin), so
    the thread's order is previous-then-new. Safe to call from inside an outer
    ``atomic()`` that already holds ``lock_thread`` (re-locking a row this
    transaction owns is immediate).
    """
    with transaction.atomic():
        lock_thread(thread.pk)
        existing = {
            str(sid)
            for sid in ChatThreadSkill.objects.filter(thread=thread).values_list(
                "skill_id", flat=True
            )
        }
        new_ids: list[str] = []
        for sid in skill_ids:
            s = str(sid)
            if s not in existing and s not in new_ids:
                new_ids.append(s)
        if new_ids:
            ChatThreadSkill.objects.bulk_create(
                [
                    ChatThreadSkill(thread=thread, skill_id=sid, attached_by=attached_by)
                    for sid in new_ids
                ]
            )
        return new_ids


def remove_thread_skills(thread, skill_ids) -> list[str]:
    """Detach the given ids; return the ids actually removed (as ``str``).

    Ids that aren't attached are ignored. The "may the agent detach this?"
    check is the caller's (``chat_skill_detach`` refuses user-attached rows
    before calling this).
    """
    with transaction.atomic():
        lock_thread(thread.pk)
        ids = [str(sid) for sid in skill_ids]
        qs = ChatThreadSkill.objects.filter(thread=thread, skill_id__in=ids)
        removed = [str(sid) for sid in qs.values_list("skill_id", flat=True)]
        if removed:
            qs.delete()
        return removed
