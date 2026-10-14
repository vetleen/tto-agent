"""Per-run delivery marker for sub-agent outcomes.

``delivered_at`` records that a run's outcome reached the orchestrator model:
its hidden outcome message was replayed into a turn that persisted a final
assistant message, or the blocking ``chat_subagent_create`` wait returned it
inline. ``reported_at`` stays the short-lived claim lease.

Before this, "reported" was inferred from *any* later assistant reply tagged
``subagent_response`` — so an outcome that landed while a seeded turn was
already streaming looked reported and was never surfaced.

Backfill: without it every historical thread with sub-agent runs would seed a
continuation turn the next time it is opened. Two signals reproduce what the
old code considered settled:

(a) an assistant reply tagged ``subagent_response`` after the hidden message;
(b) an inline blocking delivery (those replies were never tagged): a run with
    a wait budget that was claimed and whose claiming turn then answered. A
    FAILED run with a canvas is excluded — the inline error never carried it.

Accepted trade-off: an outcome orphaned by the old heuristic before this
migration looks delivered under (a) and is not recovered; it is still in the
thread's history and reaches the model on the next user turn. Runs with a
claim in flight at deploy time stay NULL and are re-seeded, which is what we
want.
"""

from django.db import migrations, models


def backfill_delivered_at(apps, schema_editor):
    ChatMessage = apps.get_model("chat", "ChatMessage")
    SubAgentRun = apps.get_model("chat", "SubAgentRun")

    runs = SubAgentRun.objects.filter(
        status__in=["completed", "failed"], delivered_at__isnull=True,
    ).only("id", "thread_id", "status", "timeout", "reported_at", "canvas")
    for run in runs.iterator(chunk_size=200):
        hidden_at = (
            ChatMessage.objects.filter(
                thread_id=run.thread_id,
                metadata__source="subagent",
                metadata__subagent_run_id=str(run.id),
            )
            .order_by("created_at")
            .values_list("created_at", flat=True)
            .first()
        )
        if hidden_at is None:
            continue  # cancelled: no outcome message, never claimable

        assistants = ChatMessage.objects.filter(thread_id=run.thread_id, role="assistant")
        # (a) pre-0052 heuristic: a tagged reply after the hidden message.
        delivered = (
            assistants.filter(metadata__subagent_response=True, created_at__gt=hidden_at)
            .order_by("created_at")
            .values_list("created_at", flat=True)
            .first()
        )
        # (b) inline blocking delivery: claimed by the waiting turn, which answered.
        if (
            delivered is None
            and run.timeout > 0
            and run.reported_at is not None
            and not (run.status == "failed" and run.canvas)
        ):
            delivered = (
                assistants.filter(created_at__gt=run.reported_at)
                .order_by("created_at")
                .values_list("created_at", flat=True)
                .first()
            )
        if delivered is not None:
            SubAgentRun.objects.filter(pk=run.pk, delivered_at__isnull=True).update(
                delivered_at=delivered,
            )


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0051_subagentrun_shared_attachments"),
    ]

    operations = [
        migrations.AddField(
            model_name="subagentrun",
            name="delivered_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(backfill_delivered_at, migrations.RunPython.noop),
    ]
