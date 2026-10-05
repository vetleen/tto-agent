from django.db import migrations
from django.db.models import Q


def recompute_quarantine_from_effective_version(apps, schema_editor):
    """Re-derive document-level quarantine flags from the effective version only
    (active searchable, else the working head), mirroring the new
    recompute_document_sensitivity. Before this, the rollup was a union over all
    retained versions, so a quarantined v0 kept a cleanly edited document flagged.
    Only currently flagged documents can change; few rows, a plain loop is fine.
    """
    DataRoomDocument = apps.get_model("documents", "DataRoomDocument")
    DataRoomDocumentVersion = apps.get_model("documents", "DataRoomDocumentVersion")
    flagged = DataRoomDocument.objects.filter(
        Q(is_quarantined=True) | Q(is_partially_quarantined=True)
    )
    for doc in flagged:
        vid = doc.active_searchable_version_id or doc.current_version_id
        v = DataRoomDocumentVersion.objects.filter(pk=vid).first() if vid else None
        is_q = bool(v and v.is_quarantined)
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            is_quarantined=is_q,
            is_partially_quarantined=is_q or bool(v and v.is_partially_quarantined),
            quarantine_reason=v.quarantine_reason if is_q else "",
            quarantine_detail=v.quarantine_detail if is_q else "",
        )


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0023_dataroomdocumentversion_page_render'),
    ]

    operations = [
        migrations.RunPython(
            recompute_quarantine_from_effective_version, migrations.RunPython.noop,
        ),
    ]
