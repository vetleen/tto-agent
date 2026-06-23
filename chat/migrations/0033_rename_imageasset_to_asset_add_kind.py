import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    """Rename ImageAsset -> Asset (the model now backs both inline-image and
    file-download reference tokens) and add the `kind` discriminator that keeps an
    image-ref and a file-ref for the same version as distinct rows."""

    dependencies = [
        ("chat", "0032_chatattachment_extracted_content"),
    ]

    operations = [
        migrations.RenameModel(old_name="ImageAsset", new_name="Asset"),
        migrations.AddField(
            model_name="asset",
            name="kind",
            field=models.CharField(
                choices=[("image", "image"), ("file", "file")],
                db_index=True,
                default="image",
                max_length=8,
            ),
        ),
        migrations.AlterField(
            model_name="asset",
            name="version",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="assets",
                to="documents.dataroomdocumentversion",
            ),
        ),
        migrations.AlterField(
            model_name="asset",
            name="canvas",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="assets",
                to="chat.chatcanvas",
            ),
        ),
        migrations.AlterField(
            model_name="asset",
            name="message",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="assets",
                to="chat.chatmessage",
            ),
        ),
        migrations.AlterField(
            model_name="asset",
            name="thread",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="assets",
                to="chat.chatthread",
            ),
        ),
        # Auto-named Meta.indexes follow the table name; realign them to `asset`.
        migrations.RenameIndex(
            model_name="asset",
            new_name="chat_asset_version_a477d3_idx",
            old_name="chat_imagea_version_415b00_idx",
        ),
        migrations.RenameIndex(
            model_name="asset",
            new_name="chat_asset_canvas__0d68fc_idx",
            old_name="chat_imagea_canvas__93554c_idx",
        ),
        migrations.RenameIndex(
            model_name="asset",
            new_name="chat_asset_thread__201688_idx",
            old_name="chat_imagea_thread__3821f3_idx",
        ),
    ]
