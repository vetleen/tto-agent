from django.contrib import admin

from .models import (
    Meeting,
    MeetingArtifact,
    MeetingAttachment,
    MeetingDataRoom,
    MeetingTranscriptSegment,
)


@admin.register(Meeting)
class MeetingAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "transcript_source", "created_by", "updated_at", "is_archived")
    list_filter = ("status", "transcript_source", "is_archived")
    search_fields = ("name", "slug", "agenda", "participants")
    readonly_fields = ("uuid", "created_at", "updated_at", "started_at", "ended_at")


@admin.register(MeetingTranscriptSegment)
class MeetingTranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = ("meeting", "segment_index", "status", "transcribed_at")
    list_filter = ("status",)
    search_fields = ("meeting__name",)


@admin.register(MeetingArtifact)
class MeetingArtifactAdmin(admin.ModelAdmin):
    list_display = ("meeting", "kind", "title", "created_by", "created_at")
    list_filter = ("kind",)
    search_fields = ("meeting__name", "title")


@admin.register(MeetingAttachment)
class MeetingAttachmentAdmin(admin.ModelAdmin):
    list_display = ("meeting", "original_filename", "size_bytes", "uploaded_by", "uploaded_at")
    search_fields = ("meeting__name", "original_filename")


@admin.register(MeetingDataRoom)
class MeetingDataRoomAdmin(admin.ModelAdmin):
    list_display = ("meeting", "data_room", "attached_at")
