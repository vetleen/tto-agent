from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/meetings/<uuid:meeting_uuid>/transcribe/", consumers.MeetingTranscribeConsumer.as_asgi()),
]
