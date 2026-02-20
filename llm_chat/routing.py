from django.urls import re_path

from . import consumers


websocket_urlpatterns = [
    # e.g. ws://.../ws/chat/<thread_id>/
    re_path(r"^ws/chat/(?P<thread_id>[0-9a-f-]+)/$", consumers.ChatConsumer.as_asgi()),
]

