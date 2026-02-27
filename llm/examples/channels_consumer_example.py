"""
Example: minimal Channels AsyncWebsocketConsumer that uses LLMService.stream()
and forwards deltas to the client.

This file is for reference only; it is not wired into Django/Channels routing.
To use it, register the consumer in your ASGI routing and call the URL from the client.

In production, run the blocking service.stream() loop in a thread pool so the
async event loop is not blocked, e.g. with asyncio.to_thread() or
loop.run_in_executor().
"""

import json
from channels.generic.websocket import AsyncWebsocketConsumer

# In your consumer:
#
# from llm import get_llm_service
# from llm.types import ChatRequest, Message, RunContext
#
#
# class ChatConsumer(AsyncWebsocketConsumer):
#     async def receive(self, text_data=None):
#         if not text_data:
#             return
#         try:
#             body = json.loads(text_data)
#             conversation_id = body.get("conversation_id")
#             pipeline_id = body.get("pipeline_id", "simple_chat")
#             model = body.get("model")
#             message = (body.get("message") or "").strip()
#             if not message:
#                 await self.send(json.dumps({"error": "message required"}))
#                 return
#         except json.JSONDecodeError as e:
#             await self.send(json.dumps({"error": str(e)}))
#             return
#
#         request = ChatRequest(
#             messages=[Message(role="user", content=message)],
#             stream=True,
#             model=model,
#             context=RunContext.create(conversation_id=conversation_id),
#         )
#         service = get_llm_service()
#
#         # Run blocking stream in executor to avoid blocking the event loop
#         import asyncio
#         def run_stream():
#             return list(service.stream(pipeline_id, request))
#         events = await asyncio.to_thread(run_stream)
#
#         for event in events:
#             await self.send(json.dumps(event.model_dump()))
