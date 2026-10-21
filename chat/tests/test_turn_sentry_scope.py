"""A chat turn runs under a Sentry scope tagged with its thread id, so its events
are found with the same ``thread_id:<uuid>`` query as the thread's sub-agent runs."""

import uuid
from types import SimpleNamespace

import sentry_sdk
from django.test import SimpleTestCase

from chat.consumers import ChatConsumer


class TurnSentryScopeTests(SimpleTestCase):
    async def test_turn_body_sees_thread_tags_and_its_arguments(self):
        consumer = ChatConsumer()
        thread = SimpleNamespace(id=uuid.uuid4())
        seen = {}

        async def fake_body(passed_thread, **kwargs):
            seen["thread"] = passed_thread
            seen["kwargs"] = kwargs
            seen["tags"] = dict(sentry_sdk.get_isolation_scope()._tags)

        consumer._stream_and_finalize = fake_body
        # Compare against the shared scope as it was: tests that call Celery task
        # bodies directly (no Celery integration to isolate them) leave tags on it.
        before = dict(sentry_sdk.get_isolation_scope()._tags)
        await consumer._stream_and_finalize_tagged(thread, content="hi", seed_mode=True)

        self.assertIs(seen["thread"], thread)
        self.assertEqual(seen["kwargs"], {"content": "hi", "seed_mode": True})
        self.assertEqual(seen["tags"]["thread_id"], str(thread.id))
        self.assertEqual(seen["tags"]["agent_kind"], "main")
        # The fork is gone afterwards: nothing leaked onto the shared scope.
        self.assertEqual(dict(sentry_sdk.get_isolation_scope()._tags), before)
