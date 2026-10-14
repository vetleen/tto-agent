"""``core.observability``: Sentry tagging that never breaks the code it observes."""

import asyncio
from unittest.mock import patch

import sentry_sdk
from django.test import SimpleTestCase

from core.observability import sentry_scope, set_sentry_tags


def _tags():
    return dict(sentry_sdk.get_isolation_scope()._tags)


class SentryScopeTests(SimpleTestCase):
    def test_tags_live_only_inside_the_block(self):
        self.assertNotIn("thread_id", _tags())
        with sentry_scope(thread_id="t-1", agent_kind="main", skipped=None):
            self.assertEqual(_tags()["thread_id"], "t-1")
            self.assertEqual(_tags()["agent_kind"], "main")
            self.assertNotIn("skipped", _tags())
        self.assertNotIn("thread_id", _tags())

    def test_scope_is_dropped_when_the_block_raises(self):
        with self.assertRaises(ValueError):
            with sentry_scope(thread_id="t-2"):
                raise ValueError("boom")
        self.assertNotIn("thread_id", _tags())

    def test_tasks_created_inside_inherit_the_scope(self):
        async def child():
            return _tags().get("thread_id")

        async def main():
            with sentry_scope(thread_id="t-3"):
                return await asyncio.create_task(child())

        self.assertEqual(asyncio.run(main()), "t-3")

    def test_values_are_stringified(self):
        with sentry_scope(run=7):
            self.assertEqual(_tags()["run"], "7")

    def test_block_runs_when_the_sdk_fails(self):
        ran = False
        with patch("sentry_sdk.isolation_scope", side_effect=RuntimeError("sentry down")):
            with sentry_scope(thread_id="t-4"):
                ran = True
        self.assertTrue(ran)


class SetSentryTagsTests(SimpleTestCase):
    def test_sets_on_the_current_isolation_scope(self):
        with sentry_scope():
            set_sentry_tags(subagent_run_id="r-1", thread_id=None)
            self.assertEqual(_tags()["subagent_run_id"], "r-1")
            self.assertNotIn("thread_id", _tags())

    def test_survives_a_failing_sdk(self):
        with patch("sentry_sdk.set_tag", side_effect=RuntimeError("sentry down")):
            set_sentry_tags(thread_id="t-5")  # must not raise
