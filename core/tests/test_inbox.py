"""Tests for the inbox / up-for-deletion queue (core.views.inbox / inbox_renew)."""
from __future__ import annotations

import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from chat.models import ChatThread
from core.retention import RETENTION_PERIODS
from core.views import _relative_future
from documents.models import DataRoom
from meetings.models import Meeting

User = get_user_model()


def _set_retain(model_cls, pk, retain_until):
    """Set retain_until via .update() so save()/auto_now/signals don't interfere."""
    model_cls.objects.filter(pk=pk).update(retain_until=retain_until)


class InboxTestBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="inbox@example.com", password="pw")
        self.other = User.objects.create_user(email="other@example.com", password="pw")
        self.now = timezone.now()
        self.client.force_login(self.user)
        self._n = 0

    def _slug(self, prefix):
        self._n += 1
        return f"{prefix}-{self._n}"

    def _chat(self, *, owner=None, days=5, title="Thread", archived=False):
        t = ChatThread.objects.create(
            title=title, created_by=owner or self.user, is_archived=archived,
        )
        _set_retain(ChatThread, t.pk, self.now + timedelta(days=days))
        return t

    def _dataroom(self, *, owner=None, days=5, name="DR", archived=False):
        dr = DataRoom.objects.create(
            name=name, slug=self._slug("dr"), created_by=owner or self.user,
            is_archived=archived,
        )
        _set_retain(DataRoom, dr.pk, self.now + timedelta(days=days))
        return dr

    def _meeting(self, *, owner=None, days=5, name="Mtg", archived=False):
        m = Meeting.objects.create(
            name=name, slug=self._slug("m"), created_by=owner or self.user,
            is_archived=archived,
        )
        _set_retain(Meeting, m.pk, self.now + timedelta(days=days))
        return m

    def _keys(self, response):
        return [item["key"] for item in response.context["page_obj"]]


class InboxListingTests(InboxTestBase):
    def test_in_window_item_appears(self):
        t = self._chat(days=5)
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertIn(f"chat:{t.id}", keys)

    def test_out_of_window_item_absent(self):
        t = self._chat(days=60)
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertNotIn(f"chat:{t.id}", keys)

    def test_null_retain_until_absent(self):
        t = self._chat(days=5)
        ChatThread.objects.filter(pk=t.pk).update(retain_until=None)
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertNotIn(f"chat:{t.id}", keys)

    def test_ordering_overdue_first(self):
        overdue = self._chat(days=-1, title="overdue")
        soon = self._dataroom(days=2, name="soon")
        later = self._meeting(days=10, name="later")
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertEqual(
            keys,
            [f"chat:{overdue.id}", f"dataroom:{soon.uuid}", f"meeting:{later.uuid}"],
        )

    def test_overdue_renders_today(self):
        self._chat(days=-1)
        items = list(self.client.get(reverse("inbox")).context["page_obj"])
        self.assertTrue(items[0]["is_overdue"])
        self.assertEqual(items[0]["countdown"], "Today")

    def test_only_own_items(self):
        mine = self._chat(days=5)
        theirs = self._chat(owner=self.other, days=5)
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertIn(f"chat:{mine.id}", keys)
        self.assertNotIn(f"chat:{theirs.id}", keys)

    def test_all_three_types_listed(self):
        c = self._chat(days=3)
        d = self._dataroom(days=3)
        m = self._meeting(days=3)
        keys = set(self._keys(self.client.get(reverse("inbox"))))
        self.assertEqual(
            keys, {f"chat:{c.id}", f"dataroom:{d.uuid}", f"meeting:{m.uuid}"},
        )

    def test_empty_state(self):
        resp = self.client.get(reverse("inbox"))
        self.assertEqual(resp.context["page_obj"].paginator.count, 0)
        self.assertContains(resp, "all caught up")


class InboxArchivedTests(InboxTestBase):
    def test_archived_hidden_by_default(self):
        t = self._chat(days=5, archived=True)
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertNotIn(f"chat:{t.id}", keys)

    def test_archived_shown_with_param(self):
        t = self._chat(days=5, archived=True)
        keys = self._keys(self.client.get(reverse("inbox"), {"show_archived": "1"}))
        self.assertIn(f"chat:{t.id}", keys)


class InboxPaginationTests(InboxTestBase):
    def _make_many(self, n):
        for i in range(n):
            ChatThread.objects.create(title=f"t{i}", created_by=self.user)
        ChatThread.objects.filter(created_by=self.user).update(
            retain_until=self.now + timedelta(days=5),
        )

    def test_first_page_capped_and_two_pages(self):
        self._make_many(51)
        page = self.client.get(reverse("inbox")).context["page_obj"]
        self.assertEqual(len(page), 50)
        self.assertEqual(page.paginator.num_pages, 2)

    def test_second_page_has_remainder(self):
        self._make_many(51)
        page = self.client.get(reverse("inbox"), {"page": "2"}).context["page_obj"]
        self.assertEqual(len(page), 1)

    def test_pager_preserves_show_archived(self):
        self._make_many(51)
        resp = self.client.get(reverse("inbox"), {"show_archived": "1"})
        # {% querystring %} output is HTML-escaped and param order may vary; assert both.
        self.assertContains(resp, "page=2")
        self.assertContains(resp, "show_archived=1")


class InboxRenewTests(InboxTestBase):
    def _renew(self, items):
        return self.client.post(
            reverse("inbox_renew"),
            data=json.dumps({"items": items}),
            content_type="application/json",
        )

    def test_renew_single_resets_timer(self):
        t = self._chat(days=2)
        resp = self._renew([f"chat:{t.id}"])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["renewed"], 1)
        t.refresh_from_db()
        expected = timezone.now() + RETENTION_PERIODS["chat.ChatThread"]
        self.assertAlmostEqual(t.retain_until, expected, delta=timedelta(seconds=30))

    def test_renew_bulk_mixed_types(self):
        c = self._chat(days=2)
        d = self._dataroom(days=2)
        m = self._meeting(days=2)
        resp = self._renew([f"chat:{c.id}", f"dataroom:{d.uuid}", f"meeting:{m.uuid}"])
        self.assertEqual(resp.json()["renewed"], 3)
        c.refresh_from_db(); d.refresh_from_db(); m.refresh_from_db()
        now = timezone.now()
        self.assertAlmostEqual(
            c.retain_until, now + RETENTION_PERIODS["chat.ChatThread"], delta=timedelta(seconds=30),
        )
        self.assertAlmostEqual(
            d.retain_until, now + RETENTION_PERIODS["documents.DataRoom"], delta=timedelta(seconds=30),
        )
        self.assertAlmostEqual(
            m.retain_until, now + RETENTION_PERIODS["meetings.Meeting"], delta=timedelta(seconds=30),
        )

    def test_renew_cannot_touch_other_users_item(self):
        theirs = self._chat(owner=self.other, days=2)
        before = ChatThread.objects.get(pk=theirs.pk).retain_until
        resp = self._renew([f"chat:{theirs.id}"])
        self.assertEqual(resp.json()["renewed"], 0)
        self.assertEqual(ChatThread.objects.get(pk=theirs.pk).retain_until, before)

    def test_renew_does_not_bump_updated_at(self):
        t = self._chat(days=2)
        t.refresh_from_db()
        before = t.updated_at
        self._renew([f"chat:{t.id}"])
        t.refresh_from_db()
        self.assertEqual(t.updated_at, before)

    def test_renew_skips_malformed_keys(self):
        t = self._chat(days=2)
        resp = self._renew([f"chat:{t.id}", "garbage", "meeting:not-a-uuid", "bogus:x"])
        self.assertEqual(resp.json()["renewed"], 1)

    def test_renew_all_malformed_returns_400(self):
        resp = self._renew(["chat", "chat:", "chat:not-a-uuid"])
        self.assertEqual(resp.status_code, 400)

    def test_renew_unknown_type_ignored(self):
        import uuid
        resp = self._renew([f"bogus:{uuid.uuid4()}"])
        self.assertEqual(resp.status_code, 400)

    def test_renew_missing_items_400(self):
        resp = self.client.post(
            reverse("inbox_renew"), data=json.dumps({}), content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_renew_empty_list_400(self):
        resp = self._renew([])
        self.assertEqual(resp.status_code, 400)

    def test_renew_non_list_items_400(self):
        resp = self.client.post(
            reverse("inbox_renew"),
            data=json.dumps({"items": "chat:x"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_renew_invalid_json_400(self):
        resp = self.client.post(
            reverse("inbox_renew"), data="not json", content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class InboxAccessTests(InboxTestBase):
    def test_inbox_requires_login(self):
        self.client.logout()
        self.assertEqual(self.client.get(reverse("inbox")).status_code, 302)

    def test_renew_requires_login(self):
        self.client.logout()
        self.assertEqual(self.client.post(reverse("inbox_renew")).status_code, 302)

    def test_renew_rejects_get(self):
        self.assertEqual(self.client.get(reverse("inbox_renew")).status_code, 405)

    def test_inbox_rejects_post(self):
        self.assertEqual(self.client.post(reverse("inbox")).status_code, 405)

    def test_nav_shows_inbox_link(self):
        resp = self.client.get(reverse("inbox"))
        self.assertContains(resp, 'href="/inbox/"')
        self.assertContains(resp, 'aria-label="Inbox"')


class InboxSubtitleTests(InboxTestBase):
    def test_subtitle_shows_org_name(self):
        from accounts.models import Membership, Organization

        org = Organization.objects.create(name="Acme TTO", slug="acme-tto")
        Membership.objects.create(user=self.user, org=org)
        resp = self.client.get(reverse("inbox"))
        self.assertContains(resp, "Acme TTO")
        self.assertContains(resp, "data retention policy")

    def test_subtitle_fallback_without_org(self):
        resp = self.client.get(reverse("inbox"))
        self.assertContains(resp, "your organization")

    def test_retention_periods_shown(self):
        resp = self.client.get(reverse("inbox"))
        self.assertContains(resp, "retained for one year")
        self.assertContains(resp, "90 days")
        self.assertContains(resp, "data retention policy")


class InboxFilterSortTests(InboxTestBase):
    def test_type_filter_limits_to_selected_type(self):
        c = self._chat(days=3)
        self._dataroom(days=3)
        self._meeting(days=3)
        keys = set(self._keys(self.client.get(reverse("inbox"), {"type": "chat"})))
        self.assertEqual(keys, {f"chat:{c.id}"})

    def test_invalid_type_falls_back_to_all(self):
        c = self._chat(days=3)
        d = self._dataroom(days=3)
        keys = set(self._keys(self.client.get(reverse("inbox"), {"type": "bogus"})))
        self.assertEqual(keys, {f"chat:{c.id}", f"dataroom:{d.uuid}"})

    def test_sort_default_soonest_first(self):
        soon = self._chat(days=2, title="soon")
        later = self._chat(days=20, title="later")
        keys = self._keys(self.client.get(reverse("inbox")))
        self.assertEqual(keys, [f"chat:{soon.id}", f"chat:{later.id}"])

    def test_sort_latest_reverses_order(self):
        soon = self._chat(days=2, title="soon")
        later = self._chat(days=20, title="later")
        keys = self._keys(self.client.get(reverse("inbox"), {"sort": "latest"}))
        self.assertEqual(keys, [f"chat:{later.id}", f"chat:{soon.id}"])

    def test_tabs_show_per_type_counts(self):
        self._chat(days=3)
        self._chat(days=4)
        self._meeting(days=3)
        resp = self.client.get(reverse("inbox"))
        counts = {t["key"]: t["count"] for t in resp.context["type_tabs"]}
        self.assertEqual(counts["all"], 3)
        self.assertEqual(counts["chat"], 2)
        self.assertEqual(counts["meeting"], 1)
        self.assertEqual(counts["dataroom"], 0)


class RelativeFutureUnitTests(TestCase):
    def setUp(self):
        # Anchor at a fixed midday so local-date boundaries don't make this flaky.
        self.now = timezone.now().replace(hour=12, minute=0, second=0, microsecond=0)

    def test_overdue(self):
        label, _abs, overdue = _relative_future(self.now - timedelta(days=1), self.now)
        self.assertEqual(label, "Today")
        self.assertTrue(overdue)

    def test_today(self):
        label, _abs, overdue = _relative_future(self.now + timedelta(hours=1), self.now)
        self.assertEqual(label, "Today")
        self.assertFalse(overdue)

    def test_tomorrow(self):
        label, _abs, _ = _relative_future(self.now + timedelta(days=1), self.now)
        self.assertEqual(label, "Tomorrow")

    def test_n_days(self):
        label, _abs, _ = _relative_future(self.now + timedelta(days=10), self.now)
        self.assertEqual(label, "in 10 days")
