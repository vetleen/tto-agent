"""Unit tests for core.spend budget helpers."""

from datetime import date
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from accounts.models import Membership, Organization
from core.spend import (
    get_budget_status,
    get_month_boundaries,
    get_org_monthly_spend,
    get_user_monthly_spend,
)

User = get_user_model()


def _create_log(user, cost, dt):
    """Create an LLMCallLog and force-set created_at (bypassing auto_now_add)."""
    from llm.models import LLMCallLog

    log = LLMCallLog.objects.create(user=user, model="test", prompt=[], cost_usd=cost)
    aware_dt = timezone.make_aware(timezone.datetime(dt.year, dt.month, dt.day, 12, 0))
    LLMCallLog.objects.filter(pk=log.pk).update(created_at=aware_dt)
    return log


class GetMonthBoundariesTests(TestCase):
    def test_mid_month(self):
        start, end, next_month = get_month_boundaries(date(2026, 3, 15))
        self.assertEqual(start.date(), date(2026, 3, 1))
        self.assertEqual(end.date(), date(2026, 4, 1))
        self.assertEqual(next_month, date(2026, 4, 1))

    def test_first_of_month(self):
        start, end, next_month = get_month_boundaries(date(2026, 1, 1))
        self.assertEqual(start.date(), date(2026, 1, 1))
        self.assertEqual(end.date(), date(2026, 2, 1))
        self.assertEqual(next_month, date(2026, 2, 1))

    def test_last_of_month(self):
        start, end, next_month = get_month_boundaries(date(2026, 1, 31))
        self.assertEqual(start.date(), date(2026, 1, 1))
        self.assertEqual(end.date(), date(2026, 2, 1))

    def test_december_rolls_to_next_year(self):
        start, end, next_month = get_month_boundaries(date(2026, 12, 25))
        self.assertEqual(start.date(), date(2026, 12, 1))
        self.assertEqual(end.date(), date(2027, 1, 1))
        self.assertEqual(next_month, date(2027, 1, 1))

    def test_february_leap_year(self):
        start, end, next_month = get_month_boundaries(date(2028, 2, 29))
        self.assertEqual(start.date(), date(2028, 2, 1))
        self.assertEqual(end.date(), date(2028, 3, 1))


class GetUserMonthlySpendTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="spend@test.com", password="pass123")
        self.start, self.end, _ = get_month_boundaries(date(2026, 3, 15))

    def test_no_records_returns_zero(self):
        result = get_user_monthly_spend(self.user, self.start, self.end)
        self.assertEqual(result, Decimal("0"))

    def test_sums_records_within_month(self):
        _create_log(self.user, Decimal("1.50"), date(2026, 3, 10))
        _create_log(self.user, Decimal("2.25"), date(2026, 3, 20))
        result = get_user_monthly_spend(self.user, self.start, self.end)
        self.assertEqual(result, Decimal("3.75"))

    def test_excludes_other_months(self):
        _create_log(self.user, Decimal("10.00"), date(2026, 2, 28))  # February
        _create_log(self.user, Decimal("5.00"), date(2026, 4, 1))    # April
        _create_log(self.user, Decimal("1.00"), date(2026, 3, 15))   # March
        result = get_user_monthly_spend(self.user, self.start, self.end)
        self.assertEqual(result, Decimal("1.00"))

    def test_excludes_other_users(self):
        other_user = User.objects.create_user(email="other@test.com", password="pass123")
        _create_log(other_user, Decimal("100.00"), date(2026, 3, 10))
        result = get_user_monthly_spend(self.user, self.start, self.end)
        self.assertEqual(result, Decimal("0"))


class GetOrgMonthlySpendTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        self.user1 = User.objects.create_user(email="u1@test.com", password="pass123")
        self.user2 = User.objects.create_user(email="u2@test.com", password="pass123")
        Membership.objects.create(user=self.user1, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.user2, org=self.org, role=Membership.Role.MEMBER)
        self.start, self.end, _ = get_month_boundaries(date(2026, 3, 15))

    def test_sums_all_org_members(self):
        _create_log(self.user1, Decimal("3.00"), date(2026, 3, 10))
        _create_log(self.user2, Decimal("7.00"), date(2026, 3, 15))
        result = get_org_monthly_spend(self.org.id, self.start, self.end)
        self.assertEqual(result, Decimal("10.00"))

    def test_excludes_non_members(self):
        outsider = User.objects.create_user(email="outsider@test.com", password="pass123")
        _create_log(outsider, Decimal("99.00"), date(2026, 3, 10))
        result = get_org_monthly_spend(self.org.id, self.start, self.end)
        self.assertEqual(result, Decimal("0"))


class GetBudgetStatusTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Budget Org", slug="budget-org")
        self.user = User.objects.create_user(email="budget@test.com", password="pass123")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

    def test_no_budget_returns_none(self):
        self.org.preferences = {}
        self.org.save()
        self.assertIsNone(get_budget_status(self.user))

    def test_zero_budgets_returns_none(self):
        self.org.preferences = {"monthly_budget_per_user": 0, "monthly_budget_org": 0}
        self.org.save()
        self.assertIsNone(get_budget_status(self.user))

    def test_no_org_returns_none(self):
        user_no_org = User.objects.create_user(email="noorg@test.com", password="pass123")
        self.assertIsNone(get_budget_status(user_no_org))

    def test_user_budget_not_exceeded(self):
        from llm.models import LLMCallLog

        self.org.preferences = {"monthly_budget_per_user": 50}
        self.org.save()
        LLMCallLog.objects.create(
            user=self.user, model="test", prompt=[], cost_usd=Decimal("10.00"),
        )
        status = get_budget_status(self.user)
        self.assertIsNotNone(status)
        self.assertFalse(status["exceeded"])
        self.assertEqual(status["user_budget"], Decimal("50"))
        self.assertEqual(status["user_spend"], Decimal("10.00"))
        self.assertEqual(status["percentage"], 20)

    def test_user_budget_exceeded(self):
        from llm.models import LLMCallLog

        self.org.preferences = {"monthly_budget_per_user": 5}
        self.org.save()
        LLMCallLog.objects.create(
            user=self.user, model="test", prompt=[], cost_usd=Decimal("5.00"),
        )
        status = get_budget_status(self.user)
        self.assertTrue(status["exceeded"])
        self.assertEqual(status["exceeded_reason"], "user")
        self.assertEqual(status["percentage"], 100)

    def test_org_budget_exceeded(self):
        from llm.models import LLMCallLog

        other_user = User.objects.create_user(email="other@test.com", password="pass123")
        Membership.objects.create(user=other_user, org=self.org, role=Membership.Role.MEMBER)

        self.org.preferences = {"monthly_budget_org": 20}
        self.org.save()
        LLMCallLog.objects.create(
            user=self.user, model="test", prompt=[], cost_usd=Decimal("8.00"),
        )
        LLMCallLog.objects.create(
            user=other_user, model="test", prompt=[], cost_usd=Decimal("15.00"),
        )
        status = get_budget_status(self.user)
        self.assertTrue(status["exceeded"])
        self.assertEqual(status["exceeded_reason"], "org")

    def test_both_budgets_picks_tighter(self):
        from llm.models import LLMCallLog

        self.org.preferences = {"monthly_budget_per_user": 10, "monthly_budget_org": 100}
        self.org.save()
        LLMCallLog.objects.create(
            user=self.user, model="test", prompt=[], cost_usd=Decimal("8.00"),
        )
        status = get_budget_status(self.user)
        # User spend $8 of $10 = 80%, org spend $8 of $100 = 8% → user is tighter
        self.assertEqual(status["effective_budget"], Decimal("10"))
        self.assertEqual(status["effective_spend"], Decimal("8.00"))
        self.assertEqual(status["percentage"], 80)

    def test_percentage_capped_at_100(self):
        from llm.models import LLMCallLog

        self.org.preferences = {"monthly_budget_per_user": 5}
        self.org.save()
        LLMCallLog.objects.create(
            user=self.user, model="test", prompt=[], cost_usd=Decimal("25.00"),
        )
        status = get_budget_status(self.user)
        self.assertEqual(status["percentage"], 100)

    def test_reset_date_format(self):
        self.org.preferences = {"monthly_budget_per_user": 50}
        self.org.save()
        status = get_budget_status(self.user)
        # Should contain month name and year
        self.assertRegex(status["reset_date"], r"^[A-Z][a-z]+ \d+, \d{4}$")
