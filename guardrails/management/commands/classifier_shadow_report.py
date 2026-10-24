"""Report on the inline classifier shadow (guardrails.shadow_classifier).

Usage:
    python manage.py classifier_shadow_report                        # last 7 days up to yesterday, all families
    python manage.py classifier_shadow_report --date 2026-10-25 --days 1
    python manage.py classifier_shadow_report --family web_content --family chat_message
    python manage.py classifier_shadow_report --days 14 --disagreements

Per family, per day and in total: rows the quick pathway answered (headline), the
confusion matrix at 0.5, agreement at 0.3/0.5/0.7, quick-probability deciles split by
the production verdict, false-positive rates (the reviewer's "allow" share of each
system's positives — production's own reviewer for the classifier, the blind reviewer
for both), the blind reviewer's verdict distribution and the cost of the shadow's own
calls. ``--disagreements`` lists the one-sided rows; only chat messages have a live text
preview (nothing textual is stored, and web/skill text exists nowhere else).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, timedelta

from django.core.management.base import BaseCommand, CommandError

from guardrails.shadow_stats import (
    agreement,
    allow_share,
    confusion,
    deciles,
    family_verdicts,
    mean,
    pct,
)

FAMILIES = ("chat_message", "web_content", "skill_content", "identity_field")


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CommandError(f"Not a date (YYYY-MM-DD): {value}") from exc


def summarize_family(rows, family: str) -> dict:
    """Aggregate one family's rows into the report's numbers (pure; tested directly)."""
    verdicts = family_verdicts(family)
    allow, act = verdicts["allow"], verdicts["act"]
    headline = [r for r in rows if r.quick_pathway == "quick" and r.quick_probability is not None]
    pairs = [(r.prod_flagged, r.quick_probability) for r in headline]
    out = {
        "rows": len(rows),
        "compared": len(headline),
        "users": len({r.user_id for r in headline if r.user_id is not None}),
        "orgs": len({r.org_id for r in headline if r.org_id is not None}),
        "non_quick": sum(1 for r in rows if r.quick_pathway != "quick" and not r.decide_error),
        "refusals": sum(1 for r in rows if r.quick_pathway == "quick" and r.quick_probability is None),
        "decide_errors": sum(1 for r in rows if r.decide_error),
        "confusion": confusion((r.prod_flagged, r.quick_flagged) for r in headline),
        "agreement": agreement(pairs),
        "deciles": deciles(pairs),
    }
    clf_pos = [r for r in headline if r.prod_flagged]
    quick_pos = [r for r in headline if r.quick_flagged]
    both_judged = [
        r for r in clf_pos
        if r.blind_review in allow | act and r.prod_review in allow | act
    ]
    out["fp"] = {
        "classifier_blind": allow_share((r.blind_review for r in clf_pos), allow, act),
        "quick_blind": allow_share((r.blind_review for r in quick_pos), allow, act),
        "classifier_prod": allow_share((r.prod_review for r in clf_pos), allow, act),
        "prod_vs_blind": pct(
            sum(1 for r in both_judged if (r.blind_review in allow) == (r.prod_review in allow)), len(both_judged),
        ),
    }
    out["prod_reviews"] = dict(Counter(r.prod_review or "-" for r in rows if r.prod_flagged))
    reviewed = [r for r in rows if r.blind_review in allow | act]
    by_verdict = {}
    for verdict in sorted({r.blind_review for r in reviewed}):
        subset = [r for r in reviewed if r.blind_review == verdict]
        by_verdict[verdict] = {"n": len(subset), "mean_confidence": mean(r.blind_review_confidence for r in subset)}
    out["blind"] = {
        "by_verdict": by_verdict,
        "severity": dict(Counter(r.blind_review_severity for r in reviewed if r.blind_review_severity)),
        "errors": sum(1 for r in rows if r.blind_review == "error"),
        "no_model": sum(1 for r in rows if r.blind_review == "no_model"),
        "not_reviewed": sum(1 for r in rows if r.blind_review == "not_reviewed"),
    }
    return out


class Command(BaseCommand):
    help = "Report on the inline classifier shadow (decisions quick pathway vs. the text classifiers)."

    def add_arguments(self, parser):
        parser.add_argument("--date", help="Last day to report, YYYY-MM-DD (default: yesterday).")
        parser.add_argument("--days", type=int, default=7, help="Number of days ending at --date (default: 7).")
        parser.add_argument("--family", action="append", choices=FAMILIES,
                            help="Family to report (repeatable; default: all).")
        parser.add_argument("--disagreements", action="store_true",
                            help="List classifier-only / quick-only rows (chat messages with a live preview).")

    def handle(self, *args, **options):
        from django.utils import timezone

        from guardrails.models import ClassifierShadowResult
        from guardrails.shadow import day_bounds

        end_day = _parse_day(options["date"]) if options["date"] else timezone.localdate() - timedelta(days=1)
        days = max(1, options["days"])
        start_day = end_day - timedelta(days=days - 1)
        families = tuple(dict.fromkeys(options["family"])) if options["family"] else FAMILIES
        start, _ = day_bounds(start_day)
        _, end = day_bounds(end_day)

        rows = list(
            ClassifierShadowResult.objects.filter(
                family__in=families, created_at__gte=start, created_at__lt=end,
            ).order_by("created_at")
        )
        if not rows:
            self.stdout.write(f"Nothing to report for {start_day}..{end_day} ({', '.join(families)}).")
            return

        for family in families:
            family_rows = [r for r in rows if r.family == family]
            if not family_rows:
                continue
            self.stdout.write(f"##### {family} #####")
            by_day: dict[date, list] = {}
            for row in family_rows:
                by_day.setdefault(timezone.localdate(row.created_at), []).append(row)
            for day in sorted(by_day):
                self._write_stats(f"== {family} {day} ==", summarize_family(by_day[day], family), family)
            if days > 1:
                self._write_stats(
                    f"== {family} TOTAL {start_day}..{end_day} ==", summarize_family(family_rows, family), family,
                )
            self._write_cost(family, [start_day + timedelta(days=i) for i in range(days)])
            if options["disagreements"]:
                self._write_disagreements(family_rows)
        self.stdout.write(self.style.SUCCESS("Done."))

    # ---- output -----------------------------------------------------------

    def _write_stats(self, title: str, s: dict, family: str) -> None:
        w = self.stdout.write
        w(title)
        w(f"compared: {s['compared']} (quick pathway)   users: {s['users']}   orgs: {s['orgs']}"
          f"   non-quick rows: {s['non_quick']}   refusals: {s['refusals']}   decide errors: {s['decide_errors']}")
        c = s["confusion"]
        w("confusion @0.50: " + "  ".join(
            f"{k}={c.get(k, 0)}" for k in ("both_negative", "both_positive", "classifier_only", "quick_only")))
        w("agreement: " + "  ".join(f"@{t:.2f} {v}" for t, v in s["agreement"].items()))
        w("quick probability deciles (0.0-0.1 .. 0.9-1.0):")
        w("  classifier flagged     : " + " ".join(str(n) for n in s["deciles"]["prod_flagged"]))
        w("  classifier not flagged : " + " ".join(str(n) for n in s["deciles"]["prod_not_flagged"]))
        verdicts = family_verdicts(family)
        if verdicts["allow"]:
            fp = s["fp"]
            w('false-positive rates (reviewer "allow" share: ' + "/".join(sorted(verdicts["allow"])) + "):")
            w(f"  classifier positives, blind review : {fp['classifier_blind']}")
            w(f"  quick positives, blind review      : {fp['quick_blind']}")
            w(f"  classifier positives, prod review  : {fp['classifier_prod']}")
            w(f"  prod vs blind agreement on classifier positives: {fp['prod_vs_blind']}")
        prod_reviews = " ".join(f"{k}={v}" for k, v in sorted(s["prod_reviews"].items())) or "-"
        w(f"production verdicts on classifier positives: {prod_reviews}")
        b = s["blind"]
        by_verdict = " | ".join(
            f"{v} n={d['n']} mean conf " + (f"{d['mean_confidence']:.2f}" if d["mean_confidence"] is not None else "-")
            for v, d in b["by_verdict"].items()
        ) or "-"
        severity = " ".join(f"{k}={v}" for k, v in sorted(b["severity"].items())) or "-"
        w(f"blind reviewer: {by_verdict} | severity {severity} | errors={b['errors']} no_model={b['no_model']}"
          f" not_reviewed={b['not_reviewed']}")
        w("")

    def _write_cost(self, family: str, days: list[date]) -> None:
        from django.db.models import Count, Sum

        from guardrails.shadow_classifier import day_trace_id
        from llm.models import LLMCallLog

        per_model = (
            LLMCallLog.objects.filter(trace_id__in=[day_trace_id(family, d) for d in days])
            .values("model").annotate(cost=Sum("cost_usd"), calls=Count("id")).order_by("model")
        )
        parts = [f"{m['model']} {m['calls']} call(s) ${float(m['cost'] or 0):.4f}" for m in per_model]
        total = sum(float(m["cost"] or 0) for m in per_model)
        self.stdout.write(f"cost ({family}): " + (" | ".join(parts) + f" | total ${total:.4f}" if parts else "no LLM rows"))
        self.stdout.write("")

    def _write_disagreements(self, rows) -> None:
        disagreeing = [
            r for r in rows
            if r.quick_pathway == "quick" and r.quick_probability is not None and r.prod_flagged != r.quick_flagged
        ]
        self.stdout.write(f"disagreements (classifier_only / quick_only): {len(disagreeing)}")
        previews = self._chat_previews(disagreeing)
        for r in disagreeing:
            kind = "classifier_only" if r.prod_flagged else "quick_only"
            p = f"{r.quick_probability:.2f}"
            blind = f"{r.blind_review or '-'}({r.blind_review_source})" if r.blind_review_source else (r.blind_review or "-")
            subject = " ".join(f"{k}={v}" for k, v in sorted((r.subject or {}).items()) if v not in (None, ""))
            line = (
                f"  {r.created_at:%Y-%m-%d %H:%M} {kind} p={p} concern={r.quick_concern or '-'}"
                f" tags={r.prod_tags} prod={r.prod_review or '-'} blind={blind} | {subject or '-'}"
                f" sha={r.text_sha256[:12]}"
            )
            if r.family == "chat_message":
                line += f' | "{previews.get(str((r.subject or {}).get("message_id")), "(message deleted)")}"'
            self.stdout.write(line)
        self.stdout.write("")

    @staticmethod
    def _chat_previews(rows) -> dict[str, str]:
        from chat.models import ChatMessage

        ids = [str((r.subject or {}).get("message_id")) for r in rows if r.family == "chat_message"]
        ids = [i for i in ids if i and i != "None"]
        if not ids:
            return {}
        previews = {}
        for pk, content, redacted in ChatMessage.objects.filter(pk__in=ids).values_list("pk", "content", "is_redacted"):
            previews[str(pk)] = "(redacted)" if redacted else re.sub(r"\s+", " ", content or "").strip()[:200]
        return previews
