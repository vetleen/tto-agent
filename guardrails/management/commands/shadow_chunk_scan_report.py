"""Report on the nightly chunk-scan shadow comparison (guardrails.shadow).

Usage:
    python manage.py shadow_chunk_scan_report                       # last 7 days up to yesterday
    python manage.py shadow_chunk_scan_report --date 2026-10-23 --days 1
    python manage.py shadow_chunk_scan_report --days 14 --disagreements
    python manage.py shadow_chunk_scan_report --run 2026-10-23      # run the shadow for that day first

Headline numbers cover rows the quick pathway answered (``quick_pathway == "quick"``
with a probability); structured/refused rows are counted separately. False-positive
rate = the reviewer's "allow" share of a system's positives. ``--disagreements`` fetches
a live text preview per chunk — nothing textual is stored in the shadow tables.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, timedelta

from django.core.management.base import BaseCommand, CommandError

from guardrails.shadow_stats import agreement, allow_share, confusion, deciles, mean, pct

# Reviewer verdicts on a chunk; the threshold outcomes (no reviewer) are not judged.
_ALLOW, _ACT = {"allow"}, {"quarantine"}


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CommandError(f"Not a date (YYYY-MM-DD): {value}") from exc


def summarize(rows) -> dict:
    """Aggregate shadow rows into the report's numbers (pure; tested directly).

    The arithmetic lives in ``guardrails.shadow_stats`` (shared with the classifier and
    PII reports); this maps chunk rows onto it and keeps this report's key names.
    """
    headline = [r for r in rows if r.quick_pathway == "quick" and r.quick_probability is not None]
    pairs = [(r.classifier_flagged, r.quick_probability) for r in headline]
    out = {
        "rows": len(rows),
        "chunks": len(headline),
        "versions": len({r.version_id for r in headline}),
        "orgs": len({r.org_id for r in headline}),
        "non_quick": sum(1 for r in rows if r.quick_pathway != "quick"),
        "refusals": sum(1 for r in rows if r.quick_probability is None),
        "confusion": confusion((r.classifier_flagged, r.quick_flagged) for r in headline),
        "agreement": agreement(pairs),
    }
    hist = deciles(pairs)
    out["deciles"] = {"classifier_flagged": hist["prod_flagged"], "classifier_not_flagged": hist["prod_not_flagged"]}

    clf_pos = [r for r in headline if r.classifier_flagged]
    quick_pos = [r for r in headline if r.quick_flagged]
    judged = _ALLOW | _ACT
    both_judged = [r for r in clf_pos if r.blind_review in judged and r.prod_review in judged]
    out["fp"] = {
        "classifier_blind": allow_share((r.blind_review for r in clf_pos), _ALLOW, _ACT),
        "quick_blind": allow_share((r.blind_review for r in quick_pos), _ALLOW, _ACT),
        "classifier_prod": allow_share((r.prod_review for r in clf_pos), _ALLOW, _ACT),
        "prod_vs_blind": pct(sum(1 for r in both_judged if r.blind_review == r.prod_review), len(both_judged)),
    }
    reviewed = [r for r in rows if r.blind_review in judged]
    by_verdict = {}
    for verdict in ("allow", "quarantine"):
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
    help = "Report on the nightly chunk-scan shadow comparison (decisions quick pathway vs. the classifier)."

    def add_arguments(self, parser):
        parser.add_argument("--date", help="Last day to report, YYYY-MM-DD (default: yesterday).")
        parser.add_argument("--days", type=int, default=7, help="Number of days ending at --date (default: 7).")
        parser.add_argument("--disagreements", action="store_true",
                            help="List classifier-only / quick-only chunks with a live text preview.")
        parser.add_argument("--run", metavar="DATE",
                            help="Run the shadow for DATE first (fills in a skipped or partial night), then report.")

    def handle(self, *args, **options):
        from django.utils import timezone

        from guardrails.models import ChunkScanShadowResult, ChunkScanShadowRun

        end_day = _parse_day(options["date"]) if options["date"] else timezone.localdate() - timedelta(days=1)
        if options["run"]:
            from guardrails.shadow import run_shadow_for_day

            run_day = _parse_day(options["run"])
            summary = run_shadow_for_day(run_day)
            self.stdout.write(f"Ran shadow for {run_day}: {json.dumps(summary, default=str, sort_keys=True)}")
            if not options["date"]:
                end_day = run_day
        days = max(1, options["days"])
        start_day = end_day - timedelta(days=days - 1)

        rows = list(
            ChunkScanShadowResult.objects.filter(run_date__range=(start_day, end_day))
            .order_by("run_date", "version_id", "chunk_index")
        )
        runs = list(ChunkScanShadowRun.objects.filter(run_date__range=(start_day, end_day)).order_by("started_at"))
        if not rows and not runs:
            self.stdout.write(f"Nothing to report for {start_day}..{end_day}.")
            return

        by_day: dict[date, list] = {}
        for row in rows:
            by_day.setdefault(row.run_date, []).append(row)
        for day in sorted(by_day):
            self._write_stats(f"== {day} ==", summarize(by_day[day]))
        if days > 1:
            self._write_stats(f"== TOTAL {start_day}..{end_day} ==", summarize(rows))

        self._write_runs(runs)
        if options["disagreements"]:
            self._write_disagreements(rows)
        self.stdout.write(self.style.SUCCESS("Done."))

    # ---- output -----------------------------------------------------------

    def _write_stats(self, title: str, s: dict) -> None:
        w = self.stdout.write
        w(title)
        w(f"chunks compared: {s['chunks']} (quick pathway)   versions: {s['versions']}   orgs: {s['orgs']}"
          f"   non-quick rows: {s['non_quick']}   refusals: {s['refusals']}")
        c = s["confusion"]
        w("confusion @0.50: " + "  ".join(
            f"{k}={c.get(k, 0)}" for k in ("both_negative", "both_positive", "classifier_only", "quick_only")))
        w("agreement: " + "  ".join(f"@{t:.2f} {v}" for t, v in s["agreement"].items()))
        w("quick probability deciles (0.0-0.1 .. 0.9-1.0):")
        w("  classifier flagged     : " + " ".join(str(n) for n in s["deciles"]["classifier_flagged"]))
        w("  classifier not flagged : " + " ".join(str(n) for n in s["deciles"]["classifier_not_flagged"]))
        fp = s["fp"]
        w('false-positive rates (reviewer "allow" share):')
        w(f"  classifier positives, blind review : {fp['classifier_blind']}")
        w(f"  quick positives, blind review      : {fp['quick_blind']}")
        w(f"  classifier positives, prod review  : {fp['classifier_prod']}")
        w(f"  prod vs blind agreement on classifier positives: {fp['prod_vs_blind']}")
        b = s["blind"]
        verdicts = " | ".join(
            f"{v} n={d['n']} mean conf " + (f"{d['mean_confidence']:.2f}" if d["mean_confidence"] is not None else "-")
            for v, d in b["by_verdict"].items()
        )
        severity = " ".join(f"{k}={v}" for k, v in sorted(b["severity"].items())) or "-"
        w(f"blind reviewer: {verdicts} | severity {severity} | errors={b['errors']} no_model={b['no_model']}"
          f" not_reviewed={b['not_reviewed']}")
        w("")

    def _write_runs(self, runs) -> None:
        from django.db.models import Count, Sum

        from llm.models import LLMCallLog

        skipped, decide_errors, reviews_capped = Counter(), 0, 0
        for run in runs:
            summary = run.summary or {}
            skipped.update(summary.get("skipped") or {})
            decide_errors += int(summary.get("decide_errors") or 0)
            reviews_capped += int(summary.get("reviews_capped") or 0)
        skipped_text = " ".join(f"{k}={v}" for k, v in sorted(skipped.items())) or "-"
        self.stdout.write(f"runs: {len(runs)}   skipped: {skipped_text}   decide_errors: {decide_errors}"
                          f"   reviews_capped: {reviews_capped}")
        per_model = (
            LLMCallLog.objects.filter(trace_id__in=[r.trace_id for r in runs])
            .values("model").annotate(cost=Sum("cost_usd"), calls=Count("id")).order_by("model")
        )
        parts = [f"{m['model']} {m['calls']} call(s) ${float(m['cost'] or 0):.4f}" for m in per_model]
        total = sum(float(m["cost"] or 0) for m in per_model)
        self.stdout.write("cost: " + (" | ".join(parts) + f" | total ${total:.4f}" if parts else "no LLM rows"))
        self.stdout.write("")

    def _write_disagreements(self, rows) -> None:
        from documents.models import DataRoomDocumentChunk

        disagreeing = [r for r in rows if r.classifier_flagged != r.quick_flagged]
        self.stdout.write(f"disagreements (classifier_only / quick_only): {len(disagreeing)}")
        texts = dict(
            DataRoomDocumentChunk.objects.filter(pk__in=[r.chunk_id for r in disagreeing])
            .values_list("pk", "text")
        )
        for r in disagreeing:
            kind = "classifier_only" if r.classifier_flagged else "quick_only"
            text = texts.get(r.chunk_id)
            preview = re.sub(r"\s+", " ", text).strip()[:200] if text is not None else "(chunk deleted)"
            p = f"{r.quick_probability:.2f}" if r.quick_probability is not None else "refused"
            blind = f"{r.blind_review or '-'}({r.blind_review_source})" if r.blind_review_source else (r.blind_review or "-")
            self.stdout.write(
                f"  {r.run_date} v={r.version_id} chunk={r.chunk_id} idx={r.chunk_index} {kind} p={p}"
                f" concern={r.quick_concern or '-'} tags={r.classifier_tags} prod={r.prod_review or '-'}"
                f" blind={blind} | \"{preview}\""
            )
        self.stdout.write("")
