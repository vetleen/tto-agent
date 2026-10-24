"""Report on the nightly PII shadow comparison (documents.services.pii_shadow).

Usage:
    python manage.py pii_shadow_report                        # last 7 days up to yesterday
    python manage.py pii_shadow_report --date 2026-10-25 --days 1
    python manage.py pii_shadow_report --days 14 --disagreements
    python manage.py pii_shadow_report --run 2026-10-25       # run the shadow for that day first

Production stores the PII verdict per VERSION (its category tags) plus a review event
per Art. 9/10 candidate window, so the headline comparison is version-level: a category
counts as "quick positive" for a version when any window's probability reaches 0.5, and
is compared with the version's tag. Per category: confusion at 0.5, agreement at
0.3/0.5/0.7 and deciles of the per-version maximum probability split by tag. The gated
section compares the two systems on Article 9/10 directly: what the production reviewer
decided on the classifier's candidate windows (its review is already blind) and what the
same reviewer decided on quick-only candidates. ``--disagreements`` lists the version /
category pairs the two systems disagree on with a live 200-char preview of the strongest
window's first chunk — nothing textual is stored in the shadow tables.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from documents.services.pii_scan import PII_CATEGORIES
from documents.services.pii_shadow import SHADOW_FLAG_THRESHOLD
from guardrails.shadow_stats import agreement, confusion, deciles, mean, pct

_ARTICLE_ATTR = {"pii_special_category": "article_9", "pii_criminal_offence": "article_10"}


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CommandError(f"Not a date (YYYY-MM-DD): {value}") from exc


def version_maxima(rows) -> dict[int, dict]:
    """Per version: ``{"tags": [...], "max": {category: max probability | None}}`` over
    the quick-pathway rows (a category is None when every window refused it)."""
    out: dict[int, dict] = {}
    for r in rows:
        if r.quick_pathway != "quick":
            continue
        entry = out.setdefault(r.version_id, {"tags": list(r.prod_version_categories or []), "max": {}})
        for category in PII_CATEGORIES:
            p = (r.quick_probabilities or {}).get(category)
            if p is None:
                entry["max"].setdefault(category, None)
            else:
                current = entry["max"].get(category)
                entry["max"][category] = p if current is None else max(current, p)
    return out


def summarize(rows) -> dict:
    """Aggregate shadow rows into the report's numbers (pure; tested directly)."""
    quick_rows = [r for r in rows if r.quick_pathway == "quick"]
    out = {
        "rows": len(rows),
        "windows": len(quick_rows),
        "versions": len({r.version_id for r in quick_rows}),
        "orgs": len({r.org_id for r in quick_rows if r.org_id is not None}),
        "non_quick": sum(1 for r in rows if r.quick_pathway != "quick"),
        "refusals": sum(1 for r in quick_rows if any(p is None for p in (r.quick_probabilities or {}).values())),
    }
    maxima = version_maxima(rows)
    per_category = {}
    for category in PII_CATEGORIES:
        pairs = [
            (category in v["tags"], v["max"][category])
            for v in maxima.values() if v["max"].get(category) is not None
        ]
        per_category[category] = {
            "versions": len(pairs),
            "confusion": confusion((tagged, p >= SHADOW_FLAG_THRESHOLD) for tagged, p in pairs),
            "agreement": agreement(pairs),
            "deciles": deciles(pairs),
        }
    out["per_category"] = per_category

    # Gated categories: the production reviewer on the classifier's candidate windows
    # vs. the same reviewer on quick-only candidates.
    prod_windows = [r for r in quick_rows if r.prod_candidates]
    prod_actions = Counter(r.prod_action or "-" for r in prod_windows)
    candidate_probs = [
        (r.quick_probabilities or {}).get(c)
        for r in prod_windows for c in r.prod_candidates
        if (r.quick_probabilities or {}).get(c) is not None
    ]
    quick_only = [r for r in quick_rows if r.blind_review]
    reviewed = [r for r in quick_only if r.blind_review == "reviewed"]
    confirmed = [
        r for r in reviewed
        if any(getattr(r, f"blind_{_ARTICLE_ATTR[c]}") for c in r.blind_candidates if c in _ARTICLE_ATTR)
    ]
    out["gated"] = {
        "prod_candidate_windows": len(prod_windows),
        "prod_actions": dict(prod_actions),
        "prod_candidates_quick_flagged": pct(
            sum(1 for p in candidate_probs if p >= SHADOW_FLAG_THRESHOLD), len(candidate_probs),
        ),
        "prod_candidates_quick_mean": mean(candidate_probs),
        "quick_only_windows": len(quick_only),
        "quick_only_confirmed": pct(len(confirmed), len(reviewed)),
        "quick_only_mean_confidence": mean(r.blind_confidence for r in reviewed),
        "quick_only_no_model": sum(1 for r in quick_only if r.blind_review == "no_model"),
        "quick_only_errors": sum(1 for r in quick_only if r.blind_review == "error"),
        "quick_only_not_reviewed": sum(1 for r in quick_only if r.blind_review == "not_reviewed"),
    }
    return out


class Command(BaseCommand):
    help = "Report on the nightly PII shadow comparison (decisions quick pathway vs. the PII classifier)."

    def add_arguments(self, parser):
        parser.add_argument("--date", help="Last day to report, YYYY-MM-DD (default: yesterday).")
        parser.add_argument("--days", type=int, default=7, help="Number of days ending at --date (default: 7).")
        parser.add_argument("--disagreements", action="store_true",
                            help="List version/category disagreements with a live text preview.")
        parser.add_argument("--run", metavar="DATE",
                            help="Run the shadow for DATE first (fills in a skipped or partial night), then report.")

    def handle(self, *args, **options):
        from django.utils import timezone

        from documents.models import PIIScanShadowResult, PIIScanShadowRun

        end_day = _parse_day(options["date"]) if options["date"] else timezone.localdate() - timedelta(days=1)
        if options["run"]:
            from documents.services.pii_shadow import run_pii_shadow_for_day

            run_day = _parse_day(options["run"])
            summary = run_pii_shadow_for_day(run_day)
            self.stdout.write(f"Ran PII shadow for {run_day}: {json.dumps(summary, default=str, sort_keys=True)}")
            if not options["date"]:
                end_day = run_day
        days = max(1, options["days"])
        start_day = end_day - timedelta(days=days - 1)

        rows = list(
            PIIScanShadowResult.objects.filter(run_date__range=(start_day, end_day))
            .order_by("run_date", "version_id", "window_index")
        )
        runs = list(PIIScanShadowRun.objects.filter(run_date__range=(start_day, end_day)).order_by("started_at"))
        if not rows and not runs:
            self.stdout.write(f"Nothing to report for {start_day}..{end_day}.")
            return

        by_day: dict[date, list] = {}
        for row in rows:
            by_day.setdefault(row.run_date, []).append(row)
        for day in sorted(by_day):
            self._write_stats(f"== {day} ==", summarize(by_day[day]))
        if days > 1 and rows:
            self._write_stats(f"== TOTAL {start_day}..{end_day} ==", summarize(rows))

        self._write_runs(runs)
        if options["disagreements"]:
            self._write_disagreements(rows)
        self.stdout.write(self.style.SUCCESS("Done."))

    # ---- output -----------------------------------------------------------

    def _write_stats(self, title: str, s: dict) -> None:
        w = self.stdout.write
        w(title)
        w(f"windows compared: {s['windows']} (quick pathway)   versions: {s['versions']}   orgs: {s['orgs']}"
          f"   non-quick rows: {s['non_quick']}   windows with a refusal: {s['refusals']}")
        w(f"per category, version level (quick max probability @{SHADOW_FLAG_THRESHOLD:.2f} vs. production tag):")
        for category, c in s["per_category"].items():
            cells = c["confusion"]
            w(f"  {category:<28} versions={c['versions']:<4} both_negative={cells['both_negative']} both_positive={cells['both_positive']}"
              f" classifier_only={cells['classifier_only']} quick_only={cells['quick_only']}"
              f" | agreement " + " ".join(f"@{t:.1f} {v}" for t, v in c["agreement"].items()))
            w(f"  {'':<28} deciles tagged     : " + " ".join(str(n) for n in c["deciles"]["prod_flagged"]))
            w(f"  {'':<28} deciles not tagged : " + " ".join(str(n) for n in c["deciles"]["prod_not_flagged"]))
        g = s["gated"]
        actions = " ".join(f"{k}={v}" for k, v in sorted(g["prod_actions"].items())) or "-"
        quick_mean = f"{g['prod_candidates_quick_mean']:.2f}" if g["prod_candidates_quick_mean"] is not None else "-"
        w("gated (Art. 9/10):")
        w(f"  production candidate windows: {g['prod_candidate_windows']}   reviewer: {actions}"
          f"   quick on those candidates: flagged {g['prod_candidates_quick_flagged']}, mean p {quick_mean}")
        conf = f"{g['quick_only_mean_confidence']:.2f}" if g["quick_only_mean_confidence"] is not None else "-"
        w(f"  quick-only candidate windows: {g['quick_only_windows']}   blind reviewer confirmed: {g['quick_only_confirmed']}"
          f" (mean conf {conf})   no_model={g['quick_only_no_model']} errors={g['quick_only_errors']}"
          f" not_reviewed={g['quick_only_not_reviewed']}")
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

        quick_rows = [r for r in rows if r.quick_pathway == "quick"]
        by_version: dict[int, list] = defaultdict(list)
        for r in quick_rows:
            by_version[r.version_id].append(r)
        maxima = version_maxima(quick_rows)
        lines = []
        for version_id, entry in sorted(maxima.items()):
            for category in PII_CATEGORIES:
                p = entry["max"].get(category)
                if p is None:
                    continue
                tagged = category in entry["tags"]
                if tagged == (p >= SHADOW_FLAG_THRESHOLD):
                    continue
                strongest = max(
                    (r for r in by_version[version_id] if (r.quick_probabilities or {}).get(category) is not None),
                    key=lambda r: r.quick_probabilities[category],
                )
                lines.append((version_id, category, p, tagged, strongest))
        self.stdout.write(f"disagreements (version/category, classifier_only / quick_only): {len(lines)}")
        chunks = {}
        if lines:
            # Fetch exactly the previewed chunks, one (version, chunk_index) pair each.
            wanted = {(s.version_id, s.chunk_index_from) for *_, s in lines}
            selector = Q()
            for version_id, chunk_index in wanted:
                selector |= Q(version_id=version_id, chunk_index=chunk_index)
            for version_id, chunk_index, text in DataRoomDocumentChunk.objects.filter(selector).values_list(
                "version_id", "chunk_index", "text",
            ):
                chunks[(version_id, chunk_index)] = text
        for version_id, category, p, tagged, strongest in lines:
            kind = "classifier_only" if tagged else "quick_only"
            text = chunks.get((version_id, strongest.chunk_index_from))
            preview = re.sub(r"\s+", " ", text).strip()[:200] if text is not None else "(chunk deleted)"
            self.stdout.write(
                f"  {strongest.run_date} v={version_id} {category} {kind} p_max={p:.2f}"
                f" window={strongest.window_index} chunks={strongest.chunk_index_from}..{strongest.chunk_index_to}"
                f" tags={entry_tags(strongest)} | \"{preview}\""
            )
        self.stdout.write("")


def entry_tags(row) -> list[str]:
    return [t.replace("pii_ordinary_", "").replace("pii_", "") for t in (row.prod_version_categories or [])]
