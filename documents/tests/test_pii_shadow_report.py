"""Tests for the pii_shadow_report management command."""

from datetime import date
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from accounts.models import User
from documents.management.commands.pii_shadow_report import summarize, version_maxima
from documents.models import DataRoom, DataRoomDocument, PIIScanShadowResult, PIIScanShadowRun
from documents.services.pii_scan import PII_CATEGORIES
from documents.tests._helpers import make_version
from llm.models import LLMCallLog

DAY = date(2026, 10, 25)
TRACE = "shadow-pii-20261025-abcdef123456"


def _probs(**overrides):
    probs = {c: 0.05 for c in PII_CATEGORIES}
    probs.update(overrides)
    return probs


class PIIShadowReportTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="pii-report@example.com", password="test1234")
        room = DataRoom.objects.create(name="Report Room", slug="pii-report-room", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="r.txt", status="ready",
        )
        self.v1 = make_version(self.doc, version_index=0, chunks=[
            {"text": "Alice Smith   lives at\n1 Main St", "token_count": 10},
            {"text": "diagnosed with diabetes", "token_count": 10},
        ])
        self.v2 = make_version(self.doc, version_index=1, chunks=[{"text": "plain clause", "token_count": 10}])

        def row(version, index, **kw):
            base = dict(
                run_date=DAY, trace_id=TRACE, org_id=1, version_id=version.pk, document_id=self.doc.pk,
                window_index=index, chunk_index_from=index, chunk_index_to=index, window_tokens=10, window_chars=20,
                quick_probabilities=_probs(), quick_pathway="quick", prod_version_categories=[],
            )
            base.update(kw)
            return PIIScanShadowResult.objects.create(**base)

        # v1: tagged identity (agreement) + special category confirmed by production on window 1
        # (quick agrees, 0.8). Window 0 also has a quick-only criminal-offence positive the blind
        # reviewer dismissed.
        v1_tags = ["pii_ordinary_identity", "pii_special_category"]
        row(self.v1, 0, prod_version_categories=v1_tags,
            quick_probabilities=_probs(pii_ordinary_identity=0.9, pii_criminal_offence=0.7),
            blind_review="reviewed", blind_candidates=["pii_criminal_offence"], blind_article_9=False,
            blind_article_10=False, blind_confidence=0.8)
        row(self.v1, 1, prod_version_categories=v1_tags,
            quick_probabilities=_probs(pii_special_category=0.8, pii_ordinary_identity=0.6),
            prod_candidates=["pii_special_category"], prod_action="confirmed", prod_article_9=True, prod_article_10=False)
        # v2: production tagged financial (quick 0.2 → classifier_only); quick-only special category
        # (0.9) confirmed by the blind reviewer; one refusal; plus a structured-pathway row.
        row(self.v2, 0, prod_version_categories=["pii_ordinary_financial"],
            quick_probabilities=_probs(pii_ordinary_financial=0.2, pii_special_category=0.9, pii_ordinary_social=None),
            blind_review="reviewed", blind_candidates=["pii_special_category"], blind_article_9=True,
            blind_article_10=False, blind_confidence=0.95)
        row(self.v2, 1, quick_pathway="structured", quick_probabilities=_probs(pii_special_category=0.9))
        PIIScanShadowRun.objects.create(
            run_date=DAY, trace_id=TRACE,
            summary={"windows": 3, "skipped": {"cap_windows": 2, "quick_off": 1}, "decide_errors": 1, "reviews_capped": 0},
        )
        LLMCallLog.objects.create(model="openai/gpt-6-luna", prompt=[], trace_id=TRACE, cost_usd="0.0030")
        LLMCallLog.objects.create(model="openai/gpt-6-sol", prompt=[], trace_id=TRACE, cost_usd="0.0400")

    def _run(self, *args):
        out = StringIO()
        call_command("pii_shadow_report", *args, stdout=out)
        return out.getvalue()

    def test_version_maxima(self):
        maxima = version_maxima(list(PIIScanShadowResult.objects.all()))
        self.assertEqual(set(maxima), {self.v1.pk, self.v2.pk})
        self.assertAlmostEqual(maxima[self.v1.pk]["max"]["pii_ordinary_identity"], 0.9)
        self.assertAlmostEqual(maxima[self.v1.pk]["max"]["pii_special_category"], 0.8)
        self.assertIsNone(maxima[self.v2.pk]["max"]["pii_ordinary_social"])  # refused everywhere
        self.assertEqual(maxima[self.v2.pk]["tags"], ["pii_ordinary_financial"])
        # The structured row does not feed the maxima.
        self.assertAlmostEqual(maxima[self.v2.pk]["max"]["pii_special_category"], 0.9)

    def test_summarize(self):
        s = summarize(list(PIIScanShadowResult.objects.all()))
        self.assertEqual(s["windows"], 3)
        self.assertEqual(s["versions"], 2)
        self.assertEqual(s["non_quick"], 1)
        self.assertEqual(s["refusals"], 1)
        identity = s["per_category"]["pii_ordinary_identity"]
        self.assertEqual(identity["versions"], 2)
        self.assertEqual(identity["confusion"], {"both_negative": 1, "both_positive": 1, "classifier_only": 0, "quick_only": 0})
        self.assertEqual(identity["agreement"][0.5], "100.0% (2/2)")
        self.assertEqual(identity["deciles"]["prod_flagged"][9], 1)
        special = s["per_category"]["pii_special_category"]
        self.assertEqual(special["confusion"], {"both_negative": 0, "both_positive": 1, "classifier_only": 0, "quick_only": 1})
        financial = s["per_category"]["pii_ordinary_financial"]
        self.assertEqual(financial["confusion"]["classifier_only"], 1)
        self.assertEqual(s["per_category"]["pii_ordinary_social"]["versions"], 1)  # v2 refused it
        g = s["gated"]
        self.assertEqual(g["prod_candidate_windows"], 1)
        self.assertEqual(g["prod_actions"], {"confirmed": 1})
        self.assertEqual(g["prod_candidates_quick_flagged"], "100.0% (1/1)")
        self.assertAlmostEqual(g["prod_candidates_quick_mean"], 0.8)
        self.assertEqual(g["quick_only_windows"], 2)
        self.assertEqual(g["quick_only_confirmed"], "50.0% (1/2)")
        self.assertAlmostEqual(g["quick_only_mean_confidence"], 0.875)

    def test_report_output(self):
        out = self._run("--date", DAY.isoformat(), "--days", "1")
        self.assertIn(f"== {DAY} ==", out)
        self.assertIn("windows compared: 3 (quick pathway)   versions: 2", out)
        self.assertIn("non-quick rows: 1   windows with a refusal: 1", out)
        self.assertIn("pii_special_category         versions=2    both_negative=0 both_positive=1 classifier_only=0 quick_only=1", out)
        self.assertIn("production candidate windows: 1   reviewer: confirmed=1   quick on those candidates: flagged 100.0% (1/1), mean p 0.80", out)
        self.assertIn("quick-only candidate windows: 2   blind reviewer confirmed: 50.0% (1/2) (mean conf 0.88)", out)
        self.assertIn("runs: 1   skipped: cap_windows=2 quick_off=1   decide_errors: 1   reviews_capped: 0", out)
        self.assertIn("openai/gpt-6-luna 1 call(s) $0.0030", out)
        self.assertIn("total $0.0430", out)
        self.assertNotIn("== TOTAL", out)
        self.assertNotIn("disagreements", out)
        self.assertIn("Done.", out)

    def test_total_block(self):
        out = self._run("--date", DAY.isoformat(), "--days", "3")
        self.assertIn("== TOTAL 2026-10-23..2026-10-25 ==", out)

    def test_disagreements_with_live_previews(self):
        out = self._run("--date", DAY.isoformat(), "--days", "1", "--disagreements")
        self.assertIn("disagreements (version/category, classifier_only / quick_only): 3", out)
        # v1 quick-only criminal offence → window 0's first chunk, whitespace collapsed.
        self.assertIn(f"v={self.v1.pk} pii_criminal_offence quick_only p_max=0.70 window=0 chunks=0..0", out)
        self.assertIn('| "Alice Smith lives at 1 Main St"', out)
        self.assertIn(f"v={self.v2.pk} pii_ordinary_financial classifier_only p_max=0.20", out)
        self.assertIn(f"v={self.v2.pk} pii_special_category quick_only p_max=0.90", out)
        self.assertIn('| "plain clause"', out)

    def test_disagreement_preview_when_chunk_deleted(self):
        from documents.models import DataRoomDocumentChunk

        DataRoomDocumentChunk.objects.filter(version=self.v2).delete()
        out = self._run("--date", DAY.isoformat(), "--days", "1", "--disagreements")
        self.assertIn("(chunk deleted)", out)

    def test_nothing_to_report(self):
        out = self._run("--date", "2026-01-01", "--days", "1")
        self.assertIn("Nothing to report for 2026-01-01..2026-01-01.", out)

    def test_run_option_runs_the_shadow_first(self):
        with patch("documents.services.pii_shadow.run_pii_shadow_for_day", return_value={"windows": 0, "trace_id": "t"}) as run:
            out = self._run("--run", DAY.isoformat(), "--days", "1")
        run.assert_called_once_with(DAY)
        self.assertIn(f"Ran PII shadow for {DAY}:", out)
        self.assertIn(f"== {DAY} ==", out)

    def test_bad_date(self):
        from django.core.management.base import CommandError

        with self.assertRaises(CommandError):
            self._run("--date", "25/10/2026")
