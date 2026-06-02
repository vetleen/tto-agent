"""Tests for documents.pii_labels — the shared PII display/label module.

Pure functions (no DB, no LLM): ``summarize_pii_keys`` groups tag keys into
display buckets, and ``format_thread_pii_report`` renders the ``/pii`` Markdown.
"""
from django.test import SimpleTestCase

from documents.pii_labels import (
    BUCKET_LABEL,
    CRIMINAL,
    ORDINARY,
    SPECIAL,
    format_thread_pii_report,
    summarize_pii_keys,
)


class SummarizePiiKeysTests(SimpleTestCase):
    def test_empty_keys_returns_all_false(self):
        s = summarize_pii_keys([])
        self.assertFalse(s["has_ordinary"])
        self.assertEqual(s["ordinary_tooltip"], "")
        self.assertEqual(s["ordinary_descriptions"], [])
        self.assertFalse(s["special"])
        self.assertFalse(s["criminal"])

    def test_ordinary_keys_build_tooltip_and_descriptions(self):
        s = summarize_pii_keys(["pii_ordinary_identity", "pii_ordinary_communication"])
        self.assertTrue(s["has_ordinary"])
        self.assertEqual(len(s["ordinary_descriptions"]), 2)
        self.assertIn("personal identity", s["ordinary_tooltip"])
        self.assertIn("the content of communications", s["ordinary_tooltip"])
        self.assertTrue(s["ordinary_tooltip"].endswith("."))

    def test_descriptions_follow_canonical_order_not_input_order(self):
        # identity precedes communication in PII_CATEGORIES regardless of input order.
        s = summarize_pii_keys(["pii_ordinary_communication", "pii_ordinary_identity"])
        joined = "\n".join(s["ordinary_descriptions"])
        self.assertLess(joined.index("Identity data"), joined.index("Communication content"))

    def test_special_and_criminal_flags(self):
        s = summarize_pii_keys(["pii_special_category", "pii_criminal_offence"])
        self.assertTrue(s["special"])
        self.assertTrue(s["criminal"])
        self.assertFalse(s["has_ordinary"])

    def test_unknown_keys_are_ignored(self):
        s = summarize_pii_keys(["document_type", "source", "pii_special_category"])
        self.assertTrue(s["special"])
        self.assertFalse(s["has_ordinary"])
        self.assertFalse(s["criminal"])


class FormatThreadPiiReportTests(SimpleTestCase):
    def test_empty_returns_friendly_message(self):
        msg = format_thread_pii_report([])
        self.assertIn("No personal data categories were detected", msg)

    def test_all_buckets_present(self):
        msg = format_thread_pii_report(
            ["pii_ordinary_identity", "pii_special_category", "pii_criminal_offence"]
        )
        self.assertIn(BUCKET_LABEL[ORDINARY], msg)
        self.assertIn(BUCKET_LABEL[SPECIAL], msg)
        self.assertIn(BUCKET_LABEL[CRIMINAL], msg)
        self.assertIn("- Identity data", msg)

    def test_ordinary_only_omits_article_sections(self):
        msg = format_thread_pii_report(["pii_ordinary_identity"])
        self.assertIn(BUCKET_LABEL[ORDINARY], msg)
        self.assertNotIn(BUCKET_LABEL[SPECIAL], msg)
        self.assertNotIn(BUCKET_LABEL[CRIMINAL], msg)
