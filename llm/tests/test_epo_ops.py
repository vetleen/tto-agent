"""Tests for the EPO OPS patent tools (llm/tools/epo_ops.py).

Uses mocked HTTP throughout — no live OPS credentials. The OPS JSON fixtures
below follow the assumed OPS v3.2 shape and double as the contract the parsers
target; if a live response differs, fixtures and parsers get corrected together.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from llm.tools.epo_ops import (
    PatentEpoOpsFamilyTool,
    PatentEpoOpsGetTool,
    PatentEpoOpsSearchTool,
    _as_list,
    _build_cql,
    _collect_text,
    _docdb_ref,
    _espacenet_url,
    _format_family,
    _format_get,
    _format_search,
    _get_access_token,
    _log_ops_usage,
    _normalize_cpc,
    _normalize_pubnumber,
    _ops_request,
    _parse_family,
    _parse_ops_fault,
    _parse_search_results,
    _pubnumber_from_doc_ids,
    _rank_legal,
    _sanitize_date,
    _text,
)

User = get_user_model()

_DUMMY_CACHE = {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}}
_LOCMEM_CACHE = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

_EXCHANGE_DOC = {
    "@country": "EP",
    "@doc-number": "1000000",
    "@kind": "A1",
    "bibliographic-data": {
        "invention-title": [
            {"@lang": "de", "$": "Ein Gerät"},
            {"@lang": "en", "$": "A widget"},
        ],
        "publication-reference": {
            "document-id": [
                {"@document-id-type": "docdb", "date": {"$": "20000101"}},
            ]
        },
        "parties": {
            "applicants": {"applicant": [{"applicant-name": {"name": {"$": "ACME Corp"}}}]},
            "inventors": {"inventor": {"inventor-name": {"name": {"$": "Jane Doe"}}}},
        },
    },
    "abstract": {"@lang": "en", "p": {"$": "An improved widget."}},
}

SEARCH_FIXTURE = {
    "ops:world-patent-data": {
        "ops:biblio-search": {
            "@total-result-count": "42",
            "ops:search-result": {
                "exchange-documents": {"exchange-document": _EXCHANGE_DOC},
            },
        }
    }
}

GET_FIXTURE = {
    "ops:world-patent-data": {
        "exchange-documents": {"exchange-document": _EXCHANGE_DOC},
    }
}

# Real OPS family members carry the number in document-id CHILD elements
# (country/doc-number/kind as {"$": ...}), not top-level attributes — this shape
# is exactly what broke pub-number extraction in live validation.
FAMILY_FIXTURE = {
    "ops:world-patent-data": {
        "ops:patent-family": {
            "ops:family-member": [
                {
                    "publication-reference": {
                        "document-id": [
                            {
                                "@document-id-type": "docdb",
                                "country": {"$": "EP"},
                                "doc-number": {"$": "1000000"},
                                "kind": {"$": "A1"},
                            },
                            {"@document-id-type": "epodoc", "doc-number": {"$": "EP1000000"}},
                        ]
                    },
                    "ops:legal": [
                        {"@code": "17Q ", "@desc": "First examination report despatched"},
                        {"@code": "PGFP", "@desc": "Annual fee paid"},
                    ],
                },
                {
                    "publication-reference": {
                        "document-id": {
                            "@document-id-type": "docdb",
                            "country": {"$": "US"},
                            "doc-number": {"$": "6093011"},
                            "kind": {"$": "A"},
                        }
                    },
                    "ops:legal": {"@code": "LAPS", "@desc": "Lapse for failure to pay maintenance fees"},
                },
            ]
        }
    }
}


def _mock_ok(payload):
    body = json.dumps(payload).encode()
    m = MagicMock()
    m.status_code = 200
    m.headers = {}
    m.iter_content.return_value = [body]
    m.content = body
    m.json.return_value = payload
    m.raise_for_status = MagicMock()
    return m


def _fault_xml(code, message):
    return (
        '<?xml version="1.0" encoding="UTF-8"?><fault xmlns="http://ops.epo.org">'
        f"<code>{code}</code><message>{message}</message></fault>"
    )


def _mock_http_error(status, text="", headers=None):
    import requests as req

    m = MagicMock()
    m.status_code = status
    m.headers = headers or {}
    m.text = text
    m.raise_for_status.side_effect = req.exceptions.HTTPError(response=m)
    return m


def _mock_token(token="tok", expires_in="1200"):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"access_token": token, "token_type": "Bearer", "expires_in": expires_in}
    return m


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class BuildCqlTests(TestCase):
    def test_keywords_only(self):
        self.assertEqual(_build_cql(keywords="battery"), 'txt="battery"')

    def test_multiple_fields_anded(self):
        self.assertEqual(
            _build_cql(keywords="battery", applicant="acme", cpc="H01M"),
            'txt="battery" and pa="acme" and cpc=H01M',
        )

    def test_multi_word_keywords_use_all_not_phrase(self):
        # txt="a b c" is an exact-phrase search (OPS 404 in prod); `all` ANDs the words.
        self.assertEqual(
            _build_cql(keywords="ultrasound vessel centerline angle correction"),
            'txt all "ultrasound vessel centerline angle correction"',
        )

    def test_multi_word_applicant_stays_quoted(self):
        self.assertEqual(
            _build_cql(applicant="NTNU Technology Transfer"), 'pa="NTNU Technology Transfer"'
        )

    def test_date_range(self):
        cql = _build_cql(keywords="x", date_from="20200101", date_to="20201231")
        self.assertIn('pd within "20200101 20201231"', cql)

    def test_date_year_expanded(self):
        cql = _build_cql(keywords="x", date_from="2020", date_to="2021")
        self.assertIn('pd within "20200101 20211231"', cql)

    def test_single_sided_dates_use_comparison(self):
        self.assertEqual(_build_cql(keywords="x", date_from="2020"), 'txt="x" and pd>=20200101')
        self.assertEqual(_build_cql(keywords="x", date_to="2021"), 'txt="x" and pd<=20211231')

    def test_cpc_keeps_slash_and_drops_spaces(self):
        self.assertEqual(_build_cql(cpc="A61B 8/06"), "cpc=A61B8/06")
        self.assertEqual(_build_cql(cpc="g01s15/8984"), "cpc=G01S15/8984")

    def test_invalid_cpc_dropped(self):
        self.assertEqual(_build_cql(keywords="x", cpc="not a cpc"), 'txt="x"')

    def test_production_failures_no_longer_emitted(self):
        """Inputs whose CQL OPS answered with 500 SERVER.DomainAccess in prod (WILFRED-7M)."""
        cases = [
            dict(keywords="geothermal pile", date_from="2000"),
            dict(applicant="NTNU Technology Transfer", date_from="2000"),
            dict(keywords="blood flow vessel", cpc="G01S15/8984"),
            dict(keywords="vector Doppler flow", inventor="Tortoli", date_to="2010"),
            dict(keywords="ultrasound flow centerline", cpc="A61B8/06"),
        ]
        for kwargs in cases:
            cql = _build_cql(**kwargs)
            self.assertNotIn("30001231", cql)
            self.assertNotIn("10000101", cql)
            self.assertNotRegex(cql, r'cpc="[^"]* [^"]*"')
        self.assertEqual(
            _build_cql(keywords="vector Doppler flow", inventor="Tortoli", date_to="2010"),
            'txt all "vector Doppler flow" and in="Tortoli" and pd<=20101231',
        )

    def test_empty_returns_blank(self):
        self.assertEqual(_build_cql(), "")

    def test_injection_stripped(self):
        # Quotes / '=' / parens can't escape the clause.
        cql = _build_cql(keywords='foo" or pa="bar')
        self.assertEqual(cql, 'txt all "foo or pa bar"')
        self.assertNotIn('="bar"', cql)


class NormalizeCpcTests(TestCase):
    def test_valid_symbols(self):
        for raw, expected in [
            ("H01M", "H01M"),
            ("a61b8", "A61B8"),
            ("A61B 8/06", "A61B8/06"),
            ("H01M10/0525", "H01M10/0525"),
            ("Y02E10/10", "Y02E10/10"),
        ]:
            self.assertEqual(_normalize_cpc(raw), expected)

    def test_invalid_symbols(self):
        for raw in ("", "battery", "A61B8 06", "Z01A", "A61B8/06 OR G01S15", "A61"):
            self.assertEqual(_normalize_cpc(raw), "", raw)


class ParseOpsFaultTests(TestCase):
    def test_parses_xml_fault(self):
        resp = MagicMock()
        resp.text = _fault_xml("SERVER.DomainAccess", "The request could not be processed.")
        resp.headers = {}
        self.assertEqual(
            _parse_ops_fault(resp),
            ("SERVER.DomainAccess", "The request could not be processed.", ""),
        )

    def test_rejection_header(self):
        resp = MagicMock()
        resp.text = ""
        resp.headers = {"X-Rejection-Reason": "IndividualQuotaPerHour"}
        self.assertEqual(_parse_ops_fault(resp), ("", "", "IndividualQuotaPerHour"))

    def test_unreadable_body_never_raises(self):
        resp = MagicMock()
        type(resp).text = property(lambda self: (_ for _ in ()).throw(ValueError("boom")))
        resp.headers = None
        self.assertEqual(_parse_ops_fault(resp), ("", "", ""))


class SanitizeDateTests(TestCase):
    def test_year_padded(self):
        self.assertEqual(_sanitize_date("2020", is_end=False), "20200101")
        self.assertEqual(_sanitize_date("2020", is_end=True), "20201231")

    def test_full_date_passthrough(self):
        self.assertEqual(_sanitize_date("2020-03-15", is_end=False), "20200315")

    def test_junk_dropped(self):
        self.assertEqual(_sanitize_date("soon", is_end=False), "")
        self.assertEqual(_sanitize_date("", is_end=True), "")


class NormalizePubNumberTests(TestCase):
    def test_variants_normalize(self):
        for raw in ("EP 1000000 A1", "ep1000000a1", "EP.1000000.A1", "EP-1000000-A1"):
            self.assertEqual(_normalize_pubnumber(raw), "EP1000000A1")

    def test_us_number(self):
        self.assertEqual(_normalize_pubnumber("US-9,876,543-B2"), "US9876543B2")

    def test_empty(self):
        self.assertEqual(_normalize_pubnumber(""), "")


class DocdbRefTests(TestCase):
    """Retrieval uses docdb dotted form — OPS 404s on epodoc + kind (validated live)."""

    def test_standard_with_kind(self):
        self.assertEqual(_docdb_ref("WO2026120190A1"), ("docdb", "WO.2026120190.A1"))

    def test_spaced_input(self):
        self.assertEqual(_docdb_ref("EP 1000000 A1"), ("docdb", "EP.1000000.A1"))

    def test_no_kind(self):
        # A kind-less number must NOT produce a trailing dot ("EP.1000000."),
        # which OPS rejects with a 404.
        self.assertEqual(_docdb_ref("EP1000000"), ("docdb", "EP.1000000"))

    def test_empty(self):
        self.assertIsNone(_docdb_ref(""))

    def test_injection_chars_rejected(self):
        # ?, #, % survive the separator strip and would inject into the OPS URL
        # path; normalization must reject them (-> None), not pass them through.
        for bad in ("EP1000000A1#x", "EP1000000A1?q=1", "EP1000000%2e"):
            self.assertEqual(_normalize_pubnumber(bad), "")
            self.assertIsNone(_docdb_ref(bad))


class EspacenetUrlTests(TestCase):
    def test_url(self):
        self.assertEqual(
            _espacenet_url("WO 2026120190 A1"),
            "https://worldwide.espacenet.com/patent/search?q=pn%3DWO2026120190A1",
        )

    def test_empty(self):
        self.assertEqual(_espacenet_url(""), "")


class ListAndTextHelperTests(TestCase):
    def test_as_list(self):
        self.assertEqual(_as_list(None), [])
        self.assertEqual(_as_list({"a": 1}), [{"a": 1}])
        self.assertEqual(_as_list([1, 2]), [1, 2])

    def test_text(self):
        self.assertEqual(_text({"$": "hi"}), "hi")
        self.assertEqual(_text("bare"), "bare")
        self.assertEqual(_text(5), "")

    def test_collect_text_skips_attributes(self):
        acc: list[str] = []
        _collect_text({"@lang": "en", "p": {"$": "body"}, "nested": [{"$": "more"}]}, acc)
        self.assertIn("body", acc)
        self.assertIn("more", acc)
        self.assertNotIn("en", acc)


# --------------------------------------------------------------------------- #
# Parsers & formatters (no HTTP).
# --------------------------------------------------------------------------- #
class ParserTests(TestCase):
    def test_parse_search_results(self):
        parsed = _parse_search_results(SEARCH_FIXTURE)
        self.assertEqual(parsed["count"], 1)
        self.assertEqual(parsed["total"], "42")
        r = parsed["results"][0]
        self.assertEqual(r["publication_number"], "EP1000000A1")
        self.assertEqual(r["title"], "A widget")
        self.assertEqual(r["applicants"], ["ACME Corp"])
        self.assertEqual(r["inventors"], ["Jane Doe"])
        self.assertEqual(r["date"], "20000101")
        self.assertIn("improved widget", r["abstract"])

    def test_parse_family(self):
        parsed = _parse_family(FAMILY_FIXTURE)
        self.assertEqual(parsed["count"], 2)
        nums = [m["publication_number"] for m in parsed["members"]]
        # Numbers come from document-id CHILD elements (the live shape).
        self.assertIn("EP1000000A1", nums)
        self.assertIn("US6093011A", nums)
        self.assertTrue(any("PGFP" in ev for m in parsed["members"] for ev in m["legal_events"]))

    def test_pubnumber_from_doc_ids_child_element_form(self):
        doc_ids = [
            {"@document-id-type": "docdb", "country": {"$": "DE"}, "doc-number": {"$": "69905327"}, "kind": {"$": "D1"}},
            {"@document-id-type": "epodoc", "doc-number": {"$": "DE69905327"}},
        ]
        # Prefers docdb (country + number + kind).
        self.assertEqual(_pubnumber_from_doc_ids(doc_ids), "DE69905327D1")

    def test_pubnumber_from_doc_ids_attribute_form(self):
        self.assertEqual(
            _pubnumber_from_doc_ids([{"@country": "EP", "@doc-number": "1000000", "@kind": "A1"}]),
            "EP1000000A1",
        )

    def test_rank_legal_surfaces_status_events(self):
        ranked = _rank_legal(["17Q First examination", "PGFP Annual fee paid", "AK Designated states"])
        self.assertEqual(ranked[0], "PGFP Annual fee paid")  # FEE keyword first

    def test_parse_empty_search(self):
        self.assertEqual(_parse_search_results({})["count"], 0)


class FormatterTests(TestCase):
    def test_format_search_wraps_and_attributes(self):
        out = _format_search(SEARCH_FIXTURE)
        self.assertIn("=== BEGIN EXTERNAL WEB CONTENT", out)
        self.assertIn("=== END EXTERNAL WEB CONTENT ===", out)
        self.assertIn("EPO / Espacenet", out)
        self.assertIn("EP1000000A1", out)
        self.assertIn("A widget", out)
        self.assertIn("ACME Corp", out)

    def test_format_search_error(self):
        self.assertIn("boom", _format_search({"error": "boom"}))

    def test_format_search_no_results(self):
        self.assertEqual(_format_search({}), "No matching patents found.")

    def test_format_get(self):
        out = _format_get(GET_FIXTURE, "EP1000000A1", "biblio")
        self.assertIn("A widget", out)
        self.assertIn("Abstract", out)
        self.assertIn("=== BEGIN EXTERNAL WEB CONTENT", out)

    def test_format_family(self):
        out = _format_family(FAMILY_FIXTURE, "EP1000000A1")
        self.assertIn("EP1000000A1", out)
        self.assertIn("US6093011A", out)
        self.assertIn("legal", out)


# --------------------------------------------------------------------------- #
# _ops_request error handling (token stubbed).
# --------------------------------------------------------------------------- #
@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_DUMMY_CACHE)
class OpsRequestTests(TestCase):
    def setUp(self):
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    @patch("llm.tools.epo_ops.requests.get")
    def test_success_returns_json_and_logs_usage(self, mock_get):
        from llm.models import OpsUsageLog

        mock_get.return_value = _mock_ok({"ok": 1})
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(OpsUsageLog.objects.filter(tool_name="patent_epoops_search").count(), 1)

    @patch("llm.tools.epo_ops.requests.get")
    def test_404_graceful_no_retry(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        data = _ops_request("published-data/publication/epodoc/EP0/biblio", {}, tool_name="patent_epoops_get")
        self.assertIn("error", data)
        self.assertIn("404", data["error"])
        self.assertEqual(mock_get.call_count, 1)

    @patch("llm.tools.epo_ops.requests.get")
    def test_400_graceful_no_retry(self, mock_get):
        mock_get.return_value = _mock_http_error(400)
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("error", data)
        self.assertEqual(mock_get.call_count, 1)

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_429_retries_then_succeeds(self, mock_get, _sleep):
        mock_get.side_effect = [_mock_http_error(429), _mock_ok({"ok": 1})]
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(mock_get.call_count, 2)

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_429_exhausted(self, mock_get, _sleep):
        mock_get.return_value = _mock_http_error(429)
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("error", data)
        self.assertEqual(mock_get.call_count, 4)  # 1 + 3 retries
        # A rate limit is transient: the final-attempt 429 must NOT be reported
        # as a permanent client error (which would make the agent give up).
        self.assertNotIn("will not resolve by retrying", data["error"])
        self.assertIn("unavailable after retries", data["error"])

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_500_retries_then_exhausts(self, mock_get, _sleep):
        mock_get.return_value = _mock_http_error(500)
        with self.assertLogs("llm.tools.epo_ops", level="INFO") as cm:
            data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("error", data)
        self.assertIn("unavailable after retries", data["error"])
        self.assertEqual(mock_get.call_count, 4)  # 1 + 3 retries
        # Per-attempt backoff chatter must stay at INFO (Sentry breadcrumbs);
        # exactly one WARNING — the terminal failure — becomes the event.
        warnings = [r for r in cm.records if r.levelno >= logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("failed after retries", warnings[0].getMessage())

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_500_domain_access_fails_fast(self, mock_get, sleep):
        """OPS's bad-query 500 is deterministic: no retries, no backoff, actionable message."""
        mock_get.return_value = _mock_http_error(
            500, text=_fault_xml("SERVER.DomainAccess", "The request could not be processed")
        )
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(mock_get.call_count, 1)
        sleep.assert_not_called()
        self.assertIn("SERVER.DomainAccess", data["error"])
        self.assertNotIn("unavailable", data["error"])

    @patch("llm.tools.epo_ops._invalidate_token")
    @patch("llm.tools.epo_ops.requests.get")
    def test_403_quota_rejection_skips_token_refresh(self, mock_get, invalidate):
        mock_get.return_value = _mock_http_error(403, headers={"X-Rejection-Reason": "IndividualQuotaPerHour"})
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(mock_get.call_count, 1)
        invalidate.assert_not_called()
        self.assertIn("IndividualQuotaPerHour", data["error"])

    @patch("llm.tools.epo_ops._invalidate_token")
    @patch("llm.tools.epo_ops.requests.get")
    def test_403_without_rejection_refreshes_once(self, mock_get, invalidate):
        mock_get.return_value = _mock_http_error(403)
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(mock_get.call_count, 2)
        invalidate.assert_called_once()
        self.assertIn("403", data["error"])

    @patch("llm.tools.epo_ops.requests.get")
    def test_client_error_message_carries_fault_code(self, mock_get):
        mock_get.return_value = _mock_http_error(400, text=_fault_xml("CLIENT.CQL", "bad query"))
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("400 CLIENT.CQL: bad query", data["error"])

    @patch("llm.tools.epo_ops.requests.get")
    def test_404_logs_info_not_warning(self, mock_get):
        """A missing record is a normal search outcome, not a Sentry event."""
        mock_get.return_value = _mock_http_error(404)
        with self.assertLogs("llm.tools.epo_ops", level="INFO") as cm:
            _ops_request("published-data/publication/epodoc/EP0/biblio", {}, tool_name="patent_epoops_get")
        self.assertTrue(all(r.levelno < logging.WARNING for r in cm.records))

    @patch("llm.tools.epo_ops.requests.get")
    def test_other_client_errors_still_warn(self, mock_get):
        """A 400 means our query-building is broken — keep it Sentry-visible."""
        mock_get.return_value = _mock_http_error(400)
        with self.assertLogs("llm.tools.epo_ops", level="WARNING") as cm:
            _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("client error 400", cm.output[0])

    @patch("llm.tools.epo_ops.requests.get")
    def test_oversized_response(self, mock_get):
        from llm.tools.web_fetch import _max_response_bytes

        oversized = MagicMock()
        oversized.status_code = 200
        oversized.headers = {"Content-Length": str(_max_response_bytes() + 1)}
        oversized.raise_for_status = MagicMock()
        mock_get.return_value = oversized
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("error", data)
        self.assertIn("large", data["error"])


# --------------------------------------------------------------------------- #
# Token flow.
# --------------------------------------------------------------------------- #
@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_LOCMEM_CACHE)
class TokenTests(TestCase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)

    @patch("llm.tools.epo_ops.requests.post")
    def test_token_fetched_once_then_cached(self, mock_post):
        mock_post.return_value = _mock_token("tok")
        self.assertEqual(_get_access_token(), "tok")
        self.assertEqual(_get_access_token(), "tok")
        mock_post.assert_called_once()

    @patch("llm.tools.epo_ops.requests.post")
    def test_token_failure_returns_none(self, mock_post):
        mock_post.return_value.raise_for_status.side_effect = Exception("boom")
        self.assertIsNone(_get_access_token())

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    @patch("llm.tools.epo_ops.requests.post")
    def test_401_refreshes_token_and_retries(self, mock_post, mock_get, _sleep):
        mock_post.return_value = _mock_token("tok")
        mock_get.side_effect = [_mock_http_error(401), _mock_ok({"ok": 1})]
        data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(mock_get.call_count, 2)
        # Initial fetch + one forced refresh after the 401.
        self.assertEqual(mock_post.call_count, 2)


# --------------------------------------------------------------------------- #
# Tools.
# --------------------------------------------------------------------------- #
@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_DUMMY_CACHE)
class PatentToolTests(TestCase):
    def setUp(self):
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_success(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        result = PatentEpoOpsSearchTool().invoke({"keywords": "widget"})
        self.assertIsInstance(result, str)
        self.assertIn("EP1000000A1", result)
        self.assertIn("A widget", result)
        self.assertIn("Espacenet: https://worldwide.espacenet.com/patent/search?q=pn%3DEP1000000A1", result)
        params = mock_get.call_args.kwargs["params"]
        self.assertIn('txt="widget"', params["q"])

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_invalid_cpc_rejected_without_calling_ops(self, mock_get):
        result = PatentEpoOpsSearchTool().invoke({"keywords": "pile", "cpc": "E02D5 30 OR E02D5/24"})
        self.assertIn("not a CPC symbol", result)
        mock_get.assert_not_called()

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_sends_fixed_cql(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        PatentEpoOpsSearchTool().invoke(
            {"keywords": "blood flow vessel", "cpc": "G01S15/8984", "date_from": "2000"}
        )
        self.assertEqual(
            mock_get.call_args.kwargs["params"]["q"],
            'txt all "blood flow vessel" and cpc=G01S15/8984 and pd>=20000101',
        )

    def test_search_requires_an_input(self):
        result = PatentEpoOpsSearchTool().invoke({"keywords": ""})
        self.assertIn("provide at least one", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_count_capped(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        PatentEpoOpsSearchTool().invoke({"keywords": "x", "count": 500})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1-25")

    @patch("llm.tools.epo_ops.requests.get")
    def test_get_success(self, mock_get):
        mock_get.return_value = _mock_ok(GET_FIXTURE)
        result = PatentEpoOpsGetTool().invoke({"publication_number": "EP 1000000 A1"})
        self.assertIn("A widget", result)
        # Retrieval uses the docdb dotted path (kind preserved), not epodoc+kind.
        self.assertIn("publication/docdb/EP.1000000.A1/biblio", mock_get.call_args.args[0])

    def test_get_invalid_parts(self):
        result = PatentEpoOpsGetTool().invoke({"publication_number": "EP1000000A1", "parts": "bogus"})
        self.assertIn("parts must be one of", result)

    def test_get_missing_number(self):
        result = PatentEpoOpsGetTool().invoke({"publication_number": ""})
        self.assertIn("publication number is required", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_family_success(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_FIXTURE)
        result = PatentEpoOpsFamilyTool().invoke({"publication_number": "EP1000000A1"})
        self.assertIn("US6093011A", result)
        self.assertIn("legal", result)

    def test_labels_are_static(self):
        # Marker-wrapped markdown is not a JSON dict, so dynamic labels never
        # fire — end_label_for_result must return None (static labels).
        for tool in (PatentEpoOpsSearchTool(), PatentEpoOpsGetTool(), PatentEpoOpsFamilyTool()):
            self.assertIsNone(tool.end_label_for_result({"anything": 1}))
            self.assertEqual(tool.section, "skills")
            self.assertEqual(tool.audience, "shared")


# --------------------------------------------------------------------------- #
# Usage log org resolution.
# --------------------------------------------------------------------------- #
class UsageLogTests(TestCase):
    def test_resolves_org_from_membership(self):
        from accounts.models import Membership, Organization
        from llm.models import OpsUsageLog
        from llm.types.context import RunContext

        user = User.objects.create_user(email="ops@example.com", password="pw")
        org = Organization.objects.create(name="Org", slug="org-ops")
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        ctx = RunContext.create(user_id=user.id)
        _log_ops_usage(ctx, "patent_epoops_search", 321)

        row = OpsUsageLog.objects.get(tool_name="patent_epoops_search")
        self.assertEqual(row.org_id, org.id)
        self.assertEqual(row.user_id, user.id)
        self.assertEqual(row.response_bytes, 321)

    def test_membership_less_user_logs_null_org(self):
        from llm.models import OpsUsageLog
        from llm.types.context import RunContext

        user = User.objects.create_user(email="noorg@example.com", password="pw")
        _log_ops_usage(RunContext.create(user_id=user.id), "patent_epoops_get", 10)
        row = OpsUsageLog.objects.get(tool_name="patent_epoops_get")
        self.assertIsNone(row.org_id)

    def test_db_error_swallowed(self):
        from llm.types.context import RunContext

        with patch("llm.models.OpsUsageLog.objects.create", side_effect=Exception("db down")):
            # Must not raise.
            _log_ops_usage(RunContext.create(user_id=1), "patent_epoops_search", 5)


# --------------------------------------------------------------------------- #
# Outcome logging: every exit of _ops_request writes one row.
# --------------------------------------------------------------------------- #
class OpsOutcomeLogTests(TestCase):
    PATH = "published-data/search/biblio"

    def setUp(self):
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    def _row(self):
        from llm.models import OpsUsageLog

        rows = list(OpsUsageLog.objects.all())
        self.assertEqual(len(rows), 1)
        return rows[0]

    @patch("llm.tools.epo_ops.requests.get")
    def test_success_row(self, mock_get):
        mock_get.return_value = _mock_ok({"ok": 1})
        _ops_request(self.PATH, {"q": 'txt="x"'}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.outcome, "ok")
        self.assertEqual(row.http_status, 200)
        self.assertEqual(row.query, 'txt="x"')
        self.assertEqual(row.request_path, self.PATH)
        self.assertEqual(row.attempts, 1)
        self.assertGreater(row.response_bytes, 0)

    @patch("llm.tools.epo_ops.requests.get")
    def test_404_is_no_results(self, mock_get):
        mock_get.return_value = _mock_http_error(404, text=_fault_xml("SERVER.EntityNotFound", "No results found"))
        _ops_request(self.PATH, {"q": 'txt="x"'}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.outcome, "no_results")
        self.assertEqual(row.http_status, 404)
        self.assertEqual(row.error_code, "SERVER.EntityNotFound")

    @patch("llm.tools.epo_ops.requests.get")
    def test_domain_access_row_keeps_fault_and_query(self, mock_get):
        mock_get.return_value = _mock_http_error(
            500, text=_fault_xml("SERVER.DomainAccess", "The request could not be processed")
        )
        _ops_request(self.PATH, {"q": 'cpc="A61B8 06"'}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.outcome, "error")
        self.assertEqual(row.http_status, 500)
        self.assertEqual(row.error_code, "SERVER.DomainAccess")
        self.assertEqual(row.error_message, "The request could not be processed")
        self.assertEqual(row.query, 'cpc="A61B8 06"')

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_exhausted_retries_one_row_with_attempts(self, mock_get, _sleep):
        mock_get.return_value = _mock_http_error(503)
        _ops_request(self.PATH, {"q": "x"}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.outcome, "error")
        self.assertEqual(row.http_status, 503)
        self.assertEqual(row.attempts, 4)

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_timeout_row(self, mock_get, _sleep):
        import requests as req

        mock_get.side_effect = req.exceptions.Timeout("read timed out")
        data = _ops_request(self.PATH, {"q": "x"}, tool_name="patent_epoops_search")
        self.assertIn("Timeout", data["error"])
        row = self._row()
        self.assertEqual(row.error_code, "Timeout")
        self.assertIsNone(row.http_status)

    def test_auth_failure_row(self):
        with patch("llm.tools.epo_ops._get_access_token", return_value=None):
            _ops_request(self.PATH, {"q": "x"}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.error_code, "AuthFailed")

    @patch("llm.tools.epo_ops.requests.get")
    def test_get_path_recorded_without_query(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        path = "published-data/publication/docdb/US.5701898.A/claims"
        _ops_request(path, {}, tool_name="patent_epoops_get")
        row = self._row()
        self.assertEqual(row.request_path, path)
        self.assertEqual(row.query, "")

    @patch("llm.tools.epo_ops.requests.get")
    def test_log_failure_never_breaks_the_call(self, mock_get):
        mock_get.return_value = _mock_ok({"ok": 1})
        with patch("llm.models.OpsUsageLog.objects.create", side_effect=Exception("db down")):
            self.assertEqual(
                _ops_request(self.PATH, {"q": "x"}, tool_name="patent_epoops_search"), {"ok": 1}
            )
