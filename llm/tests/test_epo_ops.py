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
    PatentEpoOpsClassificationTool,
    PatentEpoOpsFamilyTool,
    PatentEpoOpsGetTool,
    PatentEpoOpsSearchTool,
    _as_list,
    _build_cql,
    _collect_text,
    _citation_lines,
    _citation_number,
    _cited_references,
    _cpc_codes,
    _cpc_line,
    _cpc_lookup_symbol,
    _group_families,
    _parse_cpc_scheme,
    _split_cpc,
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

def _cpc_entry(section, cls, subclass, main_group, subgroup, value="I", office="EP", scheme="CPCI"):
    return {
        "classification-scheme": {"@office": "EP", "@scheme": scheme},
        "section": {"$": section},
        "class": {"$": cls},
        "subclass": {"$": subclass},
        "main-group": {"$": main_group},
        "subgroup": {"$": subgroup},
        "classification-value": {"$": value},
        "generating-office": {"$": office},
    }


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
        # Real OPS shape: CPC as parts, repeated per generating office.
        "patent-classifications": {
            "patent-classification": [
                _cpc_entry("A", "61", "B", "8", "06", "I", "US"),
                _cpc_entry("A", "61", "B", "8", "06", "I", "EP"),
                _cpc_entry("A", "61", "B", "8", "4254", "I", "US"),
                _cpc_entry("G", "01", "S", "15", "8979", "A", "EP"),
            ]
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
        self.assertEqual(_build_cql(keywords="battery"), 'ta="battery"')

    def test_multiple_fields_anded(self):
        self.assertEqual(
            _build_cql(keywords="battery", applicant="acme", cpc="H01M"),
            'ta="battery" and pa="acme" and cpc=H01M',
        )

    def test_multi_word_keywords_use_all_not_phrase(self):
        # ta="a b c" is an exact-phrase search (OPS 404 in prod); `all` ANDs the words.
        self.assertEqual(
            _build_cql(keywords="ultrasound vessel centerline angle correction"),
            'ta all "ultrasound vessel centerline angle correction"',
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
        self.assertEqual(_build_cql(keywords="x", date_from="2020"), 'ta="x" and pd>=20200101')
        self.assertEqual(_build_cql(keywords="x", date_to="2021"), 'ta="x" and pd<=20211231')

    def test_cpc_keeps_slash_and_drops_spaces(self):
        self.assertEqual(_build_cql(cpc="A61B 8/06", include_subgroups=False), "cpc=A61B8/06")
        self.assertEqual(_build_cql(cpc="g01s15/8984", include_subgroups=False), "cpc=G01S15/8984")

    def test_subgroups_included_by_default_only_on_subgroup_codes(self):
        # /low widens a subgroup; main groups/subclasses already include theirs.
        self.assertEqual(_build_cql(cpc="A61B8/06"), "cpc=A61B8/06/low")
        self.assertEqual(_build_cql(cpc="A61B8"), "cpc=A61B8")
        self.assertEqual(_build_cql(cpc="A61B"), "cpc=A61B")

    def test_several_codes_or_ed_in_parentheses(self):
        self.assertEqual(
            _build_cql(keywords="doppler angle", cpc=["A61B8/06", "G01S15/8984", "A61B8"], date_from="2000"),
            'ta all "doppler angle" and (cpc=A61B8/06/low or cpc=G01S15/8984/low or cpc=A61B8)'
            " and pd>=20000101",
        )

    def test_never_emits_cpc_any_with_low(self):
        # OPS 400s on `cpc any "X/low …"` (CLIENT.InvalidClassificationRelation).
        cql = _build_cql(cpc=["A61B8/06", "G01S15/8984"])
        self.assertNotIn("any", cql)

    def test_duplicate_codes_collapsed(self):
        self.assertEqual(_build_cql(cpc=["A61B8/06", "a61b 8/06"]), "cpc=A61B8/06/low")

    def test_codes_capped(self):
        codes = [f"A61B8/{n:02d}" for n in range(2, 30, 2)]  # 14 codes
        self.assertEqual(_build_cql(cpc=codes).count("cpc="), 10)

    def test_invalid_cpc_dropped(self):
        self.assertEqual(_build_cql(keywords="x", cpc="not a cpc"), 'ta="x"')

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
            'ta all "vector Doppler flow" and in="Tortoli" and pd<=20101231',
        )

    def test_empty_returns_blank(self):
        self.assertEqual(_build_cql(), "")

    def test_injection_stripped(self):
        # Quotes / '=' / parens can't escape the clause.
        cql = _build_cql(keywords='foo" or pa="bar')
        self.assertEqual(cql, 'ta all "foo or pa bar"')
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
    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_403_robot_detected_backs_off_then_asks_to_wait(self, mock_get, sleep, invalidate):
        """EPO's fair-use detector is transient: retry with backoff, no token refresh,
        and a final message that says to wait rather than 'will not resolve'."""
        mock_get.return_value = _mock_http_error(
            403, text=_fault_xml("CLIENT.RobotDetected", "Recent behaviour implies you are a robot")
        )
        with self.assertLogs("llm.tools.epo_ops", level="INFO") as cm:
            data = _ops_request("published-data/search/biblio", {"q": "x"}, tool_name="patent_epoops_search")
        self.assertEqual(mock_get.call_count, 4)
        invalidate.assert_not_called()
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5.0, 15.0, 30.0])
        self.assertIn("Wait a few minutes", data["error"])
        self.assertNotIn("will not resolve", data["error"])
        warnings = [r for r in cm.records if r.levelno >= logging.WARNING]
        self.assertEqual(len(warnings), 1)

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
        self.assertIn('ta="widget"', params["q"])

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_invalid_cpc_rejected_without_calling_ops(self, mock_get):
        result = PatentEpoOpsSearchTool().invoke({"keywords": "pile", "cpc": "E02D5 30 OR E02D5/24"})
        self.assertIn("not CPC symbols: 'E02D5 30'", result)
        mock_get.assert_not_called()

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_sends_fixed_cql(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        PatentEpoOpsSearchTool().invoke(
            {"keywords": "blood flow vessel", "cpc": "G01S15/8984", "date_from": "2000"}
        )
        self.assertEqual(
            mock_get.call_args.kwargs["params"]["q"],
            'ta all "blood flow vessel" and cpc=G01S15/8984/low and pd>=20000101',
        )

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_accepts_cpc_as_string_or_list(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        expected = "(cpc=A61B8/06/low or cpc=G01S15/8984/low)"
        for cpc in ("A61B8/06, G01S15/8984", "A61B8/06 or G01S15/8984", ["A61B8/06", "G01S15/8984"]):
            PatentEpoOpsSearchTool().invoke({"cpc": cpc})
            self.assertEqual(mock_get.call_args.kwargs["params"]["q"], expected, cpc)

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_include_subgroups_false(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        PatentEpoOpsSearchTool().invoke({"cpc": ["A61B8/06"], "include_subgroups": False})
        self.assertEqual(mock_get.call_args.kwargs["params"]["q"], "cpc=A61B8/06")

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_names_every_invalid_code(self, mock_get):
        result = PatentEpoOpsSearchTool().invoke({"cpc": ["A61B8/06", "doppler", "Z99"]})
        self.assertIn("'doppler'", result)
        self.assertIn("'Z99'", result)
        mock_get.assert_not_called()

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_rejects_more_than_ten_codes(self, mock_get):
        codes = [f"A61B8/{n:02d}" for n in range(2, 30, 2)]
        result = PatentEpoOpsSearchTool().invoke({"cpc": codes})
        self.assertIn("at most 10", result)
        mock_get.assert_not_called()

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_hits_show_cpc_line(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        result = PatentEpoOpsSearchTool().invoke({"keywords": "widget"})
        self.assertIn("CPC: A61B8/06, A61B8/4254 (additional: G01S15/8979)", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_get_shows_cpc_line(self, mock_get):
        mock_get.return_value = _mock_ok(GET_FIXTURE)
        result = PatentEpoOpsGetTool().invoke({"publication_number": "EP1000000A1"})
        self.assertIn("CPC: A61B8/06, A61B8/4254 (additional: G01S15/8979)", result)

    def test_search_requires_an_input(self):
        result = PatentEpoOpsSearchTool().invoke({"keywords": ""})
        self.assertIn("provide at least one", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_search_count_capped(self, mock_get):
        mock_get.return_value = _mock_ok(SEARCH_FIXTURE)
        PatentEpoOpsSearchTool().invoke({"keywords": "x", "count": 500})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1-50")

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
        _ops_request(self.PATH, {"q": 'ta="x"'}, tool_name="patent_epoops_search")
        row = self._row()
        self.assertEqual(row.outcome, "ok")
        self.assertEqual(row.http_status, 200)
        self.assertEqual(row.query, 'ta="x"')
        self.assertEqual(row.request_path, self.PATH)
        self.assertEqual(row.attempts, 1)
        self.assertGreater(row.response_bytes, 0)

    @patch("llm.tools.epo_ops.requests.get")
    def test_404_is_no_results(self, mock_get):
        mock_get.return_value = _mock_http_error(404, text=_fault_xml("SERVER.EntityNotFound", "No results found"))
        _ops_request(self.PATH, {"q": 'ta="x"'}, tool_name="patent_epoops_search")
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


# --------------------------------------------------------------------------- #
# CPC codes on documents.
# --------------------------------------------------------------------------- #
class CpcCodesTests(TestCase):
    def test_parts_joined_deduped_and_split(self):
        biblio = _EXCHANGE_DOC["bibliographic-data"]
        self.assertEqual(
            _cpc_codes(biblio),
            {"inventive": ["A61B8/06", "A61B8/4254"], "additional": ["G01S15/8979"]},
        )

    def test_inventive_wins_over_additional(self):
        biblio = {"patent-classifications": {"patent-classification": [
            _cpc_entry("A", "61", "B", "8", "06", "A"),
            _cpc_entry("A", "61", "B", "8", "06", "I"),
        ]}}
        self.assertEqual(_cpc_codes(biblio), {"inventive": ["A61B8/06"], "additional": []})

    def test_non_cpc_schemes_and_incomplete_entries_skipped(self):
        biblio = {"patent-classifications": {"patent-classification": [
            _cpc_entry("A", "61", "B", "8", "06", scheme="UC"),
            {"section": {"$": "A"}, "class": {"$": "61"}},
            "junk",
        ]}}
        self.assertEqual(_cpc_codes(biblio), {"inventive": [], "additional": []})

    def test_missing_block(self):
        self.assertEqual(_cpc_codes({}), {"inventive": [], "additional": []})

    def test_search_line_capped(self):
        cpc = {"inventive": [f"A61B8/{n}" for n in range(10)], "additional": []}
        self.assertIn("… (+4)", _cpc_line(cpc, max_inventive=6))


class SplitCpcTests(TestCase):
    def test_variants(self):
        self.assertEqual(_split_cpc("A61B8/06, G01S15/8984"), ["A61B8/06", "G01S15/8984"])
        self.assertEqual(_split_cpc("A61B8/06; G01S15/8984"), ["A61B8/06", "G01S15/8984"])
        self.assertEqual(_split_cpc("A61B8/06 OR G01S15/8984"), ["A61B8/06", "G01S15/8984"])
        self.assertEqual(_split_cpc("A61B8/06 G01S15/8984"), ["A61B8/06", "G01S15/8984"])
        self.assertEqual(
            _split_cpc(["A61B8/06", "G01S15/8984, H01M"]), ["A61B8/06", "G01S15/8984", "H01M"]
        )

    def test_space_inside_one_symbol_kept(self):
        self.assertEqual(_split_cpc("A61B 8/06"), ["A61B 8/06"])

    def test_empty(self):
        self.assertEqual(_split_cpc(""), [])
        self.assertEqual(_split_cpc(None), [])


# --------------------------------------------------------------------------- #
# Classification tool.
# --------------------------------------------------------------------------- #
def _cpc_item(symbol, level, title_xml, children="", **attrs):
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<cpc:classification-item level="{level}" sort-key="{symbol}" {extra}>'
        f"<cpc:classification-symbol>{symbol}</cpc:classification-symbol>"
        f"<cpc:class-title>{title_xml}</cpc:class-title>{children}"
        "<cpc:meta-data>+</cpc:meta-data></cpc:classification-item>"
    )


def _tp(text):
    return f"<cpc:title-part><cpc:text>{text}</cpc:text></cpc:title-part>"


_A61B_TITLE = (
    "<cpc:title-part><cpc:text>IDENTIFICATION </cpc:text><cpc:explanation><cpc:text>"
    'usefulness limited to only animals <cpc:class-ref scheme="cpc">A61D</cpc:class-ref>'
    "</cpc:text></cpc:explanation></cpc:title-part>"
)
_A61B8_065_TITLE = (
    "<cpc:title-part><cpc:comment><cpc:text>to determine blood output from the heart"
    "</cpc:text></cpc:comment></cpc:title-part>"
)

# Trimmed from the live response to classification/cpc/A61B8/06?ancestors=true&depth=1.
_A61B8_06 = _cpc_item(
    "A61B8/06", 8, _tp("Measuring blood flow"), has_children="true",
    children=_cpc_item("A61B8/065", 9, _A61B8_065_TITLE),
)
_A61B8_00 = _cpc_item(
    "A61B8/00", 7, _tp("Diagnosis using ultrasonic, sonic or infrasonic waves"),
    has_children="true", children=_A61B8_06,
)
_A61B1_00 = _cpc_item(
    "A61B1/00", 6, _tp("Diagnosis"), has_children="true", not_allocatable="true", children=_A61B8_00,
)
_A61B = _cpc_item("A61B", 5, _A61B_TITLE, has_children="true", children=_A61B1_00)
_A61_4 = _cpc_item("A61", 4, _tp("MEDICAL OR VETERINARY SCIENCE"), has_children="true", children=_A61B)
_A61_3 = _cpc_item("A61", 3, _tp("HEALTH") + _tp("AMUSEMENT"), has_children="true", children=_A61_4)
_A = _cpc_item("A", 2, _tp("HUMAN NECESSITIES"), has_children="true", children=_A61_3)
CPC_SCHEME_XML = (
    '<?xml version="1.0" encoding="utf-8" standalone="yes"?>'
    "<?xml-stylesheet type='text/xsl' href='../../../../style/cpc.xsl' ?>"
    '<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns:cpc="http://www.epo.org/cpcexport">'
    '<ops:classification-scheme><ops:cpc><cpc:class-scheme scheme-type="cpc">'
    + _A
    + "</cpc:class-scheme></ops:cpc></ops:classification-scheme></ops:world-patent-data>"
)


def _cpc_stat(symbol, score, title):
    return {
        "@classification-symbol": symbol,
        "@percentage": score,
        "cpc:class-title": {"cpc:title-part": {"cpc:text": {"$": title}}},
    }


CPC_SEARCH_FIXTURE = {
    "ops:world-patent-data": {
        "ops:classification-search": {
            "@total-result-count": "2",
            "ops:search-result": {
                "ops:classification-statistics": [
                    _cpc_stat(
                        "Y02A90/00", "0.5243757",
                        "Technologies having an indirect contribution to adaptation to climate change",
                    ),
                    _cpc_stat("A61B8/00", "0.2364201", "Diagnosis using ultrasonic, sonic or infrasonic waves"),
                ]
            },
        }
    }
}


def _mock_xml_ok(text):
    m = MagicMock()
    m.status_code = 200
    m.headers = {}
    m.content = text.encode()
    m.text = text
    m.raise_for_status = MagicMock()
    return m


class CpcLookupSymbolTests(TestCase):
    def test_main_group_gets_00(self):
        self.assertEqual(_cpc_lookup_symbol("A61B8"), "A61B8/00")
        self.assertEqual(_cpc_lookup_symbol("A61B"), "A61B")
        self.assertEqual(_cpc_lookup_symbol("a61b 8/06"), "A61B8/06")
        self.assertEqual(_cpc_lookup_symbol("doppler"), "")


class ParseCpcSchemeTests(TestCase):
    def test_path_entry_children(self):
        parsed = _parse_cpc_scheme(CPC_SCHEME_XML, "A61B8/06")
        self.assertEqual(parsed["entry"]["symbol"], "A61B8/06")
        self.assertEqual(parsed["entry"]["title"], "Measuring blood flow")
        self.assertTrue(parsed["entry"]["has_children"])
        self.assertEqual(
            [p["symbol"] for p in parsed["path"]], ["A", "A61", "A61B", "A61B1/00", "A61B8/00"]
        )
        # The two A61 levels fold into one step.
        self.assertEqual(parsed["path"][1]["title"], "HEALTH; AMUSEMENT; MEDICAL OR VETERINARY SCIENCE")
        self.assertEqual(
            parsed["path"][2]["title"], "IDENTIFICATION (usefulness limited to only animals A61D)"
        )
        self.assertTrue(parsed["path"][3]["not_allocatable"])
        # Subgroup titles often sit in cpc:comment.
        self.assertEqual(
            parsed["children"],
            [{"symbol": "A61B8/065", "title": "to determine blood output from the heart",
              "has_children": False, "not_allocatable": False}],
        )

    def test_symbol_absent(self):
        self.assertIsNone(_parse_cpc_scheme(CPC_SCHEME_XML, "A61B8/999"))

    def test_garbage(self):
        self.assertIsNone(_parse_cpc_scheme("not xml <", "A61B8/06"))
        self.assertIsNone(_parse_cpc_scheme("", "A61B8/06"))

    def test_entities_not_expanded(self):
        evil = (
            '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
            '<x xmlns:cpc="http://www.epo.org/cpcexport"><cpc:classification-item>'
            "<cpc:classification-symbol>A61B8/06</cpc:classification-symbol>"
            "<cpc:class-title><cpc:title-part><cpc:text>&e;</cpc:text></cpc:title-part></cpc:class-title>"
            "</cpc:classification-item></x>"
        )
        parsed = _parse_cpc_scheme(evil, "A61B8/06")
        self.assertNotIn("root:", (parsed or {}).get("entry", {}).get("title", ""))


@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_DUMMY_CACHE)
class PatentClassificationToolTests(TestCase):
    def setUp(self):
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    def test_metadata(self):
        tool = PatentEpoOpsClassificationTool()
        self.assertEqual(tool.name, "patent_epoops_classification")
        self.assertEqual(tool.section, "skills")
        self.assertEqual(tool.audience, "shared")
        self.assertEqual(tool.start_label, "Looking up patent classification...")
        self.assertEqual(tool.end_label, "Looked up patent classification")

    def test_exactly_one_mode(self):
        tool = PatentEpoOpsClassificationTool()
        self.assertIn("exactly one", tool.invoke({}))
        self.assertIn("exactly one", tool.invoke({"query": "x", "symbol": "A61B8"}))

    @patch("llm.tools.epo_ops.requests.get")
    def test_symbol_mode(self, mock_get):
        mock_get.return_value = _mock_xml_ok(CPC_SCHEME_XML)
        result = PatentEpoOpsClassificationTool().invoke({"symbol": "a61b8/06"})
        self.assertIn("CPC A61B8/06 — Measuring blood flow [has narrower subgroups]", result)
        self.assertIn("- A61B8/00 — Diagnosis using ultrasonic", result)
        self.assertIn("- A61B1/00 — Diagnosis", result)
        self.assertIn("- A61B8/065 — to determine blood output from the heart", result)
        self.assertIn("classification/cpc/A61B8/06", mock_get.call_args.args[0])
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Accept"], "application/cpc+xml")
        self.assertEqual(mock_get.call_args.kwargs["params"], {"ancestors": "true", "depth": "1"})

    @patch("llm.tools.epo_ops.requests.get")
    def test_main_group_requested_with_00(self, mock_get):
        mock_get.return_value = _mock_xml_ok(CPC_SCHEME_XML.replace("A61B8/06", "A61B8/0X"))
        PatentEpoOpsClassificationTool().invoke({"symbol": "A61B8"})
        self.assertTrue(mock_get.call_args.args[0].endswith("classification/cpc/A61B8/00"))

    def test_invalid_symbol(self):
        result = PatentEpoOpsClassificationTool().invoke({"symbol": "doppler"})
        self.assertIn("not a CPC symbol", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_unknown_symbol_404(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        result = PatentEpoOpsClassificationTool().invoke({"symbol": "A61B8/999"})
        self.assertIn("No CPC entry for A61B8/999", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_symbol_missing_from_200_response(self, mock_get):
        mock_get.return_value = _mock_xml_ok("<x/>")
        result = PatentEpoOpsClassificationTool().invoke({"symbol": "A61B8/999"})
        self.assertIn("No CPC entry for A61B8/999", result)

    @patch("llm.tools.epo_ops.requests.get")
    def test_query_mode(self, mock_get):
        mock_get.return_value = _mock_ok(CPC_SEARCH_FIXTURE)
        result = PatentEpoOpsClassificationTool().invoke({"query": "doppler angle correction", "count": 50})
        self.assertIn("- Y02A90/00 — Technologies having an indirect contribution", result)
        self.assertIn(
            "- A61B8/00 — Diagnosis using ultrasonic, sonic or infrasonic waves (score 0.24)", result
        )
        self.assertIn("candidates to verify", result)
        self.assertIn("classification/cpc/search", mock_get.call_args.args[0])
        self.assertEqual(
            mock_get.call_args.kwargs["params"], {"q": "doppler angle correction", "Range": "1-20"}
        )

    @patch("llm.tools.epo_ops.requests.get")
    def test_query_no_results(self, mock_get):
        mock_get.return_value = _mock_ok({"ops:world-patent-data": {"ops:classification-search": {}}})
        result = PatentEpoOpsClassificationTool().invoke({"query": "zzqxqzz"})
        self.assertIn("No CPC groups found", result)

    @patch("llm.tools.epo_ops.time.sleep")
    @patch("llm.tools.epo_ops.requests.get")
    def test_ops_error_passthrough_and_logged(self, mock_get, _sleep):
        from llm.models import OpsUsageLog

        mock_get.return_value = _mock_http_error(503)
        result = PatentEpoOpsClassificationTool().invoke({"query": "doppler"})
        self.assertIn("Classification lookup error", result)
        row = OpsUsageLog.objects.get(tool_name="patent_epoops_classification")
        self.assertEqual(row.outcome, "error")
        self.assertEqual(row.request_path, "classification/cpc/search")


# --------------------------------------------------------------------------- #
# Search-table (concept) searching, counts, paging, families, seen, citations.
# --------------------------------------------------------------------------- #
_K1 = ["hydrochlor?thiazid*", "HCTZ"]
_K2 = ["bilayer*", "bi layer*", "multilayer*", "multi layer*"]
_C1 = ["A61K31/549"]
_C2 = ["A61K9/209"]
_K1_CQL = '(ta="hydrochlor?thiazid*" or ta="HCTZ")'
_K2_CQL = '(ta="bilayer*" or ta="bi layer*" or ta="multilayer*" or ta="multi layer*")'


class ConceptCqlTests(TestCase):
    """The paper's strategies A-D (Marttin & Derrien, Table 4) as concepts —
    the exact CQL that reproduced 33/8/27/11 (union 46) hits live."""

    def test_paper_strategies(self):
        cases = {
            "A": ([{"cpc": _C1}, {"cpc": _C2}], "cpc=A61K31/549/low and cpc=A61K9/209/low"),
            "B": ([{"cpc": _C1}, {"keywords": _K2}], f"cpc=A61K31/549/low and {_K2_CQL}"),
            "C": ([{"keywords": _K1}, {"cpc": _C2}], f"{_K1_CQL} and cpc=A61K9/209/low"),
            "D": ([{"keywords": _K1}, {"keywords": _K2}], f"{_K1_CQL} and {_K2_CQL}"),
        }
        for label, (concepts, expected) in cases.items():
            self.assertEqual(_build_cql(concepts=concepts), expected, label)

    def test_union_ors_keywords_and_codes_inside_a_concept(self):
        cql = _build_cql(concepts=[{"keywords": _K1, "cpc": _C1}, {"keywords": _K2, "cpc": _C2}])
        self.assertEqual(
            cql,
            '(ta="hydrochlor?thiazid*" or ta="HCTZ" or cpc=A61K31/549/low) and '
            '(ta="bilayer*" or ta="bi layer*" or ta="multilayer*" or ta="multi layer*" or cpc=A61K9/209/low)',
        )

    def test_full_text_field(self):
        self.assertEqual(
            _build_cql(concepts=[{"keywords": ["bilayer*"]}], keyword_field="full_text"), 'txt="bilayer*"'
        )
        self.assertEqual(_build_cql(keywords="a b", keyword_field="full_text"), 'txt all "a b"')

    def test_concepts_and_with_filters(self):
        cql = _build_cql(
            concepts=[{"keywords": ["tablet*"]}], applicant="Acme", date_to="2003", cites="EP1000000A1"
        )
        self.assertEqual(cql, 'ta="tablet*" and pa="Acme" and ct=EP1000000 and pd<=20031231')

    def test_keyword_injection_and_reserved_words_neutralised(self):
        self.assertEqual(_build_cql(concepts=[{"keywords": ['x") or (pa="y']}]), 'ta="x or pa y"')
        self.assertEqual(_build_cql(concepts=[{"keywords": ["or"]}]), 'ta="or"')

    def test_short_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 3 letters"):
            _build_cql(concepts=[{"keywords": ["hy*"]}])
        # Left truncation with enough letters is fine.
        self.assertEqual(_build_cql(concepts=[{"keywords": ["*thiazide"]}]), 'ta="*thiazide"')

    def test_long_phrase_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 4 words"):
            _build_cql(concepts=[{"keywords": ["one two three four five"]}])

    def test_over_length_raises_instead_of_truncating(self):
        concepts = [{"keywords": ["x" * 900 + str(i) for i in range(10)]}]
        with self.assertRaisesRegex(ValueError, "too long"):
            _build_cql(concepts=concepts)

    def test_no_cap_on_the_number_of_concepts(self):
        cql = _build_cql(concepts=[{"keywords": [f"word{i}"]} for i in range(7)])
        self.assertEqual(cql.count(" and "), 6)

    def test_field_per_concept_overrides_the_search_default(self):
        cql = _build_cql(
            concepts=[{"keywords": ["tablet*"], "field": "title"}, {"keywords": ["bilayer*"]}],
            keyword_field="abstract",
        )
        self.assertEqual(cql, 'ti="tablet*" and ab="bilayer*"')
        self.assertEqual(_build_cql(keywords="a b", keyword_field="title"), 'ti all "a b"')
        self.assertEqual(_build_cql(concepts=[{"keywords": ["x"], "field": "full_text"}]), 'txt="x"')

    def test_proximity_forms(self):
        cases = {
            "zero NEAR/3 order": "(ta=zero prox/distance<=3 ta=order)",
            "zero NEAR order": "(ta=zero prox/unit=sentence ta=order)",
            "zero NEAR/S order": "(ta=zero prox/unit=sentence ta=order)",
            "zero NEAR/P order": "(ta=zero prox/unit=paragraph ta=order)",
            "print* NEAR/5 tablet*": "(ta=print* prox/distance<=5 ta=tablet*)",
        }
        for raw, expected in cases.items():
            self.assertEqual(_build_cql(concepts=[{"keywords": [raw]}]), expected, raw)
        # OR-ed with its sibling synonyms; follows the concept's field.
        self.assertEqual(
            _build_cql(concepts=[{"keywords": ["zero NEAR/3 order", "zero order"], "field": "title"}]),
            '((ti=zero prox/distance<=3 ti=order) or ti="zero order")',
        )

    def test_proximity_invalid_forms(self):
        for raw, msg in [
            ("zero order NEAR/3 release", "one word on each side"),
            ("zero NEAR", "one word on each side"),
            ("hy* NEAR/2 order", "at least 3 letters"),
            ("zero NEAR/0 order", "at least 1"),
        ]:
            with self.assertRaisesRegex(ValueError, msg, msg=raw):
                _build_cql(concepts=[{"keywords": [raw]}])

    def test_lowercase_near_is_an_ordinary_phrase(self):
        self.assertEqual(_build_cql(concepts=[{"keywords": ["antenna near field"]}]), 'ta="antenna near field"')
        self.assertEqual(_build_cql(concepts=[{"keywords": ["nearly zero"]}]), 'ta="nearly zero"')

    def test_exclude_is_not_ed_off_the_whole_query(self):
        cql = _build_cql(
            concepts=[{"keywords": ["bilayer*"]}, {"cpc": ["A61K9/209"]}],
            exclude={"keywords": ["tablet*"], "cpc": ["A61K31/549"]},
        )
        self.assertEqual(
            cql, '(ta="bilayer*" and cpc=A61K9/209/low) not (ta="tablet*" or cpc=A61K31/549/low)'
        )
        # A single positive clause is not double-parenthesised; exclude has its own field.
        self.assertEqual(
            _build_cql(concepts=[{"keywords": ["bilayer*"]}], exclude={"keywords": ["tablet*"], "field": "title"}),
            'ta="bilayer*" not ti="tablet*"',
        )
        with self.assertRaisesRegex(ValueError, "exclude needs"):
            _build_cql(exclude={"keywords": ["tablet*"]})

    def test_terms_deduplicated(self):
        self.assertEqual(_build_cql(concepts=[{"keywords": ["tablet", " tablet "]}]), 'ta="tablet"')


class CitationNumberTests(TestCase):
    def test_forms(self):
        self.assertEqual(_citation_number("WO 03/059327 A1"), "WO03059327")
        self.assertEqual(_citation_number("WO2003059327A1"), "WO03059327")  # long form 404s up to 2003
        self.assertEqual(_citation_number("WO2019154667A1"), "WO2019154667")
        self.assertEqual(_citation_number("EP1000000A1"), "EP1000000")
        self.assertEqual(_citation_number("not a number!"), "")

    def test_docdb_ref_uses_wo_short_form(self):
        self.assertEqual(_docdb_ref("WO2003059327A1"), ("docdb", "WO.03059327.A1"))
        self.assertEqual(_docdb_ref("WO2019154667A1"), ("docdb", "WO.2019154667.A1"))


def _doc(country, number, kind, family, title="T", date="20200101"):
    return {
        "@country": country,
        "@doc-number": number,
        "@kind": kind,
        "@family-id": family,
        "bibliographic-data": {
            "invention-title": {"@lang": "en", "$": title},
            "publication-reference": {"document-id": {"@document-id-type": "docdb", "date": {"$": date}}},
            "parties": {"applicants": {"applicant": {"applicant-name": {"name": {"$": "ACME"}}}}},
            "patent-classifications": {"patent-classification": [_cpc_entry("A", "61", "K", "9", "209")]},
        },
        "abstract": {"@lang": "en", "p": {"$": "Abstract text. " * 60}},
    }


def _search_page(docs, total):
    return {
        "ops:world-patent-data": {
            "ops:biblio-search": {
                "@total-result-count": str(total),
                "ops:search-result": {"exchange-documents": [{"exchange-document": d} for d in docs]},
            }
        }
    }


FAMILY_PAGE = _search_page(
    [
        _doc("US", "2020000001", "A1", "111", "Bilayer tablet"),
        _doc("EP", "3000001", "A1", "111", "Bilayer tablet"),
        _doc("CN", "100000001", "A", "222", "Other tablet"),
    ],
    total=150,
)

COUNT_FIXTURE = {"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "46"}}}


class GroupFamiliesTests(TestCase):
    def test_ep_represents_family(self):
        groups = _group_families(_parse_search_results(FAMILY_PAGE)["results"])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["rep"]["publication_number"], "EP3000001A1")
        self.assertEqual(groups[0]["others"], ["US2020000001A1"])
        self.assertEqual(groups[1]["rep"]["publication_number"], "CN100000001A")

    def test_missing_family_id_kept_separate(self):
        results = [{"publication_number": "EP1A1", "family_id": ""}, {"publication_number": "EP2A1", "family_id": ""}]
        self.assertEqual(len(_group_families(results)), 2)


@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_LOCMEM_CACHE)
class SearchToolPagingTests(TestCase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    def _tool(self, conversation="conv-1"):
        from llm.types.context import RunContext

        tool = PatentEpoOpsSearchTool()
        tool.context = RunContext.create(user_id=None, conversation_id=conversation)
        return tool

    @patch("llm.tools.epo_ops.requests.get")
    def test_count_only(self, mock_get):
        mock_get.return_value = _mock_ok(COUNT_FIXTURE)
        out = self._tool().invoke({"concepts": [{"cpc": _C1}, {"cpc": _C2}], "count_only": True})
        self.assertEqual(
            out, "46 results (patent families) match. Query: cpc=A61K31/549/low and cpc=A61K9/209/low"
        )
        self.assertTrue(mock_get.call_args.args[0].endswith("published-data/search"))
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1-1")

    @patch("llm.tools.epo_ops.requests.get")
    def test_count_only_404_is_zero(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        out = self._tool().invoke({"keywords": "zzqx", "count_only": True})
        self.assertTrue(out.startswith("0 results (patent families) match."))

    @patch("llm.tools.epo_ops.requests.get")
    def test_publication_membership_check(self, mock_get):
        # Each row shows one family member; pn= finds the family of a known number.
        mock_get.return_value = _mock_ok({"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "1"}}})
        out = self._tool().invoke({
            "concepts": [{"cpc": _C1}, {"cpc": _C2}], "publication": "WO 2003/059327 A1", "count_only": True,
        })
        self.assertEqual(
            out, "1 results (patent families) match. Query: cpc=A61K31/549/low and cpc=A61K9/209/low and pn=WO03059327"
        )

    def test_publication_validated(self):
        self.assertIn("not a publication number (publication)", self._tool().invoke({"publication": "???"}))

    @patch("llm.tools.epo_ops.requests.get")
    def test_offset_and_view_caps(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        tool = self._tool()
        tool.invoke({"keywords": "tablet", "view": "list", "count": 30, "offset": 100})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "101-130")
        tool.invoke({"keywords": "tablet", "view": "list", "count": 500})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1-100")
        tool.invoke({"keywords": "tablet", "view": "abstracts", "count": 500})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1-50")
        tool.invoke({"keywords": "tablet", "view": "list", "count": 50, "offset": 1990})
        self.assertEqual(mock_get.call_args.kwargs["params"]["Range"], "1991-2000")

    def test_offset_past_2000_rejected(self):
        out = self._tool().invoke({"keywords": "tablet", "offset": 2000})
        self.assertIn("first 2000 positions", out)

    @patch("llm.tools.epo_ops.requests.get")
    def test_header_and_family_lines(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        out = self._tool().invoke({"keywords": "tablet", "view": "list", "count": 3})
        self.assertIn("150 results — one per patent family", out)
        self.assertIn("Positions 1-3. Ordered by family, newest families first — NOT by relevance.", out)
        self.assertIn("repeat the search with offset=3", out)
        self.assertIn('Query: ta="tablet"', out)
        self.assertIn("[1] EP3000001A1 (20200101) Bilayer tablet — ACME — CPC: A61K9/209", out)
        self.assertIn("    family: US2020000001A1", out)

    @patch("llm.tools.epo_ops.requests.get")
    def test_abstracts_view_trims_abstract(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        out = self._tool().invoke({"keywords": "tablet", "count": 3})
        self.assertIn("Family: US2020000001A1", out)
        self.assertIn("Abstract text.", out)
        self.assertNotIn("Abstract text. " * 40, out)

    @patch("llm.tools.epo_ops.requests.get")
    def test_seen_marks_and_hide_seen(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        first = self._tool().invoke({"keywords": "tablet", "view": "list", "count": 3})
        self.assertNotIn("[seen]", first)
        # A later sub-agent in the same conversation sees the marks.
        second = self._tool().invoke({"keywords": "other", "view": "list", "count": 3})
        self.assertEqual(second.count("[seen]"), 2)
        hidden = self._tool().invoke({"keywords": "other", "view": "list", "count": 3, "hide_seen": True})
        self.assertIn("2 families already seen in this conversation hidden", hidden)
        self.assertIn("Every family on this page was already seen", hidden)
        # Another conversation starts fresh.
        other = self._tool("conv-2").invoke({"keywords": "other", "view": "list", "count": 3})
        self.assertNotIn("[seen]", other)

    @patch("llm.tools.epo_ops.requests.get")
    def test_seen_cache_failure_tolerated(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        with patch("django.core.cache.cache.get_many", side_effect=Exception("redis down")), \
                patch("django.core.cache.cache.set_many", side_effect=Exception("redis down")):
            out = self._tool().invoke({"keywords": "tablet", "view": "list", "count": 3})
        self.assertIn("EP3000001A1", out)

    @patch("llm.tools.epo_ops.requests.get")
    def test_page_cached_slim(self, mock_get):
        mock_get.return_value = _mock_ok(FAMILY_PAGE)
        tool = self._tool()
        tool.invoke({"keywords": "tablet", "count": 3})
        tool.invoke({"keywords": "tablet", "count": 3})
        self.assertEqual(mock_get.call_count, 1)

    @patch("llm.tools.epo_ops.requests.get")
    def test_404_is_no_results_with_query(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        out = self._tool().invoke({"concepts": [{"keywords": ["zzqx"]}]})
        self.assertEqual(out, 'No matching patents found. Query: ta="zzqx"')

    def test_concept_validation(self):
        tool = self._tool()
        self.assertIn("neither keywords nor CPC", tool.invoke({"concepts": [{"name": "empty"}]}))
        self.assertIn("not CPC symbols: 'tablet'", tool.invoke({"concepts": [{"cpc": ["tablet"]}]}))
        self.assertIn("not CPC symbols: 'x'", tool.invoke({"concepts": [{"keywords": ["a1b"]}], "exclude": {"cpc": ["x"]}}))
        self.assertIn("exclude needs something", tool.invoke({"exclude": {"keywords": ["tablet*"]}}))
        self.assertIn("not a publication number", tool.invoke({"cites": "???"}))
        self.assertIn("provide at least one concept", tool.invoke({}))

    @patch("llm.tools.epo_ops.requests.get")
    def test_exclude_and_field_reach_the_query(self, mock_get):
        mock_get.return_value = _mock_ok(COUNT_FIXTURE)
        out = self._tool().invoke({
            "concepts": [{"keywords": ["bilayer*"], "field": "title"}, {"keywords": ["zero NEAR/3 order"]}],
            "exclude": {"keywords": ["capsule*"]},
            "count_only": True,
        })
        self.assertIn(
            'Query: (ti="bilayer*" and (ta=zero prox/distance<=3 ta=order)) not ta="capsule*"', out
        )

    @patch("llm.tools.epo_ops.requests.get")
    def test_concept_keywords_accept_a_string(self, mock_get):
        mock_get.return_value = _mock_ok(COUNT_FIXTURE)
        out = self._tool().invoke({"concepts": [{"keywords": "bilayer*, multilayer*"}], "count_only": True})
        self.assertIn('(ta="bilayer*" or ta="multilayer*")', out)


_CITED_BIBLIO = {
    "references-cited": {
        "citation": [
            {"@cited-phase": "undefined", "@cited-by": "applicant",
             "patcit": {"document-id": [{"@document-id-type": "docdb", "country": {"$": "EP"},
                                         "doc-number": {"$": "0502314"}, "kind": {"$": "A1"}}]}},
            {"@cited-phase": "undefined", "@cited-by": "applicant",
             "nplcit": {"text": {"$": "- LACOURSIERE ET AL., CAN J CARDIOL, vol. 16, 2000"}}},
            {"@cited-phase": "international-search-report", "@cited-by": "examiner",
             "patcit": {"document-id": [{"@document-id-type": "docdb", "country": {"$": "WO"},
                                         "doc-number": {"$": "0027397"}, "kind": {"$": "A1"}}]},
             "category": [{"$": "X"}, {"$": "Y"}], "rel-claims": [{"$": "1,4-9,14"}, {"$": "12"}]},
        ]
    }
}


class CitationParsingTests(TestCase):
    def test_cited_references(self):
        refs = _cited_references(_CITED_BIBLIO)
        self.assertEqual(refs[0]["number"], "EP0502314A1")
        self.assertEqual(refs[1]["npl"], "LACOURSIERE ET AL., CAN J CARDIOL, vol. 16, 2000")
        self.assertEqual(refs[2]["category"], "X/Y")
        self.assertEqual(refs[2]["claims"], "1,4-9,14, 12")

    def test_lines_examiner_first(self):
        lines = _citation_lines(_cited_references(_CITED_BIBLIO))
        self.assertTrue(lines[1].startswith("- WO0027397A1 (examiner, international search report, category X/Y"))
        self.assertEqual(lines[2], "- EP0502314A1 (applicant)")
        self.assertIn("Non-patent literature cited: 1", lines)

    def test_cap(self):
        refs = [{"number": f"EP{i}A1", "npl": "", "by": "applicant", "phase": "", "category": "", "claims": ""}
                for i in range(40)]
        lines = _citation_lines(refs)
        self.assertIn("- … +10 more cited patents", lines)

    def test_empty(self):
        self.assertEqual(_cited_references({}), [])
        self.assertEqual(_citation_lines([]), [])

    def test_get_output_includes_citations(self):
        doc = dict(_EXCHANGE_DOC)
        doc["bibliographic-data"] = {**_EXCHANGE_DOC["bibliographic-data"], **_CITED_BIBLIO}
        out = _format_get(
            {"ops:world-patent-data": {"exchange-documents": {"exchange-document": doc}}}, "EP1000000A1", "biblio"
        )
        self.assertIn("Cited references", out)
        self.assertIn("category X/Y", out)


@override_settings(EPO_OPS_KEY="k", EPO_OPS_SECRET="s", CACHES=_DUMMY_CACHE)
class GetClaimsNotFoundTests(TestCase):
    def setUp(self):
        p = patch("llm.tools.epo_ops._ops_rate_limiter")
        p.start()
        self.addCleanup(p.stop)
        p2 = patch("llm.tools.epo_ops._get_access_token", return_value="tok")
        p2.start()
        self.addCleanup(p2.stop)

    @patch("llm.tools.epo_ops.requests.get")
    def test_claims_404_points_to_wo_member(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        out = PatentEpoOpsGetTool().invoke({"publication_number": "EP2252273A1", "parts": "claims"})
        self.assertIn("No claims text at EPO for EP2252273A1", out)
        self.assertIn("patent_epoops_family", out)

    @patch("llm.tools.epo_ops.requests.get")
    def test_biblio_404_unchanged(self, mock_get):
        mock_get.return_value = _mock_http_error(404)
        out = PatentEpoOpsGetTool().invoke({"publication_number": "EP2252273A1"})
        self.assertIn("No matching patent record was found", out)
