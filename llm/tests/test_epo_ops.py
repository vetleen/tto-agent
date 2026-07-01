"""Tests for the EPO OPS patent tools (llm/tools/epo_ops.py).

Uses mocked HTTP throughout — no live OPS credentials. The OPS JSON fixtures
below follow the assumed OPS v3.2 shape and double as the contract the parsers
target; if a live response differs, fixtures and parsers get corrected together.
"""

from __future__ import annotations

import json
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
    _normalize_pubnumber,
    _ops_request,
    _parse_family,
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


def _mock_http_error(status):
    import requests as req

    m = MagicMock()
    m.status_code = status
    m.headers = {}
    m.text = ""
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
            'txt="battery" and pa="acme" and cpc="H01M"',
        )

    def test_date_range(self):
        cql = _build_cql(keywords="x", date_from="20200101", date_to="20201231")
        self.assertIn('pd within "20200101 20201231"', cql)

    def test_date_year_expanded(self):
        cql = _build_cql(keywords="x", date_from="2020", date_to="2021")
        self.assertIn('pd within "20200101 20211231"', cql)

    def test_single_sided_dates(self):
        self.assertIn('pd within "20200101 30001231"', _build_cql(keywords="x", date_from="2020"))
        self.assertIn('pd within "10000101 20211231"', _build_cql(keywords="x", date_to="2021"))

    def test_empty_returns_blank(self):
        self.assertEqual(_build_cql(), "")

    def test_injection_stripped(self):
        # Quotes / '=' / parens can't escape the clause.
        cql = _build_cql(keywords='foo" or pa="bar')
        self.assertEqual(cql, 'txt="foo or pa bar"')
        self.assertNotIn('="bar"', cql)


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
        self.assertEqual(_docdb_ref("EP1000000"), ("docdb", "EP.1000000."))

    def test_empty(self):
        self.assertIsNone(_docdb_ref(""))


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
