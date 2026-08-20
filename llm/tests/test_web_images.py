"""Tests for web_fetch image discovery: candidate extraction, the manifest
(handles / boundary / pagination / concurrency), and the redirect helper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from bs4 import BeautifulSoup
from django.test import TestCase, override_settings

from llm.tests.test_web_fetch import _mock_response
from llm.tools.web_fetch import (
    WebFetchTool,
    _RedirectBlocked,
    _build_image_manifest,
    _extract_image_candidates,
    _pinned_get_following_redirects,
    _strip_hidden_elements,
    _truncate_filename,
)
from llm.types.context import RunContext

_BASE = "https://example.com/article/"

_PAGE = """
<html><head><title>Doc</title></head><body>
  <header><img src="logo.png" class="site-logo" alt="Logo"></header>
  <nav><a href="/other"><img src="thumb.png" alt="Other article"></a></nav>
  <main>
    <figure><img src="diagram.png" alt="A detailed diagram of the process"></figure>
    <p><img src="/media/photo.jpg" alt="A photo"></p>
    <img src="cart.png" class="icon icon-cart" alt="cart">
    <img src="pixel.gif" width="1" height="1" alt="">
    <img src="data:image/png;base64,iVBORw0KGgo=" alt="inline">
    <img src="chart.svg" alt="svg chart">
    <img src="diagram.png" alt="A detailed diagram of the process">
  </main>
</body></html>
"""


def _candidates(html, base=_BASE):
    soup = BeautifulSoup(html, "html.parser")
    _strip_hidden_elements(soup)
    return _extract_image_candidates(soup, base)


class ExtractImageCandidatesTests(TestCase):
    def test_keeps_only_content_images(self):
        cands = _candidates(_PAGE)
        urls = [c["url"] for c in cands]
        self.assertEqual(
            urls,
            [
                "https://example.com/article/diagram.png",  # figure+alt → highest score, first
                "https://example.com/media/photo.jpg",
            ],
        )

    def test_chrome_icons_pixels_svg_data_excluded(self):
        joined = " ".join(c["url"] for c in _candidates(_PAGE))
        for bad in ("logo.png", "thumb.png", "cart.png", "pixel.gif", "chart.svg", "data:"):
            self.assertNotIn(bad, joined)

    def test_dedup_by_url(self):
        # diagram.png appears twice in the fixture but collapses to one entry.
        urls = [c["url"] for c in _candidates(_PAGE)]
        self.assertEqual(urls.count("https://example.com/article/diagram.png"), 1)

    def test_figure_alt_scored_first(self):
        cands = _candidates(_PAGE)
        self.assertEqual(cands[0]["filename"], "diagram.png")

    def test_filename_and_alt_truncated(self):
        long_stem = "x" * 80
        html = (
            f'<main><figure><img src="{long_stem}.png" alt="{"a" * 200}">'
            "</figure></main>"
        )
        cand = _candidates(html)[0]
        self.assertTrue(cand["filename"].endswith(".png"))
        self.assertIn("…", cand["filename"])
        self.assertLessEqual(len(cand["alt"]), 120)

    def test_cap_enforced(self):
        imgs = "".join(f'<figure><img src="i{i}.png" alt="a{i}"></figure>' for i in range(25))
        cands = _candidates(f"<main>{imgs}</main>")
        self.assertLessEqual(len(cands), 10)

    def test_truncate_filename_keeps_extension(self):
        out = _truncate_filename("a" * 100 + ".jpeg")
        self.assertTrue(out.endswith(".jpeg"))
        self.assertIn("…", out)


class BuildImageManifestTests(TestCase):
    def test_monotonic_handles_no_urls(self):
        ctx = RunContext.create()
        images = [
            {"url": "https://x/a.png", "filename": "a.png", "alt": "Alpha"},
            {"url": "https://x/b.png", "filename": "b.png", "alt": ""},
        ]
        lines = _build_image_manifest(images, "https://x/page", ctx)
        text = "\n".join(lines)
        self.assertIn("[img-1] a.png", text)
        self.assertIn("[img-2] b.png", text)
        # Handle registry resolves back to the URLs …
        self.assertEqual(ctx.web_image_manifest["img-1"]["url"], "https://x/a.png")
        self.assertEqual(ctx.web_image_manifest["img-1"]["page_url"], "https://x/page")
        # … but the manifest TEXT never leaks a URL.
        self.assertNotIn("https://x/a.png", text)
        self.assertNotIn("https://x/b.png", text)

    def test_no_context_returns_empty(self):
        self.assertEqual(_build_image_manifest([{"url": "u", "filename": "f", "alt": ""}], "p", None), [])

    def test_concurrent_allocation_unique(self):
        ctx = RunContext.create()

        def alloc(i):
            return ctx.allocate_web_image_handle({"url": f"u{i}", "page_url": "p", "filename": "f", "alt": ""})

        with ThreadPoolExecutor(max_workers=8) as ex:
            handles = list(ex.map(alloc, range(200)))
        self.assertEqual(len(set(handles)), 200)
        self.assertEqual(len(ctx.web_image_manifest), 200)


@override_settings(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}},
    JINA_API_KEY="",
)
@patch("llm.tools.web_fetch._run_web_scan", lambda *a, **k: None)
class WebFetchIncludeImagesTests(TestCase):
    def setUp(self):
        self.tool = WebFetchTool()

    def _long_page(self):
        return (
            "<html><head><title>Doc</title></head><body><main>"
            f"<p>{'lorem ipsum ' * 300}</p>"
            '<figure><img src="diagram.png" alt="A diagram"></figure>'
            '<img src="/media/photo.jpg" alt="Photo">'
            "</main></body></html>"
        )

    @patch("llm.tools.web_fetch._pinned_get")
    def test_manifest_inside_boundary_no_urls(self, mock_get):
        mock_get.return_value = _mock_response(text=self._long_page())
        ctx = RunContext.create(user_id="1", conversation_id="00000000-0000-0000-0000-000000000000")
        self.tool.set_context(ctx)
        result = self.tool.invoke({"url": _BASE, "include_images": True})

        self.assertIn("[img-1]", result)
        self.assertIn("diagram.png", result)
        # No raw image URL leaks into the text.
        self.assertNotIn("https://example.com/article/diagram.png", result)
        # Manifest sits INSIDE the external-content wrapper.
        self.assertLess(result.index("[img-1]"), result.index("=== END EXTERNAL"))
        self.assertTrue(ctx.web_image_manifest)

    @patch("llm.tools.web_fetch._pinned_get")
    def test_default_is_lean(self, mock_get):
        mock_get.return_value = _mock_response(text=self._long_page())
        ctx = RunContext.create(user_id="1", conversation_id="00000000-0000-0000-0000-000000000000")
        self.tool.set_context(ctx)
        result = self.tool.invoke({"url": _BASE})  # include_images defaults False
        self.assertNotIn("Images on this page", result)
        self.assertNotIn("[img-1]", result)
        self.assertEqual(ctx.web_image_manifest, {})

    @patch("llm.tools.web_fetch._pinned_get")
    def test_pagination_allocates_no_new_handles(self, mock_get):
        mock_get.return_value = _mock_response(text=self._long_page())
        ctx = RunContext.create(user_id="1", conversation_id="00000000-0000-0000-0000-000000000000")
        self.tool.set_context(ctx)
        self.tool.invoke({"url": _BASE, "include_images": True, "start_index": 0})
        size_after_first = len(ctx.web_image_manifest)
        self.assertGreater(size_after_first, 0)
        result2 = self.tool.invoke({"url": _BASE, "include_images": True, "start_index": 50})
        self.assertEqual(len(ctx.web_image_manifest), size_after_first)
        self.assertNotIn("Images on this page", result2)


class RedirectHelperTests(TestCase):
    @patch("llm.tools.web_fetch._pinned_get")
    def test_follows_redirect(self, mock_get):
        final = _mock_response(text="ok")
        redirect = _mock_response(is_redirect=True, location="https://cdn.example.com/final")
        mock_get.side_effect = [redirect, final]
        resp, final_url = _pinned_get_following_redirects(
            "https://example.com", headers={}, max_bytes=1000
        )
        self.assertIs(resp, final)
        self.assertEqual(final_url, "https://cdn.example.com/final")

    @patch("llm.tools.web_fetch._pinned_get")
    def test_bad_redirect_scheme_blocked(self, mock_get):
        mock_get.return_value = _mock_response(is_redirect=True, location="ftp://evil/x")
        with self.assertRaises(_RedirectBlocked):
            _pinned_get_following_redirects("https://example.com", headers={}, max_bytes=1000)
