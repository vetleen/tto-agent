"""Tests for web_fetch image discovery: candidate extraction, the manifest
(handles / boundary / pagination / concurrency), and the redirect helper."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests as req_lib
from bs4 import BeautifulSoup
from django.test import TestCase, override_settings

from llm.tests.test_web_fetch import _mock_response


def _jina_resp_with_images(content, images):
    """A mocked Jina JSON response carrying a data.images summary."""
    payload = {"code": 200, "status": 20000,
               "data": {"title": "T", "content": content, "images": images}}
    r = _mock_response(content_type="application/json", text=json.dumps(payload))
    r.json.return_value = payload
    return r
from llm.tools.web_fetch import (
    WebFetchTool,
    _RedirectBlocked,
    _build_image_manifest,
    _extract_image_candidates,
    _images_from_jina,
    _pinned_get,
    _pinned_get_following_redirects,
    _strip_for_images,
    _strip_hidden_elements,
    _truncate_filename,
)
from llm.types.context import RunContext

_BASE = "https://example.com/article/"

_PAGE = """
<html><head><title>Doc</title></head><body>
  <header><img src="logo.png" class="site-logo" alt="Logo"></header>
  <nav><a href="/other"><img src="thumb.png" class="teaser-thumbnail" alt="Other article"></a></nav>
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
    # Mirror the real image path: the lighter clean that keeps chrome/aria-hidden.
    soup = BeautifulSoup(html, "html.parser")
    _strip_for_images(soup)
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

    def test_logo_and_plugin_urls_dropped_by_path(self):
        # With chrome kept, WordPress header/partner logos come in — filter them
        # by the URL (filename 'logo', or a plugin/flag asset dir), keeping the
        # real content image.
        html = """
        <main>
          <img src="/wp-content/uploads/2020/inven2_logo.png" alt="Inven2 logo">
          <img src="/wp-content/plugins/wpml/res/flags/no.png" alt="Norsk">
          <figure><img src="/wp-content/uploads/2026/story-photo.jpg" alt="A photo"></figure>
        </main>
        """
        urls = [c["url"] for c in _candidates(html)]
        self.assertEqual(urls, ["https://example.com/wp-content/uploads/2026/story-photo.jpg"])


class PortalAndAriaHiddenTests(TestCase):
    """The image path keeps semantic chrome and aria-hidden figures (portal/
    homepage content lives there), while the text path still strips both."""

    def test_content_images_in_chrome_and_aria_hidden_survive(self):
        # Mirrors NRK's homepage: teaser images inside nav/aside, each wrapped in
        # an aria-hidden <figure> (decorative but visible).
        html = """
        <html><body>
          <nav><figure aria-hidden="true"><img src="/lead.jpg" alt=""></figure></nav>
          <aside><figure aria-hidden="true"><img src="/side.png" alt=""></figure></aside>
          <header><img src="brand.png" class="site-logo"></header>
        </body></html>
        """
        urls = [c["url"] for c in _candidates(html)]
        self.assertIn("https://example.com/lead.jpg", urls)   # nav + aria-hidden
        self.assertIn("https://example.com/side.png", urls)   # aside + aria-hidden
        self.assertNotIn("https://example.com/brand.png", urls)  # logo class filtered

    def test_visually_hidden_images_still_dropped_on_image_path(self):
        html = """
        <main>
          <div style="display:none"><img src="/invisible.png" alt="x"></div>
          <figure aria-hidden="true"><img src="/visible.png" alt="ok"></figure>
        </main>
        """
        urls = [c["url"] for c in _candidates(html)]
        self.assertIn("https://example.com/visible.png", urls)
        self.assertNotIn("https://example.com/invisible.png", urls)

    def test_text_path_still_strips_aria_hidden(self):
        # aria-hidden hides injection payloads from the reading text — must stay
        # removed on the text path even though the image path keeps it.
        html = (
            "<html><body><main><p>Visible body</p>"
            '<div aria-hidden="true">SECRET-INJECTION</div></main></body></html>'
        )
        soup = BeautifulSoup(html, "html.parser")
        _strip_hidden_elements(soup)
        text = soup.get_text()
        self.assertIn("Visible body", text)
        self.assertNotIn("SECRET-INJECTION", text)


class ImagesFromJinaTests(TestCase):
    """Jina Reader is the dominant fetch path (many sites 400/421 the direct,
    IP-pinned fetch), so image discovery must work from its data.images summary."""

    def test_parses_summary_dict(self):
        data = {"images": {
            "Image 1": "https://x/a.jpg",
            "Image 2: A red car": "https://x/b.png",
            "logo": "https://x/site-logo.png",         # 'logo' in path -> dropped
            "Image 3": "https://x/icon-sprite.svg",    # svg -> dropped
            "Image 4": "data:image/png;base64,AAA",    # data uri -> dropped
            "Image 5": "https://x/a.jpg",              # dup -> collapsed
            "Image 6": "https://x/wp-content/plugins/wpml/res/flags/no.png",  # plugin flag
            "Image 7": "https://x/wp-content/uploads/2020/hero_logo.png",     # logo filename
        }}
        out = _images_from_jina(data)
        self.assertEqual([c["url"] for c in out], ["https://x/a.jpg", "https://x/b.png"])
        self.assertEqual(out[0]["alt"], "")            # generic "Image 1" -> no alt
        self.assertEqual(out[1]["alt"], "A red car")   # "Image 2:" prefix stripped

    def test_non_dict_returns_empty(self):
        self.assertEqual(_images_from_jina({"images": None}), [])
        self.assertEqual(_images_from_jina({}), [])

    def test_cap(self):
        data = {"images": {f"Image {i}": f"https://x/{i}.jpg" for i in range(25)}}
        self.assertLessEqual(len(_images_from_jina(data)), 10)


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

    @override_settings(JINA_API_KEY="test-jina-key")
    @patch("llm.tools.web_fetch.requests.get")
    @patch("llm.tools.web_fetch._pinned_get")
    def test_jina_path_surfaces_images(self, mock_pinned, mock_requests_get):
        # Real sites (NRK/NTNU) 400/421 the direct fetch → Jina wins; the image
        # summary must still reach the manifest.
        mock_pinned.side_effect = req_lib.exceptions.ConnectionError("refused")
        mock_requests_get.return_value = _jina_resp_with_images(
            "Body text.", {"Image 1": "https://cdn/x.jpg", "Image 2: A cat": "https://cdn/cat.png"},
        )
        ctx = RunContext.create(user_id="1", conversation_id="00000000-0000-0000-0000-000000000000")
        self.tool.set_context(ctx)
        result = self.tool.invoke({"url": "https://example.com/p", "include_images": True})

        self.assertIn("[img-1]", result)
        self.assertIn("A cat", result)                 # alt parsed from "Image 2: A cat"
        self.assertNotIn("https://cdn/cat.png", result)  # no URL leak
        self.assertTrue(ctx.web_image_manifest)

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


@override_settings(JINA_API_KEY="test-jina-key")
class AltTextScanTests(TestCase):
    """Alt text is untrusted, page-supplied content: EVERY path that produces
    image candidates must run it through the web-content scan alongside the
    body — direct (covered in test_web_content_adversarial), Jina, and the
    js-rendered fallback that merges direct-extracted candidates."""

    def setUp(self):
        self.tool = WebFetchTool()
        self.ctx = RunContext.create(
            user_id="1", conversation_id="00000000-0000-0000-0000-000000000000",
        )
        self.tool.set_context(self.ctx)

    @patch("guardrails.tasks.scan_web_content_task.delay")
    @patch("llm.tools.web_fetch.requests.get")
    @patch("llm.tools.web_fetch._pinned_get")
    def test_jina_path_alts_scanned(self, mock_pinned, mock_requests_get, mock_scan):
        mock_pinned.side_effect = req_lib.exceptions.ConnectionError("refused")
        mock_requests_get.return_value = _jina_resp_with_images(
            "Jina body text.", {"Image 1: Injected alt content": "https://cdn/inj.jpg"},
        )
        self.tool.invoke({"url": "https://example.com/alt-scan-jina"})

        # .delay(text, user_id, thread_id, source_label) — join all enqueued texts.
        scanned = "\n".join(c.args[0] for c in mock_scan.call_args_list)
        self.assertIn("Jina body text.", scanned)
        self.assertIn("Injected alt content", scanned)

    @patch("guardrails.tasks.scan_web_content_task.delay")
    @patch("llm.tools.web_fetch.requests.get")
    @patch("llm.tools.web_fetch._pinned_get")
    def test_merged_direct_alts_scanned_on_jina_fallback(
        self, mock_pinned, mock_requests_get, mock_scan,
    ):
        # Direct fetch succeeds but looks JS-rendered (big HTML, thin text), so
        # the Jina result wins; Jina has no images, so the direct-extracted
        # candidates are merged in — their alts must still be scanned.
        html = (
            "<html><head><title>T</title></head><body>"
            "<!--" + "pad " * 2000 + "-->"
            "<main><p>Tiny.</p>"
            '<figure><img src="figure-shot.png" alt="Sneaky alt payload"></figure>'
            "</main></body></html>"
        )
        mock_pinned.return_value = _mock_response(text=html)
        mock_requests_get.return_value = _jina_resp_with_images("Jina body text.", {})
        self.tool.invoke({"url": "https://example.com/alt-scan-merge"})

        scanned = "\n".join(c.args[0] for c in mock_scan.call_args_list)
        self.assertIn("Sneaky alt payload", scanned)


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

    @patch("llm.tools.web_fetch._resolve_and_validate", return_value=("1.2.3.4", None))
    @patch("requests.Session.get")
    def test_pinned_get_sends_hostname_host_header(self, mock_get, _mock_resolve):
        # Connecting to an IP literal makes urllib3 send Host:<ip>, which CDNs
        # 404. _pinned_get must send the real hostname as Host.
        mock_get.return_value = _mock_response(text="ok")
        _pinned_get("https://example.com/p", timeout=5, headers={"User-Agent": "x"}, max_bytes=1000)
        sent = mock_get.call_args.kwargs["headers"]
        self.assertEqual(sent["Host"], "example.com")
        self.assertEqual(sent["User-Agent"], "x")  # caller headers preserved
