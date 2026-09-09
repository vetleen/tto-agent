"""Tests pinning how the chat page loads its JavaScript and reveals its messages.

The chat page used to paint garbled raw markdown for about a second before settling.
Two things caused it: assistant turns ship as raw markdown text that JS rewrites at
DOMContentLoaded, and DOMContentLoaded sat behind ~4.3 MB of parser-blocking scripts
(mermaid alone is 3.3 MB). These tests pin the fixes, which are all template-level and
therefore easy to undo by accident:

* mermaid + dom-to-image are referenced as lazy URLs, never as blocking <script src>
* the scripts that can defer do, and the ones that must not, don't
* server-rendered markdown carries ``md-pending`` and the ``md-gate`` rule that hides it
  is emitted in <head>, ahead of the message markup it applies to

They follow the ``EditorBundleScopingTests`` precedent in
``accounts/tests/test_settings_views.py``. Expected strings are built with ``static()``
rather than hardcoded, so they stay honest if the storage backend ever changes.
"""

from django.contrib.auth import get_user_model
from django.templatetags.static import static
from django.test import TestCase
from django.urls import reverse

from chat.models import ChatMessage, ChatThread

User = get_user_model()


class ChatPageLoadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="perf@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _get(self):
        response = self.client.get(reverse("chat_home"), {"thread": str(self.thread.id)})
        self.assertEqual(response.status_code, 200)
        return response

    # -- Lazy mermaid ----------------------------------------------------------

    def test_mermaid_is_not_a_blocking_script_tag(self):
        """mermaid.min.js is 3.3 MB and only threads containing a ```mermaid fence need
        it. Assert the *tag shape* is absent, not the bare path — the path legitimately
        survives as the MERMAID_SRC constant the lazy loader reads."""
        response = self._get()
        self.assertNotContains(
            response, '<script src="%s"' % static("js/vendor/mermaid.min.js")
        )

    def test_domtoimage_is_not_a_blocking_script_tag(self):
        """dom-to-image exists only to rasterize a rendered mermaid SVG, so it rides the
        same lazy path."""
        response = self._get()
        self.assertNotContains(
            response, '<script src="%s"' % static("js/vendor/dom-to-image-more.min.js")
        )

    def test_mermaid_url_reaches_the_lazy_loader(self):
        """The URL must still be present, and must come from {% static %}.

        Paired with the two tests above, this is what distinguishes "lazily loaded" from
        "accidentally deleted". It also catches someone replacing the {% static %} tag
        with a literal /static/... path, which 404s under the hashed prod storage.
        """
        response = self._get()
        self.assertContains(response, static("js/vendor/mermaid.min.js"))
        self.assertContains(response, "ensureMermaid")

    # -- Deferred / blocking scripts -------------------------------------------

    def test_defer_safe_scripts_are_deferred(self):
        response = self._get()
        for path in ("js/vendor/diff.min.js", "js/slide-theme-editor.js"):
            with self.subTest(path=path):
                self.assertContains(response, '<script src="%s" defer>' % static(path))

    def test_marked_and_purify_stay_blocking(self):
        """The page calls marked.use() at parse time, twice unguarded. Deferring marked
        would throw a ReferenceError there and kill the rest of the script block, turning
        a slow page into a blank one. purify is left blocking for the same class of
        reason — it buys ~8 KB gzipped and renderMarkdown depends on it."""
        response = self._get()
        for path in ("js/vendor/marked.min.js", "js/vendor/purify.min.js"):
            with self.subTest(path=path):
                self.assertContains(response, '<script src="%s"></script>' % static(path))

    def test_editor_bundle_is_loaded_from_the_body_and_not_deferred(self):
        """560 KB of CodeMirror must not block first paint from <head>, but must stay
        synchronous: the canvas mounts window.WilfredEditor at parse time behind a guard
        that fails *silently*, so deferring it would break the canvas with a clean
        console. EditorBundleScopingTests only pins presence, so it catches neither
        mistake."""
        response = self._get()
        html = response.content.decode()
        self.assertNotContains(response, '%s" defer' % static("js/editor.bundle.js"))
        head, _, body = html.partition("</head>")
        self.assertNotIn("js/editor.bundle", head)
        self.assertIn("js/editor.bundle", body)

    # -- Hide-until-hydrated gate ----------------------------------------------

    def test_server_rendered_markdown_is_marked_pending(self):
        ChatMessage.objects.create(thread=self.thread, role="user", content="Hi")
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="## Heading\n\n**bold**",
        )
        response = self._get()
        self.assertContains(response, "markdown-content md-pending")

    def test_hide_gate_css_precedes_the_message_markup(self):
        """The rule has to be in <head>. The page's other <style> blocks sit *after* the
        message list, so a rule placed there would let the raw markdown paint first —
        exactly the bug. This ordering is invisible on a fast machine in a browser, which
        is why it is asserted here."""
        html = self._get().content.decode()
        self.assertLess(
            html.index("md-gate .md-pending"), html.index('id="chat-messages"')
        )

    def test_gate_is_armed_only_by_script(self):
        """Fail-open by construction: the hiding rule is gated on a class that only JS
        adds, so with JS off (or the inline script CSP-blocked) the text stays visible."""
        self.assertContains(self._get(), "classList.add('md-gate')")

    def test_canvas_preview_is_not_gated(self):
        """#canvas-preview is also .markdown-content but is JS-populated and lives
        outside the message list, so gating it would strand it invisible forever."""
        html = self._get().content.decode()
        preview = html[html.index('id="canvas-preview"'):][:400]
        self.assertNotIn("md-pending", preview)

    def test_streaming_bubble_is_not_gated(self):
        """Live-streamed bubbles are built by createAssistantBubble() with
        .markdown-content. On a thread with a pending turn the socket can deliver one
        before DOMContentLoaded, so a container-wide rule would render it invisible."""
        html = self._get().content.decode()
        self.assertIn("message-content markdown-content", html)
        bubble = html[html.index("message-content markdown-content"):][:120]
        self.assertNotIn("md-pending", bubble)
