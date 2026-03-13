"""Tests for email block rendering in canvas export."""

from django.test import TestCase

from chat.services import EMAIL_BLOCK_RE, replace_email_with_html


class EmailBlockRegexTests(TestCase):
    """Test the regex that detects email code blocks."""

    def test_matches_simple_email_block(self):
        text = "```email\nTo: a@b.com\nSubject: Hi\n\nHello\n```"
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 1)

    def test_matches_multiple_blocks(self):
        text = (
            "Text\n\n```email\nTo: a@b.com\nSubject: Hi\n\nHello\n```\n\n"
            "More\n\n```email\nTo: c@d.com\nSubject: Bye\n\nGoodbye\n```"
        )
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 2)

    def test_does_not_match_other_code_blocks(self):
        text = "```python\nprint('hello')\n```"
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 0)

    def test_does_not_match_plain_text(self):
        text = "Just plain text, no code blocks."
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 0)

    def test_does_not_match_mermaid_blocks(self):
        text = "```mermaid\ngraph TD\n    A --> B\n```"
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 0)

    def test_matches_with_extra_whitespace_after_email(self):
        text = "```email  \nTo: a@b.com\nSubject: Hi\n\nBody\n```"
        matches = list(EMAIL_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 1)


class ReplaceEmailWithHtmlTests(TestCase):
    """Test replace_email_with_html output."""

    def test_no_email_blocks_returns_unchanged(self):
        content = "# Hello\n\nSome regular markdown."
        result = replace_email_with_html(content)
        self.assertEqual(result, content)

    def test_replaces_single_block_with_html(self):
        content = (
            "Before\n\n"
            "```email\nTo: alice@example.com\nSubject: Update\n\nHi Alice,\n\nUpdate here.\n```"
            "\n\nAfter"
        )
        result = replace_email_with_html(content)

        self.assertNotIn("```email", result)
        self.assertIn("Before", result)
        self.assertIn("After", result)
        self.assertIn("alice@example.com", result)
        self.assertIn("Update", result)
        self.assertIn("Hi Alice,", result)

    def test_parses_to_header(self):
        content = "```email\nTo: bob@example.com\nSubject: Test\n\nBody\n```"
        result = replace_email_with_html(content)
        self.assertIn("<td", result)
        self.assertIn("To:", result)
        self.assertIn("bob@example.com", result)

    def test_parses_cc_header(self):
        content = "```email\nTo: a@b.com\nCc: c@d.com\nSubject: Test\n\nBody\n```"
        result = replace_email_with_html(content)
        self.assertIn("Cc:", result)
        self.assertIn("c@d.com", result)

    def test_parses_bcc_header(self):
        content = "```email\nTo: a@b.com\nBcc: secret@b.com\nSubject: Test\n\nBody\n```"
        result = replace_email_with_html(content)
        self.assertIn("Bcc:", result)
        self.assertIn("secret@b.com", result)

    def test_parses_subject_header(self):
        content = "```email\nTo: a@b.com\nSubject: Important Matter\n\nBody\n```"
        result = replace_email_with_html(content)
        self.assertIn("Subject:", result)
        self.assertIn("Important Matter", result)

    def test_body_extraction(self):
        content = "```email\nTo: a@b.com\nSubject: Hi\n\nLine one\nLine two\n```"
        result = replace_email_with_html(content)
        self.assertIn("Line one", result)
        self.assertIn("Line two", result)

    def test_html_escapes_content(self):
        content = '```email\nTo: a@b.com\nSubject: <script>alert("x")</script>\n\nBody with <b>html</b>\n```'
        result = replace_email_with_html(content)
        self.assertNotIn("<script>", result)
        self.assertNotIn("<b>html</b>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_replaces_multiple_blocks(self):
        content = (
            "```email\nTo: a@b.com\nSubject: First\n\nBody1\n```\n\n"
            "Middle\n\n"
            "```email\nTo: c@d.com\nSubject: Second\n\nBody2\n```"
        )
        result = replace_email_with_html(content)
        self.assertNotIn("```email", result)
        self.assertIn("Middle", result)
        self.assertIn("Body1", result)
        self.assertIn("Body2", result)

    def test_preserves_surrounding_content(self):
        content = (
            "# Title\n\nParagraph\n\n"
            "```email\nTo: a@b.com\nSubject: Hi\n\nBody\n```"
            "\n\n- List item"
        )
        result = replace_email_with_html(content)
        self.assertIn("# Title", result)
        self.assertIn("Paragraph", result)
        self.assertIn("- List item", result)

    def test_email_only_headers_no_body(self):
        content = "```email\nTo: a@b.com\nSubject: Hi\n```"
        result = replace_email_with_html(content)
        self.assertNotIn("```email", result)
        self.assertIn("a@b.com", result)
        self.assertIn("<table", result)
