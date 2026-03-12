"""Tests for mermaid diagram rendering in canvas export."""

import base64
from unittest.mock import patch

from django.test import TestCase

from chat.services import MERMAID_BLOCK_RE, replace_mermaid_with_images


class MermaidRegexTests(TestCase):
    """Test the regex that detects mermaid code blocks."""

    def test_matches_simple_mermaid_block(self):
        text = "```mermaid\ngraph TD\n    A --> B\n```"
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group(1), "graph TD\n    A --> B\n")

    def test_matches_multiple_blocks(self):
        text = (
            "Some text\n\n```mermaid\ngraph TD\n    A --> B\n```\n\n"
            "More text\n\n```mermaid\nsequenceDiagram\n    A->>B: Hello\n```"
        )
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 2)

    def test_does_not_match_other_code_blocks(self):
        text = "```python\nprint('hello')\n```"
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 0)

    def test_does_not_match_plain_text(self):
        text = "This is just plain text with no code blocks."
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 0)

    def test_matches_with_extra_whitespace_after_mermaid(self):
        text = "```mermaid  \ngraph TD\n    A --> B\n```"
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        self.assertEqual(len(matches), 1)


class ReplaceMermaidWithImagesTests(TestCase):
    """Test replace_mermaid_with_images with mocked Playwright."""

    def test_no_mermaid_blocks_returns_unchanged(self):
        content = "# Hello\n\nSome regular markdown content."
        result = replace_mermaid_with_images(content)
        self.assertEqual(result, content)

    @patch("chat.services._render_mermaid_pngs")
    def test_replaces_single_block_with_image(self, mock_render):
        png_data = b"\x89PNG\r\n\x1a\nfake"
        mock_render.return_value = [png_data]

        content = "Before\n\n```mermaid\ngraph TD\n    A --> B\n```\n\nAfter"
        result = replace_mermaid_with_images(content)

        self.assertNotIn("```mermaid", result)
        self.assertIn("Before", result)
        self.assertIn("After", result)
        b64 = base64.b64encode(png_data).decode()
        self.assertIn(f'<img src="data:image/png;base64,{b64}"', result)
        self.assertIn('alt="Diagram"', result)

    @patch("chat.services._render_mermaid_pngs")
    def test_replaces_multiple_blocks(self, mock_render):
        png_data = b"\x89PNG\r\n\x1a\nfake"
        mock_render.return_value = [png_data, png_data]

        content = (
            "```mermaid\ngraph TD\n    A --> B\n```\n\n"
            "Middle text\n\n"
            "```mermaid\nsequenceDiagram\n    A->>B: Hi\n```"
        )
        result = replace_mermaid_with_images(content)

        self.assertNotIn("```mermaid", result)
        self.assertIn("Middle text", result)
        self.assertEqual(result.count("<img"), 2)

    @patch("chat.services._render_mermaid_pngs")
    def test_keeps_block_on_render_failure(self, mock_render):
        mock_render.return_value = [None]  # Render failed

        content = "```mermaid\ngraph TD\n    A --> B\n```"
        result = replace_mermaid_with_images(content)

        # Original block is preserved
        self.assertIn("```mermaid", result)
        self.assertNotIn("<img", result)

    @patch("chat.services._render_mermaid_pngs")
    def test_preserves_surrounding_content(self, mock_render):
        png_data = b"fakepng"
        mock_render.return_value = [png_data]

        content = "# Title\n\nParagraph 1\n\n```mermaid\ngraph TD\n    A --> B\n```\n\nParagraph 2\n\n- List item"
        result = replace_mermaid_with_images(content)

        self.assertIn("# Title", result)
        self.assertIn("Paragraph 1", result)
        self.assertIn("Paragraph 2", result)
        self.assertIn("- List item", result)

    @patch("chat.services._render_mermaid_pngs")
    def test_passes_sources_to_renderer(self, mock_render):
        """Verify all diagram sources are passed in a single batch call."""
        mock_render.return_value = [b"png1", b"png2"]

        content = (
            "```mermaid\ngraph TD\n    A --> B\n```\n\n"
            "```mermaid\nsequenceDiagram\n    X->>Y: Z\n```"
        )
        replace_mermaid_with_images(content)

        mock_render.assert_called_once()
        sources = mock_render.call_args[0][0]
        self.assertEqual(len(sources), 2)
        self.assertIn("graph TD", sources[0])
        self.assertIn("sequenceDiagram", sources[1])
