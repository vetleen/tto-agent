"""Markdown → .docx helpers for canvas export, adding `==text==` highlighting.

`html2docx` and Python-Markdown don't understand the `==highlight==` convention
out of the box. ``MarkExtension`` turns ``==text==`` into ``<mark>text</mark>``
during the markdown → HTML step, and ``_HighlightHTML2Docx`` teaches html2docx
to render that ``<mark>`` as a yellow Word highlight. ``html2docx_with_highlight``
is a drop-in for ``html2docx.html2docx`` that wires the two together.
"""

from __future__ import annotations

import xml.etree.ElementTree as etree
from io import BytesIO
from typing import List, Optional, Tuple

from docx.enum.text import WD_COLOR_INDEX
from html2docx import HTML2Docx
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor

# Non-greedy, single-line (Python `.` excludes newline): `==a== ==b==` stays two
# spans. Code spans/blocks are already stashed by Markdown before inline
# processing runs, so `==` inside code is left literal.
_MARK_RE = r"==(.+?)=="


class _MarkInline(InlineProcessor):
    def handleMatch(self, m, data):
        el = etree.Element("mark")
        el.text = m.group(1)  # child text is re-processed, so ==**bold**== nests
        return el, m.start(0), m.end(0)


class MarkExtension(Extension):
    """Render ``==text==`` as ``<mark>text</mark>``."""

    def extendMarkdown(self, md):
        # Priority 75 keeps us below the backtick/code patterns (≈190) so code
        # spans win and their `==` is never re-interpreted.
        md.inlinePatterns.register(_MarkInline(_MARK_RE, md), "mark", 75)


class _HighlightHTML2Docx(HTML2Docx):
    """html2docx that maps ``<mark>`` to a yellow run highlight."""

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "mark":
            self.init_run([("highlight_color", WD_COLOR_INDEX.YELLOW)])
        else:
            super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "mark":
            self.finish_run()
        else:
            super().handle_endtag(tag)


def html2docx_with_highlight(content: str, title: str) -> BytesIO:
    """Drop-in for ``html2docx.html2docx`` that also renders ``<mark>`` highlights."""
    parser = _HighlightHTML2Docx(title)
    parser.feed(content.strip())

    buf = BytesIO()
    parser.doc.save(buf)
    return buf
