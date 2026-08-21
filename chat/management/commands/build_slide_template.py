"""Regenerate the branded slide-deck base template.

Writes ``chat/slides/assets/templates/wilfred_default.pptx`` from the current
:data:`chat.slides.theme.WILFRED_BASE_THEME`. The output is committed to the
repo (package data read at build time by python-pptx) — run this after editing
the theme's colours or fonts, then commit the regenerated ``.pptx``.
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from chat.slides.template_gen import build_template


class Command(BaseCommand):
    help = "Regenerate the branded slide-deck template (wilfred_default.pptx)."

    def handle(self, *args, **options):
        out_dir = Path(__file__).resolve().parents[2] / "slides" / "assets" / "templates"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wilfred_default.pptx"
        data = build_template()
        out_path.write_bytes(data)
        self.stdout.write(self.style.SUCCESS(f"Wrote {out_path} ({len(data):,} bytes)"))
