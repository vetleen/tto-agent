"""Tests for core.images.sanitize_raster_image."""

from __future__ import annotations

import io
from unittest.mock import patch

from django.test import TestCase
from PIL import Image

from core.images import sanitize_raster_image


def _img_bytes(fmt="PNG", size=(4, 4), color=(255, 0, 0), mode="RGB"):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class SanitizeRasterImageTests(TestCase):
    def test_valid_png_roundtrips(self):
        out = sanitize_raster_image(_img_bytes("PNG"))
        self.assertIsNotNone(out)
        safe_bytes, ctype = out
        self.assertEqual(ctype, "image/png")
        # Re-decodes as a real PNG.
        with Image.open(io.BytesIO(safe_bytes)) as img:
            self.assertEqual(img.format, "PNG")

    def test_valid_jpeg(self):
        out = sanitize_raster_image(_img_bytes("JPEG"))
        self.assertIsNotNone(out)
        self.assertEqual(out[1], "image/jpeg")

    def test_valid_webp(self):
        out = sanitize_raster_image(_img_bytes("WEBP"))
        self.assertIsNotNone(out)
        self.assertEqual(out[1], "image/webp")

    def test_garbage_bytes_rejected(self):
        self.assertIsNone(sanitize_raster_image(b"this is not an image"))

    def test_html_spoofed_as_image_rejected(self):
        # An attacker serving HTML under Content-Type image/png.
        self.assertIsNone(sanitize_raster_image(b"<html><body>hi</body></html>"))

    def test_svg_rejected(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        self.assertIsNone(sanitize_raster_image(svg))

    def test_empty_rejected(self):
        self.assertIsNone(sanitize_raster_image(b""))

    def test_pixel_bomb_rejected(self):
        # A real (small) image, but with the pixel guard lowered so its
        # dimensions exceed the cap — exercises the header-based bomb guard.
        png = _img_bytes("PNG", size=(50, 50))
        with patch("core.images._MAX_IMAGE_PIXELS", 100):  # 50*50 = 2500 > 100
            self.assertIsNone(sanitize_raster_image(png))

    def test_animated_gif_collapses_to_first_frame(self):
        # Two-frame animated GIF; sanitized output is a single-frame image.
        frames = [Image.new("P", (4, 4), 0), Image.new("P", (4, 4), 1)]
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
        out = sanitize_raster_image(buf.getvalue())
        self.assertIsNotNone(out)
        safe_bytes, ctype = out
        self.assertEqual(ctype, "image/gif")
        with Image.open(io.BytesIO(safe_bytes)) as img:
            self.assertEqual(getattr(img, "n_frames", 1), 1)
