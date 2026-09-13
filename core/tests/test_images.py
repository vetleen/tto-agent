"""Tests for core.images.sanitize_raster_image."""

from __future__ import annotations

import io
from unittest.mock import patch

from django.test import TestCase, override_settings
from PIL import Image

from core.images import optimize_for_vision, sanitize_raster_image


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


def _noisy(fmt="PNG", size=(200, 120), mode="RGB"):
    """An image with varied pixels so re-encoding actually changes size."""
    import random

    random.seed(1)
    img = Image.new(mode, size)
    px = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            if mode == "RGBA":
                px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 200)
            else:
                px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@override_settings(
    VISION_IMAGE_MAX_EDGE=50, VISION_IMAGE_MAX_PIXELS=100_000, VISION_IMAGE_JPEG_QUALITY=82
)
class OptimizeForVisionTests(TestCase):
    def _dims(self, data):
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height

    def test_downscales_to_edge_cap(self):
        out = optimize_for_vision(_noisy(size=(200, 120)))
        self.assertIsNotNone(out)
        data, _media = out
        w, h = self._dims(data)
        self.assertLessEqual(max(w, h), 50)

    def test_opaque_transcodes_to_jpeg(self):
        out = optimize_for_vision(_noisy("PNG", size=(200, 120)))
        self.assertIsNotNone(out)
        data, media = out
        self.assertEqual(media, "image/jpeg")

    def test_alpha_stays_png(self):
        out = optimize_for_vision(_noisy("PNG", size=(200, 120), mode="RGBA"))
        self.assertIsNotNone(out)
        _data, media = out
        self.assertEqual(media, "image/png")

    def test_keep_format_preserves_png(self):
        out = optimize_for_vision(_noisy("PNG", size=(200, 120)), allow_transcode=False)
        self.assertIsNotNone(out)
        data, media = out
        self.assertEqual(media, "image/png")
        with Image.open(io.BytesIO(data)) as img:
            self.assertEqual(img.format, "PNG")

    def test_never_inflates_tiny_image(self):
        raw = _img_bytes("PNG", size=(4, 4))
        out = optimize_for_vision(raw)
        self.assertIsNotNone(out)
        data, _media = out
        self.assertLessEqual(len(data), len(raw))

    def test_no_upscale(self):
        raw = _noisy("PNG", size=(20, 12))  # well under the 50px cap
        out = optimize_for_vision(raw)
        self.assertIsNotNone(out)
        w, h = self._dims(out[0])
        self.assertEqual((w, h), (20, 12))

    def test_exif_orientation_applied(self):
        img = Image.new("RGB", (40, 20), (10, 20, 30))
        exif = img.getexif()
        exif[274] = 6  # Orientation: rotate 90° → dims swap to (20, 40)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        out = optimize_for_vision(buf.getvalue(), allow_transcode=False)
        self.assertIsNotNone(out)
        w, h = self._dims(out[0])
        self.assertEqual((w, h), (20, 40))

    def test_pixel_bomb_rejected(self):
        png = _img_bytes("PNG", size=(50, 50))
        with patch("core.images._MAX_IMAGE_PIXELS", 100):
            self.assertIsNone(optimize_for_vision(png))

    def test_garbage_rejected(self):
        self.assertIsNone(optimize_for_vision(b"not an image"))
