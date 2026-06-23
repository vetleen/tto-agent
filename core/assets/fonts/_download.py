"""Provenance + downloader for the bundled OFL/Apache substitute fonts.

The PDF export (WeasyPrint) needs the actual font *file* on the server — unlike
DOCX, which only records a font *name* the reader substitutes locally. The
default org font is Calibri, which is proprietary and cannot be shipped, so we
bundle metric-compatible / visually-close fonts under OFL/Apache licenses and
map proprietary names to them in ``core/styles.py`` (FONT_SUBSTITUTES).

All families come from the Google Fonts repository (single reliable host, clear
licensing). Some families ship only as variable fonts upstream; we instance
them to static Regular/Bold/Italic/BoldItalic with fontTools so the bundle is
uniform and doesn't depend on WeasyPrint's variable-font handling.

This script is committed for reproducibility; the resulting .ttf files are
committed alongside it (they must be in the slug at runtime).

Run from the repo root:  python core/assets/fonts/_download.py
"""
from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google/fonts/main"
HERE = Path(__file__).resolve().parent

# family_key -> (dir, token, kind, license)   kind: "static" | "variable"
FAMILIES = {
    "Carlito": ("ofl/carlito", "Carlito", "static", "OFL-1.1"),       # Calibri-metric
    "Caladea": ("ofl/caladea", "Caladea", "static", "OFL-1.1"),       # Cambria-metric
    "Tinos": ("ofl/tinos", "Tinos", "static", "Apache-2.0"),          # Times-metric
    "Cousine": ("ofl/cousine", "Cousine", "static", "Apache-2.0"),    # Courier-metric
    "Arimo": ("ofl/arimo", "Arimo", "variable", "Apache-2.0"),        # Arial-metric
    "Gelasio": ("ofl/gelasio", "Gelasio", "variable", "OFL-1.1"),     # Georgia-metric
    "EBGaramond": ("ofl/ebgaramond", "EBGaramond", "variable", "OFL-1.1"),  # Garamond-visual
}

# style -> (variable upright|italic file, weight to instance at)
VAR_INSTANCES = {
    "Regular": ("upright", 400),
    "Bold": ("upright", 700),
    "Italic": ("italic", 400),
    "BoldItalic": ("italic", 700),
}


def fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "wilfred-font-bundler"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    if data[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf"):
        return data
    return None


def instance(var_bytes: bytes, weight: int) -> bytes:
    from fontTools import ttLib
    from fontTools.varLib.instancer import instantiateVariableFont

    font = ttLib.TTFont(io.BytesIO(var_bytes))
    instantiateVariableFont(font, {"wght": weight}, inplace=True, updateFontNames=True)
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


def main() -> int:
    ok, missing = [], []
    for key, (dir_, token, kind, lic) in FAMILIES.items():
        dest = HERE / key
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "LICENSE.txt").write_text(
            f"{key} is licensed under {lic}.\nSource: {BASE}/{dir_}\n", encoding="utf-8"
        )
        if kind == "static":
            for style in VAR_INSTANCES:
                data = fetch(f"{BASE}/{dir_}/{token}-{style}.ttf")
                if data:
                    (dest / f"{key}-{style}.ttf").write_bytes(data)
                    ok.append(f"{key}-{style} ({len(data)//1024} KB)")
                else:
                    missing.append(f"{key}-{style}")
        else:  # variable: download upright + italic masters, instance to statics
            upright = fetch(f"{BASE}/{dir_}/{token}[wght].ttf")
            italic = fetch(f"{BASE}/{dir_}/{token}-Italic[wght].ttf")
            masters = {"upright": upright, "italic": italic}
            for style, (which, weight) in VAR_INSTANCES.items():
                master = masters.get(which)
                if not master:
                    missing.append(f"{key}-{style}")
                    continue
                try:
                    data = instance(master, weight)
                except Exception as exc:  # noqa: BLE001
                    missing.append(f"{key}-{style} (instancing failed: {exc})")
                    continue
                (dest / f"{key}-{style}.ttf").write_bytes(data)
                ok.append(f"{key}-{style} ({len(data)//1024} KB, instanced @{weight})")

    print("Downloaded:")
    for line in ok:
        print("  +", line)
    if missing:
        print("MISSING:")
        for line in missing:
            print("  -", line)
    print(f"\n{len(ok)} files OK, {len(missing)} missing.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
