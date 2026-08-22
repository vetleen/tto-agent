"""Slide-deck feature package.

An AI-authored PowerPoint feature that runs parallel to the markdown canvas: the
assistant writes/edits a deck as JSON (``SlideSet.content``), the builder turns
that JSON into a real ``.pptx`` via python-pptx, and the worker renders the
``.pptx`` to per-slide PNGs (LibreOffice) for a read-only preview.

Module map:

* ``theme``        — the base theme + the CSS-like cascade resolution (pure).
* ``schema``       — pydantic validation, the canonical serializer, id minting,
                     content hashing (pure).
* ``layouts``      — seed JSON for ``slides_add_slide``.
* ``oxml_pokes``   — the small OOXML surgery python-pptx lacks.
* ``pptx_build``   — ``build_pptx(slide_set) -> bytes`` (JSON -> .pptx).
* ``template_gen`` — generate the branded base template ``wilfred_default.pptx``.
* ``render``       — worker-side ``.pptx`` -> PDF (LibreOffice) -> PNG (pypdfium2).

Nothing here imports Django at module import time except where noted, so the
pure modules (``theme``, ``schema``, ``oxml_pokes``) unit-test on any platform.
"""
