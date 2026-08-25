"""Per-sheet vision pass: describe the sheet and adjudicate its header row
from a rendered top tile.

One structured call per sheet (capped by ``XLSX_MAX_DESCRIBED_SHEETS``) via
the org's ``spreadsheet_description`` model. The image exists for LAYOUT
comprehension only — exact values come from openpyxl; the model is explicitly
told not to transcribe numbers.

**Degrade, do not block**: no vision model, WeasyPrint unavailable (Windows
dev, misconfigured dyno), a render failure, or a bad structured response all
fall back to the heuristic headers with an empty description — the upload
never fails because of this pass.
"""
from __future__ import annotations

import base64
import logging

from pydantic import BaseModel, Field

from documents.services.spreadsheets.headers import HeaderInfo, _labels_from_row
from documents.services.spreadsheets.model import WorkbookModel

logger = logging.getLogger(__name__)


class SheetVisionOutput(BaseModel):
    """Structured adjudication of one rendered sheet tile."""

    description: str = Field(
        description=(
            "2-3 sentences describing what this sheet contains and how it is "
            "laid out (what kind of table, what the columns represent, any "
            "title block or grouping). Do NOT transcribe cell values or "
            "numbers — describe the layout and purpose only."
        )
    )
    header_row: int | None = Field(
        default=None,
        description=(
            "The 1-based Excel row number that holds the column headers, or "
            "null if the sheet has no header row. The row numbers are not "
            "shown in the image; count using the candidate row given in the "
            "prompt as your anchor."
        ),
    )
    transposed: bool = Field(
        default=False,
        description="True when the labels run DOWN the first column and each data record is a column.",
    )
    header_confirmed: bool = Field(
        default=True,
        description="True when the candidate header row named in the prompt is correct.",
    )
    notes: str = Field(
        default="",
        description="Anything structurally unusual worth recording (merged title block, grouped sections, multiple tables).",
    )


def _prompt_for_sheet(sheet, signal: HeaderInfo) -> str:
    candidate = (
        f"row {signal.header_row}" if signal.header_row is not None else "none detected"
    )
    labels = ", ".join(list(signal.labels.values())[:12]) or "(none)"
    return (
        f"This image is the top-left tile of spreadsheet sheet '{sheet.name}' "
        f"({len(sheet.rows)} rows). A heuristic scan (signal: {signal.signal}) "
        f"suggests the column header is {candidate}, with labels: {labels}. "
        "The tile shows the sheet from its first visible row; the suggested "
        "header row is rendered as the shaded top row of the image. "
        "Confirm or correct the header row, note if the sheet is transposed, "
        "and describe the sheet's content and layout. Never transcribe cell "
        "values — exact data is read separately."
    )


def _adjudicated_header(sheet, signal: HeaderInfo, parsed: SheetVisionOutput) -> HeaderInfo:
    """Fold the vision verdict into a new HeaderInfo (or return ``signal``
    unchanged when the model confirmed it)."""
    if parsed.transposed and not signal.transposed:
        return HeaderInfo(
            header_row=None, transposed=True,
            confidence=signal.confidence, signal=signal.signal,
        )
    row = parsed.header_row
    if parsed.header_confirmed or row is None or row == signal.header_row:
        return signal
    if not (1 <= row <= max(sheet.max_data_row, 1)):
        return signal
    return HeaderInfo(
        header_row=row, labels=_labels_from_row(sheet, row),
        transposed=False, confidence=signal.confidence, signal=signal.signal,
    )


def describe_sheets(
    model: WorkbookModel, signal_headers: dict, plans: dict, version, doc
) -> tuple[dict, dict]:
    """Run the vision pass. Returns ``(final_headers, descriptions)`` —
    ``final_headers[idx] is signal_headers[idx]`` whenever no adjudication
    changed anything (the manifest uses identity to mark vision overrides).
    """
    from django.conf import settings

    from chat.pdf_export import weasyprint_available
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import org_id_for_document

    final = dict(signal_headers)
    descriptions: dict = {}

    org_id = org_id_for_document(doc)
    vision_model = resolve_org_feature_model(org_id, "spreadsheet_description")
    if not vision_model:
        logger.info("spreadsheets: no spreadsheet_description model for org; heuristic headers only")
        return final, descriptions
    if not weasyprint_available():
        logger.info("spreadsheets: WeasyPrint unavailable; heuristic headers only")
        return final, descriptions

    from documents.services.spreadsheets.tiles import render_and_store_band

    max_described = getattr(settings, "XLSX_MAX_DESCRIBED_SHEETS", 10)
    described = 0
    for sheet in model.sheets:
        if described >= max_described:
            break
        plan = plans.get(sheet.index)
        if sheet.is_empty or plan is None or not plan.bands:
            continue
        described += 1
        try:
            rendered = render_and_store_band(version, doc, sheet, plan, 0, y_limit=1)
            if not rendered:
                continue
            _, _, png = rendered[0]
            parsed = _call_vision(png, sheet, signal_headers[sheet.index], vision_model, doc)
            if parsed is None:
                continue
            if parsed.description.strip():
                descriptions[sheet.index] = parsed.description.strip()
            adjusted = _adjudicated_header(sheet, signal_headers[sheet.index], parsed)
            if adjusted is not signal_headers[sheet.index]:
                final[sheet.index] = adjusted
        except Exception:
            logger.exception(
                "spreadsheets: vision pass failed for sheet %r (version %s); using heuristics",
                sheet.name, version.id,
            )
    return final, descriptions


def _call_vision(png: bytes, sheet, signal: HeaderInfo, vision_model: str, doc):
    from chat.services import build_image_content_block
    from llm import get_llm_service
    from llm.core.model_factory import detect_provider
    from llm.types import ChatRequest, Message, RunContext

    provider = detect_provider(vision_model)
    content = [
        {"type": "text", "text": _prompt_for_sheet(sheet, signal)},
        build_image_content_block(base64.b64encode(png).decode("ascii"), "image/png", provider),
    ]
    request = ChatRequest(
        messages=[Message(role="user", content=content)],
        model=vision_model,
        stream=False,
        tools=[],
        context=RunContext.create(user_id=doc.uploaded_by_id),
    )
    try:
        parsed, _usage = get_llm_service().run_structured(request, SheetVisionOutput)
        return parsed
    except Exception:
        logger.exception("spreadsheets: structured vision call failed for sheet %r", sheet.name)
        return None
