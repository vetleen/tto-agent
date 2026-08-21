"""Orchestrate a SlideRenderRun on the worker: build -> render -> cache -> notify.

Called by ``chat.tasks.render_deck_task``. Uses the per-slide ``SlideRender``
cache so unchanged slides are not re-rendered — which is why an agent-previewed
deck costs almost nothing at the turn-end user render.
"""

from __future__ import annotations

import logging

from django.db.utils import OperationalError
from django.utils import timezone

from chat.assets import store_slide_set_file, store_slide_set_image
from chat.models import Asset, SlideRender, SlideRenderRun
from chat.slides import schema
from chat.slides.pptx_build import build_pptx
from chat.slides.render import render_pptx

logger = logging.getLogger(__name__)


def execute_render_run(run_id: str) -> None:
    run = SlideRenderRun.objects.select_related("slide_set").filter(pk=run_id).first()
    if run is None:
        return
    if run.status == SlideRenderRun.Status.COMPLETED:
        return  # idempotent (duplicate dispatch)

    SlideRenderRun.objects.filter(pk=run_id).update(
        status=SlideRenderRun.Status.RUNNING, started_at=timezone.now()
    )
    deck = run.slide_set
    try:
        result = _render(run, deck)
    except OperationalError:
        raise  # DB blip -> let Celery autoretry (leaves the row RUNNING)
    except Exception as exc:  # noqa: BLE001 — render failures are terminal, not retried
        logger.warning("slide render run %s failed: %s", run_id, exc, exc_info=True)
        SlideRenderRun.objects.filter(pk=run_id).update(
            status=SlideRenderRun.Status.FAILED, error=str(exc)[:2000], finished_at=timezone.now()
        )
        if run.purpose in (SlideRenderRun.Purpose.USER_PREVIEW, SlideRenderRun.Purpose.PDF_EXPORT):
            notify_render_event(deck, run, "slidedeck.render_failed")
        return

    SlideRenderRun.objects.filter(pk=run_id).update(
        status=SlideRenderRun.Status.COMPLETED, result=result, finished_at=timezone.now()
    )
    if run.purpose == SlideRenderRun.Purpose.USER_PREVIEW:
        notify_render_event(deck, run, "slidedeck.rendered")
    elif run.purpose == SlideRenderRun.Purpose.PDF_EXPORT:
        notify_render_event(deck, run, "slidedeck.pdf_ready")


def _is_dirty(row, current_hash) -> bool:
    return row is None or row.asset_id is None or row.content_hash != current_hash


def _render(run, deck) -> dict:
    content = deck.content or {}
    schema.mint_ids(content)
    slides = content.get("slides") or []
    all_ids = [s.get("id") for s in slides]
    index_of = {sid: i for i, sid in enumerate(all_ids)}
    hashes = {sid: schema.slide_content_hash(content, index_of[sid]) for sid in all_ids}

    is_pdf = run.purpose == SlideRenderRun.Purpose.PDF_EXPORT
    requested = [sid for sid in (run.slide_ids or all_ids) if sid in index_of]

    existing = {r.slide_id: r for r in SlideRender.objects.filter(slide_set=deck)}
    dirty = all_ids if is_pdf else [sid for sid in requested if _is_dirty(existing.get(sid), hashes.get(sid))]

    warnings: list[str] = []
    pdf_bytes = None
    if dirty:
        only = None if is_pdf else dirty
        pptx_bytes, warnings = build_pptx(deck, only_slide_ids=only)
        pdf_bytes, pngs = render_pptx(pptx_bytes)
        built_order = [sid for sid in all_ids if (only is None or sid in set(only))]
        if len(pngs) != len(built_order):
            logger.warning(
                "render produced %d pages for %d built slides (deck %s)",
                len(pngs), len(built_order), deck.pk,
            )
        for sid, (png, w, h) in zip(built_order, pngs):
            asset = store_slide_set_image(deck, img_bytes=png)
            prev = existing.get(sid)
            prev_asset_id = prev.asset_id if prev else None
            SlideRender.objects.update_or_create(
                slide_set=deck, slide_id=sid,
                defaults={"content_hash": hashes.get(sid), "asset": asset, "width": w, "height": h},
            )
            # Free the superseded render asset (blob cleaned by the Asset
            # post_delete signal) unless dedup reused it or another slide shares it.
            if prev_asset_id and prev_asset_id != asset.id:
                if not SlideRender.objects.filter(asset_id=prev_asset_id).exists():
                    Asset.objects.filter(pk=prev_asset_id).delete()

    # Prune render rows (and their assets) for slides removed from the deck.
    _prune_deleted_slide_renders(deck, all_ids)

    renders = {r.slide_id: r for r in SlideRender.objects.filter(slide_set=deck)}
    ordered = all_ids if (is_pdf or not run.slide_ids) else requested
    result_slides = []
    for sid in ordered:
        row = renders.get(sid)
        if row and row.asset_id:
            result_slides.append({
                "slide_id": sid, "asset_id": str(row.asset_id),
                "width": row.width, "height": row.height, "page": index_of[sid] + 1,
            })

    result = {"slides": result_slides, "warnings": warnings}
    if is_pdf and pdf_bytes is not None:
        pdf_asset = store_slide_set_file(deck, file_bytes=pdf_bytes)
        result["pdf_asset_id"] = str(pdf_asset.id)
    return result


def _prune_deleted_slide_renders(deck, live_slide_ids) -> None:
    """Delete SlideRender rows (+ their assets) for slides no longer in the deck."""
    stale = list(
        SlideRender.objects.filter(slide_set=deck)
        .exclude(slide_id__in=live_slide_ids)
    )
    for row in stale:
        asset_id = row.asset_id
        row.delete()
        if asset_id and not SlideRender.objects.filter(asset_id=asset_id).exists():
            Asset.objects.filter(pk=asset_id).delete()


def notify_render_event(deck, run, event: str) -> None:
    """Best-effort WebSocket push to the deck's thread group."""
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer

        layer = get_channel_layer()
        if layer is None:
            return
        async_to_sync(layer.group_send)(
            f"thread_{deck.thread_id}",
            {
                "type": event,
                "deck_id": str(deck.pk),
                "run_id": str(run.pk),
                "purpose": run.purpose,
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not notify consumer of render %s", run.pk, exc_info=True)
