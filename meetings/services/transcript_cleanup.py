"""Collapse degenerate repetition loops *within a single* transcription segment.

Speech-to-text models (Whisper / gpt-4o-transcribe, both the realtime and the
batch/upload paths) occasionally degenerate on low-quality or near-silent audio
and emit the same sentence or short phrase dozens of times inside one
utterance/chunk. Observed in production meeting 331: a single ~8s utterance
expanded to 7,200 chars of ``"Ja, det er en del penger til å gjøre det."``
repeated 40+ times. Our pipeline stores model output verbatim, so without a
cleanup the loop ends up in the saved transcript, the minutes Wilfred drafts
from it, and the data-room export.

Scope is deliberately ONE segment's text at a time. Cross-utterance repetition
(e.g. "Ja." said in 96 separate utterances across a long meeting) is legitimate
and must never be touched — that property falls out of this being called per
segment from ``recompute_meeting_transcript`` and from the per-utterance live
WS push, never across the joined transcript.

Design goals:
  * Conservative — only collapse runs of >= ``MIN_RUN`` consecutive equal units,
    gated behind a cheap redundancy check, so ordinary speech is returned
    untouched. Correctness does not depend on the gate: a non-looping segment
    that slips through the gate is collapsed by nothing and returned essentially
    as-is.
  * Idempotent — ``collapse_repetitions(collapse_repetitions(x))`` equals
    ``collapse_repetitions(x)``. Required because the assembly chokepoint
    re-runs this on every segment after every new segment lands.
  * Non-destructive at the call sites — callers keep the raw text on the segment
    row and only swap in the cleaned text for display / assembly.

This is a different problem from :func:`meetings.services.audio_transcription.stitch_transcripts`,
which merges *inter-chunk* audio overlap; hence a separate helper.
"""
from __future__ import annotations

import re
import zlib

# Minimum number of consecutive normalized-equal units before a run is treated
# as a loop rather than legitimate emphasis ("Nei. Nei."). Three is safe: a
# genuine triple repeat inside a single short utterance is vanishingly rare, and
# the raw text is preserved on the segment row regardless.
MIN_RUN = 3

# Only do real work on segments that are actually redundant. zlib's
# compressed/raw size ratio is a cheap proxy: ordinary prose sits around
# 0.45-0.7, while a phrase looped 40x compresses to a tiny fraction. This is a
# performance gate only — anything above it is returned untouched; correctness
# for texts below it rests on MIN_RUN / the tandem floor, not on this number.
REDUNDANCY_GATE = 0.5

# Texts shorter than this can't host a meaningful loop worth the work.
_MIN_TEXT_CHARS = 80

# Optional fuzzy merge of *near*-equal consecutive units (handles loops where
# each repeat varies slightly). 0.0 disables it (default); set e.g. 0.92 to
# enable. Off by default because exact/normalized matching is the low-risk path.
FUZZY_THRESHOLD = 0.0

# A substring of 4..200 chars repeated 3+ times back-to-back (the unit + 2 more).
# Non-greedy so it locks onto the *shortest* repeating unit; DOTALL so a loop
# spanning newlines still matches.
_TANDEM_RE = re.compile(r"(.{4,200}?)\1{2,}", re.DOTALL)

# Sentence/clause boundary: whitespace after . ! ? … or one-or-more newlines.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

_WS_BLANK_RE = re.compile(r"\n{3,}")
_WS_INLINE_RE = re.compile(r"[^\S\n]{2,}")

# Characters stripped from the ends of a unit before comparison so that
# "Ja, det er bra." and "ja det er bra" compare equal.
_NORM_STRIP = " \t.,!?…- "


def _normalize_unit(unit: str) -> str:
    """Comparison key for a sentence/clause: case-, space- and edge-punctuation-insensitive."""
    return re.sub(r"\s+", " ", unit).strip().lower().strip(_NORM_STRIP)


def _keys_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    if FUZZY_THRESHOLD > 0 and a and b:
        import difflib

        return difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_THRESHOLD
    return False


def _redundant_enough(text: str) -> bool:
    raw = text.encode("utf-8", "ignore")
    if not raw:
        return False
    return (len(zlib.compress(raw, 6)) / len(raw)) <= REDUNDANCY_GATE


def _collapse_tandem(text: str) -> str:
    """Collapse immediately-repeated substrings to a single instance.

    Iterated to a fixpoint because collapsing one tandem can place two formerly
    separated copies next to each other and expose another.
    """
    prev = None
    out = text
    while out != prev:
        prev = out
        out = _TANDEM_RE.sub(lambda m: m.group(1), out)
    return out


def _collapse_sentence_runs(text: str) -> str:
    """Collapse runs of >= MIN_RUN consecutive normalized-equal sentences/clauses.

    Keeps the first (original-cased) occurrence of each collapsed run. Operates
    on a single segment only, so legitimate repetition across separate
    utterances is never in scope here.
    """
    units = _SENTENCE_SPLIT_RE.split(text)
    if len(units) < MIN_RUN:
        return text

    out: list[str] = []
    i, n = 0, len(units)
    while i < n:
        unit = units[i]
        key = _normalize_unit(unit)
        if not key:
            out.append(unit)
            i += 1
            continue
        j = i + 1
        while j < n and _keys_equal(key, _normalize_unit(units[j])):
            j += 1
        if j - i >= MIN_RUN:
            out.append(unit)  # keep only the first occurrence of the run
        else:
            out.extend(units[i:j])
        i = j

    return " ".join(p for p in out if p)


def collapse_repetitions(text: str) -> str:
    """Remove within-segment repetition loops from one transcription segment.

    Returns *text* unchanged when it is empty, short, or not redundant enough to
    plausibly contain a loop. Otherwise collapses tandem repeats and repeated
    sentence runs and tidies whitespace. Pure and idempotent.
    """
    if not text or len(text) < _MIN_TEXT_CHARS:
        return text
    if not _redundant_enough(text):
        return text

    cleaned = _collapse_tandem(text)
    cleaned = _collapse_sentence_runs(cleaned)
    cleaned = _WS_BLANK_RE.sub("\n\n", cleaned)
    cleaned = _WS_INLINE_RE.sub(" ", cleaned)
    return cleaned.strip()
