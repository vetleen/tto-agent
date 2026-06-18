"""Detect model degeneration (repetition loops) in streamed chat output.

LLMs occasionally fall into a degeneration loop — emitting the same phrase or
sentence over and over until they hit their output ceiling. It's rare, but when
it happens the result is a wall of repeated text that wastes tokens and looks
broken. This helper lets the chat consumer detect such a loop mid-stream and
stop it early instead of waiting for the user to hit Stop.

It reuses ``collapse_repetitions`` from the meetings transcript pipeline, which
already solves the same problem for transcription degeneration: it collapses
tandem repeats and repeated sentence runs (conservatively — MIN_RUN >= 3) and
has a cheap zlib redundancy pre-gate, so calling it on ordinary prose is nearly
free. We treat output as degenerate when collapsing it removes a large fraction
of the text.
"""

from __future__ import annotations

from meetings.services.transcript_cleanup import collapse_repetitions

# Don't judge short outputs — a few repeated words is normal, and the underlying
# collapse pass no-ops below ~80 chars anyway.
_MIN_CHARS = 1000

# Abort only when collapsing removes more than 40% of the text. Conservative on
# purpose: degeneration loops are rare, so a false positive (aborting a genuine
# response) is worse than occasionally missing one.
_COLLAPSE_RATIO = 0.6


def is_degenerate(text: str) -> bool:
    """Return True if *text* looks like a model repetition loop.

    Pure and side-effect-free so it can be called repeatedly on accumulating
    stream content (the caller should throttle how often it runs).
    """
    if len(text) < _MIN_CHARS:
        return False
    return len(collapse_repetitions(text)) < len(text) * _COLLAPSE_RATIO
