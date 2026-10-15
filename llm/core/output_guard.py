"""Degenerate-output guard for provider streams.

A model occasionally falls into a repetition loop — the same short run of
characters over and over — and keeps going until the provider's output cap.
Inside a tool-call argument (a ``canvas_edit`` replacement text, a deck JSON)
nothing on the visible-text path sees it, so a 255,000-character ``" | | |"``
runaway streams for ten minutes at full price before the tool rejects it.

``OutputGuard`` watches every fragment of a stream (visible text, thinking and
tool-call argument deltas) and flags two conditions:

* ``repetition`` — once at least ``MIN_CHARS`` have streamed, every
  ``STEP_CHARS`` the last ``WINDOW_CHARS`` are zlib-compressed; a ratio below
  ``MAX_COMPRESSION_RATIO`` means the window is a loop. Calibration (zlib level
  6, 12k window): ``" |"`` loop 0.003, a looped 60-char sentence 0.010, a looped
  200-char sentence 0.018, a looped 400-char paragraph 0.031; ordinary prose
  0.38, a varied markdown table 0.34, a list of CPC codes 0.29, a realistic slide
  deck JSON 0.38 and a deliberately templated one 0.054. The threshold sits
  between the two populations with margin on both sides. (The transcript
  cleaner uses the same gate: ``meetings.services.transcript_cleanup``.)
* ``output_ceiling`` — visible text plus tool arguments exceed
  ``HARD_CEILING_CHARS``, a backstop for a non-repetitive runaway when a
  provider ignores the output cap. One full-cap canvas write
  (``chat.services.CANVAS_MAX_CHARS`` = 225,000 chars) plus JSON escaping stays
  under it; a second one in the same round does not, which is intended. The
  provider output cap (``llm.context_budget._EFFORT_OUTPUT_CAP``, 24k–64k
  tokens) normally cuts a runaway first — this is defence in depth; thinking is
  not counted (bounded by the cap).

Pure Python, no Django; one zlib call per ``STEP_CHARS`` of output.
"""

from __future__ import annotations

import zlib


class OutputGuard:
    MIN_CHARS = 12_000
    STEP_CHARS = 4_000
    WINDOW_CHARS = 12_000
    MAX_COMPRESSION_RATIO = 0.04
    HARD_CEILING_CHARS = 250_000

    REPETITION = "repetition"
    OUTPUT_CEILING = "output_ceiling"

    def __init__(self) -> None:
        self._parts: list[str] = []  # everything fed, for token estimation
        self._tail_parts: list[str] = []
        self._tail_len = 0
        self.total_chars = 0
        self.ceiling_chars = 0
        self._next_check = self.MIN_CHARS
        self.reason: str | None = None

    # -- feeding -----------------------------------------------------------

    def feed(self, text: str, *, counts_toward_ceiling: bool = True) -> str | None:
        """Account for *text* and return the abort reason, if any (sticky)."""
        if self.reason or not text:
            return self.reason
        self._parts.append(text)
        self.total_chars += len(text)
        if counts_toward_ceiling:
            self.ceiling_chars += len(text)
        self._tail_parts.append(text)
        self._tail_len += len(text)
        # Keep the tail bounded without re-slicing a string per token.
        while (
            len(self._tail_parts) > 1
            and self._tail_len - len(self._tail_parts[0]) >= self.WINDOW_CHARS
        ):
            self._tail_len -= len(self._tail_parts.pop(0))

        if self.ceiling_chars > self.HARD_CEILING_CHARS:
            self.reason = self.OUTPUT_CEILING
        elif self.total_chars >= self._next_check:
            while self._next_check <= self.total_chars:
                self._next_check += self.STEP_CHARS
            if self._window_is_repetitive():
                self.reason = self.REPETITION
        return self.reason

    def observe(self, chunk, parsed_events) -> str | None:
        """Feed one provider chunk: its parsed token/thinking events plus the raw
        tool-call argument fragments (``AIMessageChunk.tool_call_chunks``; ``args``
        is None on the name/id-only fragments)."""
        for event_type, data in parsed_events:
            if event_type == "token":
                self.feed(str(data.get("text") or ""))
            elif event_type == "thinking":
                self.feed(str(data.get("text") or ""), counts_toward_ceiling=False)
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if args:
                self.feed(str(args))
        return self.reason

    # -- inspection --------------------------------------------------------

    def text(self) -> str:
        """Everything fed so far (for usage estimation after an abort)."""
        return "".join(self._parts)

    def _window_is_repetitive(self) -> bool:
        window = "".join(self._tail_parts)[-self.WINDOW_CHARS:]
        if len(window) < self.WINDOW_CHARS:
            return False
        raw = window.encode("utf-8", "ignore")
        if not raw:
            return False
        return len(zlib.compress(raw, 6)) / len(raw) < self.MAX_COMPRESSION_RATIO


__all__ = ["OutputGuard"]
