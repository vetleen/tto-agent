"""Server-side end-of-speech detection for the realtime transcription path.

``gpt-live-transcribe`` does not support server VAD: the client decides where
a turn ends and sends ``input_audio_buffer.commit`` itself. ``TurnDetector``
makes that decision from the decoded PCM16 frames the consumer already pipes
through ffmpeg, using a frame-energy (RMS) gate against an adaptive noise
floor. It is pure and synchronous so it can be driven frame by frame from
``OpenAIRealtimeSession.send_pcm`` and tested with synthetic PCM.

Per frame it reports at most one of each: speech started, speech stopped,
commit (the audio since the last commit/clear is a turn worth transcribing)
and clear (the buffered audio is silence; drop it upstream instead of paying
to transcribe it).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# 24 kHz mono PCM16 — the format PcmPipe produces.
PCM_BYTES_PER_SECOND = 24_000 * 2
_BYTES_PER_MS = PCM_BYTES_PER_SECOND / 1000
_MIN_FALLBACK_WINDOW_MS = 1_000


@dataclass(frozen=True)
class TurnDetectorConfig:
    # Trailing silence that closes a turn (the old server_vad silence_duration_ms).
    silence_ms: int = 600
    # Turns with less speech than this are noise (a cough, a click): they are
    # not committed on their own and ride along with the next real turn.
    min_speech_ms: int = 400
    # Past this length any silent frame closes the turn, so a long monologue
    # splits at a breath instead of mid-word at the hard cap.
    soft_max_turn_ms: int = 12_000
    # Past this length the turn closes regardless.
    hard_max_turn_ms: int = 20_000
    # Uncommitted audio older than this with no open turn is cleared upstream,
    # so a quiet room isn't transcribed (and billed) as one long commit.
    idle_window_ms: int = 15_000
    # Full-scale RMS below which a frame is never speech (≈ -54 dBFS; speech
    # recorded at -30 dB still clears it).
    rms_floor: float = 0.002
    # A frame is speech when its RMS is this many times the noise floor.
    rms_speech_ratio: float = 3.0
    # The noise floor is the quietest non-speech frame in this rolling window
    # (or, once a second of audio holds no non-speech frame at all, the
    # quietest frame of any kind, so a loud fan from the first second can't
    # lock every frame as speech). It adapts in both directions.
    noise_window_ms: int = 5_000


@dataclass
class TurnDecision:
    speech_started: bool = False
    speech_stopped: bool = False
    commit: bool = False
    clear: bool = False
    # Set when ``commit``: byte offset (in the session's audio timeline) where
    # the committed turn's speech began, and how many bytes the commit covers.
    turn_start_bytes: int | None = None
    turn_bytes: int = 0


@dataclass
class TurnDetectorStats:
    frames: int = 0
    speech_frames: int = 0
    commits: int = 0
    clears: int = 0


def frame_rms(frame: bytes) -> float:
    """Full-scale RMS (0.0–1.0) of a little-endian PCM16 frame."""
    usable = len(frame) - (len(frame) % 2)
    if usable <= 0:
        return 0.0
    samples = np.frombuffer(frame[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


@dataclass
class TurnDetector:
    config: TurnDetectorConfig = field(default_factory=TurnDetectorConfig)
    stats: TurnDetectorStats = field(default_factory=TurnDetectorStats)

    def __post_init__(self) -> None:
        self._recent_rms: deque[tuple[float, float, bool]] = deque()  # (frame_ms, rms, is_speech)
        self._recent_ms = 0.0
        self._total_bytes = 0
        # Audio sent upstream since the last commit/clear.
        self._buffered_bytes = 0
        # Earliest speech onset since the last commit/clear (blips included,
        # since their audio is part of what the next commit transcribes).
        self._pending_start_bytes: int | None = None
        self._pending_speech_ms = 0.0
        # Current open turn.
        self.in_speech = False
        self._turn_ms = 0.0
        self._turn_speech_ms = 0.0
        self._trailing_silence_ms = 0.0

    @property
    def turn_has_speech(self) -> bool:
        """Whether uncommitted audio contains any speech (an open turn or a blip)."""
        return self._pending_start_bytes is not None

    @property
    def buffered_bytes(self) -> int:
        return self._buffered_bytes

    def feed(self, frame: bytes) -> TurnDecision:
        cfg = self.config
        decision = TurnDecision()
        frame_ms = len(frame) / _BYTES_PER_MS
        if frame_ms <= 0:
            return decision
        offset = self._total_bytes
        self._total_bytes += len(frame)
        self._buffered_bytes += len(frame)
        self.stats.frames += 1

        rms = frame_rms(frame)
        threshold = max(cfg.rms_floor, self.noise_floor * cfg.rms_speech_ratio)
        is_speech = rms >= threshold
        self._remember_rms(frame_ms, rms, is_speech)
        if is_speech:
            self.stats.speech_frames += 1

        if not self.in_speech:
            if is_speech:
                self.in_speech = True
                self._turn_ms = frame_ms
                self._turn_speech_ms = frame_ms
                self._trailing_silence_ms = 0.0
                # Blips before this turn ride along in the commit, but the
                # turn's timestamp is where real speech began.
                if self._pending_start_bytes is None or self._pending_speech_ms < cfg.min_speech_ms:
                    self._pending_start_bytes = offset
                decision.speech_started = True
                return decision
            if self._buffered_bytes / _BYTES_PER_MS >= cfg.idle_window_ms:
                decision.clear = True
                self.stats.clears += 1
                self._reset_pending()
            return decision

        self._turn_ms += frame_ms
        if is_speech:
            self._turn_speech_ms += frame_ms
            self._trailing_silence_ms = 0.0
        else:
            self._trailing_silence_ms += frame_ms
        closes = (
            self._trailing_silence_ms >= cfg.silence_ms
            or (self._turn_ms >= cfg.soft_max_turn_ms and self._trailing_silence_ms > 0)
            or self._turn_ms >= cfg.hard_max_turn_ms
        )
        if not closes:
            return decision
        self.in_speech = False
        decision.speech_stopped = True
        self._pending_speech_ms += self._turn_speech_ms
        if self._pending_speech_ms >= cfg.min_speech_ms:
            self._fill_commit(decision, start=self._pending_start_bytes)
        return decision

    def flush(self) -> TurnDecision:
        """Close whatever is open (on Stop). Commits only if it contains speech."""
        decision = TurnDecision()
        if self.in_speech:
            self.in_speech = False
            decision.speech_stopped = True
            self._pending_speech_ms += self._turn_speech_ms
        if self.turn_has_speech:
            self._fill_commit(decision, start=self._pending_start_bytes)
        return decision

    @property
    def noise_floor(self) -> float:
        """Noise estimate from the frames before the current one."""
        quiet = [rms for _, rms, speech in self._recent_rms if not speech]
        if quiet:
            return min(quiet)
        # A second or more of nothing but "speech" is more likely a loud room
        # than a speaker who never dips: re-base on it. A misread speaker
        # loses nothing — the audio stays buffered and rides along in the
        # next commit — while a misread room would be committed as speech.
        if self._recent_ms >= _MIN_FALLBACK_WINDOW_MS:
            return min(rms for _, rms, _ in self._recent_rms)
        return 0.0

    def _remember_rms(self, frame_ms: float, rms: float, is_speech: bool) -> None:
        self._recent_rms.append((frame_ms, rms, is_speech))
        self._recent_ms += frame_ms
        while len(self._recent_rms) > 1 and self._recent_ms - self._recent_rms[0][0] >= self.config.noise_window_ms:
            old_ms, _, _ = self._recent_rms.popleft()
            self._recent_ms -= old_ms

    def _fill_commit(self, decision: TurnDecision, *, start: int | None) -> None:
        decision.commit = True
        decision.turn_start_bytes = start
        decision.turn_bytes = self._buffered_bytes
        self.stats.commits += 1
        self._reset_pending()

    def _reset_pending(self) -> None:
        self._buffered_bytes = 0
        self._pending_start_bytes = None
        self._pending_speech_ms = 0.0


def config_from_settings() -> TurnDetectorConfig:
    """Build the detector config, letting settings override the tunables."""
    from django.conf import settings

    defaults = TurnDetectorConfig()
    return TurnDetectorConfig(
        silence_ms=int(getattr(settings, "MEETING_REALTIME_TURN_SILENCE_MS", defaults.silence_ms)),
        min_speech_ms=defaults.min_speech_ms,
        soft_max_turn_ms=defaults.soft_max_turn_ms,
        hard_max_turn_ms=int(getattr(settings, "MEETING_REALTIME_TURN_MAX_MS", defaults.hard_max_turn_ms)),
        idle_window_ms=defaults.idle_window_ms,
        rms_floor=float(getattr(settings, "MEETING_REALTIME_TURN_RMS_FLOOR", defaults.rms_floor)),
        rms_speech_ratio=defaults.rms_speech_ratio,
        noise_window_ms=defaults.noise_window_ms,
    )


__all__ = [
    "PCM_BYTES_PER_SECOND",
    "TurnDecision",
    "TurnDetector",
    "TurnDetectorConfig",
    "TurnDetectorStats",
    "config_from_settings",
    "frame_rms",
]
