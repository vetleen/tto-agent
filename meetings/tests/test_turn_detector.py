"""Tests for the server-side end-of-speech detector (synthetic PCM)."""
from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase, override_settings

from meetings.services.turn_detector import (
    PCM_BYTES_PER_SECOND,
    TurnDetector,
    TurnDetectorConfig,
    config_from_settings,
    frame_rms,
)

FRAME_MS = 200


def _tone_ms(ms: int, amp: float = 0.2) -> bytes:
    n = 24 * ms
    t = np.arange(n) / 24_000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def _silence_ms(ms: int) -> bytes:
    return b"\x00" * (48 * ms)


def _feed(detector: TurnDetector, audio: bytes) -> list[tuple[float, object]]:
    """Feed 200 ms frames; return (seconds at frame end, decision) for non-empty decisions."""
    step = 48 * FRAME_MS
    out = []
    for i in range(0, len(audio), step):
        d = detector.feed(audio[i:i + step])
        if d.speech_started or d.speech_stopped or d.commit or d.clear:
            out.append(((i + step) / PCM_BYTES_PER_SECOND, d))
    return out


class FrameRmsTests(SimpleTestCase):
    def test_silence_is_zero_and_tone_is_scaled(self):
        self.assertEqual(frame_rms(_silence_ms(200)), 0.0)
        self.assertAlmostEqual(frame_rms(_tone_ms(200, amp=0.2)), 0.2 / np.sqrt(2), places=2)

    def test_odd_length_and_empty_frames(self):
        self.assertEqual(frame_rms(b""), 0.0)
        self.assertEqual(frame_rms(b"\x01"), 0.0)


class TurnDetectorTests(SimpleTestCase):
    def test_silence_never_commits_and_clears_after_idle_window(self):
        d = TurnDetector()
        events = _feed(d, _silence_ms(31_000))
        self.assertFalse(any(e.commit or e.speech_started for _, e in events))
        clears = [t for t, e in events if e.clear]
        self.assertEqual(clears, [15.0, 30.0])
        self.assertFalse(d.turn_has_speech)

    def test_speech_then_pause_commits_with_onset_offset(self):
        d = TurnDetector()
        events = _feed(d, _silence_ms(1000) + _tone_ms(2000) + _silence_ms(1000))
        started = [t for t, e in events if e.speech_started]
        commits = [(t, e) for t, e in events if e.commit]
        self.assertEqual(started, [1.2])
        self.assertEqual(len(commits), 1)
        t, decision = commits[0]
        # Closed by the 600 ms pause after 3 s.
        self.assertAlmostEqual(t, 3.6)
        self.assertTrue(decision.speech_stopped)
        self.assertEqual(decision.turn_start_bytes, PCM_BYTES_PER_SECOND)
        # The commit covers everything since the stream started (3.6 s).
        self.assertEqual(decision.turn_bytes, int(3.6 * PCM_BYTES_PER_SECOND))
        self.assertFalse(d.turn_has_speech)

    def test_blip_below_min_speech_is_not_committed_but_rides_along(self):
        d = TurnDetector()
        events = _feed(d, _silence_ms(1000) + _tone_ms(200) + _silence_ms(1000))
        self.assertTrue(any(e.speech_stopped for _, e in events))
        self.assertFalse(any(e.commit for _, e in events))
        self.assertTrue(d.turn_has_speech)
        # The next real turn commits, timestamped at its own onset.
        events = _feed(d, _tone_ms(1000) + _silence_ms(800))
        commits = [e for _, e in events if e.commit]
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].turn_start_bytes, int(2.2 * PCM_BYTES_PER_SECOND))

    def test_soft_max_closes_at_first_short_pause(self):
        d = TurnDetector()
        # 14 s of speech with a single 200 ms gap every 2 s (never 600 ms).
        speech = _silence_ms(200)
        for _ in range(7):
            speech += _tone_ms(1800) + _silence_ms(200)
        events = _feed(d, speech)
        commits = [t for t, e in events if e.commit]
        # The turn opens at 0.2 s; the gap ending at 12.2 s is the first one
        # past the 12 s soft cap.
        self.assertEqual(commits, [12.2])

    def test_hard_max_closes_without_any_pause(self):
        cfg = TurnDetectorConfig(noise_window_ms=60_000)
        d = TurnDetector(config=cfg)
        events = _feed(d, _silence_ms(400) + _tone_ms(25_000))
        commits = [t for t, e in events if e.commit]
        self.assertEqual(commits, [20.4])

    def test_steady_noise_from_the_start_does_not_lock_as_speech(self):
        # A loud room from the first frame reads as speech only until the
        # noise window fills; then it is re-based as the floor, and from then
        # on the room is cleared, not committed.
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 0.01, 24_000 * 40) * 32767).astype("<i2").tobytes()
        d = TurnDetector()
        events = _feed(d, noise)
        commits = [t for t, e in events if e.commit]
        self.assertLessEqual(len(commits), 1)
        self.assertTrue(all(t <= 2.0 for t in commits))
        self.assertTrue(any(e.clear for _, e in events))
        self.assertFalse(d.in_speech)

    def test_speech_from_the_first_frame_is_committed(self):
        # A perfectly steady tone from frame one is re-based as noise after a
        # second, but the turn it opened is still committed from its onset.
        d = TurnDetector()
        commits = [e for _, e in _feed(d, _tone_ms(2000) + _silence_ms(600)) if e.commit]
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].turn_start_bytes, 0)

    def test_speech_over_noise_is_detected(self):
        rng = np.random.default_rng(1)
        n = 24_000 * 6
        noise = rng.normal(0, 0.01, n)
        t = np.arange(n) / 24_000
        speech = np.where((t >= 2) & (t < 4), np.sin(2 * np.pi * 220 * t) * 0.2, 0.0)
        audio = (np.clip(noise + speech, -1, 1) * 32767).astype("<i2").tobytes()
        d = TurnDetector()
        commits = [e for _, e in _feed(d, audio) if e.commit]
        # The room may be read as one short turn before the floor settles
        # (≤ the first second); the speech itself is its own turn.
        self.assertIn(len(commits), (1, 2))
        self.assertEqual(commits[-1].turn_start_bytes, 2 * PCM_BYTES_PER_SECOND)
        if len(commits) == 2:
            self.assertLessEqual(commits[0].turn_bytes, int(1.8 * PCM_BYTES_PER_SECOND))

    def test_flush_commits_open_turn_only_when_it_has_speech(self):
        d = TurnDetector()
        _feed(d, _silence_ms(2000))
        self.assertFalse(d.flush().commit)
        _feed(d, _tone_ms(1000))
        flushed = d.flush()
        self.assertTrue(flushed.commit)
        self.assertTrue(flushed.speech_stopped)
        self.assertFalse(d.turn_has_speech)

    @override_settings(
        MEETING_REALTIME_TURN_SILENCE_MS=900,
        MEETING_REALTIME_TURN_MAX_MS=15000,
        MEETING_REALTIME_TURN_RMS_FLOOR=0.01,
    )
    def test_config_from_settings(self):
        cfg = config_from_settings()
        self.assertEqual(cfg.silence_ms, 900)
        self.assertEqual(cfg.hard_max_turn_ms, 15000)
        self.assertEqual(cfg.rms_floor, 0.01)
