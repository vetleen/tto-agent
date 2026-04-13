"""Low-level ffprobe/ffmpeg subprocess helpers for audio processing.

These helpers never load the full audio file into memory — they rely on
ffprobe metadata reads and ffmpeg streaming extraction.  A 27 MB
compressed file that would expand to ~1.3 GB of raw PCM under pydub
uses only a few MB of ffmpeg internal buffers here.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-call subprocess timeouts (seconds).
FFPROBE_TIMEOUT = 30
FFMPEG_CHUNK_TIMEOUT = 120


def ffmpeg_available() -> bool:
    """Return True if both ffmpeg and ffprobe are on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ffprobe_duration_ms(file_path: Path) -> int | None:
    """Return the audio duration in milliseconds, or None on any failure.

    Uses ffprobe to read container metadata only — never decodes the
    audio data, so memory usage is negligible.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning(
                "ffprobe failed (rc=%d) for %s: %s",
                result.returncode, file_path, result.stderr[:300],
            )
            return None
        data = json.loads(result.stdout)
        duration_str = data.get("format", {}).get("duration")
        if duration_str is None:
            logger.warning("ffprobe returned no duration for %s", file_path)
            return None
        return int(float(duration_str) * 1000)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("ffprobe_duration_ms failed for %s: %s", file_path, exc)
        return None


def ffmpeg_extract_chunk(
    file_path: Path,
    start_ms: int,
    end_ms: int,
    index: int,
    *,
    output_prefix: str = "chunk",
    timeout: int = FFMPEG_CHUNK_TIMEOUT,
) -> Path:
    """Extract a time range from *file_path* to a temporary MP3 file.

    Uses ffmpeg in streaming mode — memory usage is bounded to ffmpeg's
    internal buffers (a few MB), regardless of input file size.

    Always re-encodes to MP3 at 128 kbps mono 16 kHz.  This is safe for
    all input formats and the OpenAI transcription API accepts MP3.

    Returns the Path to the output temp file.  Caller must delete it.
    Raises ``subprocess.CalledProcessError`` on ffmpeg failure.
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False, prefix=f"{output_prefix}{index}_",
    )
    tmp.close()
    out_path = Path(tmp.name)

    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", f"{start_sec:.3f}",
                "-t", f"{duration_sec:.3f}",
                "-i", str(file_path),
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                "-ac", "1",
                "-ar", "16000",
                "-f", "mp3",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            out_path.unlink(missing_ok=True)
            raise subprocess.CalledProcessError(
                result.returncode, "ffmpeg", result.stdout, result.stderr,
            )
        return out_path
    except Exception:
        out_path.unlink(missing_ok=True)
        raise
