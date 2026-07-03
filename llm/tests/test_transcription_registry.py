"""Tests for the transcription model registry."""

from django.test import TestCase

from core.file_types import FILE_TYPES, KIND_AUDIO
from llm.transcription_registry import AUDIO_EXTENSIONS


class AudioExtensionParityTests(TestCase):
    """``transcription_registry.AUDIO_EXTENSIONS`` and ``core.file_types``
    KIND_AUDIO are independent lists of the same audio extensions — one gates
    transcription routing, the other gates uploads. They cannot be merged
    (``core.file_types`` is a no-Django-imports module loaded at settings time
    and must not import ``llm``), so this test guards against silent drift: a
    format present in one but not the other means files are either rejected on
    upload or accepted and then fail transcription.
    """

    def test_audio_extensions_match_file_types(self):
        file_type_audio = {ft.ext for ft in FILE_TYPES if ft.kind == KIND_AUDIO}
        self.assertEqual(
            set(AUDIO_EXTENSIONS),
            file_type_audio,
            "transcription_registry.AUDIO_EXTENSIONS drifted from core.file_types "
            "KIND_AUDIO; update both (they are hand-synced copies of the same set).",
        )
