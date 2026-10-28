"""Tests for the transcription model registry."""

from django.test import SimpleTestCase, TestCase

from core.file_types import FILE_TYPES, KIND_AUDIO
from llm.transcription_registry import (
    AUDIO_EXTENSIONS,
    TRANSCRIPTION_MODEL_REPLACEMENTS,
    _TRANSCRIPTION_MODELS,
    canonical_transcription_model_id,
    get_transcription_model_info,
    normalize_transcription_model_ids,
    partition_transcription_models,
)

FILE = "openai/gpt-transcribe"
LIVE = "openai/gpt-live-transcribe"


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


class ForwardingTests(SimpleTestCase):
    def test_retired_ids_are_not_registered(self):
        for retired in TRANSCRIPTION_MODEL_REPLACEMENTS:
            with self.subTest(model=retired):
                self.assertNotIn(retired, _TRANSCRIPTION_MODELS)

    def test_successors_are_registered(self):
        for retired, successors in TRANSCRIPTION_MODEL_REPLACEMENTS.items():
            for successor in successors:
                with self.subTest(model=retired, successor=successor):
                    self.assertIn(successor, _TRANSCRIPTION_MODELS)

    def test_retired_ids_forward_per_path(self):
        for retired in ("openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"):
            with self.subTest(model=retired):
                self.assertEqual(canonical_transcription_model_id(retired, path="file"), FILE)
                self.assertEqual(canonical_transcription_model_id(retired, path="live"), LIVE)

    def test_diarize_forwards_to_file_model_and_its_live_counterpart(self):
        diarize = "openai/gpt-4o-transcribe-diarize"
        self.assertEqual(canonical_transcription_model_id(diarize, path="file"), FILE)
        # A meeting stamped with the diarize model can still go live.
        self.assertEqual(canonical_transcription_model_id(diarize, path="live"), LIVE)
        self.assertEqual(canonical_transcription_model_id(diarize), FILE)

    def test_registered_models_resolve_to_their_counterpart_for_the_other_path(self):
        self.assertEqual(canonical_transcription_model_id(FILE, path="file"), FILE)
        self.assertEqual(canonical_transcription_model_id(FILE, path="live"), LIVE)
        self.assertEqual(canonical_transcription_model_id(LIVE, path="live"), LIVE)
        self.assertEqual(canonical_transcription_model_id(LIVE, path="file"), FILE)
        self.assertEqual(canonical_transcription_model_id(LIVE), LIVE)

    def test_unknown_and_empty(self):
        self.assertIsNone(canonical_transcription_model_id("openai/nope", path="file"))
        self.assertIsNone(canonical_transcription_model_id("", path="live"))
        self.assertIsNone(canonical_transcription_model_id(None))
        self.assertIsNone(get_transcription_model_info("openai/nope"))

    def test_get_info_follows_the_first_successor(self):
        info = get_transcription_model_info("openai/gpt-4o-mini-transcribe")
        self.assertEqual(info.api_model, "gpt-transcribe")
        self.assertEqual(get_transcription_model_info(LIVE).api_model, "gpt-live-transcribe")

    def test_normalize_expands_retired_ids_to_every_successor_once(self):
        self.assertEqual(
            normalize_transcription_model_ids([
                "openai/gpt-4o-transcribe",
                "openai/gpt-4o-mini-transcribe",
                "openai/gpt-4o-transcribe-diarize",
                "openai/typo",
            ]),
            [FILE, LIVE],
        )
        self.assertEqual(normalize_transcription_model_ids(["openai/gpt-4o-transcribe-diarize"]), [FILE])
        self.assertEqual(normalize_transcription_model_ids([LIVE, FILE]), [LIVE, FILE])

    def test_partition_by_capability(self):
        self.assertEqual(partition_transcription_models([FILE, LIVE]), ([LIVE], [FILE]))
        self.assertEqual(partition_transcription_models([FILE]), ([], [FILE]))
        self.assertEqual(
            partition_transcription_models(["openai/gpt-4o-mini-transcribe"]), ([LIVE], [FILE]),
        )
