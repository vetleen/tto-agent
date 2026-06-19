"""Tests for input-modality helpers in llm.display and the registry."""

from django.test import SimpleTestCase

from llm.display import input_modalities, supports_modality
from llm.model_registry import ModelInfo, get_model_info


class ModalityHelpersTests(SimpleTestCase):
    def test_registry_models_declare_image_and_pdf(self):
        info = get_model_info("anthropic/claude-opus-4-8")
        self.assertIn("text", info.input_modalities)
        self.assertIn("image", info.input_modalities)
        self.assertIn("pdf", info.input_modalities)
        self.assertEqual(info.output_modalities, ("text",))

    def test_supports_vision_property_matches_modalities(self):
        info = get_model_info("openai/gpt-5.4-mini")
        self.assertEqual(info.supports_vision, "image" in info.input_modalities)
        self.assertTrue(info.supports_vision)

    def test_supports_modality_registry(self):
        self.assertTrue(supports_modality("gemini/gemini-3.5-flash", "image"))
        self.assertTrue(supports_modality("gemini/gemini-3.5-flash", "pdf"))
        self.assertTrue(supports_modality("gemini/gemini-3.5-flash", "text"))

    def test_supports_modality_unknown_text_only(self):
        self.assertTrue(supports_modality("openai/whisper-1", "text"))
        self.assertFalse(supports_modality("openai/whisper-1", "image"))
        self.assertFalse(supports_modality("openai/whisper-1", "pdf"))

    def test_input_modalities_unknown_vision_heuristic(self):
        mods = input_modalities("anthropic/claude-future-99")
        self.assertIn("image", mods)
        self.assertIn("pdf", mods)

    def test_default_modelinfo_is_text_only(self):
        mi = ModelInfo("X", "openai", "x")
        self.assertEqual(mi.input_modalities, ("text",))
        self.assertFalse(mi.supports_vision)
