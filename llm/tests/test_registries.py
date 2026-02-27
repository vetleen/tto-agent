"""Tests for ModelRegistry and PipelineRegistry."""

from unittest.mock import MagicMock

from django.test import TestCase

from llm.core.interfaces import ChatModel
from llm.core.registry import ModelRegistry, get_model_registry
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import PipelineRegistry, get_pipeline_registry
from llm.service.errors import LLMConfigurationError
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse


class ModelRegistryTests(TestCase):
    """Test ModelRegistry prefix registration and get_model."""

    def test_register_model_prefix_empty_raises(self):
        registry = ModelRegistry()
        factory = lambda name: MagicMock(spec=ChatModel)
        with self.assertRaises(ValueError) as ctx:
            registry.register_model_prefix("", factory)
        self.assertIn("non-empty", str(ctx.exception))

    def test_get_model_unknown_prefix_raises_configuration_error(self):
        registry = ModelRegistry()
        with self.assertRaises(LLMConfigurationError) as ctx:
            registry.get_model("unknown-model")
        self.assertIn("unknown-model", str(ctx.exception))
        self.assertIn("Configured prefixes", str(ctx.exception))

    def test_get_model_matching_prefix_returns_from_factory(self):
        registry = ModelRegistry()
        fake = MagicMock(spec=ChatModel)
        fake.name = "gpt-4o"
        registry.register_model_prefix("gpt-", lambda name: fake)
        result = registry.get_model("gpt-4o-mini")
        self.assertIs(result, fake)

    def test_get_model_longest_prefix_wins_regardless_of_registration_order(self):
        registry = ModelRegistry()
        short = MagicMock(spec=ChatModel)
        short.name = "gpt"
        long = MagicMock(spec=ChatModel)
        long.name = "gpt-4"
        # Register the shorter prefix first — longest should still win.
        registry.register_model_prefix("gpt", lambda name: short)
        registry.register_model_prefix("gpt-4", lambda name: long)
        result = registry.get_model("gpt-4o")
        self.assertIs(result, long)

    def test_clear_empties_all_prefixes(self):
        registry = ModelRegistry()
        registry.register_model_prefix("gpt-", lambda name: MagicMock(spec=ChatModel))
        registry.clear()
        with self.assertRaises(LLMConfigurationError):
            registry.get_model("gpt-4o")

    def test_get_model_registry_returns_singleton(self):
        a = get_model_registry()
        b = get_model_registry()
        self.assertIs(a, b)


class PipelineRegistryTests(TestCase):
    """Test PipelineRegistry register_pipeline and get_pipeline."""

    def test_get_pipeline_unknown_raises_configuration_error(self):
        registry = PipelineRegistry()
        with self.assertRaises(LLMConfigurationError) as ctx:
            registry.get_pipeline("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))

    def test_register_and_get_pipeline(self):
        registry = PipelineRegistry()
        pipeline = MagicMock(spec=BasePipeline)
        pipeline.id = "test_pipeline"
        pipeline.capabilities = {}
        registry.register_pipeline(pipeline)
        self.assertIs(registry.get_pipeline("test_pipeline"), pipeline)

    def test_register_pipeline_empty_id_raises(self):
        registry = PipelineRegistry()
        pipeline = MagicMock(spec=BasePipeline)
        pipeline.id = ""
        pipeline.capabilities = {}
        with self.assertRaises(ValueError) as ctx:
            registry.register_pipeline(pipeline)
        self.assertIn("non-empty", str(ctx.exception))

    def test_clear_empties_all_pipelines(self):
        registry = PipelineRegistry()
        pipeline = MagicMock(spec=BasePipeline)
        pipeline.id = "test_pipeline"
        registry.register_pipeline(pipeline)
        registry.clear()
        with self.assertRaises(LLMConfigurationError):
            registry.get_pipeline("test_pipeline")

    def test_get_pipeline_registry_returns_singleton(self):
        a = get_pipeline_registry()
        b = get_pipeline_registry()
        self.assertIs(a, b)
