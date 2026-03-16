"""Tests for PipelineRegistry."""

from unittest.mock import MagicMock

from django.test import TestCase

from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import PipelineRegistry, get_pipeline_registry
from llm.service.errors import LLMConfigurationError


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
