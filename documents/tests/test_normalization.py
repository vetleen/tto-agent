"""Tests for the document normalization service."""

from unittest.mock import patch, MagicMock, call

from django.test import TestCase, override_settings

from documents.services.normalization import (
    MAX_BATCH_TOKENS,
    _compute_batches,
    _compute_overlaps,
    _get_tail_overlap,
    _is_normalization_enabled,
    _process_single_batch,
    normalize_text,
)


class ComputeBatchesTests(TestCase):
    def test_single_small_batch(self):
        text = "Hello world, this is a short document."
        batches = _compute_batches(text)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], text)

    def test_equal_distribution(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # Build text with a known token count (~14000 tokens)
        words = ["testing"] * 14000
        text = " ".join(words)
        total = len(enc.encode(text))
        batches = _compute_batches(text)
        # Should produce multiple batches
        self.assertGreater(len(batches), 1)
        # All text content should be preserved (whitespace may differ)
        reconstructed = " ".join(batches)
        self.assertEqual(reconstructed, text)

    def test_just_over_limit(self):
        # Create text just over MAX_BATCH_TOKENS
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # Build text that's just over the limit
        words = []
        while len(enc.encode(" ".join(words))) <= MAX_BATCH_TOKENS:
            words.append("testing")
        text = " ".join(words)
        batches = _compute_batches(text)
        self.assertEqual(len(batches), 2)

    def test_compute_batches_sentence_aware(self):
        """Verify that batches split on sentence boundaries."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Build sentences that together exceed MAX_BATCH_TOKENS
        # Each sentence is ~50 tokens (short enough to fit many per batch)
        sentence = "The quick brown fox jumps over the lazy dog near the riverbank. "
        sent_tokens = len(enc.encode(sentence.strip()))

        # Enough sentences to require multiple batches
        num_sentences = (MAX_BATCH_TOKENS // sent_tokens) * 3
        sentences = [sentence.strip() for _ in range(num_sentences)]
        text = " ".join(sentences)

        batches = _compute_batches(text)
        self.assertGreater(len(batches), 1)

        # Each batch (except possibly the last) should end at a sentence boundary
        for batch in batches[:-1]:
            self.assertTrue(
                batch.rstrip().endswith("."),
                f"Batch should end at sentence boundary: ...{batch[-40:]!r}",
            )


class ComputeOverlapsTests(TestCase):
    def test_compute_overlaps(self):
        """Overlaps are precomputed from raw batch text."""
        batches = ["First batch text here.", "Second batch text.", "Third batch."]
        overlaps = _compute_overlaps(batches)

        self.assertEqual(len(overlaps), len(batches))
        # First overlap is always empty
        self.assertEqual(overlaps[0], "")
        # Second overlap is derived from the first batch
        self.assertEqual(overlaps[1], _get_tail_overlap(batches[0]))
        # Third overlap is derived from the second batch
        self.assertEqual(overlaps[2], _get_tail_overlap(batches[1]))

    def test_compute_overlaps_single_batch(self):
        overlaps = _compute_overlaps(["Only one batch."])
        self.assertEqual(overlaps, [""])


class NormalizationEnabledTests(TestCase):
    @override_settings(LLM_DEFAULT_CHEAP_MODEL="")
    def test_disabled_when_no_cheap_model(self):
        self.assertFalse(_is_normalization_enabled(user_id=None))

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_enabled_when_no_user(self):
        self.assertTrue(_is_normalization_enabled(user_id=None))

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_enabled_when_user_has_no_org(self):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.create_user(email="norm@example.com", password="test123")
        self.assertTrue(_is_normalization_enabled(user_id=user.id))

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_disabled_when_org_disables_tool(self):
        from django.contrib.auth import get_user_model
        from accounts.models import Membership, Organization
        User = get_user_model()
        user = User.objects.create_user(email="normdis@example.com", password="test123")
        org = Organization.objects.create(
            name="TestOrg", slug="normtest",
            preferences={"tools": {"normalize_document": False}},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        self.assertFalse(_is_normalization_enabled(user_id=user.id))

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_enabled_when_org_enables_tool(self):
        from django.contrib.auth import get_user_model
        from accounts.models import Membership, Organization
        User = get_user_model()
        user = User.objects.create_user(email="normen@example.com", password="test123")
        org = Organization.objects.create(
            name="TestOrg2", slug="normtest2",
            preferences={"tools": {"normalize_document": True}},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        self.assertTrue(_is_normalization_enabled(user_id=user.id))


class NormalizeBatchTests(TestCase):
    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._normalize_batch")
    def test_normalize_batch_applies_edits(self, mock_batch, mock_enabled):
        mock_batch.return_value = "# Hello World\n\nThis is formatted."
        result = normalize_text("Hello World\n\nThis is unformatted.")
        self.assertEqual(result, "# Hello World\n\nThis is formatted.")
        mock_batch.assert_called_once()

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._compute_batches")
    @patch("documents.services.normalization._normalize_batch")
    def test_normalize_text_multi_batch(self, mock_batch, mock_batches, mock_enabled):
        mock_batches.return_value = ["batch one", "batch two"]
        mock_batch.side_effect = ["# Batch One", "# Batch Two"]
        result = normalize_text("batch one batch two")
        self.assertEqual(result, "# Batch One\n\n# Batch Two")
        # Both batches should be processed (order may vary due to parallelism)
        self.assertEqual(mock_batch.call_count, 2)

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._normalize_batch")
    def test_normalize_text_retry_on_failure(self, mock_batch, mock_enabled):
        mock_batch.side_effect = [Exception("LLM error"), "# Recovered text"]
        result = normalize_text("Some text")
        self.assertEqual(result, "# Recovered text")
        self.assertEqual(mock_batch.call_count, 2)

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._normalize_batch")
    def test_normalize_text_fallback_after_max_retries(self, mock_batch, mock_enabled):
        mock_batch.side_effect = Exception("persistent failure")
        result = normalize_text("Raw text that should be returned")
        # Should fall back to the raw batch text after all retries
        self.assertEqual(result, "Raw text that should be returned")
        self.assertEqual(mock_batch.call_count, 3)  # initial + 2 retries

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=False)
    def test_normalize_text_skipped_when_tool_disabled(self, mock_enabled):
        result = normalize_text("Unchanged text")
        self.assertEqual(result, "Unchanged text")

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="")
    def test_normalize_text_skipped_when_no_cheap_model(self):
        result = normalize_text("Unchanged text")
        self.assertEqual(result, "Unchanged text")

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._compute_batches")
    @patch("documents.services.normalization._normalize_batch")
    def test_overlap_context_in_second_batch(self, mock_batch, mock_batches, mock_enabled):
        mock_batches.return_value = ["first batch", "second batch"]
        mock_batch.side_effect = ["# First Batch", "# Second Batch"]
        normalize_text("first batch second batch")
        # Second call should include overlap_context from raw first batch text
        second_call_args = mock_batch.call_args_list[1]
        overlap_arg = second_call_args[0][2] if len(second_call_args[0]) > 2 else second_call_args[1].get("overlap_context", "")
        # The overlap context should be derived from the raw first batch (not LLM output)
        self.assertTrue(len(overlap_arg) > 0, "Second batch should receive overlap context")
        # Since overlap is from raw text, it should come from "first batch", not "# First Batch"
        self.assertNotIn("# First", overlap_arg)

    @override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    @patch("documents.services.normalization._is_normalization_enabled", return_value=True)
    @patch("documents.services.normalization._compute_batches")
    @patch("documents.services.normalization._normalize_batch")
    def test_parallel_partial_failure(self, mock_batch, mock_batches, mock_enabled):
        """Some batches fail all retries, others succeed — order is preserved."""
        mock_batches.return_value = ["batch A", "batch B", "batch C"]

        def side_effect(batch_text, description, overlap, user_id, data_room_id):
            if batch_text == "batch B":
                raise Exception("batch B always fails")
            return f"# {batch_text.title()}"

        mock_batch.side_effect = side_effect
        result = normalize_text("batch A batch B batch C")

        parts = result.split("\n\n")
        self.assertEqual(len(parts), 3)
        # Batch A and C should be normalized
        self.assertEqual(parts[0], "# Batch A")
        # Batch B should fall back to raw text
        self.assertEqual(parts[1], "batch B")
        self.assertEqual(parts[2], "# Batch C")


class TailOverlapTests(TestCase):
    def test_short_text_returns_full(self):
        result = _get_tail_overlap("short text")
        self.assertEqual(result, "short text")

    def test_long_text_returns_tail(self):
        # Build text with more than OVERLAP_TOKENS tokens
        words = ["word"] * 1000
        text = " ".join(words)
        result = _get_tail_overlap(text)
        self.assertIn("word", result)
        # Should be shorter than the original
        self.assertLess(len(result), len(text))
