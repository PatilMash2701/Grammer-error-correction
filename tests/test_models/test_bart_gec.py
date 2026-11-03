"""
Tests for the BART-based grammar correction model.
"""

import pytest

from src.models.bart_gec import BartGEC, CorrectionOutput


class TestBartGEC:
    """Tests for the BartGEC model wrapper."""

    def test_from_pretrained_loads_model(self):
        """from_pretrained should load model and tokenizer."""
        # TODO: implement (may need mock to avoid downloading model in CI)
        pass

    def test_correct_text_returns_correction_output(self):
        """correct_text should return a CorrectionOutput dataclass."""
        # TODO: implement
        pass

    def test_correct_batch_handles_multiple_texts(self):
        """correct_batch should process a list of texts."""
        # TODO: implement
        pass

    def test_generate_produces_token_ids(self):
        """generate should return tensor of token IDs."""
        # TODO: implement
        pass


class TestCorrectionOutput:
    """Tests for the CorrectionOutput dataclass."""

    def test_creation(self):
        """Should create CorrectionOutput with required fields."""
        output = CorrectionOutput(
            corrected_text="She went to school.",
            confidence=0.95,
            corrections=[{"original": "go", "corrected": "went"}],
        )
        assert output.corrected_text == "She went to school."
        assert output.confidence == 0.95
        assert len(output.corrections) == 1
