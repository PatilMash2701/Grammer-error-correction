"""
Tests for data preprocessing functions.
"""

import pytest

from src.data.preprocess import create_data_splits


class TestCreateDataSplits:
    """Tests for the create_data_splits function."""

    def test_split_ratios(self, sample_sentence_pairs):
        """Splits should respect the given ratios."""
        # TODO: implement when create_data_splits is ready
        pass

    def test_split_ratios_must_sum_to_one(self):
        """Should raise ValueError if ratios don't sum to 1.0."""
        # TODO: implement
        pass

    def test_reproducibility_with_seed(self, sample_sentence_pairs):
        """Same seed should produce identical splits."""
        # TODO: implement
        pass

    def test_no_data_leakage(self, sample_sentence_pairs):
        """Train, val, test sets should not overlap."""
        # TODO: implement
        pass
