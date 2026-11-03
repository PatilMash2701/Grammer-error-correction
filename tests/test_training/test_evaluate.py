"""
Tests for evaluation metric functions.
"""

import pytest

from src.training.utils import EarlyStopping


class TestEarlyStopping:
    """Tests for the EarlyStopping class."""

    def test_no_stop_when_improving(self):
        """Should not stop when loss is decreasing."""
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            es(loss)
            assert not es.should_stop

    def test_stops_after_patience(self):
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3)
        es(1.0)  # best
        es(1.1)  # no improvement
        es(1.2)  # no improvement
        assert not es.should_stop
        es(1.3)  # no improvement -> patience exceeded
        assert es.should_stop

    def test_resets_counter_on_improvement(self):
        """Counter should reset when loss improves."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # counter = 1
        es(1.2)  # counter = 2
        es(0.9)  # improvement -> counter = 0
        assert not es.should_stop
        assert es.counter == 0

    def test_min_delta(self):
        """Improvement must exceed min_delta to count."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        es(1.0)
        es(0.95)  # improvement < min_delta -> no reset
        assert es.counter == 1
