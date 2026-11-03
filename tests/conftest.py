"""
Shared pytest fixtures for the grammar correction test suite.
"""

import pytest


@pytest.fixture
def sample_texts():
    """Sample erroneous texts for testing."""
    return [
        "She go to school yesterday.",
        "He have a big cat.",
        "They was playing in park.",
        "I readed the book last night.",
        "The informations is incorrect.",
    ]


@pytest.fixture
def sample_corrections():
    """Expected corrections for sample texts."""
    return [
        "She went to school yesterday.",
        "He has a big cat.",
        "They were playing in the park.",
        "I read the book last night.",
        "The information is incorrect.",
    ]


@pytest.fixture
def sample_sentence_pairs(sample_texts, sample_corrections):
    """Source-target sentence pairs for testing."""
    return [
        {"source": src, "target": tgt}
        for src, tgt in zip(sample_texts, sample_corrections)
    ]
