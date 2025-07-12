"""
Pytest fixtures.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_texts():
    """Sample financial texts for testing."""
    return [
        ("Revenue increased by 25% this quarter", "positive"),
        ("Company reported significant losses", "negative"),
        ("Board appointed new CEO effective January", "neutral"),
        ("Strong earnings exceeded analyst expectations", "positive"),
        ("Profit warning issued due to supply chain issues", "negative"),
    ]


@pytest.fixture
def sample_predictions():
    return {
        "predictions": ["positive", "negative", "neutral", "positive", "negative"],
        "references": ["positive", "negative", "neutral", "positive", "negative"],
    }


@pytest.fixture
def sample_predictions_mixed():
    return {
        "predictions": ["positive", "positive", "neutral", "negative", "negative"],
        "references": ["positive", "negative", "neutral", "positive", "negative"],
    }
