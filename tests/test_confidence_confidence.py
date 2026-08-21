import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.confidence import ConfidenceEngine
import numpy as np


def test_confidence_engine_init():
    engine = ConfidenceEngine()
    assert engine is not None


def test_compute_token_confidence():
    engine = ConfidenceEngine()
    embeddings = np.random.randn(2, 10).astype('float32')
    result = engine.compute_token_confidence(embeddings)
    assert result is not None


def test_compute_sequence_confidence():
    engine = ConfidenceEngine()
    embeddings = [np.random.randn(10).astype('float32') for _ in range(5)]
    result = engine.compute_sequence_confidence(embeddings)
    assert result is not None


def test_fit():
    engine = ConfidenceEngine()
    # Create dummy data
    embeddings = np.random.randn(10, 10).astype('float32')
    labels = np.random.randint(0, 2, 10)
    engine.fit(embeddings, labels)
    assert engine is not None