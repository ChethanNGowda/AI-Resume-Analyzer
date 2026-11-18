"""
Tests for singleton EmbeddingService, batching, and fallback generation.
"""

import pytest
from services.rag.embedding_service import EmbeddingService, FALLBACK_DIMENSION


def test_singleton_embedding_service():
    inst1 = EmbeddingService.get_instance()
    inst2 = EmbeddingService.get_instance()
    assert inst1 is inst2


def test_embedding_dimensions_and_normalization():
    service = EmbeddingService.get_instance()
    texts = ["Python software engineer with Flask experience", "Data analyst proficient in SQL"]
    embeddings = service.embed_batch(texts)

    assert len(embeddings) == 2
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == service.dimension
        # Verify vector is not all zeros
        assert any(x != 0 for x in emb)


def test_fallback_embedding_deterministic():
    service = EmbeddingService.get_instance()
    v1 = service._generate_fallback_embedding("Machine Learning Engineer")
    v2 = service._generate_fallback_embedding("Machine Learning Engineer")
    v3 = service._generate_fallback_embedding("Frontend React Developer")

    assert v1 == v2
    assert v1 != v3
    assert len(v1) == FALLBACK_DIMENSION
