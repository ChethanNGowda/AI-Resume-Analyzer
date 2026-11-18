"""
Tests for ChromaDB VectorStoreService with user isolation, semantic search, and document deletion.
"""

import pytest
from services.rag.vector_store import VectorStoreService, sanitize_collection_name


@pytest.fixture
def store():
    return VectorStoreService.get_instance()


def test_sanitize_collection_name():
    assert sanitize_collection_name("user@example.com") == "user_example_com"
    assert sanitize_collection_name("a") == "col_a"
    assert len(sanitize_collection_name("x" * 100)) <= 63


def test_vector_store_user_isolation(store):
    if not store.is_available:
        pytest.skip("ChromaDB not available in current environment.")

    user_a = "test_user_alice"
    user_b = "test_user_bob"

    # Clean existing test data
    store.clear_user_collection(user_a, "resume")
    store.clear_user_collection(user_b, "resume")

    chunks_a = [
        {
            "chunk_id": "alice_chunk_0",
            "source_id": "src_alice_1",
            "user_id": user_a,
            "document_name": "alice_resume.pdf",
            "document_type": "resume",
            "chunk_index": 0,
            "total_chunks": 1,
            "page_number": 1,
            "section_hint": "Skills",
            "text": "Alice is an expert in Golang, Kubernetes, and Distributed Systems.",
            "char_count": 60,
        }
    ]

    chunks_b = [
        {
            "chunk_id": "bob_chunk_0",
            "source_id": "src_bob_1",
            "user_id": user_b,
            "document_name": "bob_resume.pdf",
            "document_type": "resume",
            "chunk_index": 0,
            "total_chunks": 1,
            "page_number": 1,
            "section_hint": "Skills",
            "text": "Bob is a specialist in UI/UX design, Figma, and React Native.",
            "char_count": 60,
        }
    ]

    store.ingest_chunks(user_a, chunks_a, "resume")
    store.ingest_chunks(user_b, chunks_b, "resume")

    # Alice queries for Golang -> should find her chunk
    alice_results = store.query(user_a, "Golang Kubernetes", collection_type="resume")
    assert len(alice_results) >= 1
    assert "Alice" in alice_results[0]["text"]

    # Bob queries for Golang -> should NOT find Alice's chunk (Strict User Isolation)
    bob_results = store.query(user_b, "Golang Kubernetes", collection_type="resume")
    for r in bob_results:
        assert "Alice" not in r["text"]

    # Bob queries for Figma -> finds his chunk
    bob_figma = store.query(user_b, "Figma design", collection_type="resume")
    assert len(bob_figma) >= 1
    assert "Bob" in bob_figma[0]["text"]

    # Test list sources
    alice_sources = store.list_sources(user_a, "resume")
    assert len(alice_sources) == 1
    assert alice_sources[0]["source_id"] == "src_alice_1"

    # Test deletion
    deleted = store.delete_source(user_a, "src_alice_1", "resume")
    assert deleted is True
    assert len(store.list_sources(user_a, "resume")) == 0

    # Clean up
    store.clear_user_collection(user_a, "resume")
    store.clear_user_collection(user_b, "resume")
