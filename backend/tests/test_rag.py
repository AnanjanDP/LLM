import pytest
from app.services.rag_service import rag_service
from app.services.query_service import query_service
from app.services.reranker_service import reranker_service


def test_rag_chunking_strategies():
    sample_text = (
        "FastAPI is a modern web framework for Python. It is fast and easy to learn. "
        "FAISS is Meta's vector similarity search library. BM25 is a popular sparse keyword retrieval algorithm."
    )
    word_chunks = rag_service.chunk_text(sample_text, strategy="word", chunk_size=10, overlap=2)
    assert len(word_chunks) > 0

    sentence_chunks = rag_service.chunk_text(sample_text, strategy="sentence", chunk_size=50, overlap=10)
    assert len(sentence_chunks) > 0

    para_chunks = rag_service.chunk_text(sample_text, strategy="paragraph", chunk_size=100, overlap=20)
    assert len(para_chunks) > 0


def test_rag_ingest_and_hybrid_retrieve():
    doc1 = "FastAPI supports high concurrency with async def handlers and automatic OpenAPI documentation generation."
    doc2 = "BM25Okapi scores documents based on query term frequency and inverse document frequency."
    doc3 = "FAISS indexes vector embeddings into dense spatial structures for ultra-fast k-NN search."

    rag_service.add_document(doc1, doc_name="FastAPI Doc")
    rag_service.add_document(doc2, doc_name="BM25 Doc")
    rag_service.add_document(doc3, doc_name="FAISS Doc")

    # Vector search test
    citations_vec, ctx_vec, lat_vec = rag_service.retrieve("FastAPI concurrency", top_k=2, retrieval_mode="vector")
    assert len(citations_vec) > 0

    # BM25 search test
    citations_bm25, ctx_bm25, lat_bm25 = rag_service.retrieve("BM25Okapi frequency", top_k=2, retrieval_mode="bm25")
    assert len(citations_bm25) > 0

    # Hybrid search test
    citations_hyb, ctx_hyb, lat_hyb = rag_service.retrieve("FAISS vector embeddings", top_k=2, retrieval_mode="hybrid")
    assert len(citations_hyb) > 0
    assert "vector_search_latency" in lat_hyb


def test_query_service_cleaning_and_vague_detection():
    cleaned = query_service.clean_query("   What is    FAISS?   ")
    assert cleaned == "What is FAISS?"

    vague = query_service.detect_unsupported_or_vague("hi")
    assert vague["is_vague"] is True

    valid = query_service.detect_unsupported_or_vague("Explain Reciprocal Rank Fusion")
    assert valid["is_vague"] is False
