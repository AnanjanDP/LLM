import json
import os
import numpy as np
import pytest
from app.core.config import settings
from app.services.rag_service import rag_service, tokenize
from app.services.query_service import query_service


def test_rag_chunking_strategies():
    sample_text = (
        "FastAPI is a modern web framework for Python. It is fast and easy to learn. "
        "FAISS is Meta's vector similarity search library. BM25 is a popular sparse keyword retrieval algorithm."
    )
    assert len(rag_service.chunk_text(sample_text, strategy="word", chunk_size=10, overlap=2)) > 0
    assert len(rag_service.chunk_text(sample_text, strategy="sentence", chunk_size=50, overlap=10)) > 0
    assert len(rag_service.chunk_text(sample_text, strategy="paragraph", chunk_size=100, overlap=20)) > 0


def test_word_chunking_overlap_and_no_redundant_tail():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = rag_service.chunk_text(words, strategy="word", chunk_size=40, overlap=10)
    assert chunks[0].split()[:3] == ["w0", "w1", "w2"]
    assert chunks[1].split()[0] == "w30"                      # stride is size - overlap
    assert chunks[-1].split()[-1] == "w99"                    # the end of the text is covered
    assert not set(chunks[-1].split()) <= set(chunks[-2].split())  # the last chunk adds something new
    # An overlap of 0 is honoured (it used to be silently replaced by the default).
    no_overlap = rag_service.chunk_text(words, strategy="word", chunk_size=50, overlap=0)
    assert [len(c.split()) for c in no_overlap] == [50, 50]
    assert rag_service.chunk_text("", strategy="word") == []


def test_tokenize_ignores_punctuation():
    assert tokenize("Index, search; and FAISS!") == ["index", "search", "and", "faiss"]


def test_rag_ingest_and_hybrid_retrieve():
    rag_service.add_document("FastAPI supports high concurrency with async def handlers and automatic OpenAPI documentation generation.", doc_name="FastAPI Doc")
    rag_service.add_document("BM25Okapi scores documents based on query term frequency and inverse document frequency.", doc_name="BM25 Doc")
    rag_service.add_document("FAISS indexes vector embeddings into dense spatial structures for ultra-fast k-NN search.", doc_name="FAISS Doc")

    citations_vec, _, _ = rag_service.retrieve("FastAPI concurrency", top_k=2, retrieval_mode="vector")
    assert citations_vec and citations_vec[0].source_doc == "FastAPI Doc"

    citations_bm25, _, _ = rag_service.retrieve("BM25Okapi frequency", top_k=2, retrieval_mode="bm25")
    assert citations_bm25 and citations_bm25[0].source_doc == "BM25 Doc"

    citations_hyb, _, lat_hyb = rag_service.retrieve("FAISS vector embeddings", top_k=2, retrieval_mode="hybrid")
    assert citations_hyb and citations_hyb[0].source_doc == "FAISS Doc"
    assert "vector_search_latency" in lat_hyb


# ------------------------------------------------------------ session scoping
def test_scope_is_applied_before_ranking_so_other_sessions_cannot_crowd_results_out():
    # Twenty strong matches in other sessions would fill a post-filtered top-k completely.
    for n in range(20):
        rag_service.add_document(f"zeppelin propeller maintenance schedule number {n}", doc_name=f"other-{n}", session_id=f"other-{n}")
    rag_service.add_document("my own note about the zeppelin propeller", doc_name="mine", session_id="mine")

    for mode in ("vector", "bm25", "hybrid"):
        citations, _, _ = rag_service.retrieve("zeppelin propeller", top_k=3, retrieval_mode=mode, session_id="mine")
        assert [c.source_doc for c in citations] == ["mine"], mode


def test_shared_chunks_are_visible_to_everyone_but_private_ones_are_not():
    rag_service.add_document("shared handbook entry about gaskets", doc_name="shared")
    rag_service.add_document("private note about gaskets", doc_name="private", session_id="s1")

    assert {c.source_doc for c in rag_service.retrieve("gaskets", top_k=5, session_id="s1")[0]} == {"shared", "private"}
    assert {c.source_doc for c in rag_service.retrieve("gaskets", top_k=5, session_id="s2")[0]} == {"shared"}
    assert {c.source_doc for c in rag_service.retrieve("gaskets", top_k=5)[0]} == {"shared"}
    assert {c.source_doc for c in rag_service.retrieve("gaskets", top_k=5, session_id="s1", include_shared=False)[0]} == {"private"}


def test_retrieval_from_an_empty_index_is_empty():
    assert rag_service.retrieve("anything at all", session_id="s1")[0] == []


def test_clear_index_removes_only_the_requested_session_without_re_embedding(monkeypatch):
    rag_service.add_document("alpha notes about turbines", doc_name="a", session_id="s1")
    rag_service.add_document("beta notes about turbines", doc_name="b", session_id="s2")
    rag_service.add_document("shared notes about turbines", doc_name="shared")

    calls = []
    original = rag_service._embed
    monkeypatch.setattr(rag_service, "_embed", lambda texts: (calls.append(texts), original(texts))[1])

    removed_before = rag_service.index.ntotal
    assert rag_service.clear_index("s1") == 1
    assert rag_service.index.ntotal == removed_before - 1
    assert calls == []  # nothing was embedded again

    names = {c.source_doc for c in rag_service.retrieve("turbines", top_k=5, session_id="s1")[0]}
    assert names == {"shared"}
    assert {c.source_doc for c in rag_service.retrieve("turbines", top_k=5, session_id="s2")[0]} == {"b", "shared"}
    # Vectors still line up with their records after the rebuild.
    assert rag_service.retrieve("beta notes", top_k=1, retrieval_mode="vector", session_id="s2")[0][0].source_doc == "b"
    assert rag_service.clear_index("does-not-exist") == 0


def test_clear_with_no_session_only_touches_the_shared_chunks():
    rag_service.add_document("shared entry", doc_name="shared")
    rag_service.add_document("private entry", doc_name="private", session_id="s1")
    assert rag_service.clear_index() == 1
    assert {c.source_doc for c in rag_service.retrieve("entry", top_k=5, session_id="s1")[0]} == {"private"}


# ------------------------------------------------------------ persistence
def test_documents_are_saved_as_json_and_reloaded():
    rag_service.add_document("persisted fact about lighthouses", doc_name="p", session_id="s1")
    with open(settings.FAISS_DOCS_PATH, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved[0]["text"].startswith("persisted fact") and saved[0]["session_id"] == "s1"

    rag_service.index = None
    rag_service.documents = []
    rag_service._load_index()
    assert rag_service.index.ntotal == 1
    assert rag_service.retrieve("lighthouses", top_k=1, session_id="s1")[0][0].source_doc == "p"


def test_a_legacy_pickled_store_is_migrated_to_json():
    rag_service.add_document("legacy fact about windmills", doc_name="old", session_id=None)
    docs = list(rag_service.documents)
    legacy = rag_service._legacy_docs_file()
    np.save(legacy, np.array(docs, dtype=object))
    os.remove(settings.FAISS_DOCS_PATH)

    rag_service.index = None
    rag_service.documents = []
    rag_service._load_index()
    assert rag_service.index.ntotal == 1
    assert os.path.exists(settings.FAISS_DOCS_PATH)  # converted
    os.remove(legacy)


def test_a_corrupt_store_is_ignored_instead_of_crashing_startup():
    rag_service.add_document("some fact", doc_name="x")
    with open(settings.FAISS_DOCS_PATH, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    rag_service.index = None
    rag_service.documents = []
    rag_service._load_index()
    assert rag_service.index is None and rag_service.documents == []


def test_query_service_cleaning_and_vague_detection():
    assert query_service.clean_query("   What is    FAISS?   ") == "What is FAISS?"
    assert query_service.detect_unsupported_or_vague("hi")["is_vague"] is True
    assert query_service.detect_unsupported_or_vague("Explain Reciprocal Rank Fusion")["is_vague"] is False
