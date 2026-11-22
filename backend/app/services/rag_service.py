import json
import os
import re
import time
import threading
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    # BM25+ rather than BM25Okapi: Okapi gives a *negative* weight to any word that appears in more than half of
    # the documents, so on the small per-session corpora here (often a single uploaded file) it returned nothing.
    from rank_bm25 import BM25Plus
except ImportError:
    BM25Plus = None

from app.core.config import settings
from app.core.logging import logger
from app.schemas.chat import SourceCitation
from app.services.reranker_service import reranker_service
from app.services.query_service import query_service
from app.services.cache_service import cache_service

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens (punctuation is not part of a word, so "index," matches "index")."""
    return _TOKEN.findall(text.lower())


class RAGService:
    """Modular RAG retrieval service.

    Supports:
    - Vector search (FAISS)
    - BM25 keyword search
    - Reciprocal Rank Fusion (RRF) Hybrid Retrieval
    - Cross-Encoder Reranking
    - Configurable chunking (Word, Sentence, Paragraph)
    - Per-stage latency measurement

    Scoping: every chunk carries a `session_id`. A chunk with `session_id=None` belongs to the shared knowledge
    base and is visible to everyone; any other chunk is visible only to retrieval for that session. The scope is
    applied *before* ranking (inside FAISS and on the BM25 scores), so a busy index never crowds out the
    caller's own documents.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.embedding_model = None
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        # BM25 statistics (document frequencies) must come from the chunks a caller may see, never from other
        # people's documents, so there is one small BM25 index per scope, rebuilt lazily when the content changes.
        self._bm25_cache: Dict[Tuple[Optional[str], bool], Tuple[int, List[int], Any]] = {}
        # Bumped whenever the indexed content changes; part of the response-cache key so a cached answer
        # can never outlive the documents it was computed from.
        self.version = 0

        self.index_file = settings.FAISS_INDEX_PATH
        self.docs_file = settings.FAISS_DOCS_PATH

        self._ensure_storage_dir()
        self._load_embedding_model()
        self._load_index()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.index_file)), exist_ok=True)

    def _load_embedding_model(self):
        try:
            logger.info(f"Loading SentenceTransformer model '{settings.EMBEDDING_MODEL_NAME}'")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def _require_model(self):
        if self.embedding_model is None:
            self._load_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not available.")
        return self.embedding_model

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = np.asarray(self._require_model().encode(texts), dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    # ------------------------------------------------------------------ persistence
    def _persist(self):
        """Write the index and the chunk metadata. Metadata is JSON (never pickle), written atomically."""
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_file)
        tmp = f"{self.docs_file}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.documents, fh)
        os.replace(tmp, self.docs_file)

    def _legacy_docs_file(self) -> str:
        return os.path.splitext(self.docs_file)[0] + ".npy"

    def _read_documents(self) -> Optional[List[Any]]:
        if os.path.exists(self.docs_file):
            with open(self.docs_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        legacy = self._legacy_docs_file()
        if os.path.exists(legacy):
            # Older versions saved a pickled numpy array. Unpickling can execute code, so this is read once, only
            # from the local data directory, and immediately converted to JSON.
            logger.warning(f"Migrating legacy pickled document store '{legacy}' to JSON.")
            return list(np.load(legacy, allow_pickle=True))
        return None

    def _load_index(self):
        with self.lock:
            if not os.path.exists(self.index_file):
                return
            try:
                raw_docs = self._read_documents()
                if raw_docs is None:
                    return
                self.index = faiss.read_index(self.index_file)
                self.documents = []
                for doc in raw_docs:
                    if isinstance(doc, dict):
                        self.documents.append(doc)
                    else:
                        self.documents.append({
                            "text": str(doc),
                            "doc_name": "Knowledge Base",
                            "chunk_id": len(self.documents),
                            "session_id": None,
                        })
                if self.index.ntotal != len(self.documents):
                    raise ValueError(
                        f"index has {self.index.ntotal} vectors but {len(self.documents)} chunk records"
                    )
                if not os.path.exists(self.docs_file):
                    self._persist()  # completes the legacy migration
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                self.version += 1
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self.index = None
                self.documents = []

    # ------------------------------------------------------------------ chunking
    def chunk_text(
        self,
        text: str,
        strategy: str = None,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[str]:
        """Split text into chunks using requested strategy (word, sentence, paragraph)."""
        if not text or not text.strip():
            return []

        strat = strategy or settings.CHUNK_STRATEGY
        size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        step_overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
        step_overlap = max(0, min(step_overlap, size - 1))

        if strat == "sentence":
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sentences:
                return [text]
            chunks = []
            sentence_window = max(1, size // 25)  # estimate ~25 words per sentence
            step = max(1, sentence_window - max(0, step_overlap // 25))
            for i in range(0, len(sentences), step):
                chunk = " ".join(sentences[i : i + sentence_window])
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        elif strat == "paragraph":
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                return [text]
            chunks = []
            para_window = max(1, size // 100)
            step = max(1, para_window - 1)
            for i in range(0, len(paragraphs), step):
                chunk = "\n\n".join(paragraphs[i : i + para_window])
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        else:
            # Word sliding window strategy
            words = text.split()
            if not words:
                return []
            chunks = []
            stride = max(1, size - step_overlap)
            for i in range(0, len(words), stride):
                chunk = " ".join(words[i : i + size])
                if chunk.strip():
                    chunks.append(chunk)
                if i + size >= len(words):
                    break
            return chunks

    # ------------------------------------------------------------------ ingestion
    def add_document(
        self,
        text: str,
        doc_name: str = "Uploaded Document",
        chunk_strategy: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        session_id: Optional[str] = None
    ) -> int:
        """Process document into chunks, embed, insert into FAISS & BM25, and persist. Returns chunk count."""
        if not text or not text.strip():
            return 0

        chunks = self.chunk_text(
            text, strategy=chunk_strategy, chunk_size=chunk_size, overlap=chunk_overlap
        )
        if not chunks:
            return 0

        embeddings = self._embed(chunks)  # CPU heavy: done outside the lock

        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(embeddings.shape[1])

            start_idx = len(self.documents)
            self.index.add(embeddings)

            for idx, chunk in enumerate(chunks):
                self.documents.append({
                    "text": chunk,
                    "doc_name": doc_name,
                    "chunk_id": start_idx + idx,
                    "session_id": session_id
                })

            self.version += 1
            self._persist()
            total = self.index.ntotal

        logger.info(f"Added {len(chunks)} chunks from '{doc_name}' for session '{session_id}' (Total vectors: {total})")
        return len(chunks)

    # ------------------------------------------------------------------ search
    def _allowed_positions(self, session_id: Optional[str], include_shared: bool = True) -> List[int]:
        """Chunk positions visible to `session_id`: that session's own chunks, plus the shared ones if asked."""
        return [
            i for i, d in enumerate(self.documents)
            if (include_shared and d.get("session_id") is None)
            or (session_id is not None and d.get("session_id") == session_id)
        ]

    def _vector_search(
        self, query: str, top_k: int, allowed: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float], float]:
        """FAISS dense search restricted (inside the search) to `allowed` positions. Returns (indices, scores, latency)."""
        t0 = time.time()
        query_embedding = self._embed([query])  # outside the lock

        with self.lock:
            if self.index is None or self.index.ntotal == 0 or not self.documents:
                return [], [], 0.0
            if allowed is not None and not allowed:
                return [], [], 0.0

            candidate_count = len(allowed) if allowed is not None else self.index.ntotal
            k = min(top_k, candidate_count)
            if allowed is not None and candidate_count < self.index.ntotal:
                params = faiss.SearchParameters()
                params.sel = faiss.IDSelectorBatch(np.asarray(allowed, dtype=np.int64))
                distances, indices = self.index.search(query_embedding, k, params=params)
            else:
                distances, indices = self.index.search(query_embedding, k)

        latency = round(time.time() - t0, 4)
        pairs = [(int(i), float(s)) for i, s in zip(indices[0], distances[0]) if i >= 0]
        return [i for i, _ in pairs], [s for _, s in pairs], latency

    def _scoped_bm25(self, session_id: Optional[str], include_shared: bool, allowed: List[int]):
        """BM25 index over exactly the `allowed` chunks, cached until the index content changes (caller holds the lock)."""
        key = (session_id, include_shared)
        cached = self._bm25_cache.get(key)
        if cached and cached[0] == self.version:
            return cached[1], cached[2]
        bm25 = None
        if allowed and BM25Plus is not None:
            try:
                bm25 = BM25Plus([tokenize(self.documents[i]["text"]) for i in allowed])
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")
        if len(self._bm25_cache) >= 32:
            self._bm25_cache.pop(next(iter(self._bm25_cache)))
        self._bm25_cache[key] = (self.version, list(allowed), bm25)
        return list(allowed), bm25

    def _bm25_search(
        self, query: str, top_k: int, allowed: List[int], session_id: Optional[str] = None, include_shared: bool = True
    ) -> Tuple[List[int], List[float], float]:
        """BM25 search over the `allowed` chunks only. Returns (document positions, scores, latency)."""
        t0 = time.time()
        with self.lock:
            positions, bm25 = self._scoped_bm25(session_id, include_shared, allowed)
            if bm25 is None or not positions:
                return [], [], 0.0
            scores = np.asarray(bm25.get_scores(tokenize(query)), dtype="float64")

        order = np.argsort(scores)[::-1][:top_k]
        valid = [int(i) for i in order if scores[i] > 0]
        return [positions[i] for i in valid], [float(scores[i]) for i in valid], round(time.time() - t0, 4)

    def _rrf_fuse(
        self, vector_indices: List[int], bm25_indices: List[int], rrf_k: int = 60
    ) -> List[Tuple[int, float]]:
        """Combine Vector & BM25 rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[int, float] = {}

        for rank, doc_idx in enumerate(vector_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc_idx in enumerate(bm25_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        retrieval_mode: str = None,
        use_reranker: bool = None,
        session_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> Tuple[List[SourceCitation], str, Dict[str, float]]:
        """Execute the complete retrieval pipeline with per-stage latency profiling.

        `session_id=None` searches the shared knowledge base only; a session id also includes that session's
        own uploads. `include_shared=False` restricts the search to that session's chunks alone (used by the
        benchmark so user data cannot change its results).

        Returns:
            (citations, formatted_context, latency_breakdown)
        """
        start_time = time.time()
        k = top_k or settings.RAG_TOP_K
        mode = (retrieval_mode or settings.DEFAULT_RETRIEVAL_MODE).lower()
        rerank = use_reranker if use_reranker is not None else settings.ENABLE_RERANKER

        cleaned_query = query_service.clean_query(query)
        with self.lock:
            allowed = self._allowed_positions(session_id, include_shared)

        vector_indices, vector_scores, vec_latency = [], [], 0.0
        bm25_indices, bm25_scores, bm25_latency = [], [], 0.0

        if mode in ("vector", "hybrid"):
            vector_indices, vector_scores, vec_latency = self._vector_search(cleaned_query, k * 2, allowed)
        if mode in ("bm25", "hybrid"):
            bm25_indices, bm25_scores, bm25_latency = self._bm25_search(
                cleaned_query, k * 2, allowed, session_id, include_shared
            )

        t_h0 = time.time()
        if mode == "vector":
            ranked = list(zip(vector_indices, vector_scores))
        elif mode == "bm25":
            ranked = list(zip(bm25_indices, bm25_scores))
        else:  # hybrid
            ranked = self._rrf_fuse(vector_indices, bm25_indices, rrf_k=settings.RRF_K)
        hybrid_latency = round(time.time() - t_h0, 4)

        candidate_docs: List[Dict[str, Any]] = []
        with self.lock:
            for idx, score in ranked:
                if 0 <= idx < len(self.documents):
                    doc = dict(self.documents[idx])
                    doc["score"] = score
                    candidate_docs.append(doc)

        reranker_latency = 0.0
        if rerank and candidate_docs:
            t_r0 = time.time()
            reranked = reranker_service.rerank(cleaned_query, candidate_docs, top_n=k)
            candidate_docs = []
            for doc, score in reranked:
                doc["score"] = score
                candidate_docs.append(doc)
            reranker_latency = round(time.time() - t_r0, 4)
        else:
            candidate_docs = candidate_docs[:k]

        citations: List[SourceCitation] = []
        context_snippets: List[str] = []
        for idx_rank, doc in enumerate(candidate_docs):
            citation = SourceCitation(
                id=idx_rank + 1,
                text=doc["text"],
                score=round(float(doc.get("score", 0.0)), 4),
                source_doc=doc.get("doc_name", "Document")
            )
            citations.append(citation)
            context_snippets.append(f"[{idx_rank + 1}] (Source: {citation.source_doc}): {doc['text']}")

        latency_breakdown = {
            "vector_search_latency": vec_latency,
            "bm25_search_latency": bm25_latency,
            "hybrid_latency": hybrid_latency,
            "reranker_latency": reranker_latency,
            "total_retrieval_latency": round(time.time() - start_time, 4)
        }
        return citations, "\n\n".join(context_snippets), latency_breakdown

    # ------------------------------------------------------------------ removal
    def clear_index(self, session_id: Optional[str] = None, everything: bool = False) -> int:
        """Remove chunks and return how many were removed.

        * `session_id` given: only that session's uploads.
        * `session_id=None`: only the shared knowledge base.
        * `everything=True`: all chunks (administrative use).

        The surviving vectors are copied out of the existing index, so nothing is re-embedded.
        """
        with self.lock:
            if self.index is None or not self.documents:
                return 0

            if everything:
                keep = []
            else:
                keep = [i for i, d in enumerate(self.documents) if d.get("session_id") != session_id]

            removed = len(self.documents) - len(keep)
            if removed == 0:
                return 0

            if keep:
                vectors = np.vstack([self.index.reconstruct(i) for i in keep]).astype("float32")
                new_index = faiss.IndexFlatIP(vectors.shape[1])
                new_index.add(vectors)
                self.index = new_index
                self.documents = [self.documents[i] for i in keep]
                for new_pos, doc in enumerate(self.documents):
                    doc["chunk_id"] = new_pos
                self._persist()
            else:
                self.index = None
                self.documents = []
                for path in (self.index_file, self.docs_file, self._legacy_docs_file()):
                    if os.path.exists(path):
                        os.remove(path)

            self.version += 1

        logger.info(f"Removed {removed} chunks (session_id={session_id}, everything={everything})")
        return removed


rag_service = RAGService()
