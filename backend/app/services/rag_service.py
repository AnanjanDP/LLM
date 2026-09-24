import os
import re
import time
import threading
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.core.config import settings
from app.core.logging import logger
from app.schemas.chat import SourceCitation
from app.services.reranker_service import reranker_service
from app.services.query_service import query_service
from app.services.cache_service import cache_service



class RAGService:
    """Modular, high-performance production RAG service.
    
    Supports:
    - Vector search (FAISS)
    - BM25 keyword search
    - Reciprocal Rank Fusion (RRF) Hybrid Retrieval
    - Cross-Encoder Reranking
    - Configurable chunking (Word, Sentence, Paragraph)
    - Microsecond per-stage latency measurement
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.embedding_model = None
        self.index = None
        self.bm25_index = None
        self.documents: List[Dict[str, Any]] = []

        self.index_file = settings.FAISS_INDEX_PATH
        self.docs_file = settings.FAISS_DOCS_PATH

        self._ensure_storage_dir()
        self._load_embedding_model()
        self._load_index()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)

    def _load_embedding_model(self):
        try:
            logger.info(f"Loading SentenceTransformer model '{settings.EMBEDDING_MODEL_NAME}'")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def _build_bm25_index(self):
        """Construct BM25 index over currently registered document chunks."""
        if not self.documents or BM25Okapi is None:
            self.bm25_index = None
            return

        try:
            corpus = [doc["text"].lower().split() for doc in self.documents]
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"Built BM25 index over {len(self.documents)} document chunks.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None

    def _load_index(self):
        with self.lock:
            if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
                try:
                    self.index = faiss.read_index(self.index_file)
                    raw_docs = np.load(self.docs_file, allow_pickle=True)
                    self.documents = []
                    for doc in list(raw_docs):
                        if isinstance(doc, dict):
                            self.documents.append(doc)
                        else:
                            self.documents.append({
                                "text": str(doc),
                                "doc_name": "Knowledge Base",
                                "chunk_id": len(self.documents)
                            })
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.documents)} documents.")
                    self._build_bm25_index()
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
                    self.index = None
                    self.documents = []

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
        size = chunk_size or settings.CHUNK_SIZE
        step_overlap = overlap or settings.CHUNK_OVERLAP

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
            return chunks

    def add_document(
        self,
        text: str,
        doc_name: str = "Uploaded Document",
        chunk_strategy: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        session_id: Optional[str] = None
    ) -> int:
        """Process document into chunks, generate embeddings, insert into FAISS & BM25, and persist."""
        if not text or not text.strip():
            return 0

        chunks = self.chunk_text(
            text, strategy=chunk_strategy, chunk_size=chunk_size, overlap=chunk_overlap
        )
        if not chunks:
            return 0

        if self.embedding_model is None:
            self._load_embedding_model()

        embeddings = self.embedding_model.encode(chunks)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        with self.lock:
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)

            start_idx = len(self.documents)
            self.index.add(np.array(embeddings).astype("float32"))

            for idx, chunk in enumerate(chunks):
                self.documents.append({
                    "text": chunk,
                    "doc_name": doc_name,
                    "chunk_id": start_idx + idx,
                    "session_id": session_id
                })

            faiss.write_index(self.index, self.index_file)
            np.save(self.docs_file, np.array(self.documents, dtype=object))
            self._build_bm25_index()

        logger.info(f"Added {len(chunks)} chunks from '{doc_name}' for session '{session_id}' (Total vectors: {self.index.ntotal})")
        return len(chunks)


    def _vector_search(self, query: str, top_k: int) -> Tuple[List[int], List[float], float]:
        """Perform FAISS dense vector search and return (indices, scores, latency)."""
        t0 = time.time()
        if self.index is None or self.index.ntotal == 0 or not self.documents:
            return [], [], 0.0

        if self.embedding_model is None:
            self._load_embedding_model()

        query_embedding = self.embedding_model.encode([query])
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        if norm[0][0] > 0:
            query_embedding = query_embedding / norm

        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), min(top_k, self.index.ntotal)
        )
        latency = round(time.time() - t0, 4)

        if len(indices) > 0:
            return list(indices[0]), [float(s) for s in distances[0]], latency
        return [], [], latency

    def _bm25_search(self, query: str, top_k: int) -> Tuple[List[int], List[float], float]:
        """Perform BM25 sparse keyword search and return (indices, scores, latency)."""
        t0 = time.time()
        if self.bm25_index is None or not self.documents:
            return [], [], 0.0

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        valid_indices = [int(idx) for idx in top_indices if scores[idx] > 0]
        valid_scores = [float(scores[idx]) for idx in valid_indices]
        latency = round(time.time() - t0, 4)

        return valid_indices, valid_scores, latency

    def _rrf_fuse(
        self, vector_indices: List[int], bm25_indices: List[int], rrf_k: int = 60
    ) -> List[Tuple[int, float]]:
        """Combine Vector & BM25 rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[int, float] = {}

        for rank, doc_idx in enumerate(vector_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc_idx in enumerate(bm25_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        retrieval_mode: str = None,
        use_reranker: bool = None,
        session_id: Optional[str] = None
    ) -> Tuple[List[SourceCitation], str, Dict[str, float]]:
        """Execute complete retrieval pipeline with per-stage latency profiling.
        
        Returns:
            (citations, formatted_context, latency_breakdown)
        """
        start_time = time.time()
        k = top_k or settings.RAG_TOP_K
        mode = (retrieval_mode or settings.DEFAULT_RETRIEVAL_MODE).lower()
        rerank = use_reranker if use_reranker is not None else settings.ENABLE_RERANKER

        cleaned_query = query_service.clean_query(query)
        candidate_docs: List[Dict[str, Any]] = []

        vector_indices, vector_scores, vec_latency = [], [], 0.0
        bm25_indices, bm25_scores, bm25_latency = [], [], 0.0
        hybrid_latency, reranker_latency = 0.0, 0.0

        if mode in ("vector", "hybrid"):
            vector_indices, vector_scores, vec_latency = self._vector_search(cleaned_query, top_k=k * 2)

        if mode in ("bm25", "hybrid"):
            bm25_indices, bm25_scores, bm25_latency = self._bm25_search(cleaned_query, top_k=k * 2)

        t_h0 = time.time()
        if mode == "vector":
            for idx, score in zip(vector_indices, vector_scores):
                if 0 <= idx < len(self.documents):
                    doc = dict(self.documents[idx])
                    doc["score"] = score
                    candidate_docs.append(doc)
        elif mode == "bm25":
            for idx, score in zip(bm25_indices, bm25_scores):
                if 0 <= idx < len(self.documents):
                    doc = dict(self.documents[idx])
                    doc["score"] = score
                    candidate_docs.append(doc)
        else:  # hybrid
            fused = self._rrf_fuse(vector_indices, bm25_indices, rrf_k=settings.RRF_K)
            for doc_idx, score in fused:
                if 0 <= doc_idx < len(self.documents):
                    doc = dict(self.documents[doc_idx])
                    doc["score"] = score
                    candidate_docs.append(doc)
        hybrid_latency = round(time.time() - t_h0, 4)

        # Filter documents by session_id if scoped retrieval is requested
        if session_id:
            candidate_docs = [
                d for d in candidate_docs
                if d.get("session_id") is None or d.get("session_id") == session_id
            ]

        # Reranking Stage
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


        # Build Citations and Context Snippets
        citations: List[SourceCitation] = []
        context_snippets: List[str] = []

        for idx_rank, doc in enumerate(candidate_docs):
            score = float(doc.get("score", 0.0))
            citation = SourceCitation(
                id=idx_rank + 1,
                text=doc["text"],
                score=round(score, 4),
                source_doc=doc.get("doc_name", "Document")
            )
            citations.append(citation)
            context_snippets.append(f"[{idx_rank + 1}] (Source: {citation.source_doc}): {doc['text']}")

        combined_context = "\n\n".join(context_snippets)
        total_latency = round(time.time() - start_time, 4)

        latency_breakdown = {
            "vector_search_latency": vec_latency,
            "bm25_search_latency": bm25_latency,
            "hybrid_latency": hybrid_latency,
            "reranker_latency": reranker_latency,
            "total_retrieval_latency": total_latency
        }

        return citations, combined_context, latency_breakdown

    def clear_index(self, session_id: Optional[str] = None) -> bool:
        """Clear vector store index and document memory. If session_id is provided, removes only chunks for that session."""
        with self.lock:
            if session_id:
                self.documents = [d for d in self.documents if d.get("session_id") != session_id]
            else:
                self.documents = []

            if self.documents and self.embedding_model:
                texts = [d["text"] for d in self.documents]
                embeddings = self.embedding_model.encode(texts)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(np.array(embeddings).astype("float32"))
                faiss.write_index(self.index, self.index_file)
                np.save(self.docs_file, np.array(self.documents, dtype=object))
                self._build_bm25_index()
            else:
                self.index = None
                self.bm25_index = None
                self.documents = []
                if os.path.exists(self.index_file):
                    os.remove(self.index_file)
                if os.path.exists(self.docs_file):
                    os.remove(self.docs_file)
        cache_service.clear_cache()
        logger.info(f"Cleared vector store index (session_id={session_id})")
        return True



rag_service = RAGService()

