import threading
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class RerankerService:
    """Cross-Encoder Reranking service to evaluate query-document pairs and re-order candidate context."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.model = None
        self.lock = threading.Lock()

    def _load_model(self):
        """Lazy load CrossEncoder model."""
        if self.model is not None:
            return

        with self.lock:
            if self.model is None and CrossEncoder is not None:
                try:
                    logger.info(f"Loading CrossEncoder reranker model '{self.model_name}'...")
                    self.model = CrossEncoder(self.model_name)
                    logger.info(f"CrossEncoder reranker '{self.model_name}' loaded successfully.")
                except Exception as e:
                    logger.warning(f"Failed to load CrossEncoder model '{self.model_name}': {e}. Reranking fallback active.")
                    self.model = None

    def rerank(
        self, query: str, candidate_docs: List[Dict[str, Any]], top_n: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank candidate documents based on CrossEncoder relevance scores.
        
        Returns a list of tuples: (doc_item, rerank_score).
        """
        if not candidate_docs or not query:
            return []

        n = top_n or settings.RAG_TOP_K
        self._load_model()

        if self.model is None:
            # Fallback: return original top candidate order with uniform or normalized rank scores
            return [(doc, doc.get("score", 1.0 / (idx + 1))) for idx, doc in enumerate(candidate_docs[:n])]

        try:
            pairs = [[query, doc.get("text", "")] for doc in candidate_docs]
            scores = self.model.predict(pairs)

            scored_docs = list(zip(candidate_docs, [float(s) for s in scores]))
            # Sort descending by CrossEncoder relevance score
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return scored_docs[:n]
        except Exception as e:
            logger.error(f"Error during CrossEncoder reranking: {e}")
            return [(doc, doc.get("score", 0.5)) for doc in candidate_docs[:n]]


reranker_service = RerankerService()
