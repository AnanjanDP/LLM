import os
import re
import time
import uuid
from typing import List, Dict, Any, Optional
from app.services.rag_service import rag_service
from app.services.llm_provider import llm_factory
from app.services.prompt_service import prompt_service
from app.core.config import settings
from app.core.logging import logger

# The benchmark searches only this scope of the index, so user uploads cannot change its results.
EVAL_SCOPE = "__eval__"
CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval_corpus", "platform_knowledge.md"
)

_WORD = re.compile(r"\w+")


def _content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


class RAGEvalEngine:
    """Evaluation of retrieval (Recall@K, Precision, MRR, Hit Rate) and, optionally, generation.

    What the numbers mean (and do not):
      * Retrieval metrics use *keyword coverage*: a retrieved chunk counts as relevant if it contains one of the
        question's expected keywords, and Recall@K is the share of expected keywords found in the top K chunks.
        This is a cheap, deterministic proxy for labelled relevance judgements.
      * Generation metrics are *lexical proxies* (word overlap), not model-graded or human-graded quality:
        faithfulness = share of the answer's content words that appear in the retrieved context;
        answer_relevance = share of the question's content words that appear in the answer;
        context_relevance = share of the question's content words that appear in the context;
        answer_correctness = share of the reference answer's content words that appear in the answer.
      * The benchmark is small and written against the bundled corpus. Use it to compare configurations and to
        catch regressions, not as a general quality claim.
    """

    DEFAULT_DATASET: List[Dict[str, Any]] = [
        {
            "id": "eval_1",
            "question": "What architecture does the application use?",
            "ground_truth_answer": "The application uses a FastAPI backend, a React frontend, FAISS vector search, BM25 keyword retrieval, Redis caching, and a SQLite database.",
            "relevant_keywords": ["fastapi", "faiss", "bm25", "react", "sqlite", "redis"],
            "category": "architecture"
        },
        {
            "id": "eval_2",
            "question": "How does document indexing work?",
            "ground_truth_answer": "Documents are split into configurable chunks, embedded with a SentenceTransformer, stored in a FAISS index, and also indexed with BM25.",
            "relevant_keywords": ["chunk", "embedding", "faiss", "bm25", "sentencetransformer"],
            "category": "rag_indexing"
        },
        {
            "id": "eval_3",
            "question": "What retrieval modes and reranking options are supported?",
            "ground_truth_answer": "Vector search, BM25 keyword search, hybrid retrieval with Reciprocal Rank Fusion, and an optional Cross-Encoder reranker.",
            "relevant_keywords": ["vector", "bm25", "hybrid", "reciprocal rank fusion", "reranker", "cross-encoder"],
            "category": "retrieval"
        },
        {
            "id": "eval_4",
            "question": "How is performance and latency instrumented?",
            "ground_truth_answer": "Latency is recorded per stage and reported as P50, P95 and P99, and responses are cached in Redis with a reported hit rate.",
            "relevant_keywords": ["latency", "p95", "redis", "cache", "stage"],
            "category": "performance"
        },
        {
            "id": "eval_5",
            "question": "How are uploaded documents kept separate between users?",
            "ground_truth_answer": "Documents belong to the conversation they were uploaded to, and the session filter is applied inside the search before ranking.",
            "relevant_keywords": ["conversation", "session", "filter", "id selector", "private"],
            "category": "isolation"
        },
        {
            "id": "eval_6",
            "question": "How does authentication work and what protects it?",
            "ground_truth_answer": "Passwords are stored as bcrypt hashes, login returns a signed JWT that expires after a day, and authentication endpoints are rate limited.",
            "relevant_keywords": ["bcrypt", "jwt", "expires", "rate limited", "secret key"],
            "category": "security"
        },
        {
            "id": "eval_7",
            "question": "Which prompts and language model does it use?",
            "ground_truth_answer": "Versioned prompts (standard, strict and chain of thought) with Llama 3.3 on Groq and a smaller fallback model.",
            "relevant_keywords": ["prompt", "strict", "chain of thought", "llama", "groq", "fallback"],
            "category": "generation"
        },
        {
            "id": "eval_8",
            "question": "What do the evaluation metrics measure?",
            "ground_truth_answer": "Recall at K, precision, mean reciprocal rank and hit rate for retrieval, plus lexical proxy scores for faithfulness and relevance that measure word overlap.",
            "relevant_keywords": ["recall", "mean reciprocal rank", "hit rate", "faithfulness", "word overlap"],
            "category": "evaluation"
        },
    ]

    # ------------------------------------------------------------------ corpus
    def ensure_corpus(self) -> int:
        """Index the bundled benchmark corpus once (idempotent). Returns the number of chunks in the eval scope."""
        def count() -> int:
            with rag_service.lock:
                return sum(1 for d in rag_service.documents if d.get("session_id") == EVAL_SCOPE)

        existing = count()
        if existing:
            return existing
        with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
            text = fh.read()
        rag_service.add_document(
            text, doc_name="platform_knowledge.md", chunk_strategy="word", chunk_size=60, chunk_overlap=15,
            session_id=EVAL_SCOPE,
        )
        return count()

    # ------------------------------------------------------------------ metrics
    def _eval_retrieval_query(
        self, citations: List[Any], expected_keywords: List[str]
    ) -> Dict[str, float]:
        """Recall@1/3/5/10, Precision@K, MRR and Hit Rate by keyword coverage."""
        if not expected_keywords:
            return {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "recall@10": 1.0, "precision@k": 1.0, "mrr": 1.0, "hit_rate": 1.0}

        kw_set = set(k.lower() for k in expected_keywords)
        matched_at_rank: Dict[int, set] = {}
        first_hit_rank = None

        for idx, citation in enumerate(citations):
            rank = idx + 1
            text = citation.text.lower()
            hits = {kw for kw in kw_set if kw in text}
            if hits:
                matched_at_rank[rank] = hits
                if first_hit_rank is None:
                    first_hit_rank = rank

        def recall_at_k(k: int) -> float:
            seen = set()
            for r, hits in matched_at_rank.items():
                if r <= k:
                    seen.update(hits)
            return round(len(seen) / len(kw_set), 4)

        def precision_at_k(k: int) -> float:
            k_cit = citations[:k]
            if not k_cit:
                return 0.0
            relevant_count = sum(1 for idx, _ in enumerate(k_cit) if (idx + 1) in matched_at_rank)
            return round(relevant_count / len(k_cit), 4)

        return {
            "recall@1": recall_at_k(1),
            "recall@3": recall_at_k(3),
            "recall@5": recall_at_k(5),
            "recall@10": recall_at_k(10),
            "precision@k": precision_at_k(len(citations) or 1),
            "mrr": round(1.0 / first_hit_rank, 4) if first_hit_rank else 0.0,
            "hit_rate": 1.0 if first_hit_rank is not None else 0.0,
        }

    def _eval_generation_quality(
        self, query: str, context: str, answer: str, ground_truth: str
    ) -> Dict[str, float]:
        """Lexical proxies for faithfulness, relevance and correctness (see the class docstring). No bonuses are added."""
        if not answer or not answer.strip():
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0, "answer_correctness": 0.0}

        ans = _content_words(answer)
        ctx = _content_words(context)
        qry = _content_words(query)
        gt = _content_words(ground_truth)

        def share(part: set, whole: set) -> float:
            return round(len(part & whole) / len(part), 4) if part else 0.0

        return {
            "faithfulness": share(ans, ctx),
            "answer_relevance": share(qry, ans),
            "context_relevance": share(qry, ctx),
            "answer_correctness": share(gt, ans),
        }

    # ------------------------------------------------------------------ experiment
    def run_benchmark_experiment(
        self,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = False,
        top_k: int = 4,
        prompt_version: str = "v1.0",
        model: Optional[str] = None,
        evaluate_generation: bool = True,
    ) -> Dict[str, Any]:
        """Run the benchmark. Generation is only evaluated when a real language model is configured."""
        self.ensure_corpus()
        experiment_id = f"exp_{str(uuid.uuid4())[:8]}"

        provider = llm_factory.get_provider() if evaluate_generation else None
        generation_evaluated = bool(provider is not None and provider.name != "mock")

        retrieval = {k: [] for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "hit_rate")}
        generation = {k: [] for k in ("faithfulness", "answer_relevance", "context_relevance", "answer_correctness")}
        results = []
        total_latency = 0.0

        for item in self.DEFAULT_DATASET:
            t0 = time.time()
            query = item["question"]

            citations, context, _ = rag_service.retrieve(
                query=query, top_k=top_k, retrieval_mode=retrieval_mode, use_reranker=use_reranker,
                session_id=EVAL_SCOPE, include_shared=False,
            )
            ret_metrics = self._eval_retrieval_query(citations, item["relevant_keywords"])
            for k in retrieval:
                retrieval[k].append(ret_metrics[k])

            answer = ""
            gen_metrics = {k: None for k in generation}
            if generation_evaluated:
                messages, _ = prompt_service.build_chat_messages(
                    query=query, context=context, prompt_version_name=prompt_version
                )
                answer = provider.generate(messages=messages, model=model)
                gen_metrics = self._eval_generation_quality(query, context, answer, item["ground_truth_answer"])
                for k in generation:
                    generation[k].append(gen_metrics[k])

            elapsed = round(time.time() - t0, 3)
            total_latency += elapsed
            results.append({
                "question": query,
                "category": item["category"],
                "retrieved_count": len(citations),
                "answer_snippet": (answer[:150] + "...") if answer else None,
                "recall@5": ret_metrics["recall@5"],
                "mrr": ret_metrics["mrr"],
                "faithfulness": gen_metrics["faithfulness"],
                "answer_relevance": gen_metrics["answer_relevance"],
                "latency_sec": elapsed,
                "passed": ret_metrics["hit_rate"] > 0 and (not generation_evaluated or len(answer) > 20),
            })

        avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0
        gen_avg = {k: (avg(v) if generation_evaluated else None) for k, v in generation.items()}

        return {
            "experiment_id": experiment_id,
            "config": {
                "retrieval_mode": retrieval_mode,
                "use_reranker": use_reranker,
                "top_k": top_k,
                "prompt_version": prompt_version,
                "model": (model or settings.DEFAULT_LLM_MODEL) if generation_evaluated else "retrieval-only",
                "generation_evaluated": generation_evaluated,
            },
            "retrieval_metrics": {
                "recall@1": avg(retrieval["recall@1"]),
                "recall@3": avg(retrieval["recall@3"]),
                "recall@5": avg(retrieval["recall@5"]),
                "recall@10": avg(retrieval["recall@10"]),
                "mrr": avg(retrieval["mrr"]),
                "hit_rate": avg(retrieval["hit_rate"]),
            },
            "generation_metrics": gen_avg,
            "performance_metrics": {
                "total_queries": len(self.DEFAULT_DATASET),
                "avg_latency_sec": round(total_latency / len(self.DEFAULT_DATASET), 3) if self.DEFAULT_DATASET else 0.0,
            },
            "test_cases": results,
        }


eval_service = RAGEvalEngine()
