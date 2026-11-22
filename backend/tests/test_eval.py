import pytest
from app.services.eval_service import eval_service, EVAL_SCOPE
from app.services.rag_service import rag_service


class DummyCitation:
    def __init__(self, text):
        self.text = text


def test_retrieval_metrics_calculation():
    citations = [
        DummyCitation("FastAPI is a Python framework"),
        DummyCitation("FAISS enables vector similarity search"),
        DummyCitation("BM25 performs keyword search"),
    ]
    metrics = eval_service._eval_retrieval_query(citations, ["fastapi", "faiss"])

    assert metrics["recall@1"] == 0.5   # only 'fastapi' in rank 1
    assert metrics["recall@3"] == 1.0   # both present in the top 3
    assert metrics["mrr"] == 1.0
    assert metrics["hit_rate"] == 1.0
    assert eval_service._eval_retrieval_query([], ["fastapi"])["hit_rate"] == 0.0


def test_generation_proxies_are_plain_overlap_with_no_bonus():
    # An answer sharing no content words with the context must score 0, not a flattering constant.
    scores = eval_service._eval_generation_quality(
        query="describe retrieval", context="alpha beta gamma delta", answer="completely unrelated sentence here",
        ground_truth="alpha beta"
    )
    assert scores["faithfulness"] == 0.0
    assert scores["answer_relevance"] == 0.0
    assert scores["context_relevance"] == 0.0

    perfect = eval_service._eval_generation_quality(
        query="alpha beta", context="alpha beta gamma", answer="alpha beta gamma", ground_truth="alpha beta"
    )
    assert perfect["faithfulness"] == 1.0
    assert eval_service._eval_generation_quality("q", "c", "", "gt")["faithfulness"] == 0.0


def test_benchmark_runs_on_its_own_corpus_without_a_language_model():
    result = eval_service.run_benchmark_experiment(retrieval_mode="hybrid", use_reranker=False, top_k=4)

    assert result["config"]["generation_evaluated"] is False      # no real model in the tests
    assert result["config"]["model"] == "retrieval-only"
    assert all(v is None for v in result["generation_metrics"].values())  # "not evaluated", never a made-up number
    assert result["retrieval_metrics"]["hit_rate"] == 1.0
    assert result["retrieval_metrics"]["recall@5"] > 0.5
    assert len(result["test_cases"]) == len(eval_service.DEFAULT_DATASET)


def test_user_uploads_cannot_change_benchmark_results():
    baseline = eval_service.run_benchmark_experiment(retrieval_mode="bm25", top_k=4)
    # Documents full of the benchmark's own keywords, in the shared scope and in another session.
    rag_service.add_document("fastapi faiss bm25 react sqlite redis " * 50, doc_name="noise-shared")
    rag_service.add_document("fastapi faiss bm25 react sqlite redis " * 50, doc_name="noise-session", session_id="s1")
    after = eval_service.run_benchmark_experiment(retrieval_mode="bm25", top_k=4)
    assert after["retrieval_metrics"] == baseline["retrieval_metrics"]


def test_corpus_is_indexed_once():
    first = eval_service.ensure_corpus()
    assert first > 0
    assert eval_service.ensure_corpus() == first
    assert rag_service.clear_index(EVAL_SCOPE) == first


def test_each_retrieval_mode_finds_the_relevant_chunks():
    for mode in ("vector", "bm25", "hybrid"):
        result = eval_service.run_benchmark_experiment(retrieval_mode=mode, top_k=4)
        assert result["retrieval_metrics"]["hit_rate"] == 1.0, mode


@pytest.mark.asyncio
async def test_eval_endpoints_require_a_signed_in_user_to_run(client, alice):
    body = {"retrieval_mode": "hybrid", "use_reranker": False, "top_k": 4, "prompt_version": "v1.0"}
    assert (await client.post("/api/v1/eval/experiment", json=body)).status_code == 401

    response = await client.post("/api/v1/eval/experiment", json=body, headers=alice)
    assert response.status_code == 200
    data = response.json()
    assert data["config"]["generation_evaluated"] is False
    assert data["generation_metrics"]["faithfulness"] is None
    assert "recall@5" in data["retrieval_metrics"]

    # Reading the history stays open, and retrieval-only runs show no generation score (not a fake 0).
    runs = (await client.get("/api/v1/eval/experiments")).json()
    assert len(runs) == 1 and runs[0]["faithfulness"] is None

    metrics = (await client.get("/api/v1/eval/metrics")).json()
    assert "latency_percentiles_sec" in metrics and "cache_performance" in metrics
    assert metrics["latest_evaluation"]["faithfulness"] is None


@pytest.mark.asyncio
async def test_eval_request_is_validated(client, alice):
    bad = await client.post("/api/v1/eval/experiment", json={"retrieval_mode": "nonsense"}, headers=alice)
    assert bad.status_code == 422
    bad = await client.post("/api/v1/eval/experiment", json={"top_k": 1000}, headers=alice)
    assert bad.status_code == 422
