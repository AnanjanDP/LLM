import pytest
from app.services.eval_service import eval_service


def test_retrieval_metrics_calculation():
    class DummyCitation:
        def __init__(self, text):
            self.text = text

    citations = [
        DummyCitation("FastAPI is a Python framework"),
        DummyCitation("FAISS enables vector similarity search"),
        DummyCitation("BM25 performs keyword search")
    ]
    expected_keywords = ["fastapi", "faiss"]

    metrics = eval_service._eval_retrieval_query(citations, expected_keywords)

    assert "recall@1" in metrics
    assert "recall@5" in metrics
    assert "mrr" in metrics
    assert "hit_rate" in metrics

    assert metrics["recall@1"] == 0.5  # Only 'fastapi' in rank 1
    assert metrics["recall@3"] == 1.0  # Both 'fastapi' and 'faiss' present in top 3
    assert metrics["mrr"] == 1.0
    assert metrics["hit_rate"] == 1.0


def test_benchmark_experiment_run():
    exp_result = eval_service.run_benchmark_experiment(
        retrieval_mode="hybrid", use_reranker=False, top_k=3, prompt_version="v1.0"
    )

    assert "experiment_id" in exp_result
    assert "retrieval_metrics" in exp_result
    assert "recall@5" in exp_result["retrieval_metrics"]
    assert "faithfulness" in exp_result["generation_metrics"]
    assert len(exp_result["test_cases"]) > 0


@pytest.mark.asyncio
async def test_eval_api_endpoints(client):
    response = await client.post("/api/v1/eval/experiment", json={
        "retrieval_mode": "hybrid",
        "use_reranker": False,
        "top_k": 4,
        "prompt_version": "v1.0"
    })
    assert response.status_code == 200
    data = response.json()
    assert "experiment_id" in data
    assert "retrieval_metrics" in data
    assert "recall@5" in data["retrieval_metrics"]
    assert "faithfulness" in data["generation_metrics"]

    exp_res = await client.get("/api/v1/eval/experiments")
    assert exp_res.status_code == 200
    exp_list = exp_res.json()
    assert len(exp_list) >= 1

    metrics_res = await client.get("/api/v1/eval/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert "latency_percentiles_sec" in metrics
    assert "cache_performance" in metrics
    assert "latest_evaluation" in metrics
