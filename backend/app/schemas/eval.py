from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvalExperimentRequest(BaseModel):
    retrieval_mode: str = "hybrid"  # vector, bm25, hybrid
    use_reranker: bool = False
    top_k: int = 4
    prompt_version: str = "v1.0"
    model: Optional[str] = None


class RetrievalMetricsSchema(BaseModel):
    recall_at_1: float = Field(..., alias="recall@1")
    recall_at_3: float = Field(..., alias="recall@3")
    recall_at_5: float = Field(..., alias="recall@5")
    recall_at_10: float = Field(..., alias="recall@10")
    mrr: float
    hit_rate: float

    model_config = {
        "populate_by_name": True
    }


class GenerationMetricsSchema(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    answer_correctness: float


class PerformanceMetricsSchema(BaseModel):
    total_queries: int
    avg_latency_sec: float
    p50_latency_sec: Optional[float] = None
    p95_latency_sec: Optional[float] = None
    p99_latency_sec: Optional[float] = None


class EvalExperimentResult(BaseModel):
    experiment_id: str
    config: Dict[str, Any]
    retrieval_metrics: RetrievalMetricsSchema
    generation_metrics: GenerationMetricsSchema
    performance_metrics: PerformanceMetricsSchema
    test_cases: List[Dict[str, Any]]
