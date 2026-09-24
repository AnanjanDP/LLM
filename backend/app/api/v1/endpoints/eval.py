import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.core.database import get_db
from app.schemas.eval import EvalExperimentRequest, EvalExperimentResult
from app.services.eval_service import eval_service
from app.services.cache_service import cache_service
from app.models.db_models import EvaluationRun, RequestLog
from app.core.logging import logger

router = APIRouter()


@router.post("/experiment", response_model=EvalExperimentResult)
async def run_experiment(
    payload: EvalExperimentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Run an automated evaluation benchmark experiment and persist results to DB."""
    result = eval_service.run_benchmark_experiment(
        retrieval_mode=payload.retrieval_mode,
        use_reranker=payload.use_reranker,
        top_k=payload.top_k,
        prompt_version=payload.prompt_version,
        model=payload.model
    )

    # Persist EvaluationRun record to database
    try:
        ret_m = result["retrieval_metrics"]
        gen_m = result["generation_metrics"]
        perf_m = result["performance_metrics"]

        db_run = EvaluationRun(
            experiment_id=result["experiment_id"],
            retrieval_mode=payload.retrieval_mode,
            use_reranker=payload.use_reranker,
            top_k=payload.top_k,
            prompt_version=payload.prompt_version,
            model=payload.model or "default",
            recall_at_1=ret_m["recall@1"],
            recall_at_3=ret_m["recall@3"],
            recall_at_5=ret_m["recall@5"],
            recall_at_10=ret_m["recall@10"],
            mrr=ret_m["mrr"],
            hit_rate=ret_m["hit_rate"],
            faithfulness=gen_m["faithfulness"],
            answer_relevance=gen_m["answer_relevance"],
            context_relevance=gen_m["context_relevance"],
            answer_correctness=gen_m["answer_correctness"],
            avg_latency_sec=perf_m["avg_latency_sec"]
        )
        db.add(db_run)
        await db.commit()
    except Exception as e:
        logger.warning(f"Could not persist EvaluationRun to DB: {e}")

    return result


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List historical experiment runs."""
    stmt = select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(limit)
    res = await db.execute(stmt)
    runs = res.scalars().all()

    return [
        {
            "id": r.id,
            "experiment_id": r.experiment_id,
            "retrieval_mode": r.retrieval_mode,
            "use_reranker": r.use_reranker,
            "top_k": r.top_k,
            "prompt_version": r.prompt_version,
            "model": r.model,
            "recall@5": r.recall_at_5,
            "mrr": r.mrr,
            "faithfulness": r.faithfulness,
            "answer_relevance": r.answer_relevance,
            "avg_latency_sec": r.avg_latency_sec,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in runs
    ]


@router.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(db: AsyncSession = Depends(get_db)):
    """Fetch aggregated system metrics (P50, P95, P99 latency, cache stats, evaluation summary)."""
    # Fetch recent request logs for percentile calculation
    stmt = select(RequestLog.total_latency).order_by(desc(RequestLog.created_at)).limit(500)
    res = await db.execute(stmt)
    latencies = [row[0] for row in res.all() if row[0] is not None]

    p50, p95, p99 = 0.0, 0.0, 0.0
    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

    cache_stats = cache_service.get_stats()

    # Fetch latest evaluation run
    stmt_eval = select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(1)
    res_eval = await db.execute(stmt_eval)
    latest_eval = res_eval.scalar_one_or_none()

    latest_metrics = {}
    if latest_eval:
        latest_metrics = {
            "experiment_id": latest_eval.experiment_id,
            "retrieval_mode": latest_eval.retrieval_mode,
            "recall@5": latest_eval.recall_at_5,
            "mrr": latest_eval.mrr,
            "hit_rate": latest_eval.hit_rate,
            "faithfulness": latest_eval.faithfulness,
            "answer_relevance": latest_eval.answer_relevance,
        }

    return {
        "latency_percentiles_sec": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "sample_size": len(latencies)
        },
        "cache_performance": cache_stats,
        "latest_evaluation": latest_metrics
    }
