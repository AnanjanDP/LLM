import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from starlette.concurrency import run_in_threadpool
from app.core.config import settings
from app.core.database import get_db
from app.core.limiter import limiter
from app.schemas.eval import EvalExperimentRequest, EvalExperimentResult
from app.services.auth_service import get_current_user_required
from app.services.eval_service import eval_service
from app.services.cache_service import cache_service
from app.models.db_models import EvaluationRun, RequestLog, User
from app.core.logging import logger

router = APIRouter()

RETRIEVAL_ONLY = "retrieval-only"


@router.post("/experiment", response_model=EvalExperimentResult)
@limiter.limit(settings.EVAL_RATE_LIMIT)
async def run_experiment(
    request: Request,
    payload: EvalExperimentRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_required),
):
    """Run the evaluation benchmark and store the result. Signed-in users only: it runs the model and reranker."""
    result = await run_in_threadpool(
        eval_service.run_benchmark_experiment,
        retrieval_mode=payload.retrieval_mode,
        use_reranker=payload.use_reranker,
        top_k=payload.top_k,
        prompt_version=payload.prompt_version,
        model=payload.model,
        evaluate_generation=payload.evaluate_generation,
    )

    try:
        ret_m = result["retrieval_metrics"]
        gen_m = result["generation_metrics"]
        perf_m = result["performance_metrics"]

        db.add(EvaluationRun(
            experiment_id=result["experiment_id"],
            retrieval_mode=payload.retrieval_mode,
            use_reranker=payload.use_reranker,
            top_k=payload.top_k,
            prompt_version=payload.prompt_version,
            model=result["config"]["model"],  # "retrieval-only" when no real model was evaluated
            recall_at_1=ret_m["recall@1"],
            recall_at_3=ret_m["recall@3"],
            recall_at_5=ret_m["recall@5"],
            recall_at_10=ret_m["recall@10"],
            mrr=ret_m["mrr"],
            hit_rate=ret_m["hit_rate"],
            faithfulness=gen_m["faithfulness"] or 0.0,
            answer_relevance=gen_m["answer_relevance"] or 0.0,
            context_relevance=gen_m["context_relevance"] or 0.0,
            answer_correctness=gen_m["answer_correctness"] or 0.0,
            avg_latency_sec=perf_m["avg_latency_sec"]
        ))
        await db.commit()
    except Exception as e:
        logger.warning(f"Could not persist EvaluationRun to DB: {e}")
        await db.rollback()

    return result


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List historical experiment runs."""
    res = await db.execute(select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(limit))
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
            # Runs without a real model have no generation scores; show "not evaluated", not 0.
            "faithfulness": None if r.model == RETRIEVAL_ONLY else r.faithfulness,
            "answer_relevance": None if r.model == RETRIEVAL_ONLY else r.answer_relevance,
            "avg_latency_sec": r.avg_latency_sec,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in res.scalars().all()
    ]


@router.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(db: AsyncSession = Depends(get_db)):
    """Aggregated system metrics: latency percentiles over the last 500 requests, cache stats, latest evaluation."""
    res = await db.execute(select(RequestLog.total_latency).order_by(desc(RequestLog.created_at)).limit(500))
    latencies = [row[0] for row in res.all() if row[0] is not None]

    p50 = p95 = p99 = 0.0
    if latencies:
        p50, p95, p99 = (float(np.percentile(latencies, q)) for q in (50, 95, 99))

    res_eval = await db.execute(select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(1))
    latest_eval = res_eval.scalar_one_or_none()

    latest_metrics = {}
    if latest_eval:
        latest_metrics = {
            "experiment_id": latest_eval.experiment_id,
            "retrieval_mode": latest_eval.retrieval_mode,
            "recall@5": latest_eval.recall_at_5,
            "mrr": latest_eval.mrr,
            "hit_rate": latest_eval.hit_rate,
            "faithfulness": None if latest_eval.model == RETRIEVAL_ONLY else latest_eval.faithfulness,
            "answer_relevance": None if latest_eval.model == RETRIEVAL_ONLY else latest_eval.answer_relevance,
        }

    return {
        "latency_percentiles_sec": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "sample_size": len(latencies)
        },
        "cache_performance": cache_service.get_stats(),
        "latest_evaluation": latest_metrics
    }
