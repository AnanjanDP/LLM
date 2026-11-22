"""Run the bundled retrieval benchmark for every strategy and print a Markdown table.

    python scripts/run_benchmark.py          # retrieval only: no API key needed, fully deterministic
    python scripts/run_benchmark.py --llm    # also scores generation (needs GROQ_API_KEY in backend/app/.env)

The benchmark indexes `eval_corpus/platform_knowledge.md` into a temporary directory, so it never touches your own
uploaded documents. See `RAGEvalEngine` for exactly what each metric means.
"""
import argparse
import os
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--llm", action="store_true", help="also evaluate generation with the configured model")
parser.add_argument("--top-k", type=int, default=4)
parser.add_argument("--runs", type=int, default=3, help="repeat each strategy and report the mean latency")
args = parser.parse_args()

workdir = tempfile.mkdtemp(prefix="rag-bench-")
os.environ["FAISS_INDEX_PATH"] = os.path.join(workdir, "faiss.index")
os.environ["FAISS_DOCS_PATH"] = os.path.join(workdir, "documents.json")
os.environ["CACHE_ENABLED"] = "false"
if not args.llm:
    os.environ["GROQ_API_KEY"] = ""  # an explicit empty value wins over backend/app/.env
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.services.eval_service import eval_service  # noqa: E402

STRATEGIES = [
    ("Vector only (FAISS)", "vector", False),
    ("BM25 only", "bm25", False),
    ("Hybrid (vector + BM25, RRF)", "hybrid", False),
    ("Hybrid + cross-encoder reranker", "hybrid", True),
]

eval_service.ensure_corpus()
rows = []
for label, mode, rerank in STRATEGIES:
    eval_service.run_benchmark_experiment(retrieval_mode=mode, use_reranker=rerank, top_k=args.top_k,
                                          evaluate_generation=args.llm)  # warm-up: loads models, fills caches
    runs = [eval_service.run_benchmark_experiment(retrieval_mode=mode, use_reranker=rerank, top_k=args.top_k,
                                                  evaluate_generation=args.llm) for _ in range(args.runs)]
    result = runs[-1]
    result["performance_metrics"]["avg_latency_sec"] = round(
        sum(r["performance_metrics"]["avg_latency_sec"] for r in runs) / len(runs), 4)
    rows.append((label, result))

n = len(eval_service.DEFAULT_DATASET)
generated = rows[0][1]["config"]["generation_evaluated"]
header = "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Hit rate | Avg time / query |"
rule = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
if generated:
    header = header[:-1] + " Faithfulness (proxy) | Answer relevance (proxy) |"
    rule = rule + " :---: | :---: |"

print(f"\nBenchmark: {n} questions, top_k={args.top_k}, bundled corpus, mean of {args.runs} timed runs\n")
print(header)
print(rule)
for label, r in rows:
    m, p, g = r["retrieval_metrics"], r["performance_metrics"], r["generation_metrics"]
    line = (f"| {label} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | "
            f"{m['hit_rate']:.3f} | {p['avg_latency_sec'] * 1000:.0f} ms |")
    if generated:
        line = line[:-1] + f" {g['faithfulness']:.3f} | {g['answer_relevance']:.3f} |"
    print(line)
print()
