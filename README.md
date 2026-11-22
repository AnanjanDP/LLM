# RAG Intelligence Platform — Production & Evaluation Suite

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/Frontend-React%2019-61DAFB?logo=react)](https://react.dev)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-blue)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Search-BM25-green)](https://github.com/dorianbrown/rank_bm25)
[![Redis](https://img.shields.io/badge/Cache-Redis-red?logo=redis)](https://redis.io)

A production-grade, measurable Retrieval-Augmented Generation (RAG) platform built with FastAPI, PyTorch/FAISS vector search, BM25 keyword search, Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking, Redis caching, per-stage latency profiling, and a reproducible retrieval benchmark (Recall@K, MRR, hit rate) with lexical proxy scores for generation.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
React 19 SPA Frontend
    │
    ▼  REST / SSE Stream (with Request ID & Auth)
FastAPI Gateway (`backend/app/main.py`)
    │
    ├── Query Optimization (`app/services/query_service.py`)
    │     ├── Query Cleaning & Normalization
    │     └── Vague / Unsupported Query Detection
    │
    ├── Redis Caching Layer (`app/services/cache_service.py`)
    │     ├── Response & Context Cache (TTL: 3600s)
    │     └── Hit Rate & Cache Performance Tracking
    │
    ├── Modular Retrieval Pipeline (`app/services/rag_service.py`)
    │     ┌─────────────────────────────────────────────────┐
    │     │ Dense Vector Search (FAISS + all-MiniLM-L6-v2) │
    │     │ Sparse Keyword Search (rank_bm25 BM25Plus)     │
    │     │          │                                      │
    │     │          ▼                                      │
    │     │ Reciprocal Rank Fusion (RRF, k=60)              │
    │     │          │                                      │
    │     │          ▼                                      │
    │     │ Cross-Encoder Reranking (ms-marco-MiniLM-L6)    │
    │     └─────────────────────────────────────────────────┘
    │
    ├── Prompt Management & Versioning (`app/services/prompt_service.py`)
    │     ├── Versioned Templates (v1.0, v1.1_strict, v2.0_cot)
    │     └── Token Accounting & Cost Estimator
    │
    ├── LLM Provider Engine (`app/services/llm_provider.py`)
    │     ├── Primary: Groq API (`llama-3.3-70b-versatile`)
    │     ├── Fallback models: llama-3.1-8b-instant (on unavailable / rate-limited)
    │     └── Offline mock provider (only when requested or no API key is set)
    │
    └── Evaluation Engine & Metrics Store (`app/services/eval_service.py`)
          ├── IR Metrics: Recall@1/3/5/10, Precision@K, MRR, Hit Rate
          ├── Generation proxies: lexical faithfulness / answer relevance (only with a real LLM)
          └── Per-Stage Latency (P50, P95, P99)
```

---

## 🔄 RAG Pipeline Walkthrough

1. **Document Ingestion & Chunking**: Uploaded `.pdf`, `.txt`, and `.md` files are parsed and split using configurable chunking strategies (`word`, `sentence`, `paragraph`), with adjustable `chunk_size` and `chunk_overlap`.
2. **Dual Indexing**:
   - **Dense Vectors**: `SentenceTransformer('all-MiniLM-L6-v2')` generates 384-d normalized embeddings indexed into FAISS `IndexFlatIP`.
   - **Sparse Keywords**: `BM25Plus` (a BM25 variant that stays well-behaved on very small corpora) is built per session scope over the tokenized chunks.
   - Chunks are persisted as JSON (written atomically); legacy pickled `.npy` stores are migrated on load.
3. **Session-scoped Hybrid Retrieval (RRF)**: An uploaded document is searchable only from its own conversation (plus any shared chunks). The scope is applied *before* ranking: inside FAISS with an id selector, and on the BM25 corpus. Vector and BM25 rankings are fused with Reciprocal Rank Fusion:
   $$RRF\_Score(d) = \sum_{m \in \{Vector, BM25\}} \frac{1}{k_{rrf} + rank_m(d)}$$
4. **Cross-Encoder Reranking**: Candidate chunks are scored with `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` to re-order context snippets based on deep semantic query-document alignment.
5. **Context & Prompt Injection**: The top $K$ context snippets are injected into a versioned system prompt (`v1.0`, `v1.1_strict`, or `v2.0_cot`).
6. **LLM Generation**: Model completion via Groq API (`llama-3.3-70b-versatile`) or SSE streaming.
7. **Evaluation & Profiling**: Latency across every pipeline stage (embedding, vector, BM25, hybrid, reranker, LLM) is logged, token usage & estimated USD cost are computed, and real-LLM answers are cached in Redis (the cache key includes the query, mode, model, prompt, settings, session and index version).

---

## 📊 Benchmark Results

Reproduce with `cd backend && python scripts/run_benchmark.py` (add `--llm` to also score generation; needs `GROQ_API_KEY`). The script indexes `backend/eval_corpus/platform_knowledge.md` into a temporary directory, so it never touches your own documents. It uses 8 questions, `top_k=4`, and reports the mean latency of 3 timed runs (retrieval only, CPU):

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Hit rate | Avg time / query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vector only (FAISS) | 0.708 | 0.975 | 0.975 | 1.000 | 1.000 | 6 ms |
| BM25 only | 0.408 | 0.950 | 0.975 | 0.854 | 1.000 | ~0 ms |
| Hybrid (vector + BM25, RRF) | 0.633 | 0.975 | 1.000 | 1.000 | 1.000 | 5 ms |
| Hybrid + cross-encoder reranker | 0.758 | 1.000 | 1.000 | 1.000 | 1.000 | 20 ms |

**Read these numbers with care:**
* The corpus is small (8 sections) and describes this very project, so every strategy does well; treat the table as a regression check and a way to compare strategies, not as a claim about real-world quality.
* Recall is *keyword coverage*: the share of an expected answer's key terms found in the retrieved chunks. It is not a labelled relevance judgement.
* Faithfulness and answer relevance are **lexical proxies** (word overlap with the context and the question), not an LLM judge. They are `n/a` when no real model is configured, and the API/UI then label the run "retrieval-only".
* Timings depend on the machine; they are per query after a warm-up run.

---

## 🛠️ Technology Stack

* **Backend Framework**: FastAPI + Uvicorn (blocking work runs in a threadpool)
* **Vector Index**: FAISS (`IndexFlatIP` with `all-MiniLM-L6-v2`)
* **Sparse Index**: `rank_bm25` (BM25Plus)
* **Reranker**: SentenceTransformers `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **Caching**: Redis 7 (TTL: 3600s), optional
* **Database**: Async SQLAlchemy 2.0 + SQLite
* **Frontend**: React 19 + Vite
* **Containerization**: Docker & Docker Compose

---

## ⚡ Quickstart & Setup

### Prerequisites
- Python 3.11+
- Node.js 20+
- Redis (optional locally; included in Docker Compose)

### 1. Backend
```bash
git clone https://github.com/AnanjanDP/LLM.git
cd LLM/backend
python -m venv .venv
# Activate: source .venv/bin/activate (Linux/Mac) or .venv\Scriptsctivate (Windows)
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example app/.env     # Windows: copy .env.example app\.env
```
Edit `backend/app/.env`. All settings are documented in `backend/.env.example`. Add `GROQ_API_KEY` for real answers; without it the app uses an offline mock and benchmarks run retrieval-only.

### 3. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive docs: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

### 4. Run the UI
```bash
cd ../frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## 🐳 Docker Compose

```bash
cp .env.example .env         # then set SECRET_KEY (and optionally GROQ_API_KEY)
docker compose up -d --build
```
The app (API and built UI) is served on [http://localhost:8000](http://localhost:8000). The stack runs with `ENVIRONMENT=production`, so it refuses to start without a strong `SECRET_KEY` or with wildcard CORS. Redis is reachable only from the app container; data lives in named volumes.

---

## 🔐 Security & Access Model

* **Accounts**: bcrypt-hashed passwords (8-72 chars), JWT access tokens, rate-limited register/login, lower-cased emails.
* **Conversations**: a signed-in user's conversations are private to them (others get 404). Guest conversations are reachable only via their random UUID, and a guest lists only the ids their browser remembers.
* **Documents**: uploads require a conversation id and are searchable only from it. There is no "clear everything" endpoint; clearing removes one conversation's documents after an ownership check.
* **Uploads**: type allow-list, size cap (`MAX_UPLOAD_MB`), PDF page cap, sanitised filenames.
* **Config**: production refuses the default/short secret and wildcard CORS; unexpected errors return a generic message.
* Known trade-off: the JWT is kept in `localStorage`; an httpOnly cookie would be safer.

---

## 🧪 Testing

```bash
cd backend
python -m pytest        # 57 tests: auth, ownership, uploads, retrieval isolation, eval, config
cd ../frontend
npm run lint && npm run build
```

---

## 📄 Design Decisions & Limitations

### Key Design Decisions
1. **Reciprocal Rank Fusion over raw score summation**: FAISS inner products and BM25 scores live on different scales; RRF fuses ranks (k=60).
2. **Filter before ranking**: session scope is applied inside the FAISS search and on the BM25 corpus, so other sessions can neither leak into nor crowd out results.
3. **Lazy cross-encoder**: the reranker loads on first use to keep startup fast.
4. **Graceful cache**: if Redis is unreachable, requests simply run uncached; mock answers are never cached.
5. **Safe persistence**: chunks are stored as JSON via atomic rename, not pickle.

### Limitations
1. **Single-process index**: the FAISS index and BM25 caches live in one process's memory; scaling out needs a shared vector store.
2. **SQLite and `create_all`**: fine for a demo; use PostgreSQL and Alembic migrations for real deployments.
3. **CPU reranking** adds roughly 15-20 ms per query on the small benchmark and grows with candidate count.
4. **Proxy generation metrics**: for real quality measurement, add an LLM-judge or human-labelled set.
