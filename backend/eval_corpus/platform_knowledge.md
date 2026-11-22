# RAG Intelligence Platform: Knowledge Base

## Architecture

The application uses a FastAPI backend and a React frontend built with Vite. Documents are indexed with FAISS for vector search and with BM25 for keyword retrieval. Redis caches chat responses, and a SQLite database stores users, conversations, messages, documents, request logs and evaluation runs through async SQLAlchemy.

## How document indexing works

When a user uploads a PDF, text or Markdown file, the text is split into chunks with a configurable strategy: word windows, sentence windows or paragraphs. Each chunk is turned into a normalised 384 dimensional embedding by a SentenceTransformer model and added to a FAISS IndexFlatIP index, so the inner product equals cosine similarity. The same chunks are also tokenised to build a BM25 sparse index (the BM25 plus variant, which also ranks sensibly on very small corpora). Chunk size and chunk overlap can be set per upload.

## Retrieval modes and reranking

Three retrieval modes are supported: vector search, BM25 keyword search, and hybrid retrieval. Hybrid retrieval runs both searches and merges their rankings with Reciprocal Rank Fusion, which uses ranks instead of scores so the two scoring scales do not need to be normalised. An optional Cross-Encoder reranker then rescores the candidate chunks against the query to improve the final order.

## Latency and performance instrumentation

Every request records latency for each stage: vector search, BM25 search, hybrid fusion, reranking and the language model call. The metrics endpoint reports P50, P95 and P99 latency over recent requests. Chat responses are cached in Redis for an hour, and the cache reports its hit rate, so repeated questions are answered without running retrieval or the model again.

## Session isolation and access control

Uploaded documents belong to the conversation they were uploaded to and are only searched for that conversation. The session filter is applied inside the FAISS search with an id selector and on the BM25 scores before ranking, so other sessions can never crowd out or leak into the results. A conversation created while signed in is private to its owner, and guest conversations are reachable only by their random identifier.

## Authentication and security

Users register with an email and a password that is stored as a bcrypt hash. Login returns a signed JWT access token that expires after a day. Authentication endpoints are rate limited. The server refuses to start in production with the default secret key, and CORS only allows the configured origins.

## Prompts and language model

Prompts are versioned: a standard grounded prompt, a strict prompt that refuses to answer outside the provided context, and a chain of thought prompt. Answers are generated with gpt-oss-120b through the Groq API, with a smaller model as a fallback when the first is unavailable. Token usage and estimated cost are calculated for every response.

## Evaluation

The evaluation engine runs a fixed set of questions against a bundled knowledge base and reports retrieval metrics: Recall at 1, 3, 5 and 10, precision, mean reciprocal rank and hit rate. When a real language model is configured it also reports lexical proxy scores for faithfulness and answer relevance. These measure word overlap with the retrieved context and the question, not human judgement.
