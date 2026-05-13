# Archon - FastAPI RAG System

## What This Is
This project is a retrieval-augmented generation (RAG) system built with FastAPI, PostgreSQL, pgvector, and OpenAI models. It ingests documents, chunks and embeds them, retrieves semantically relevant context for a user query, and generates grounded answers with citation and reliability checks. The goal is not just to answer questions from a corpus, but to make the answer path debuggable, cost-aware, and measurable.

## Architecture
```text
                +----------------------+
                |      Client/API      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   FastAPI /ask       |
                | request_id middleware|
                +----------+-----------+
                           |
                 +---------+---------+
                 |                   |
                 v                   v
      +-------------------+   +-------------------+
      | Semantic Cache    |   | Query Embedding   |
      | Postgres+pgvector |   | OpenAI embeddings |
      +---------+---------+   +---------+---------+
                |                         |
          cache hit/miss                  v
                |               +-------------------+
                |               | Vector Retrieval  |
                |               | embeddings table  |
                |               +---------+---------+
                |                         |
                |                         v
                |               +-------------------+
                |               | Prompt Builder    |
                |               | context + rules   |
                |               +---------+---------+
                |                         |
                |                         v
                |               +-------------------+
                |               | LLM Generation    |
                |               | gpt-4o-mini       |
                |               +---------+---------+
                |                         |
                |                         v
                |               +-------------------+
                |               | Validation Layer  |
                |               | citations/fallback|
                |               +---------+---------+
                |                         |
                +-------------> +-------------------+
                                | JSON Response     |
                                +-------------------+

Ingestion path:
Client -> /documents/ingest -> chunk_text -> batch embeddings -> batch DB insert
```

## Key Design Decisions

### 1. Foundation
What this stage implements:
- FastAPI service with request/response models and a lifespan-managed database pool.
- PostgreSQL + pgvector as the primary storage for embeddings.
- OpenAI embedding integration for both document ingestion and query embedding.

Tradeoffs considered:
- Managed vector DB vs. Postgres + pgvector.
- Separate services for ingestion and query vs. one app boundary.

Why this approach was chosen:
- PostgreSQL + pgvector was chosen over a separate vector database so retrieval, cache state, and debugging all live in one place during early iteration.
- A single service keeps the first version easy to reason about, while the code is still split into modules so the system can be broken apart later if needed.

### 2. Ingestion
What this stage implements:
- `/documents/ingest` endpoint for end-to-end ingestion.
- Token-based chunking with configurable `chunk_size` and `chunk_overlap`.
- Batch embedding and batch insert into the `embeddings` table.

Tradeoffs considered:
- Character chunking vs. token chunking.
- Very small chunks for precision vs. larger chunks for context continuity.

Why this approach was chosen:
- Chunking is token-based rather than character-based because model limits and retrieval quality are both governed by tokens, not raw text length.
- The initial `500` token chunk size with overlap is a practical baseline: large enough to preserve context, small enough to avoid bloated prompts, and configurable because chunking is one of the biggest silent drivers of RAG quality.

### 3. Retrieval + Generation
What this stage implements:
- Semantic retrieval over pgvector using cosine similarity.
- Prompt builder isolated in `prompt.py`.
- LLM answer generation isolated in `generation.py`.
- `/ask` endpoint that runs the full RAG flow.

Tradeoffs considered:
- Exact-match retrieval vs. vector search.
- Hardcoded prompts in the endpoint vs. prompt assembly in its own module.
- Streaming response vs. buffered response.

Why this approach was chosen:
- Retrieval is treated as the first accuracy gate, because in RAG most answer quality is won or lost before generation starts.
- Prompt construction is kept separate and versioned because prompt changes alter system behavior and should invalidate semantic cache entries.
- The endpoint currently buffers responses instead of streaming because post-generation citation validation needs access to the full answer before it can be trusted.

### 4. Reliability Layer
What this stage implements:
- Confidence fallback using top retrieval similarity score.
- Citation instructions in the system prompt.
- Citation parsing and fabricated-citation rejection.
- Eval harness with behavior-based assertions.

Tradeoffs considered:
- Trust the model’s answer directly vs. force citation behavior.
- Always call the LLM vs. reject weak retrieval early.
- Unit-test exact strings vs. assert properties of LLM outputs.

Why this approach was chosen:
- The reliability layer is designed around the actual failure modes of RAG systems: weak retrieval, unsupported claims, and confident-looking fabricated citations.
- The system is intentionally conservative. It is better to refuse a weakly grounded answer than to return something polished but false.

### 5. Cost Layer
What this stage implements:
- Semantic cache in Postgres + pgvector.
- Compound cache key based on query embedding, generation model, and prompt-template hash.
- Cache metadata for debugging and future policy tuning.

Tradeoffs considered:
- Redis vs. Postgres for the first cache layer.
- Exact string cache vs. semantic cache.
- Aggressive threshold vs. conservative threshold.

Why this approach was chosen:
- The first semantic cache lives in Postgres rather than Redis because correctness and inspectability matter more than raw speed at this stage.
- The cache threshold starts conservative at `0.95` because a false cache hit is more damaging than a slow miss; a bad cached answer bypasses the rest of the safety pipeline.
- Prompt hash and model name are part of the cache key because a response is only reusable under the same generation conditions that produced it.

### 6. Polish
What this stage implements:
- Dockerized app service plus pgvector service in `docker-compose.yml`.
- Structured JSON logs with request-scoped tracing.
- Request IDs attached through middleware and returned via response header.

Tradeoffs considered:
- Ad hoc `print()` debugging vs. structured logs.
- Running the app as Postgres superuser vs. least-privilege app user.

Why this approach was chosen:
- LLM systems need richer debugging than standard CRUD APIs because the failure surface includes retrieval quality, prompt construction, cache behavior, and model output.
- Request-scoped structured logs are the minimum viable observability layer because they make it possible to trace one bad answer or one cost spike across the full pipeline.

## Reliability Layer
This system currently has three explicit defenses against hallucination:

### 1. Confidence Fallback
Before calling the LLM, `/ask` checks the top retrieval similarity score. If retrieval is weak, the request short-circuits to a hardcoded fallback instead of spending tokens on an answer that is likely to be unsupported.

Why:
- Weak retrieval is an upstream reliability failure, not something generation can usually fix.
- It is cheaper and safer to refuse early than to pay for an answer that is likely to hallucinate or eventually say "not enough information" anyway.

### 2. Citation Validation
The prompt requires inline citations like `[Chunk 1]`. After generation, the answer is parsed and validated. If the model cites a chunk number outside the retrieved range, the answer is rejected as not reliably grounded.

Why:
- This catches one concrete hallucination mode: the model inventing source references that were never retrieved.
- It does not fully prove claim-level support, but it blocks a class of failures that should never reach a user-facing answer.

### 3. Eval Suite
The eval harness checks behavioral properties instead of exact strings:
- happy path grounded answer
- out-of-scope fallback
- ambiguous query with uncertainty
- fabricated citation rejection

Why:
- LLM outputs are non-deterministic, so exact-match tests are brittle and often misleading.
- What matters is whether the system behaves correctly under important scenarios: answer when it should, refuse when it should, and surface grounding signals consistently.

## Cost Engineering

### Semantic Cache Design
The semantic cache lives in Postgres using pgvector. Each cache row stores:
- original query text
- query embedding
- response text
- response metadata
- model name
- prompt template hash
- access metadata such as hit count and last accessed time

### Compound Cache Key
The cache is not keyed by exact query string. It is effectively:
- semantic similarity on `query_embedding`
- AND same `model_name`
- AND same `prompt_template_hash`

This prevents reusing answers across prompt or model changes, while still allowing paraphrased queries to hit the cache.

### Threshold Rationale
The cache threshold starts at `0.95`.

Why so high:
- false cache hits are dangerous because they bypass retrieval and generation entirely
- early in the system’s life, correctness matters more than hit rate
- a semantic cache should earn trust slowly rather than optimize for aggressive reuse too early

The intended tuning loop is:
- log `(query, matched_query, similarity, outcome)`
- sample cache hits
- manually verify whether the cached answer was actually reusable

## Observability
This project uses structured JSON logging with a request-scoped `request_id`.

Every request gets:
- a UUID generated in FastAPI middleware
- `request_started` and `request_completed` events
- request duration in milliseconds

The `/ask` path logs:
- query preview
- cache hit or miss
- retrieval score summary
- generation start and completion
- low-similarity rejection
- fabricated-citation rejection
- response return

The cache layer logs:
- semantic cache hit
- semantic cache miss
- semantic cache store

In production, I would monitor:
- request count and latency by endpoint
- retrieval top-score distribution
- fallback rate
- fabricated citation rejection rate
- cache hit rate
- average response length
- LLM call volume by model
- cost per request and cost per successful grounded answer

Why this matters:
- When an answer is wrong, the debugging question is not just "what did the model say?" but also "what did retrieval return, what prompt was built, and why did the safety checks allow it through?"
- When cost spikes, the system needs enough tracing to attribute spend back to specific request patterns, cache misses, and model calls.

## What’s Next / Known Gaps
- Retrieval currently uses vector similarity only; no reranking layer is implemented yet.
- Citation validation only checks that cited chunk numbers exist; it does not yet verify claim-to-chunk support.
- Semantic cache invalidation on corpus updates is not implemented yet.
- Token budgeting is not implemented yet; `/ask` still forwards all retrieved chunks in the current top-k.
- There is a TODO to consolidate the duplicate OpenAI client initialization into one shared module.
- Expired semantic cache rows are filtered out on read but not physically cleaned up yet.
- Eval coverage is mocked and behavior-focused; it is not yet a corpus-backed benchmark suite.

## Setup
Required environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
DB_HOST=postgres
DB_PORT=5432
DB_NAME=embeddings_db
DB_USER=app_user
DB_PASSWORD=app_password
```

Start the system:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000` and Postgres at `localhost:5432`.
