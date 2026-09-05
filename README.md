# DocsQuery

Production-oriented, domain-specific Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, Qdrant, BM25, Sentence Transformers, Cross-Encoder reranking, and Google Gemini.

DocsQuery is designed to answer questions from a controlled document corpus while minimizing hallucinations through retrieval, reranking, evidence-grounded generation, and citation validation.

---

## Table of Contents

* [Overview](#overview)
* [Goals](#goals)
* [Architecture](#architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
* [Document Ingestion](#document-ingestion)
* [Chunking](#chunking)
* [Document and Chunk IDs](#document-and-chunk-ids)
* [Embeddings](#embeddings)
* [Vector Search](#vector-search)
* [BM25 Search](#bm25-search)
* [Hybrid Retrieval](#hybrid-retrieval)
* [Reranking](#reranking)
* [Context Construction](#context-construction)
* [Grounded Generation](#grounded-generation)
* [Citation Validation](#citation-validation)
* [API](#api)
* [Evaluation System](#evaluation-system)
* [Retrieval Metrics](#retrieval-metrics)
* [Answer Metrics](#answer-metrics)
* [Quality Gates](#quality-gates)
* [Evaluation Dataset](#evaluation-dataset)
* [Corpus](#corpus)
* [Configuration](#configuration)
* [Environment Variables](#environment-variables)
* [Running the Project](#running-the-project)
* [Running Tests](#running-tests)
* [Development Checks](#development-checks)
* [Logging and Observability](#logging-and-observability)
* [Health and Readiness](#health-and-readiness)
* [Security Considerations](#security-considerations)
* [Current Limitations](#current-limitations)
* [Production Roadmap](#production-roadmap)
* [Design Principles](#design-principles)

---

# Overview

DocsQuery is a domain-specific RAG application.

Instead of giving a language model unrestricted access to arbitrary information, DocsQuery follows a controlled pipeline:

```text
User Query
    |
    v
FastAPI
    |
    v
RAGService
    |
    v
RetrievalService
    |
    +------------------+
    |                  |
    v                  v
  BM25             Vector Search
    |                  |
    +--------+---------+
             |
             v
          RRF Fusion
             |
             v
       Cross-Encoder
          Reranker
             |
             v
       Top-K Evidence
             |
             v
       Context Builder
       [C1] [C2] [C3]
             |
             v
          Gemini
             |
             v
     Citation Validator
             |
             v
        Final Answer
```

The important architectural principle is:

> The LLM is the final reasoning and language layer, not the primary source of truth.

The source of truth is the retrieved document evidence.

---

# Goals

DocsQuery is designed around the following goals:

### 1. Grounded answers

Answers should be based on retrieved document content rather than model memory.

### 2. Hybrid retrieval

Use both:

* semantic vector search
* lexical BM25 search

This allows the system to handle both conceptual queries and exact technical terminology.

### 3. High-quality retrieval

Use Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking to improve the final evidence ranking.

### 4. Citation-aware generation

Retrieved chunks are assigned citation IDs such as:

```text
[C1]
[C2]
[C3]
```

The generated answer is expected to cite the evidence it uses.

### 5. Evaluation-driven development

Retrieval and answer quality are measured instead of relying only on manual inspection.

### 6. Production-oriented architecture

Long-lived models and services are initialized once instead of being recreated for every request.

### 7. Reproducibility

Evaluation runs should record enough information to determine:

* which dataset was used
* which corpus was indexed
* which retrieval configuration was used
* which models were used
* what metrics were produced

---

# Technology Stack

| Area                     | Technology                  |
| ------------------------ | --------------------------- |
| Language                 | Python 3.12                 |
| API                      | FastAPI                     |
| Validation/configuration | Pydantic                    |
| PDF extraction           | pypdf                       |
| Embeddings               | Sentence Transformers       |
| Vector database          | Qdrant                      |
| Lexical retrieval        | BM25                        |
| Hybrid fusion            | Reciprocal Rank Fusion      |
| Reranking                | Cross-Encoder               |
| LLM                      | Google Gemini               |
| LLM SDK                  | google-genai                |
| Testing                  | pytest                      |
| Linting                  | Ruff                        |
| Formatting               | Ruff                        |
| Packaging                | setuptools                  |
| Evaluation               | Custom evaluation framework |
| Containerization         | Docker                      |
| CI                       | GitHub Actions              |

---

# Project Structure

```text
DocsQuery/
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── middleware.py
│   │   └── routes.py
│   │
│   ├── config/
│   │   └── ...
│   │
│   ├── evaluation/
│   │   ├── answer_correctness.py
│   │   ├── answer_evaluator.py
│   │   ├── answer_metrics.py
│   │   ├── answer_models.py
│   │   ├── baseline.py
│   │   ├── context.py
│   │   ├── dataset.py
│   │   ├── diagnostics.py
│   │   ├── e2e_evaluator.py
│   │   ├── e2e_results.py
│   │   ├── evaluator.py
│   │   ├── groundedness.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── quality_gate.py
│   │   └── results.py
│   │
│   ├── generation/
│   │   ├── context_builder.py
│   │   ├── citation_validator.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── ...
│   │
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── loader.py
│   │   ├── models.py
│   │   └── ...
│   │
│   ├── retrieval/
│   │   ├── bm25_index.py
│   │   ├── bm25_storage.py
│   │   ├── hybrid_retriever.py
│   │   ├── index_manager.py
│   │   ├── reranker.py
│   │   ├── vector_retriever.py
│   │   └── ...
│   │
│   ├── services/
│   │   └── rag_service.py
│   │
│   ├── container.py
│   ├── logging_config.py
│   └── main.py
│
├── configs/
│   └── evaluation.yaml
│
├── data/
│   ├── raw/
│   │   └── *.pdf
│   │
│   ├── index/
│   │   └── bm25.json
│   │
│   └── evaluation/
│       ├── retrieval_dataset.json
│       └── retrieval_baseline.json
│
├── scripts/
│   ├── analyze_retrieval.py
│   ├── benchmark_summary.py
│   ├── check_evaluation_quality.py
│   ├── corpus_stats.py
│   ├── create_retrieval_baseline.py
│   ├── evaluate_rag.py
│   ├── evaluate_retrieval.py
│   ├── ingest_corpus.py
│   ├── inspect_chunks.py
│   ├── rag_summary.py
│   ├── search.py
│   ├── search_bm25.py
│   ├── search_hybrid.py
│   └── test_gemini.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
└── README.md
```

---

# How the RAG Pipeline Works

The complete request path is:

```text
POST /api/v1/query
        |
        v
   Request validation
        |
        v
     RAGService
        |
        v
 RetrievalService
        |
        +------------------------+
        |                        |
        v                        v
      BM25                  Vector Search
        |                        |
        +-----------+------------+
                    |
                    v
              RRF Fusion
                    |
                    v
             Cross Encoder
               Reranker
                    |
                    v
               Top Results
                    |
                    v
            ContextBuilder
                    |
                    v
                 Gemini
                    |
                    v
           Citation Validation
                    |
                    v
               API Response
```

---

# Document Ingestion

PDF documents are processed using:

```text
PDF
 |
 v
pypdf
 |
 v
Extracted text
 |
 v
Text cleaning
 |
 v
Chunking
 |
 v
Document chunks
 |
 +---------> BM25
 |
 +---------> Embeddings
                 |
                 v
               Qdrant
```

The main ingestion flow is implemented under:

```text
app/ingestion/
```

The corpus ingestion script is:

```bash
python -m scripts.ingest_corpus data/raw
```

The directory is scanned deterministically so ingestion order is stable.

---

# Chunking

DocsQuery uses word-based chunking with overlap.

The overlap provides contextual continuity between neighboring chunks.

Conceptually:

```text
Chunk 1:
word word word word word word

Chunk 2:
              word word word word word word

Chunk 3:
                        word word word word word word
```

Without overlap, information near chunk boundaries can be lost.

---

# Document and Chunk IDs

Document IDs are deterministic SHA-256 identifiers.

The goal is:

```text
same document
    =
same document ID
```

Chunk IDs are globally unique within a document:

```text
<document_id>-chunk-<global_index>
```

Example:

```text
7f0b67701fa7...-chunk-0
7f0b67701fa7...-chunk-1
7f0b67701fa7...-chunk-2
```

This is important because chunk IDs are referenced by:

* retrieval results
* evaluation datasets
* citations
* diagnostics
* baselines

Changing ID semantics would invalidate evaluation data.

---

# Embeddings

Semantic embeddings are generated using Sentence Transformers.

Embeddings allow queries and document chunks to be compared in vector space.

For example:

```text
Query:
"How do I create a new branch?"

        |
        v

Embedding

        |
        v

Vector similarity

        |
        v

Relevant Git chunks
```

Embedding generation is handled by the embedding service and reused across retrieval operations.

---

# Vector Search

Qdrant is used as the vector database.

The vector index contains:

* embedding vector
* chunk metadata
* document information
* chunk text
* chunk ID

Vector search handles semantic similarity.

It is particularly useful when the query and source document do not use exactly the same words.

Example:

```text
Query:
"remove a local branch"

Document:
"git branch -d <branch>"
```

A semantic retriever can recognize the conceptual relationship even when the wording differs.

---

# BM25 Search

BM25 provides lexical retrieval.

It is stored persistently in:

```text
data/index/bm25.json
```

BM25 is particularly useful for:

* exact command names
* technical terminology
* flags
* function names
* identifiers
* uncommon keywords

For example:

```text
git rebase --interactive
```

is often better handled by lexical matching than purely semantic search.

---

# Hybrid Retrieval

DocsQuery combines:

```text
BM25
+
Vector Search
```

using Reciprocal Rank Fusion (RRF).

The current architecture uses:

```text
RRF k = 60
```

The basic idea is:

```text
score(document) =
    sum(
        1 / (k + rank)
    )
```

for documents appearing in the individual ranked lists.

The result is a combined ranking containing evidence discovered by either strategy.

---

# Reranking

After hybrid retrieval, candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

A Cross-Encoder evaluates the query and candidate document together rather than comparing independent embeddings.

The pipeline therefore becomes:

```text
BM25
   \
    \
     -> RRF -> Cross Encoder -> final ranking
    /
   /
Vector
```

The reranker is loaded once by the application container so it is not initialized on every request.

---

# Context Construction

Retrieved chunks are converted into a structured generation context.

Each retrieved item receives a citation ID:

```text
[C1]
[C2]
[C3]
```

Conceptually:

```text
[C1]
Source: progit.pdf
Chunk: ...
Text:
...

[C2]
Source: git_cheat_sheet.pdf
Chunk: ...
Text:
...
```

This allows the LLM to reference evidence explicitly.

The same `ContextBuilder` is used by production generation and evaluation to avoid creating two different context representations.

---

# Grounded Generation

Google Gemini is used as the generation model.

The model is instructed to:

1. Answer using supplied evidence.
2. Avoid unsupported claims.
3. Cite factual claims.
4. Use only citation IDs that exist.
5. State when the evidence is insufficient.

The prompt is implemented in:

```text
app/generation/prompts.py
```

The Gemini client is implemented in:

```text
app/generation/llm.py
```

---

# Citation Validation

Generated responses are checked after generation.

The validator checks:

### Citation validity

Does every referenced citation ID actually exist?

Example:

```text
Answer:
Git creates a branch using git branch [C1].
```

If `[C1]` does not exist in the supplied context, the answer is invalid.

### Citation coverage

Factual-looking answer sentences should contain supporting citations.

This is intentionally conservative.

The validator is a heuristic rather than a perfect semantic citation judge.

---

# API

## Health

```http
GET /health
```

Used for liveness.

It answers whether the application process is running.

---

## Readiness

```http
GET /ready
```

Used for readiness.

It verifies required dependencies such as:

* persisted BM25 index
* Qdrant connectivity/state

---

## Search

```http
POST /api/v1/search
```

Used when the caller wants retrieval results without LLM generation.

Typical flow:

```text
query
 |
 v
BM25 + Vector
 |
 v
RRF
 |
 v
Reranker
 |
 v
results
```

---

## Query

```http
POST /api/v1/query
```

Runs the full RAG pipeline:

```text
query
 |
 v
retrieval
 |
 v
reranking
 |
 v
context
 |
 v
Gemini
 |
 v
citation validation
 |
 v
answer
```

The response also includes timing information.

---

# Evaluation System

DocsQuery does not treat a single successful API response as proof that the system works.

There are two major evaluation layers.

## 1. Retrieval evaluation

Measures whether the correct chunks are retrieved.

## 2. End-to-end answer evaluation

Measures whether the generated answer is:

* supported by evidence
* properly cited
* grounded
* semantically aligned with the expected answer

The evaluation framework is located under:

```text
app/evaluation/
```

---

# Retrieval Metrics

The current retrieval evaluator supports:

## Recall@K

Measures how much of the relevant evidence appears in the top K results.

```text
Recall@K =
relevant retrieved chunks
-------------------------
all relevant chunks
```

---

## Precision@K

Measures how many of the retrieved results are relevant.

```text
Precision@K =
relevant retrieved chunks
-------------------------
K
```

---

## MRR

Mean Reciprocal Rank measures how quickly the first relevant result appears.

For example:

```text
first relevant result = rank 1

RR = 1 / 1 = 1.0
```

If it appears at rank 5:

```text
RR = 1 / 5 = 0.2
```

---

## nDCG@K

Normalized Discounted Cumulative Gain rewards relevant results appearing near the top of the ranking.

It is useful when ranking quality matters, rather than only whether a relevant document appears somewhere in the top K.

---

# Retrieval Strategies Evaluated

The benchmark compares:

```text
1. BM25
2. Vector
3. Hybrid RRF
4. Hybrid RRF + Cross-Encoder
```

This allows us to determine whether each architectural layer actually improves retrieval.

---

# Retrieval Diagnostics

The system can also produce query-level diagnostics.

These classify whether individual strategies:

* found relevant evidence
* missed relevant evidence
* improved over another strategy

This is useful when a metric changes but we need to understand why.

---

# Retrieval Baseline

A known-good retrieval benchmark is stored in:

```text
data/evaluation/retrieval_baseline.json
```

The baseline is tracked in Git.

Generated benchmark outputs are not treated as source-controlled truth.

The baseline provides a reference point for regression detection.

---

# Answer Metrics

The answer evaluation layer measures:

## Citation validity

Whether the answer only uses valid citation IDs.

Target:

```text
very close to 1.0
```

---

## Citation coverage

Measures whether factual answer content is adequately cited.

---

## Groundedness

A separate NLI model is used during evaluation:

```text
cross-encoder/nli-MiniLM2-L6-H768
```

This model is evaluation-only.

It should not be considered part of the production generation stack.

The current implementation estimates how much answer content is entailed by the retrieved evidence.

---

## Correctness

An embedding-based semantic similarity score is used as a cheap correctness signal.

This is deliberately considered a weak evaluation signal.

Semantic similarity is not equivalent to factual correctness.

A future production evaluation system should ideally use a stronger reference-based judge or domain-specific evaluator.

---

# Evaluation Dataset

The retrieval evaluation dataset is stored in:

```text
data/evaluation/retrieval_dataset.json
```

The dataset contains information such as:

```json
{
  "example_id": "q001",
  "query": "What does git init do?",
  "query_type": "repository-basics",
  "relevant_chunk_ids": [
    "actual-chunk-id"
  ],
  "reference_answer": "..."
}
```

The dataset version is explicitly tracked.

Chunk IDs must correspond to actual indexed chunks.

Placeholder IDs must never be used in the real benchmark.

---

# Quality Gates

Evaluation results can be checked against configured thresholds.

Configuration:

```text
configs/evaluation.yaml
```

Current engineering thresholds are:

```yaml
retrieval:
  max_mrr_drop: 0.05
  max_ndcg_at_5_drop: 0.05
  max_recall_at_5_drop: 0.05

answer:
  min_citation_validity: 0.95
  min_citation_coverage: 0.90
  min_groundedness: 0.80
  min_correctness: 0.70
```

These are initial engineering thresholds, not scientifically validated universal standards.

A future iteration should tune them using a larger representative benchmark.

---

# Corpus

The current benchmark corpus is Git documentation.

Current known PDFs include:

```text
data/raw/git_cheat_sheet.pdf
data/raw/progit.pdf
```

The corpus intentionally remains domain-specific.

A large unrelated book such as:

```text
Hands-On Machine Learning.pdf
```

is not part of the Git benchmark.

That PDF currently does not provide usable text through the normal `pypdf` extraction path and therefore requires a future OCR/alternative extraction pipeline.

---

# Configuration

Project-level configuration is handled through the application settings system.

Environment-specific values should not be hard-coded into Python modules.

Use:

```text
.env
```

for local development.

A template is provided:

```text
.env.example
```

---

# Environment Variables

Important variables include:

```text
GEMINI_API_KEY=
GEMINI_MODEL=
GEMINI_TEMPERATURE=
GEMINI_MAX_TOKENS=
```

Example:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_TEMPERATURE=0
GEMINI_MAX_TOKENS=1000
```

Secrets must never be committed to Git.

---

# Running the Project

## Create environment

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## Install project

```bash
pip install -e .
```

---

## Install development dependencies

Use the dependency configuration from `pyproject.toml`.

---

# Ingest the Corpus

First place PDFs into:

```text
data/raw/
```

Then run:

```bash
python -m scripts.ingest_corpus data/raw
```

Check corpus statistics:

```bash
python -m scripts.corpus_stats
```

---

# Inspect Chunks

Example:

```bash
python -m scripts.inspect_chunks
```

This is useful for verifying:

* extraction quality
* chunk boundaries
* metadata
* chunk IDs
* source document information

---

# BM25 Search

```bash
python -m scripts.search_bm25
```

---

# Hybrid Search

```bash
python -m scripts.search_hybrid
```

---

# Vector Search

```bash
python -m scripts.search
```

---

# Run the API

Start FastAPI using the project's configured application entrypoint.

Typical development command:

```bash
uvicorn app.main:app --reload
```

---

# Test Gemini

Real API smoke test:

```bash
python -m scripts.test_gemini
```

This requires a valid:

```text
GEMINI_API_KEY
```

Unit tests should not call the live Gemini API.

---

# Running Tests

Run the entire test suite:

```bash
pytest -v
```

Run unit tests:

```bash
pytest tests/unit -v
```

Run integration tests:

```bash
pytest tests/integration -v
```

---

# Development Checks

The project uses Ruff for linting and formatting.

Lint:

```bash
ruff check .
```

Format:

```bash
ruff format .
```

Full project check:

```bash
make check
```

---

# Evaluation Commands

## Retrieval evaluation

```bash
python -m scripts.evaluate_retrieval
```

---

## Retrieval diagnostics

```bash
python -m scripts.analyze_retrieval
```

---

## Retrieval benchmark summary

```bash
python -m scripts.benchmark_summary
```

---

## Create/update retrieval baseline

```bash
python -m scripts.create_retrieval_baseline
```

---

## End-to-end RAG evaluation

```bash
python -m scripts.evaluate_rag
```

This requires the live Gemini API.

---

## RAG evaluation summary

```bash
python -m scripts.rag_summary
```

---

## Evaluation quality gate

```bash
python -m scripts.check_evaluation_quality
```

A quality regression causes a non-zero exit code.

This is intended to become part of CI.

---

# Logging and Observability

The API includes request-level observability.

Middleware adds:

```text
X-Request-ID
X-Process-Time
```

The RAG service tracks timing for major operations such as:

```text
retrieval latency
generation latency
total request latency
```

The application has centralized logging configuration.

This makes future production monitoring easier.

---

# Health and Readiness

The project separates:

### Liveness

```http
GET /health
```

from:

### Readiness

```http
GET /ready
```

This is important for deployment systems such as Docker and Kubernetes.

An application can be alive while still not being ready to serve requests because a dependency is unavailable.

---

# Error Handling

Unexpected API exceptions are handled centrally.

The API should not leak:

* stack traces
* API secrets
* internal exception details
* implementation-specific information

to clients.

Expected application errors are converted into appropriate HTTP responses.

---

# Security Considerations

The production system should enforce:

* secret management through environment variables or secret stores
* CORS restrictions
* request validation
* payload size limits
* PDF upload validation
* rate limiting
* authentication if deployed for non-public users
* safe logging that does not expose secrets
* safe exception handling
* dependency updates
* non-root container execution

Security hardening is not yet considered complete.

---

# Current Limitations

The following areas are known limitations rather than hidden assumptions.

## 1. Citation validation is heuristic

The current citation validator uses conservative sentence-level rules.

It does not fully understand whether each citation semantically supports every individual claim.

---

## 2. Groundedness evaluation is not a perfect judge

The NLI evaluator is an engineering signal.

It should not be treated as an authoritative factuality score.

---

## 3. Correctness evaluation is approximate

Embedding similarity can identify obviously similar answers but cannot reliably determine whether every factual statement is correct.

---

## 4. PDF extraction is currently text-first

Scanned/image PDFs are not handled by the normal ingestion path.

OCR is a future capability.

---

## 5. BM25 persistence needs a production storage strategy

Local:

```text
data/index/bm25.json
```

is appropriate for development.

A deployed multi-instance system needs a more durable shared-storage or rebuild strategy.

---

## 6. Qdrant persistence must be production-grade

Local development storage is not sufficient as the final deployment architecture.

A managed or persistent Qdrant deployment should be used in production.

---

## 7. Live LLM evaluation costs API quota

End-to-end answer evaluation uses Gemini and therefore should be managed carefully in CI.

Unit and deterministic retrieval tests should remain API-free.

---

# Production Roadmap

Current development is organized approximately as follows:

```text
11.1  Retrieval evaluation foundations
11.2  Retrieval metrics
11.3  Evaluation dataset
11.4  Retrieval diagnostics
11.5  Retrieval baseline
11.6  Answer evaluation
11.7  Groundedness/correctness evaluation
11.8  End-to-end RAG evaluation
11.9  Quality gates
11.10 Reproducible evaluation runs
11.11 CI/CD
11.12 Docker
11.13 Production storage
11.14 Deployment
11.15 Frontend
11.16 Security hardening
11.17 Monitoring/observability
11.18 Final end-to-end validation
11.19 Documentation and release cleanup
```

---

# Design Principles

## Evidence before generation

The model should not be expected to know the answer before retrieval.

---

## Retrieval is measurable

Every retrieval strategy should be benchmarked.

---

## Fail closed when evidence is insufficient

The system should prefer:

```text
"I don't have enough evidence to answer this."
```

over an unsupported confident answer.

---

## Deterministic identifiers

Stable document and chunk IDs make evaluation and debugging reproducible.

---

## Long-lived dependencies

Heavy models should be loaded once and reused.

---

## Evaluation is part of the product

Quality checks should eventually run automatically in CI/CD.

---

## Production constraints are considered early

The architecture should account for:

* memory usage
* model initialization cost
* persistent storage
* deployment health checks
* secrets
* observability
* reproducibility

---

# Project Status

DocsQuery currently has the core RAG pipeline implemented:

```text
PDF ingestion
      ✓
Text cleaning
      ✓
Chunking
      ✓
Deterministic IDs
      ✓
Embeddings
      ✓
Qdrant
      ✓
BM25
      ✓
Hybrid retrieval
      ✓
RRF
      ✓
Cross-Encoder reranking
      ✓
Context builder
      ✓
Gemini generation
      ✓
Citation validation
      ✓
FastAPI
      ✓
Health/readiness
      ✓
Retrieval evaluation
      ✓
Answer evaluation
      ✓
Quality gates
      ✓
Reproducible evaluation runs
      → in progress
CI/CD
      → next
Docker
      → next
Production deployment
      → next
Frontend
      → next
Security hardening
      → next
Monitoring
      → next
Final E2E validation
      → next
```

---

# Philosophy

DocsQuery is not intended to be "just a chatbot."

It is an engineering project focused on building a measurable, testable, deployable RAG system.

The target architecture is:

```text
Reliable ingestion
        +
Reliable retrieval
        +
Strong ranking
        +
Grounded generation
        +
Citation validation
        +
Automated evaluation
        +
Regression protection
        +
Production deployment
```

That combination is what turns a basic RAG prototype into a production-oriented RAG system.
