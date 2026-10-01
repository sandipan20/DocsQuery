# DocsQuery

<div align="center">

![DocsQuery Architecture](https://img.shields.io/badge/Architecture-Hybrid%20RAG-00f2fe?style=for-the-badge&logo=diagramsdotnet)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React 19](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini%203.1%20Flash%20Lite-8E75C2?style=for-the-badge&logo=google&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-285%20Passed-success?style=for-the-badge&logo=pytest)

**An enterprise-grade, domain-specific Retrieval-Augmented Generation (RAG) system engineered for high-precision document intelligence, strict citation validation, and zero-hallucination grounded generation.**

[Beginner's Guide](#1-for-beginners-what-is-docsquery-how-does-it-work--how-was-it-built) • [Architecture](#3-system-architecture) • [Quick Start](#10-quick-start-guide) • [API Reference](#11-api-reference--curl-cookbook) • [Evaluation System](#12-evaluation--benchmarking-suite) • [Deployment](#14-production-deployment-guide)

</div>

---

## Table of Contents

- [1. For Beginners: What is DocsQuery, How Does It Work, & How Was It Built?](#1-for-beginners-what-is-docsquery-how-does-it-work--how-was-it-built)
  - [1.1 The Real-World Problem: Why Standard AI Fails on Documents](#11-the-real-world-problem-why-standard-ai-fails-on-documents)
  - [1.2 The 4-Actor Analogy: How DocsQuery Solves It](#12-the-4-actor-analogy-how-docsquery-solves-it)
  - [1.3 The Life of a PDF: From File Upload to Cited Answer](#13-the-life-of-a-pdf-from-file-upload-to-cited-answer)
  - [1.4 The Blueprint: How DocsQuery Was Built Step-by-Step](#14-the-blueprint-how-docsquery-was-built-step-by-step)
  - [1.5 Beginner's Glossary of Key Terms](#15-beginners-glossary-of-key-terms)
- [2. Executive Overview](#2-executive-overview)
- [3. System Architecture](#3-system-architecture)
- [4. End-to-End Pipeline Deep Dive](#4-end-to-end-pipeline-deep-dive)
  - [4.1 Ingestion & Document Processing](#41-ingestion--document-processing)
  - [4.2 Deterministic Identity & Chunking Strategy](#42-deterministic-identity--chunking-strategy)
  - [4.3 Dense Semantic Search (Qdrant)](#43-dense-semantic-search-qdrant)
  - [4.4 Sparse Lexical Search (BM25 Okapi)](#44-sparse-lexical-search-bm25-okapi)
  - [4.5 Reciprocal Rank Fusion (RRF)](#45-reciprocal-rank-fusion-rrf)
  - [4.6 Neural Cross-Encoder Reranking](#46-neural-cross-encoder-reranking)
  - [4.7 Evidence Context Construction](#47-evidence-context-construction)
  - [4.8 Grounded Generation with Google Gemini](#48-grounded-generation-with-google-gemini)
  - [4.9 Dual-Layer Citation Validation & Self-Correction](#49-dual-layer-citation-validation--self-correction)
  - [4.10 Step-by-Step Data Transformation & Latency Budget](#410-step-by-step-data-transformation--latency-budget)
- [5. Mathematical Formulations & Worked Examples](#5-mathematical-formulations--worked-examples)
  - [5.1 Cosine Similarity Formulation & Numerical Example](#51-cosine-similarity-formulation--numerical-example)
  - [5.2 BM25 Okapi Lexical Relevance](#52-bm25-okapi-lexical-relevance)
  - [5.3 Reciprocal Rank Fusion (RRF) Step-by-Step Calculation](#53-reciprocal-rank-fusion-rrf-step-by-step-calculation)
  - [5.4 Mean Reciprocal Rank (MRR) & nDCG@K](#54-mean-reciprocal-rank-mrr--ndcgk)
- [6. Grounded Prompt Engineering & Self-Correction Specifications](#6-grounded-prompt-engineering--self-correction-specifications)
- [7. Anonymous Session & Multi-Tenant Security Model](#7-anonymous-session--multi-tenant-security-model)
- [8. Cyber/Holo Frontend Experience & Design System](#8-cyberholo-frontend-experience--design-system)
- [9. Complete Project Structure](#9-complete-project-structure)
- [10. Quick Start Guide](#10-quick-start-guide)
  - [Local Development Setup](#local-development-setup)
  - [Running with Docker Compose](#running-with-docker-compose)
- [11. API Reference & cURL Cookbook](#11-api-reference--curl-cookbook)
- [12. Evaluation & Benchmarking Suite](#12-evaluation--benchmarking-suite)
  - [Evaluation Dataset Schema](#evaluation-dataset-schema)
  - [Quality Gate Enforcement](#quality-gate-enforcement)
- [13. CLI Scripts & Utilities Catalog](#13-cli-scripts--utilities-catalog)
- [14. Production Deployment Guide](#14-production-deployment-guide)
  - [Render Deployment (Web Service & Static Frontend)](#render-deployment-web-service--static-frontend)
  - [Single-Container Production Image](#single-container-production-image)
  - [Qdrant Cloud Configuration](#qdrant-cloud-configuration)
  - [Resource & Memory Optimization](#resource--memory-optimization)
- [15. Configuration & Environment Variables](#15-configuration--environment-variables)
- [16. Development, Testing & Code Quality](#16-development-testing--code-quality)
- [17. Troubleshooting & Operational Playbook](#17-troubleshooting--operational-playbook)
- [18. License & Credits](#18-license--credits)

---

# 1. For Beginners: What is DocsQuery, How Does It Work, & How Was It Built?

If you have **zero prior knowledge** about AI, vector databases, or backend engineering, this section is designed specifically for you.

---

### 1.1 The Real-World Problem: Why Standard AI Fails on Documents

Imagine you work at a company with a 500-page technical manual or legal contract. You ask ChatGPT or a standard AI:
> *"What is the exact penalty clause under Section 4.2 in our agreement?"*

A standard Large Language Model (LLM) faces three severe problems:
1. **Hallucination**: LLMs are text-prediction engines. If they don't know the exact answer, they will invent a believable, confident-sounding lie.
2. **Context Limits & Cost**: You cannot feed entire encyclopedias or dozens of PDFs into an AI prompt every time someone asks a question—it is too slow, too expensive, and causes the AI to forget the middle parts.
3. **No Proof (Lack of Citations)**: A generic chatbot gives you an answer, but cannot prove which sentence or page in your document backs up its statement.

**DocsQuery solves this entirely.** It acts as an unbreakable bridge between your private documents and the AI.

---

### 1.2 The 4-Actor Analogy: How DocsQuery Solves It

Think of DocsQuery as a team of 4 specialized experts working together whenever you ask a question:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. THE LIBRARIAN (Hybrid Retrieval: Qdrant + BM25)                     │
 │ Searches the library using two strategies:                             │
 │ • Meaning Search: "Find paragraphs talking about branch management"    │
 │ • Exact Word Search: "Find the exact command 'git checkout -b'"        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (Returns ~40 candidate pages)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. THE SENIOR JUDGE (Neural Cross-Encoder Reranker)                    │
 │ Carefully compares your exact question against each candidate page.    │
 │ Discards 35 irrelevant pages and keeps the TOP 5 absolute best proofs. │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (Top-5 Evidence Cards)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. THE LEGAL CLERK (Context Builder & Citation Validator)              │
 │ Pastes citation stickers on the proof ([C1], [C2], [C3]) and hands     │
 │ them to the Professor with strict orders: "Use ONLY these stickers!"   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. THE PROFESSOR (Google Gemini 3.1 Flash Lite)                        │
 │ Writes a clear, fluent answer citing every fact:                       │
 │ "To switch branches, run git checkout -b [C1] or git switch -c [C2]."  │
 │ If the proof doesn't have the answer, says: "I don't know."            │
 └────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 The Life of a PDF: From File Upload to Cited Answer

Here is what happens inside DocsQuery step-by-step:

```text
[Step 1: Upload]
User drops "user_guide.pdf" into the browser
   │
   ▼
[Step 2: Text Extraction & Cleaning]
Backend extracts text from each page, strips broken formatting and extra spaces.
   │
   ▼
[Step 3: Chunking (Flashcards)]
Text is sliced into 200-word "chunks" with a 40-word overlap so sentences aren't cut in half.
   │
   ▼
[Step 4: Fingerprinting & Vectorization]
Each chunk gets a unique SHA-256 fingerprint and is converted into a 384-number mathematical
embedding (like a GPS coordinate for its meaning).
   │
   ▼
[Step 5: Storing]
• Coordinates stored in Qdrant (Vector DB)
• Exact words stored in BM25 (Keyword Index)
   │
   ▼
[Step 6: User Asks a Question]
User types: "How do I reset my password?"
   │
   ▼
[Step 7: Dual Search & Fusion]
• Vector DB finds chunks with similar concepts.
• Keyword index finds chunks with the exact word "reset" and "password".
• Reciprocal Rank Fusion (RRF) blends both lists fairly.
   │
   ▼
[Step 8: AI Reranking]
A specialized neural model reads the question and chunks together, scoring their true relevance.
   │
   ▼
[Step 9: Answer Generation & Validation]
Google Gemini writes the answer with [C1] citation tags.
The backend inspects the citations: if valid, it displays the answer and shows the evidence cards!
```

---

### 1.4 The Blueprint: How DocsQuery Was Built Step-by-Step

If you wanted to build DocsQuery from scratch, here is the complete engineering journey:

1. **Step 1: Foundation & PDF Ingestion**:
   - We used Python 3.12 and `pypdf` to read PDF pages into raw strings.
   - Built `cleaner.py` to sanitize control characters, headers, and footers.
2. **Step 2: Intelligent Sliding-Window Chunking**:
   - Built `chunker.py` to slide a 200-word window with a 40-word overlap across the document.
   - Computed deterministic SHA-256 hashes (`<document_hash>-chunk-<index>`) so IDs never change between server restarts.
3. **Step 3: Vector Database (Qdrant) & Embeddings**:
   - Integrated HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` to turn text chunks into 384-dimensional dense vectors.
   - Connected Qdrant to store vectors with metadata (`workspace_id`, `document_id`, `page_number`, `text`).
4. **Step 4: Lexical BM25 Search**:
   - Implemented an in-memory BM25 Okapi search engine with JSON serialization (`bm25.json`) to handle exact codes, function names, and technical terms.
5. **Step 5: Hybrid Fusion & Neural Reranking**:
   - Combined vector and BM25 results using Reciprocal Rank Fusion (RRF $k=60$).
   - Added `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank top candidates based on joint attention.
6. **Step 6: Grounded Prompt & LLM Generation**:
   - Formatted evidence into `[C1]`, `[C2]` blocks.
   - Crafted strict system prompts for Google Gemini 3.1 Flash Lite forbidding external hallucinations.
7. **Step 7: Citation Safety Net & Self-Correction**:
   - Built `citation_validator.py` to verify every citation tag against the context.
   - Built an automated self-correction loop in `generation_service.py` that asks the LLM to fix invalid citations automatically.
8. **Step 8: Multi-Tenant Session Isolation**:
   - Created HTTP-only `docsquery_session` cookie authentication.
   - Enforced database-level payload filtering so User A can never see User B's documents.
9. **Step 9: Futuristic Cyber/Holo Web Frontend**:
   - Built a React 19 + TypeScript + Vite single-page application with Cyber Dark & Holo Light themes, telemetry HUD, drag-and-drop file upload, and citation preview cards.
10. **Step 10: Rigorous Testing & Production Deployment**:
    - Wrote 285+ automated unit and integration tests with `pytest`.
    - Created Docker multi-stage build and Render Blueprint (`render.yaml`).

---

### 1.5 Beginner's Glossary of Key Terms

| Term | Simple Definition |
| :--- | :--- |
| **RAG** (Retrieval-Augmented Generation) | Giving an AI an "open book" to read before asking it to answer your question, instead of relying on its memory. |
| **LLM** (Large Language Model) | The AI that writes natural language answers (in our case, Google Gemini 3.1 Flash Lite). |
| **Embedding / Vector** | A list of numbers representing the "meaning" of a sentence. Similar meanings have numbers close to each other. |
| **Vector DB (Qdrant)** | A specialized database designed to search through millions of numerical embeddings in milliseconds. |
| **BM25** | The classic algorithm behind search engines like Google that looks for exact matching words and counts how rare they are. |
| **Reranker (Cross-Encoder)** | A deep-learning model that reads a question and an answer together to decide if they truly match. |
| **Chunking** | Slicing a huge 100-page PDF into small, digestible paragraphs (chunks). |
| **Hallucination** | When an AI makes up false information and presents it as a confident fact. |
| **Groundedness** | How well an answer is supported by real proof in the source documents. |
| **Citation `[C1]`** | A marker linking a specific claim in the AI's answer to the exact page and paragraph it came from. |

---

# 2. Executive Overview

**DocsQuery** is a production-hardened Retrieval-Augmented Generation (RAG) platform. Unlike naive RAG demos that pass top-K vector matches directly into an unconstrained language model prompt, DocsQuery is built on a fundamental principle:

> **The Large Language Model is the final synthesis and reasoning layer—never the primary source of factual truth. The immutable source of truth is the retrieved, reranked, and validated document evidence.**

### Core Pillars of DocsQuery

1. **Zero-Hallucination Grounding**: Every factual assertion must be attributed to an explicitly retrieved chunk ID (e.g., `[C1]`, `[C2]`). If evidence is insufficient, the system gracefully fails closed.
2. **Hybrid Search Synergy**: Vector embeddings capture high-level semantic intent, while BM25 Okapi captures exact keywords, flags, acronyms, and technical symbols (`git checkout -b`, `CVE-2024-XXXX`). Reciprocal Rank Fusion (RRF) synergizes both.
3. **Cross-Encoder Precision**: Bi-encoders compress documents into fixed vectors; Cross-Encoders perform full joint token-level cross-attention over `(query, document)` pairs, eliminating irrelevant false positives before LLM context construction.
4. **Automated Citation Self-Correction**: When an LLM output fails syntactic or sentence-level citation validation, DocsQuery triggers an automated corrective feedback loop that forces the model to repair its citation mappings without user intervention.
5. **Multi-Tenant Privacy by Design**: Out-of-the-box support for anonymous browser sessions using cryptographically generated HTTP-only cookies. Documents uploaded in one session are completely isolated at the database index layer from all other sessions.
6. **Production Observability & Telemetry**: Millisecond-level latency breakdowns for retrieval, reranking, and generation attached to every response with standard `X-Request-ID` and `X-Process-Time` tracing headers.

---

# 3. System Architecture

```mermaid
flowchart TD
    User([User Browser]) <-->|HTTP / Cookie Session| Frontend[React 19 + TypeScript Cyber UI]
    Frontend <-->|REST API JSON| API[FastAPI Application]
    
    subgraph Ingestion Pipeline
        PDF[PDF Document] --> Extractor[pypdf Parser]
        Extractor --> Cleaner[Text Cleaner & Normalizer]
        Cleaner --> Chunker[Sliding Window Word Chunker]
        Chunker --> Hasher[Deterministic SHA-256 Hasher]
        Hasher --> DenseEmbed[Sentence Transformers all-MiniLM-L6-v2]
        Hasher --> SparseIndex[BM25 Okapi Indexer]
        DenseEmbed --> Qdrant[(Qdrant Vector DB)]
        SparseIndex --> BM25Storage[(BM25 JSON Storage)]
    end

    subgraph Retrieval & Ranking Engine
        QueryReq[User Query] --> EmbedQuery[Dense Embedding Generator]
        QueryReq --> TokenizeQuery[BM25 Query Tokenizer]
        EmbedQuery -->|Vector Search| Qdrant
        TokenizeQuery -->|Lexical Search| BM25Storage
        Qdrant -->|Dense Candidates| RRF[Reciprocal Rank Fusion k=60]
        BM25Storage -->|Sparse Candidates| RRF
        RRF -->|Fused Top-N| CrossEncoder[Cross-Encoder Reranker ms-marco-MiniLM-L-6-v2]
        CrossEncoder -->|Scored & Filtered| TopEvidence[Top-K Evidence Candidates]
    end

    subgraph Generation & Verification Engine
        TopEvidence --> CtxBuilder[Structured Context Builder [C1], [C2]...]
        CtxBuilder --> PromptGen[Strict Grounded Prompt]
        PromptGen --> Gemini[Google Gemini 3.1 Flash Lite]
        Gemini --> Validator{Citation Validator}
        Validator -->|Valid Citations| SuccessResp[Final Grounded Answer]
        Validator -->|Validation Failed| SelfCorrect[Self-Correction Feedback Loop]
        SelfCorrect --> Gemini
    end

    API --> QueryReq
    SuccessResp --> API
```

---

# 4. End-to-End Pipeline Deep Dive

### 4.1 Ingestion & Document Processing
Documents uploaded via `POST /api/v1/documents` or indexed offline via `scripts.ingest_corpus` follow an exhaustive multi-stage processing pipeline:

1. **Validation & Security Scans**:
   - Magic byte signature verification (`%PDF-`).
   - MIME type and file extension validation.
   - Max file size enforcement (default: 25 MiB per file).
   - Max file count enforcement (up to 10 files per batch).
2. **Text Extraction (`app.ingestion.loader`)**:
   - High-throughput page-by-page extraction using `pypdf`.
   - Structural metadata capture (page numbers, total page counts, file dimensions).
3. **Text Cleaning & Normalization (`app.ingestion.cleaner`)**:
   - Stripping non-printable characters and control byte sequences.
   - Normalizing multi-character whitespace, carriage returns, and tabs into clean single spaces.
   - Removing repeated pagination headers/footers and stray hyphenations.

### 4.2 Deterministic Identity & Chunking Strategy
DocsQuery relies on **deterministic, idempotent identification** to prevent duplicate chunks and guarantee 100% reproducible evaluation runs:

- **Document ID**: Hex-encoded SHA-256 hash computed over the raw PDF binary content:
  $$\text{doc\_id} = \text{SHA256}(\text{raw\_pdf\_bytes})$$
- **Chunk ID**: Formatted with document hash and zero-indexed ordinal:
  $$\text{chunk\_id} = \text{doc\_id}\text{-chunk-}N$$
- **Sliding Window Chunking (`app.ingestion.chunker`)**:
  - `chunk_size`: 200 words (configurable).
  - `chunk_overlap`: 40 words (20% overlap).
  - Sliding overlap ensures that boundary concepts (e.g., a function signature split across sentences) remain semantically intact in adjacent chunks.

```text
Document: [Word 1, Word 2, ... Word 200, Word 201, ... Word 360, Word 361, ...]
Chunk 0:  [Word 1 ---------------------> Word 200]
Chunk 1:                [Word 161 ---------------------> Word 360]
Chunk 2:                                [Word 321 ---------------------> Word 520]
```

### 4.3 Dense Semantic Search (Qdrant)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimensionality**: 384 dimensions.
- **Distance Metric**: Cosine Similarity.
- **Storage**: Qdrant Vector Database (Local embedded, Docker container, or Qdrant Cloud).
- **Point Payload**: Contains `chunk_id`, `document_id`, `workspace_id`, `text`, `page_number`, `source_filename`, and character start/end offsets.
- **Payload Indexing**: Indexed fields on `workspace_id` and `document_id` for sub-millisecond multi-tenant filtering.

### 4.4 Sparse Lexical Search (BM25 Okapi)
Dense embeddings can struggle with specialized alphanumeric tokens, CLI flags, exact configuration keys, and error codes. DocsQuery implements a parallel BM25 Okapi retrieval engine:
- **Tokenizer**: Regex-based tokenization with lowercasing and punctuation stripping.
- **Inverted Index**: Memory-mapped inverted index with term frequencies ($f(q_i, D)$) and document lengths ($|D|$).
- **Partitioning**: Dynamic partition filtering by `workspace_id` and selected `document_ids`.

### 4.5 Reciprocal Rank Fusion (RRF)
To merge candidates from disparate scoring distributions (cosine similarity in $[-1, 1]$ vs. unbounded BM25 scores), DocsQuery uses **Reciprocal Rank Fusion** with constant $k=60$:

$$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}$$

Where $r_m(d)$ is the 1-based rank of document $d$ in retrieval system $m$. RRF prevents either dense or lexical scores from dominating candidate selection.

### 4.6 Neural Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Function**: Performs full bidirectional cross-attention across the concatenated token sequence `[CLS] Query [SEP] Chunk Text [SEP]`.
- **Score Calculation**: Outputs an unconstrained logit representing relevance.
- **Dynamic Thresholding**:
  - In anonymous private workspaces: Chunks with cross-encoder scores $\ge 0.25$ are retained (falling back to top-3 if below threshold).
  - In public corpus benchmark mode: Stricter thresholding ($\ge 0.54$) ensures only high-confidence evidence enters the context window.

### 4.7 Evidence Context Construction
The `ContextBuilder` formats retrieved chunks into an unambiguous, labeled prompt block:

```text
[C1] Source: git-scm-guide.pdf (Page 14)
Chunk ID: 3a8f...-chunk-2
Content:
Git stores its data as a series of snapshots of a miniature filesystem...

[C2] Source: progit.pdf (Page 42)
Chunk ID: 7b2c...-chunk-18
Content:
To create a new branch and switch to it immediately, run git checkout -b <branch-name>...
```

### 4.8 Grounded Generation with Google Gemini
- **Model**: `gemini-3.1-flash-lite` (or user-configured `GEMINI_MODEL`).
- **Temperature**: `0.0` (strictly deterministic generation).
- **System Instructions**:
  1. Base every claim *only* on the provided context evidence.
  2. Append citation tags (e.g., `[C1]`, `[C2]`) to every factual statement.
  3. Never mention citation markers that are not in the context.
  4. If the provided context does not contain sufficient information to answer the question, explicitly state: *"I do not have enough evidence to answer this question based on the provided documents."*

### 4.9 Dual-Layer Citation Validation & Self-Correction
DocsQuery enforces strict post-generation citation validation:
1. **Syntactic Check**: Verifies that all citations in the response match the regular expression `\[C\d+\]` and exist in the supplied context block.
2. **Coverage Check**: Assesses the ratio of non-trivial factual sentences containing valid citations.
3. **Refusal Recognition**: If the model politely refuses due to insufficient evidence, citation checks are bypassed to prevent false-positive errors.
4. **Self-Correction Feedback Loop**: If validation fails (e.g. hallucinated citation `[C99]`), `GenerationService` sends the error diagnostics back to Gemini with instructions to correct the citations.

### 4.10 Step-by-Step Data Transformation & Latency Budget

| Stage | Input Representation | Output Representation | Latency Budget | Error Handling |
| :--- | :--- | :--- | :--- | :--- |
| **1. Session Resolution** | `docsquery_session` Cookie | `workspace_id: UUID` | $< 1\,\text{ms}$ | Auto-generates new session if missing |
| **2. Dense Retrieval** | `query: str` | Top-20 Qdrant Points (384-d vectors) | $10\text{--}25\,\text{ms}$ | Empty fallback if collection is empty |
| **3. BM25 Retrieval** | `query: str` (tokenized) | Top-20 Lexical Match Chunks | $2\text{--}5\,\text{ms}$ | Empty fallback if index empty |
| **4. RRF Fusion** | 2 Ranked Lists (max 40 candidates) | Fused Top-10 Ranked Candidates | $< 1\,\text{ms}$ | Union of unique chunk IDs |
| **5. Cross-Encoder Rerank** | `List[(query, chunk_text)]` | Scored & Threshold-Filtered Chunks | $15\text{--}35\,\text{ms}$ | Preserves top-3 if all fall below threshold |
| **6. Context Builder** | Top-K Scored Chunks | Formatted Context String `[C1]...` | $< 1\,\text{ms}$ | Truncates to max prompt token limits |
| **7. LLM Generation** | System Prompt + Context + Query | Raw LLM Answer String | $200\text{--}450\,\text{ms}$ | Exponential backoff on 503/429 errors |
| **8. Citation Validation** | Raw Answer + Valid Citations List | Validated Answer + Structured Citations | $< 2\,\text{ms}$ | 1x Self-Correction retry on validation failure |

---

# 5. Mathematical Formulations & Worked Examples

### 5.1 Cosine Similarity Formulation & Numerical Example
$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

**Worked Example:**
Let query embedding $\mathbf{u} = [0.6, 0.8, 0.0]$ and document chunk embedding $\mathbf{v} = [0.0, 0.6, 0.8]$:
1. Dot product: $\mathbf{u} \cdot \mathbf{v} = (0.6)(0) + (0.8)(0.6) + (0)(0.8) = 0.48$
2. Norms: $\|\mathbf{u}\| = \sqrt{0.6^2 + 0.8^2 + 0^2} = 1.0$, $\|\mathbf{v}\| = \sqrt{0^2 + 0.6^2 + 0.8^2} = 1.0$
3. Cosine Similarity: $S_C = \frac{0.48}{1.0 \times 1.0} = 0.4800$

### 5.2 BM25 Okapi Lexical Relevance
For a query $Q = \{q_1, q_2, \dots, q_n\}$ and document $D$:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$

*Constants*: $k_1 = 1.5$, $b = 0.75$, $N = \text{total chunks in workspace}$, $n(q_i) = \text{chunks matching term } q_i$.

### 5.3 Reciprocal Rank Fusion (RRF) Step-by-Step Calculation
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k = 60$.

**Worked Example:**
Suppose we retrieve candidates for query *"How to rebase branch"* across Dense and BM25 search:

| Chunk ID | Dense Rank ($r_{\text{dense}}$) | BM25 Rank ($r_{\text{BM25}}$) | Dense RRF Term ($\frac{1}{60 + r_D}$) | BM25 RRF Term ($\frac{1}{60 + r_B}$) | Final RRF Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chunk A** | 1 | 4 | $\frac{1}{61} \approx 0.01639$ | $\frac{1}{64} \approx 0.01563$ | **$0.03202$ (Rank 1)** |
| **Chunk B** | 2 | Not in Top-20 | $\frac{1}{62} \approx 0.01613$ | $0.00000$ | **$0.01613$ (Rank 3)** |
| **Chunk C** | 5 | 1 | $\frac{1}{65} \approx 0.01538$ | $\frac{1}{61} \approx 0.01639$ | **$0.03177$ (Rank 2)** |

*Notice that Chunk A, which scored well in both retrieval methods, rises to Rank 1 over Chunk C, demonstrating how RRF surfaces high-confidence consensus candidates.*

### 5.4 Mean Reciprocal Rank (MRR) & nDCG@K
- **MRR**: Measures how quickly the first relevant chunk appears across query evaluations:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **nDCG@K**: Measures overall ranking quality with logarithmic position discounting:
  $$\text{DCG}_K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}$$

---

# 6. Grounded Prompt Engineering & Self-Correction Specifications

### 6.1 System Prompt Template (`app/generation/prompts.py`)
```text
You are DocsQuery, a precise, domain-specific AI assistant.
Your task is to answer the user's question using ONLY the provided document evidence chunks.

Strict Rules:
1. Base every factual statement solely on the provided context evidence.
2. Every factual statement or sentence MUST be followed by its supporting citation tag, e.g., [C1], [C2].
3. Do NOT use any citation tags that are not present in the provided context.
4. Do NOT invent, extrapolate, or introduce external knowledge not present in the context.
5. If the context does not contain enough information to answer the question, state:
   "I do not have enough evidence to answer this question based on the provided documents."

Context Evidence:
{context_text}

User Question: {query}

Grounded Answer (with citations):
```

### 6.2 Self-Correction Feedback Loop Prompt
When `CitationValidator` detects an invalid citation (e.g. `[C8]` referenced when only `[C1]`, `[C2]` exist) or ungrounded sentences, `GenerationService` prompts the model with targeted diagnostic feedback:

```text
Your previous answer failed strict citation verification:
Validation Errors: {validation_error_summary}

Please rewrite your answer to fix these errors:
- Use ONLY citations from the valid set: {valid_citation_ids}
- Ensure every factual assertion is cited with an exact bracketed tag [C#].
- If a claim cannot be cited from the valid context, remove that claim entirely.

Previous Answer:
{previous_answer}

Corrected Grounded Answer:
```

---

# 7. Anonymous Session & Multi-Tenant Security Model

DocsQuery features an anonymous session isolation architecture:

```text
Browser Client                    FastAPI Middleware                     Qdrant / BM25
      │                                   │                                    │
      │ ─── 1. GET /api/v1/session ─────> │                                    │
      │                                   │ ── Generate UUIDv4 Workspace ID ── │
      │ <── 2. Set-Cookie: docsquery... ──│                                    │
      │                                   │                                    │
      │ ─── 3. POST /api/v1/documents ──> │                                    │
      │    (Cookie: docsquery_session)    │ ── Extract workspace_id ─────────> │
      │                                   │    Index points with payload:      │
      │                                   │    { workspace_id: "abc-123" }     │
      │                                   │                                    │
      │ ─── 4. POST /api/v1/query ──────> │                                    │
      │    (Cookie: docsquery_session)    │ ── Apply Filter: ────────────────> │
      │                                   │    MustMatch(workspace_id)         │
      │                                   │    (Zero cross-tenant leakage)     │
```

- **HTTP-Only Cookies**: Session tokens are transmitted via the `docsquery_session` cookie (`SameSite=Lax`, `HttpOnly=True`, `Max-Age=86400`, `Secure` in production).
- **Backend-Enforced Authorization**: Workspace IDs submitted in request bodies or query params are strictly ignored. The active workspace is resolved solely from the validated session cookie.
- **Complete Data Isolation**: Qdrant vector point IDs for private sessions are deterministically hashed as `UUID5(workspace_id, chunk_id)` to prevent collision across sessions even if users upload identical files.
- **Targeted Deletions**: Deleting a document executes a scoped payload filter requiring *both* `workspace_id` AND `document_id`.

---

# 8. Cyber/Holo Frontend Experience & Design System

The DocsQuery user interface is built with **React 19**, **Vite**, and **TypeScript**, designed with a futuristic command-center aesthetic.

### Key Visual & Interactive Highlights:
1. **Dynamic Theme Switcher**:
   - **Holo Light Theme**: Crisp, luminescent porcelain white canvas with neon cyan and cobalt accents, frosted glass cards, and high-contrast typography.
   - **Cyber Dark Theme**: Immersive deep onyx HUD with glowing matrix cyan borders, emerald indicators, and backdrop blur glassmorphism.
2. **Real-Time Telemetry HUD**:
   - Live system status pills (`SYSTEM ONLINE`, active session hash).
   - Document inventory monitor (total documents, selected count).
   - Millisecond-precise latency telemetry (`Retrieval: 42ms | Generation: 310ms | Total: 352ms`).
3. **Interactive Document Workspace**:
   - Drag-and-drop multi-PDF uploader with upload progress bars and instant validation error notifications.
   - Document checkbox selector allowing users to target queries to all documents or a specific subset.
   - One-click document deletion with automatic index cleanup.
4. **Query Directive Assistant**:
   - Quick-action directive chips:
     - 💡 *Explain Core Concepts*
     - 🛠️ *Step-by-Step Tutorial*
     - 🔍 *Troubleshoot & Fix Errors*
     - ⚡ *CLI & Command Reference*
5. **Interactive Evidence Viewer**:
   - Citation chips (`[C1]`, `[C2]`) in the generated answer link directly to collapsible evidence cards.
   - Evidence cards display source document name, page number, similarity/reranking score badge, and one-click text copy.

---

# 9. Complete Project Structure

```text
DocsQuery/
├── app/
│   ├── __init__.py
│   ├── container.py                 # Dependency Injection container for singletons
│   ├── logging_config.py            # Structured logging configuration
│   ├── main.py                      # FastAPI app entrypoint & static frontend mount
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── errors.py                # Global exception handlers & RFC-7807 error models
│   │   ├── middleware.py            # Request ID, timing & session resolution middleware
│   │   └── routes.py                # FastAPI route endpoints (/query, /documents, etc.)
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Pydantic Settings management (.env loader)
│   │
│   ├── evaluation/
│   │   ├── answer_correctness.py    # Semantic similarity correctness judge
│   │   ├── answer_evaluator.py      # End-to-end answer quality evaluation orchestrator
│   │   ├── answer_metrics.py        # Answer metric calculations (validity, coverage)
│   │   ├── answer_models.py         # Pydantic models for evaluation queries & runs
│   │   ├── baseline.py              # Regression baseline comparison tools
│   │   ├── context.py               # Evaluation context loader
│   │   ├── dataset.py               # Ground-truth evaluation dataset parser
│   │   ├── diagnostics.py           # Per-query failure diagnostics & strategy comparison
│   │   ├── e2e_evaluator.py         # Full pipeline evaluation runner
│   │   ├── e2e_results.py           # Serialization for E2E evaluation reports
│   │   ├── evaluator.py             # Retrieval evaluation engine (Recall, MRR, nDCG)
│   │   ├── groundedness.py          # NLI-based groundedness scoring
│   │   ├── metrics.py               # Pure mathematical metric implementations
│   │   ├── models.py                # Data structures for retrieval evaluation
│   │   ├── quality_gate.py          # CI/CD Quality Gate enforcement logic
│   │   └── results.py               # Formatted reporting for evaluation metrics
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── citation_validator.py    # Regex & coverage citation validator
│   │   ├── context_builder.py       # [C1], [C2] prompt context builder
│   │   ├── llm.py                   # Google Gemini SDK integration with backoff
│   │   ├── models.py                # Generation request/response Pydantic models
│   │   └── prompts.py               # Strict grounded system & correction prompts
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── cleaner.py               # Text normalization & artifact cleaner
│   │   ├── chunker.py               # Sliding window word chunker
│   │   ├── loader.py                # pypdf PDF parser
│   │   ├── models.py                # Ingestion models (Document, Chunk)
│   │   └── pipeline.py              # Ingestion pipeline coordinator
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_index.py            # BM25 Okapi in-memory index
│   │   ├── bm25_storage.py          # BM25 JSON persistence manager
│   │   ├── embedding.py             # SentenceTransformer embedding service
│   │   ├── hybrid_retriever.py      # RRF fusion engine
│   │   ├── index_manager.py         # Workspace indexing orchestrator
│   │   ├── models.py                # Retrieval models (ScoredChunk, RetrievalResult)
│   │   ├── qdrant_store.py          # Qdrant client wrapper & payload filters
│   │   ├── reranker.py              # Cross-Encoder neural reranker
│   │   └── vector_retriever.py      # Qdrant vector retrieval service
│   │
│   └── services/
│       ├── __init__.py
│       ├── document_service.py      # Session-scoped document upload/list/delete
│       ├── generation_service.py    # Generation & self-correction coordinator
│       ├── rag_service.py           # Unified RAG workflow coordinator
│       └── retrieval_service.py     # Multi-stage retrieval orchestrator
│
├── configs/
│   └── evaluation.yaml              # Thresholds for CI/CD Quality Gates
│
├── data/
│   ├── evaluation/                  # Ground truth datasets & golden baselines
│   │   ├── retrieval_baseline.json
│   │   └── retrieval_dataset.json
│   ├── index/                       # Persisted BM25 JSON partitions
│   └── raw/                         # Benchmark evaluation PDFs
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main Cyber/Holo UI application
│   │   ├── App.css                  # HUD layout, animations & component styles
│   │   ├── index.css                # Global Cyber/Holo design tokens & variables
│   │   ├── main.tsx                 # React DOM root
│   │   ├── api.ts                   # Typed API client for FastAPI backend
│   │   └── types.ts                 # TypeScript interfaces matching backend models
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts               # Vite proxy configuration
│
├── scripts/                         # CLI utilities for indexing & evaluation
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
│   ├── search_bm25.py
│   ├── search_hybrid.py
│   └── test_gemini.py
│
├── tests/
│   ├── unit/                        # Fast, offline isolated unit tests (250+ tests)
│   └── integration/                 # API & end-to-end integration tests
│
├── Dockerfile                       # Multi-stage production container build
├── docker-compose.yml               # Local multi-service composition (API + Qdrant)
├── Makefile                         # Automation commands
├── pyproject.toml                   # Project metadata & Python dependencies
├── render.yaml                      # Render Blueprint infrastructure specification
└── README.md                        # Master project documentation
```

---

# 10. Quick Start Guide

### Prerequisites
- **Python 3.12+**
- **Node.js 20+ & npm** (for frontend)
- **Docker & Docker Compose** (optional, recommended)
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/))

---

### Local Development Setup

#### 1. Clone the repository & create Python virtual environment:
```bash
git clone https://github.com/sandipan20/DocsQuery.git
cd DocsQuery

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

#### 2. Configure Environment:
```bash
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY
```

#### 3. Start local Qdrant container:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  --name docsquery-qdrant \
  qdrant/qdrant:latest
```

#### 4. Launch FastAPI Backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API will be live at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).*

#### 5. Launch React Frontend:
```bash
cd frontend
npm ci
npm run dev
```
*Frontend will be live at `http://localhost:5173`.*

---

### Running with Docker Compose
To spin up both the FastAPI backend and Qdrant database in isolated containers with a single command:

```bash
docker compose up --build
```

---

# 11. API Reference & cURL Cookbook

### 1. Initialize Anonymous Session
```bash
curl -i -X GET http://localhost:8000/api/v1/session
```
*Response sets the `docsquery_session` HTTP-only cookie.*

---

### 2. Upload PDF Documents
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -b "docsquery_session=my-session-token" \
  -F "files=@/path/to/guide.pdf" \
  -F "files=@/path/to/handbook.pdf"
```

**Response (`201 Created`):**
```json
{
  "documents": [
    {
      "id": "7f0b67701fa7e18b87192f15",
      "filename": "guide.pdf",
      "page_count": 12,
      "chunk_count": 28,
      "created_at": "2026-10-01T12:00:00Z"
    }
  ],
  "total_chunks": 28
}
```

---

### 3. List Active Session Documents
```bash
curl -X GET http://localhost:8000/api/v1/documents \
  -b "docsquery_session=my-session-token"
```

---

### 4. Delete a Document
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/7f0b67701fa7e18b87192f15 \
  -b "docsquery_session=my-session-token"
```

---

### 5. Execute Raw Hybrid Search (No LLM)
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -b "docsquery_session=my-session-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure branch protection rules?",
    "top_k": 3
  }'
```

**Response (`200 OK`):**
```json
{
  "query": "How do I configure branch protection rules?",
  "results": [
    {
      "chunk_id": "7f0b...-chunk-14",
      "document_id": "7f0b...",
      "score": 0.892,
      "text": "Branch protection rules enforce workflows before merges...",
      "metadata": {
        "source": "guide.pdf",
        "page_number": 8
      }
    }
  ],
  "total_results": 1,
  "timing_ms": 34.2
}
```

---

### 6. Execute Full Grounded RAG Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -b "docsquery_session=my-session-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What command creates and checks out a new Git branch simultaneously?",
    "top_k": 5
  }'
```

**Response (`200 OK`):**
```json
{
  "query": "What command creates and checks out a new Git branch simultaneously?",
  "answer": "To create and immediately switch to a new branch, use the command `git checkout -b <branch-name>` [C1]. In newer Git versions, `git switch -c <branch-name>` achieves the same result [C2].",
  "citations": [
    {
      "citation_id": "C1",
      "chunk_id": "7f0b...-chunk-2",
      "document_id": "7f0b...",
      "source": "git_guide.pdf",
      "page_number": 4,
      "snippet": "To create a new branch and switch to it immediately, run git checkout -b..."
    },
    {
      "citation_id": "C2",
      "chunk_id": "7f0b...-chunk-5",
      "document_id": "7f0b...",
      "source": "git_guide.pdf",
      "page_number": 6,
      "snippet": "Git 2.23 introduced git switch -c to simplify branch switching..."
    }
  ],
  "citation_validation": {
    "is_valid": true,
    "valid_citations": ["C1", "C2"],
    "invalid_citations": [],
    "citation_coverage": 1.0
  },
  "timing_ms": {
    "retrieval_ms": 38.4,
    "generation_ms": 284.1,
    "total_ms": 322.5
  }
}
```

---

# 12. Evaluation & Benchmarking Suite

DocsQuery does not treat a single successful query response as proof of quality. The framework includes an offline evaluation suite.

### Evaluation Dataset Schema (`data/evaluation/retrieval_dataset.json`):
```json
{
  "version": "1.0.0",
  "domain": "git_documentation",
  "examples": [
    {
      "example_id": "q001",
      "query": "How do I create and switch to a branch in a single command?",
      "query_type": "command_lookup",
      "relevant_chunk_ids": [
        "7f0b67701fa7e18b87192f15...-chunk-2",
        "7f0b67701fa7e18b87192f15...-chunk-5"
      ],
      "reference_answer": "Use git checkout -b <branch-name> or git switch -c <branch-name>."
    }
  ]
}
```

### Quality Gate Enforcement
The quality gate compares new runs against `retrieval_baseline.json`:
- **MRR Regression Guard**: Flags failure if MRR drops $> 0.05$.
- **nDCG@5 Regression Guard**: Flags failure if nDCG@5 drops $> 0.05$.
- **Recall@5 Regression Guard**: Flags failure if Recall@5 drops $> 0.05$.
- **Citation Validity Guard**: Requires $\ge 95\%$ valid citation tags.
- **NLI Groundedness Guard**: Requires $\ge 80\%$ entailment score using `cross-encoder/nli-MiniLM2-L6-H768`.

---

# 13. CLI Scripts & Utilities Catalog

DocsQuery provides an array of command-line tools under `scripts/`:

| Script Command | Purpose & Description | Output / Artifacts |
| :--- | :--- | :--- |
| `python -m scripts.ingest_corpus data/raw` | Scans PDFs in directory, chunks text, and indexes into Qdrant & BM25 | `data/index/bm25.json`, Qdrant points |
| `python -m scripts.corpus_stats` | Inspects index statistics (chunk counts, word stats, memory footprint) | Console summary table |
| `python -m scripts.inspect_chunks --limit 10` | Dumps chunk boundaries, metadata, and deterministic SHA-256 hashes | Formatted console debug logs |
| `python -m scripts.search "your query"` | Runs dense vector search only | Console scored chunk table |
| `python -m scripts.search_bm25 "your query"` | Runs BM25 lexical search only | Console scored chunk table |
| `python -m scripts.search_hybrid "your query"` | Runs full Hybrid RRF + Cross-Encoder reranking | Top-K evidence table with scores |
| `python -m scripts.test_gemini` | Verifies live Gemini API connectivity and credentials | Sample grounded answer |
| `python -m scripts.evaluate_retrieval` | Runs multi-strategy evaluation on `retrieval_dataset.json` | Detailed Recall, MRR, nDCG metrics |
| `python -m scripts.analyze_retrieval` | Performs per-query failure diagnosis (false negatives, strategy wins) | Strategy comparison report |
| `python -m scripts.benchmark_summary` | Prints comparative summary of all 4 retrieval strategies | Markdown/ASCII comparison table |
| `python -m scripts.create_retrieval_baseline` | Generates a new `retrieval_baseline.json` reference file | Updated baseline JSON |
| `python -m scripts.evaluate_rag` | Runs end-to-end evaluation including LLM generation & NLI judges | E2E quality report |
| `python -m scripts.check_evaluation_quality` | CI/CD Quality Gate script (returns exit code 1 on regression) | Pass/Fail status |

---

# 14. Production Deployment Guide

### Render Deployment (Web Service & Static Frontend)
DocsQuery includes a production-ready `render.yaml` blueprint.

1. Connect your GitHub repository to [Render.com](https://render.com).
2. Create a **Blueprint** deployment pointing to `render.yaml`.
3. Configure the following environment variables in the Render Dashboard:
   - `GEMINI_API_KEY`: Your Google AI Studio API Key.
   - `QDRANT_URL`: URL to your Qdrant Cloud cluster (e.g. `https://xyz.cloud.qdrant.io:6333`).
   - `QDRANT_API_KEY`: Your Qdrant Cloud API Key.
   - `API_CORS_ORIGINS`: `["https://your-frontend-subdomain.onrender.com"]`.

### Single-Container Production Image
DocsQuery features a multi-stage `Dockerfile` that builds the React frontend with Node 20 and packages it into a hardened Python 3.12 runtime with memory optimizations:

```bash
docker build -t docsquery:latest .
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY="your-gemini-key" \
  -e QDRANT_URL="https://your-cluster.cloud.qdrant.io" \
  -e QDRANT_API_KEY="your-qdrant-key" \
  docsquery:latest
```

### Qdrant Cloud Configuration
When deploying to Qdrant Cloud:
1. Ensure your collection is created with 384 vector dimensions and Cosine distance.
2. Ensure payload schema indexes exist on `workspace_id` (Keyword) and `document_id` (Keyword).
3. The DocsQuery application automatically ensures collection creation and payload indices on startup.

### Resource & Memory Optimization
To prevent memory leaks and thread thrashing on resource-constrained cloud servers (e.g., 512 MiB or 1 GiB RAM containers), DocsQuery sets:
```bash
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
```
*These settings restrict PyTorch/SentenceTransformers from spawning excessive OpenMP thread pools and prevent glibc memory fragmentation.*

---

# 15. Configuration & Environment Variables

| Variable Name | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | String | `development` | Application environment (`development`, `test`, `production`). |
| `APP_NAME` | String | `DocsQuery` | Name of the application. |
| `DEBUG` | Boolean | `false` | Enables verbose debug logging. |
| `API_HOST` | String | `0.0.0.0` | Bind host for FastAPI server. |
| `API_PORT` | Integer | `8000` | Bind port for FastAPI server. |
| `API_CORS_ORIGINS` | JSON Array | `["http://localhost:5173"]` | Allowed CORS origins for browser clients. |
| `GEMINI_API_KEY` | Secret | *Required for Generation* | Google AI Studio API Key. |
| `GEMINI_MODEL` | String | `gemini-3.1-flash-lite` | Target Gemini model identifier. |
| `GEMINI_TEMPERATURE` | Float | `0.0` | Sampling temperature (0.0 for deterministic output). |
| `GEMINI_MAX_TOKENS` | Integer | `1000` | Maximum token limit for LLM generation. |
| `QDRANT_URL` | String | `http://localhost:6333` | URL of Qdrant instance. |
| `QDRANT_API_KEY` | Secret | `""` | API key for Qdrant Cloud instances. |
| `QDRANT_COLLECTION` | String | `docsquery_chunks` | Name of the Qdrant vector collection. |
| `EMBEDDING_MODEL` | String | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model ID. |
| `RERANKER_MODEL` | String | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace Cross-Encoder model ID. |
| `BM25_STORAGE_PATH` | Path | `data/index/bm25.json` | Path for serialized BM25 index storage. |
| `DEFAULT_TOP_K` | Integer | `5` | Default number of final evidence chunks returned. |
| `SESSION_COOKIE_NAME` | String | `docsquery_session` | Name of the HTTP-only session cookie. |
| `SESSION_COOKIE_SECURE` | Boolean | `false` | Enables `Secure` flag on cookie (set `true` in production). |

---

# 16. Development, Testing & Code Quality

DocsQuery enforces strict test coverage and static analysis across all layers:

### Run Full Test Suite:
```bash
pytest -v
```

### Run Unit Tests:
```bash
pytest tests/unit -v
```

### Run Integration Tests:
```bash
pytest tests/integration -v
```

### Run Linter & Formatter:
```bash
ruff check .
ruff format --check .
```

### Run Automation Makefile:
```bash
make check      # Runs ruff, formatting, and pytest
make test       # Runs pytest test suite
make lint       # Runs ruff linter
make format     # Formats code using ruff
```

---

# 17. Troubleshooting & Operational Playbook

### Scenario 1: LLM Citation Validation Retries Triggering Frequently
- **Cause**: User queries ask for information not contained in the uploaded documents, prompting the LLM to either fabricate citations or omit them.
- **Remedy**: DocsQuery automatically executes a self-correction retry. If the issue persists, the system returns a safe refusal explaining that the evidence is insufficient. Ensure relevant documents covering the topic are uploaded.

### Scenario 2: Container OOM (Out Of Memory) on 512 MiB Instances
- **Cause**: PyTorch default memory allocators create multiple glibc arenas per CPU core.
- **Remedy**: Set `MALLOC_ARENA_MAX=2` and `OMP_NUM_THREADS=1` in your container environment variables (already included in `Dockerfile`).

### Scenario 3: Qdrant `Payload Index Missing` Warning
- **Cause**: Qdrant collection was created externally without payload keyword indices.
- **Remedy**: DocsQuery automatically attempts to create keyword indexes on `workspace_id` and `document_id` on startup via `app.retrieval.qdrant_store.ensure_collection()`.

### Scenario 4: CORS Errors in Browser
- **Cause**: The frontend origin is not in `API_CORS_ORIGINS`.
- **Remedy**: Update `API_CORS_ORIGINS` in `.env` to include your exact frontend URL (e.g. `API_CORS_ORIGINS='["https://my-frontend.onrender.com"]'`).

---

# 18. License & Credits

Built with precision for high-reliability RAG workflows. 

- **Core Technologies**: [FastAPI](https://fastapi.tiangolo.com/), [Qdrant](https://qdrant.tech/), [Sentence Transformers](https://www.sbert.net/), [Google Gemini](https://ai.google.dev/), [React](https://react.dev/), and [Tailwind-free Vanilla CSS](https://developer.mozilla.org/en-US/docs/Web/CSS).
- **License**: MIT License. Open source for research, development, and enterprise deployments.
