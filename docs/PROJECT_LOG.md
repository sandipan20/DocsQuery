# DocsQuery Project Log

## Current Status

### Overall Status
The local MVP includes anonymous session isolation, document upload/list/delete, selected-document query scoping, hybrid retrieval, and a usable frontend. Runtime benchmark data has been removed from local Qdrant and BM25 while private uploaded data was preserved. Startup now ignores public benchmark chunks and treats an empty BM25 index as ready for uploads. Production deployment and live two-session isolation remain unverified.

### Current Milestone
Runtime corpus isolation and reliable retrieval for newly uploaded documents.

### Current Task
Complete the application-wide verification gates, rebuild/restart the local services, and validate the document upload/search path without exposing document contents or secrets.

### Last Completed Task
Added a workspace-scoped BM25 fallback when vector similarity is below threshold. Removed the local public seed corpus from Qdrant and BM25 while preserving private uploaded workspace data.

### Immediate Next Task
Finish full tests/lint/frontend checks, rebuild the seed-free Docker image, restart the API, and perform local readiness plus API search smoke tests. Keep benchmark PDFs only as offline evaluation fixtures.

## System Status

### Backend
FastAPI exposes anonymous session, document upload/list/delete, search, and query routes. Upload validation limits each request to 10 files and each file to 25 MiB; extension, PDF signature, parser validity, and extractable text are checked. Low vector confidence now falls back to positive workspace/document-scoped BM25 matches before abstaining.

### Frontend
React, TypeScript, Vite, and TanStack Query implement session initialization, multi-PDF upload, document listing/deletion, all-or-selected query scope, loading/error states, answers, and source/page citations. `npm run build` passed after the frontend changes. A hosted frontend and production browser-to-API routing are not configured or verified. Vite's development proxy targets the Modal API URL in `frontend/vite.config.ts`.

### RAG Pipeline
Existing citation validation, evidence checks, Gemini generation, and abstention behavior were retained. `document_ids` now flows from the API through RAG/retrieval, hybrid fusion, BM25, vector retrieval, and Qdrant. No-document-ID queries use the current cookie-derived workspace.

### Qdrant
Private point IDs are UUIDv5 values derived from `workspace_id:chunk_id`. The vector store applies workspace equality and optional selected-document MatchAny filters during vector search. Filtered scroll and delete operations require workspace and, for deletion, document ID. Payload indexes are requested for `workspace_id` and `document_id` when the collection is ensured. Unit tests have verified point-ID collision prevention and filter construction; live collection/index state has not been verified in this session.

### BM25
Ranking statistics are computed only from chunks in the authorized workspace and optional document subset, so other workspaces cannot affect that query's scores. JSON persistence remains filesystem-local. In stateless/cold-start Modal instances it is not a durable distributed index; this limitation is accepted for the MVP and must not be described as durable synchronization.
### Frontend
React, TypeScript, Vite, and TanStack Query implement session initialization, multi-PDF upload, document listing/deletion, all-or-selected query scope, loading/error states, answers, and source/page citations. Frontend lint/build pass. The Render static frontend and API CORS origin are configured but have not been live-deployed/verified. The local Vite proxy defaults to `http://localhost:8000`.
### Security / Session Isolation
The `docsquery_session` cookie is HTTP-only, SameSite=Lax, and 24-hour max age. New cookies use the Secure flag when `APP_ENV=production`. API routes derive workspace IDs from the cookie and ignore client-supplied workspace fields. Selected document IDs are checked against documents in that workspace. Focused tests cover body/form spoofing, cross-session listing/deletion behavior at the route boundary, unauthorized query IDs, workspace-scoped BM25, Qdrant filters, and identical document chunk IDs across workspaces. A production two-browser upload/query/delete acceptance test is still required. The cookie is a bearer credential; authentication, revocation, and rate limiting are not implemented.

### Docker
The runtime image does not copy the generated BM25 benchmark artifact. `data/raw/*.pdf` remains an offline evaluation fixture set and is excluded from the Docker build context. The seed-free image built successfully; API and Qdrant are healthy under Compose.
Verified during the 2026-10-01 cleanup: `.venv/bin/python -m pytest -q` passed with 281 tests; `.venv/bin/ruff check app scripts tests` passed; `.venv/bin/ruff format --check app scripts tests` passed (166 files); `cd frontend && npm run lint && npm run build` passed; `git diff --check`, `docker compose config --quiet`, and workspace diagnostics passed. Seed-independent Qdrant/evaluation tests passed. The seed-free image rebuild and restarted runtime count check remain pending.

### Docker
### Frontend Deployment
`render.yaml` configures a static frontend and API CORS origin. Deployment and production browser-to-API verification remain open.

- [x] Seed-free Docker image built and local API/Qdrant readiness verified.
- [ ] Frontend hosted and production browser-to-backend flow verified.
- BM25 JSON is filesystem-local and is not durable or synchronized across arbitrary stateless Modal instances.
- The Vite proxy only applies to local development. Render hosting/CORS are configured but production routing is not live-verified.
- Full Python suite: 281 passed.
- Seed-free Docker image built; API `/ready` reports BM25 and Qdrant healthy.
- Post-restart corpus counts: 0 public BM25 chunks, 5 private BM25 chunks, 0 public Qdrant points, 5 private Qdrant points.
- End-to-end local smoke test uploaded a PDF in a temporary session, returned 2 scoped search results from that file, then deleted the test document.
## Definition of Done

#### Verification
- Focused retrieval, BM25 startup, health, corpus-integrity, and seed-independent evaluation tests passed.
- Full post-change Python suite: 281 passed.
- Ruff lint, Ruff format check, frontend lint/build, editor diagnostics, Compose configuration, and diff whitespace checks passed.
- Seed-free Docker image built successfully and local API `/ready` passed with Qdrant healthy.
- After restart: 0 public BM25 chunks, 5 private BM25 chunks, 0 public Qdrant points, 5 private Qdrant points.
- Temporary isolated upload/search/delete smoke test returned two results from the uploaded PDF and cleaned up the test document.
- [x] Query supports all current-session documents or authorized selected documents.
#### Remaining
- The PDFs in `data/raw/` remain only as offline evaluation fixtures and for dataset reproducibility. They are excluded from the image and are not loaded into runtime retrieval.
- Live Render deployment, hosted browser flow, and production two-session acceptance remain unverified.
- [x] Dense and sparse retrieval preserve workspace and selected-document scope.
- [x] Frontend local workflow and citation display implemented; production build passed.
- [x] Full unit suite passed (275 tests).
- [x] Python Ruff lint and format checks passed.
- [x] Frontend ESLint and production build passed.
- [ ] Docker image build result collected after upload dependency/route changes.
- [ ] Live backend health/readiness and document APIs verified after deployment.
- [ ] Two-session live upload/query/delete isolation test, including identical PDFs.
- [ ] Frontend hosted and production browser-to-backend flow verified.
- [x] README documents current local workflow and known limitations.
- [x] This handoff log records current status and next work.

## Architecture

The request path remains FastAPI -> RAGService -> RetrievalService -> workspace/document-scoped BM25 and Qdrant -> hybrid RRF -> cross-encoder reranking -> context builder -> Gemini -> citation validation. The document service reuses existing ingestion and indexing abstractions, writing private chunk payloads to Qdrant and updating the in-process/persisted BM25 index. Qdrant scroll is used to reconstruct document metadata.

## Security Invariants

- The client does not choose a workspace through JSON, form data, or query parameters.
- Derive workspace identity from the HTTP-only `docsquery_session` cookie at the API boundary.
- Apply workspace filtering inside Qdrant and before BM25 scoring/ranking.
- Optional selected document IDs must be verified against the current workspace and then applied to both dense and sparse retrieval.
- Delete using a Qdrant filter requiring both current workspace and requested document ID.
- Never return workspace/session IDs as document metadata.
- Private Qdrant point IDs include workspace identity to prevent deterministic chunk-ID collisions.
- Keep Gemini and Qdrant credentials out of source, frontend bundles, logs, and this handoff.

## Important Technical Decisions

- Reuse `load_pdf`, page cleaning, `chunk_pages`, the shared embedding service, Qdrant vector store, and BM25 index rather than creating a second ingestion pipeline.
- Use filtered Qdrant payload scroll to build the MVP document list; Qdrant remains the durable vector/document payload store.
- Compute BM25 statistics against the authorized subset for each retrieval request. This avoids cross-workspace corpus statistics at the cost of rebuilding a scoped BM25 object per query.
- Keep no-document-ID semantics as all documents in the current session, never the global Qdrant collection.
- Keep the existing Gemini citation validator and abstention path unchanged.

## Known Limitations

- BM25 JSON is filesystem-local and is not durable or synchronized across arbitrary stateless Modal instances.
- Anonymous cookies are bearer credentials; no user accounts, revocation, or rate limiting exist.
- Scanned PDFs without extractable text are rejected; OCR is not implemented.
- Multi-step Qdrant/BM25 writes are not transactional across the two stores.
- The Vite proxy only applies to local development. Frontend hosting and production routing/CORS are not configured here.
- Deployment, live Qdrant indexes, and the mandatory two-session production acceptance scenario have not been verified by this session.

## Current Issues / Blockers
- No known local retrieval blocker after the BM25 fallback and seed removal; full post-change verification is pending.
- Live deployment and hosted frontend verification require deployment access/configuration.
- Live deployment and hosted frontend verification require deployment access/configuration and should be run only after checking the actual current target configuration.

## Development History

### 2026-10-01 — Anonymous Document Workflow and Isolation

#### Implemented
- Threaded the cookie-derived workspace through query and search routes and through RAG/retrieval.
- Added selected-document authorization and Qdrant/BM25 filtering.
- Added a document model/service and upload/list/delete API with PDF validation and upload limits.
- Added workspace/document Qdrant payload indexes, filtered scroll/delete, and workspace-aware point IDs.
- Scoped BM25 scoring statistics to the authorized document subset.
- Built the local React document library and query interface with citation display.
- Added `python-multipart` as a declared runtime dependency.
- Updated README and created this project log.

#### Files Changed
Backend changes are in `app/api/v1/`, `app/documents/`, `app/security/`, `app/container.py`, `app/main.py`, ingestion and retrieval services/models, `pyproject.toml`, and focused tests. Frontend changes are in `frontend/src/`. Documentation is in `README.md` and `docs/PROJECT_LOG.md`.

#### Why
The prior route handlers did not pass the anonymous cookie workspace into RAG/retrieval, and there was no document-management API or usable upload/select frontend. Those gaps prevented session isolation from being enforced end to end.

#### Verification
- `.venv/bin/python -m pytest tests/unit/test_bm25_retriever.py tests/unit/test_workspace_isolation.py -q`: 12 passed.
- `.venv/bin/python -m pytest tests/unit/test_documents_api.py tests/unit/test_query_api.py tests/unit/test_vector_store.py -q`: 15 passed.
- `.venv/bin/python -m pytest tests/unit/test_query_api.py tests/unit/test_documents.py tests/unit/test_chunker.py tests/unit/test_vector_store.py -q`: 31 passed.
- `.venv/bin/python -m pytest tests/unit/test_retrieval_service.py tests/unit/test_rag_service.py -q`: 8 passed.
- `cd frontend && npm run build`: passed.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check .`: passed (173 files already formatted).
- `cd frontend && npm run lint && npm run build`: passed.
- `.venv/bin/python -m pytest tests/unit -q`: 275 passed.
- Browser check with intercepted API responses: selected query submitted only `document_ids: ["doc-a"]`; citation filename/page rendered; 390px viewport had no horizontal overflow. This is a frontend contract check, not a live backend test.
- `curl --max-time 25 .../health`: timed out with zero response bytes; live backend unavailable/unverified from this environment.
- `docker compose config --quiet`: passed. `docker compose build`: still running at last status; result pending.

#### Result
Local feature implementation is in place with focused unit/API coverage. Deployment and complete project gates are not claimed.

#### Remaining Work
The full test/style/frontend gates are complete locally. Collect Docker build outcome, then inspect deployment access and verify actual deployment, run two-session isolation (including identical PDFs), and configure/deploy the frontend if the target host is available.

## Next Steps

1. Rerun Ruff checks and frontend lint/build after the final edits.
2. Run the complete unit test suite and diagnose any failures without weakening tests.
3. Run Docker build and verify API health/readiness locally or against the current deployment.
4. Inspect Modal/frontend deployment access and current remote state before any deployment change.
5. Run two separate cookie sessions through upload, list, selected/all-session query, and cross-session delete checks; include same-content PDFs.
6. Update this log and README only with observed results.

## Development History

### 2026-10-01 — Retrieval Fallback and Runtime Seed Removal

#### Findings
- Upload requests returned `201` and Qdrant upserts succeeded. Queries reached Qdrant but returned the API's no-evidence `404` because a fixed vector threshold stopped the pipeline before BM25 could contribute.
- The Dockerfile copied `data/index` into the serving image and application startup loaded public benchmark chunks from the same BM25 file used to persist local uploaded workspaces.
- Local Qdrant contained 551 points in the `public` benchmark workspace and 5 private uploaded points. The local BM25 file contained the same 551 public chunks plus 5 private chunks.
- `app/api/dependencies.py` duplicated retrieval construction but had no callers.

#### Changes
- Added a workspace/document-scoped BM25 fallback for low-confidence vector retrieval; the reranker receives only positive lexical matches, and the system still abstains when BM25 has no positive match.
- App startup removes only `public` chunks from BM25 persistence, retains every non-public workspace, and reports ready with an empty index so users can upload the first document.
- Removed benchmark index copying and validation from the Docker image. Benchmark PDFs under `data/raw/` remain evaluation fixtures, not runtime or frontend documents.
- Removed the confirmed unreferenced `app/api/dependencies.py` implementation.
- Updated corpus integrity to use `(workspace_id, chunk_id)` as the unique point identity, matching Qdrant's workspace-aware deterministic point IDs.
- Deleted exactly 551 local public Qdrant seed points and 551 public BM25 chunks; verified 5 private vector points and 5 private BM25 chunks remained.

#### Verification
- Focused retrieval, BM25 startup, health, and corpus-integrity tests passed.
- Full post-change Python suite: 279 passed.
- Ruff and diff whitespace checks passed.
- Full Python suite: 281 passed.
- Ruff lint, Ruff format check (166 files), frontend ESLint/build, editor diagnostics, Compose config, and diff check passed.
- Isolated vector retrieval/Qdrant/evaluation fixture tests passed (7 tests).
- Docker image rebuild and local API restart/readiness verification were started; final completion/count output pending.
