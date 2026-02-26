# Document Upload, Chunking, and Vector Indexing (MVP) — Implementation Plan

## 1) Goals and Scope

### In scope (this story)
- Upload supported document types from the web app.
- Store original uploaded file and link it to user + workspace/project context.
- Extract text and chunk primarily by headings, with fallback chunking by token size.
- Persist chunks and metadata in relational DB.
- Generate embeddings for chunks using configured embedding provider.
- Write vectors to configured index (pgvector first, abstracted for future providers).
- Track processing lifecycle statuses: `Uploaded`, `Processing`, `Ready`, `Failed`.
- Provide backend retrieval primitives by workspace/project/document, preserving chunk order.

### Out of scope (explicitly deferred)
- End-user semantic search UI.
- RAG chat/answer generation.
- Re-indexing and dedup strategy for updates.
- Evaluation dashboards and quality monitoring.

---

## 2) Proposed Architecture (MVP)

### Core entities
1. **Project**
   - Minimal model with standard metadata:
     - `id`
     - `workspace` (if workspace model exists; otherwise `owner`/organization surrogate)
     - `name`
     - `created_by`
     - timestamps

2. **ProjectDocument**
   - Represents an uploaded file.
   - Fields:
     - `project` (FK)
     - `uploaded_by` (FK user)
     - `original_file` (FileField)
     - `original_filename`, `mime_type`, `size_bytes`
     - `status` (`UPLOADED`, `PROCESSING`, `READY`, `FAILED`)
     - `processing_error` (nullable text)
     - `parser_type`, `chunking_strategy`, `embedding_model`
     - timestamps (`uploaded_at`, `processed_at`)

3. **ProjectDocumentChunk**
   - Separate model (recommended) for chunk-level lifecycle and metadata.
   - Fields:
     - `document` (FK)
     - `chunk_index` (int; unique per document)
     - `heading` (nullable)
     - `text`
     - `token_count`
     - `source_page_start` / `source_page_end` (nullable)
     - `source_offset_start` / `source_offset_end` (nullable)
     - timestamps
   - Constraint/indexes:
     - unique `(document, chunk_index)`
     - index on `(document, chunk_index)`

4. **ChunkEmbeddingPointer** (optional but useful)
   - If using external vector DB, map DB chunk -> vector row id.
   - For pgvector-in-DB approach, this can be skipped and embedding can live directly on chunk/related table.

### Why separate chunk model?
- Better queryability, ordering, and metadata richness.
- Cleaner retries and partial-failure handling.
- Easier to support future re-chunking and evaluation.

---

## 3) Storage & Vector Strategy

### File storage
- Use Django storage backend (local/S3-compatible depending on environment).
- Keep original files for auditability and future parser improvements.

### Relational storage
- `ProjectDocument` and `ProjectDocumentChunk` in primary DB.

### Vector index
- **Preferred MVP:** PostgreSQL + pgvector (single-store operational simplicity).
  - Option A: vector on `ProjectDocumentChunk` table (`embedding` column).
  - Option B: separate `ChunkEmbedding` table if dimensions/providers may vary.
- Add abstraction layer in service code so alternative vector DB can be plugged in later.

---

## 4) Parsing and Chunking Design

### Supported file types (MVP recommendation)
- Start with text-first formats:
  - `.txt`, `.md`, `.html`, `.pdf` (if parser reliability acceptable)
- Return graceful validation error for unsupported formats.

### Parsing pipeline
1. Detect file type (`mime` + extension).
2. Parse to normalized intermediate structure:
   - `DocumentSection[]` with optional heading + body + source location.
3. If heading structure present, chunk by sections/headings.
4. Fallback to token-window chunker with overlap.

### Token counting and splitting
- Use `tiktoken` for token counting and limits.
- Use LangChain text splitters where beneficial, but keep deterministic wrapper:
  - Header-aware splitter first.
  - Recursive/token splitter fallback.
- Store exact chunk token counts in DB.

### Suggested defaults
- `target_chunk_tokens`: 600–900
- `max_chunk_tokens`: 1200
- `chunk_overlap_tokens`: 80–120
- Make values configurable in settings.

---

## 5) Processing Workflow

### Async workflow (recommended)
Use background worker (Celery/RQ/Django-Q; choose existing project standard):

1. User uploads document via form/API.
2. Create `ProjectDocument(status=UPLOADED)`.
3. Enqueue processing task with document id.
4. Worker sets `PROCESSING` and runs:
   - parse
   - chunk
   - persist chunks
   - embed chunks
   - write vectors
5. On success set `READY` and `processed_at`.
6. On failure set `FAILED` + capture error text for diagnostics.

### Idempotency
- Before writing new chunks for a document, clear existing chunks/embeddings for that document in same transaction boundary (or use processing version ids).

---

## 6) Web and API Surfaces

### Upload UI (MVP)
- New page in project context:
  - file input
  - submit
  - simple status badge/table
- Show statuses: Uploaded, Processing, Ready, Failed.
- Confirmation toast/message when document reaches Ready (polling or websocket optional; polling simplest).

### Backend endpoints (minimum)
- `POST /projects/<id>/documents/upload/`
- `GET /projects/<id>/documents/` (list + status)
- `GET /projects/<id>/documents/<doc_id>/chunks/` (ordered)

### Access control
- Ensure project/workspace membership checks on all endpoints.
- Ensure users can only access their authorized project documents/chunks.

---

## 7) Embedding Integration

### Provider abstraction
- Reuse current LLM service patterns if available.
- Add `EmbeddingService` interface:
  - `embed_texts(texts: list[str]) -> list[list[float]]`
- Configure model/provider via settings/env.

### Batch strategy
- Batch chunks per document to reduce latency/cost.
- Persist progress in case of partial failures only if needed; MVP can fail-fast and mark document failed.

### Vector write contract
- Upsert vectors keyed by `chunk_id` (and `workspace/project` namespace).
- Keep deterministic metadata in vector index:
  - `chunk_id`, `document_id`, `project_id`, `workspace_id`, `chunk_index`.

---

## 8) Data Model & Migration Plan

1. Add `Project` model (minimal).
2. Add `ProjectDocument` model and status enum.
3. Add `ProjectDocumentChunk` model with ordering and metadata fields.
4. Add vector schema changes (pgvector extension, embedding column/table).
5. Add DB indexes:
   - document status
   - chunk order
   - project/document foreign key paths

---

## 9) Reliability, Observability, and Errors

### Error handling
- Capture parser errors, embedding provider errors, and vector write errors.
- Surface concise user-facing error + internal verbose logs.

### Logging
- Structured logs with `document_id`, `project_id`, stage (`parse/chunk/embed/index`).

### Metrics (lightweight MVP)
- documents processed
- processing success/failure counts
- average processing duration
- average chunks per document

---

## 10) Security and Compliance Considerations

- Validate upload size/type server-side.
- Sanitize extracted text where needed for rendering.
- Enforce permissions on document/chunk retrieval.
- Ensure file storage pathing avoids traversal and uses generated names.
- Define retention/deletion behavior (even if basic now).

---

## 11) Testing Plan

### Unit tests
- Parser routing by MIME/extension.
- Heading-based chunking and fallback chunking.
- Token counting boundaries and overlap behavior.
- Status transitions and failure handling.

### Integration tests
- Upload -> async processing -> ready state.
- Chunk persistence ordering and metadata correctness.
- Embedding generation call contract.
- Vector index write/read by project/document namespace.

### Permission tests
- Unauthorized user cannot view/upload documents in other project/workspace.

---

## 12) Incremental Delivery Plan

### Phase 1: Data + upload skeleton
- Models + migrations.
- Upload form/endpoint.
- Status plumbing.

### Phase 2: Parsing + chunking
- Implement parsers for MVP formats.
- Heading-first splitter with token fallback.
- Persist chunks.

### Phase 3: Embeddings + vector indexing
- Embedding service integration.
- pgvector writes and retrieval helper API.
- Mark documents ready/failure states end-to-end.

### Phase 4: Hardening
- Logging/metrics.
- Improved error messages.
- Performance tuning and batch sizing.

---

## 13) Open Decisions to Confirm Before Build

1. **Workspace model source of truth:** existing model or introduce now?
2. **Background jobs framework:** Celery vs existing async pattern in repo.
3. **Supported file types for MVP:** include PDF immediately or text/markdown first.
4. **Vector store target:** pgvector now vs external provider.
5. **Embedding model/provider default:** confirm cost/latency constraints.

---

## 14) Suggested MVP Definition of Done

- A user uploads a supported file from the web page.
- File is stored and linked to project + user.
- System processes asynchronously to chunks (heading-first, token fallback).
- Chunks persist with order + metadata.
- Embeddings are generated and stored/indexed.
- Document reaches `Ready` status on success or `Failed` with reason.
- Backend endpoints can list documents and ordered chunks by project.
- Automated tests cover happy path + key failures + permissions.
