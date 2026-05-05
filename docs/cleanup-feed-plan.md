# Cleanup Feed Current Architecture

**Status:** current implementation notes for `codex/question-cleanup-panel`.
**Dataset:** `question_tag.csv`, 62,978 row cards.
**Primary goal:** clean the question/tag CSV while using full-dataset semantic Top-10 results to spot duplicates, bad tags, and near-overlaps.

## Product Shape

The app is a single React cleanup workbench. It no longer shows the old atlas, force graph, hierarchy views, clusters, or suggestion panels.

The main surface has:

- A virtualized feed with one card per CSV row.
- The tag name and question text visible on every row card.
- All rows for the same tag remain together because the source CSV order is preserved.
- Feed-local question search for narrowing the visible row list without running E5.
- Tag-first ordering that moves one selected tag group to the top while preserving the rest of the feed.
- Per-row actions for edit, delete, and add.
- Checkbox selection for cleanup operations, not semantic search.
- A centered search area for semantic queries.
- A semantic setup/results dock above the feed.
- A cleanup operation log so the user can see edits, adds, deletes, restores, and rollback-point changes.

## Semantic Search Contract

Top-10 results are full-index-only.

Feed-local search is not semantic search. It only filters the visible feed by literal question text. The tag-first selector is also browse-only; it does not create similarity scores or change the Top-10 query.

The UI must not show semantic scores while the user is merely browsing rows. Scores appear only after:

1. The vector index is complete for all active rows.
2. The user either searches text or clicks a row to inspect similar rows.

There is intentionally no "partial Top-10" state in the user-facing copy. During indexing, the UI shows progress and keeps search disabled because partial matches are easy to misread during data cleanup.

Row click replaces the old "pivot" wording. A clicked row copies that row's question into the search candidate and runs the same Top-10 search path as a typed query. Multi-select never becomes a search input.

## Semantic Backend

The preferred engine is the local Python 3.13 backend:

```bash
npm run semantic:server
```

It serves `http://127.0.0.1:8765` and exposes:

- `GET /health`
- `GET /status`
- `POST /configure`
- `POST /build`
- `POST /upsert`
- `POST /remove`
- `POST /search`

The backend uses:

- `intfloat/multilingual-e5-large-instruct` for SentenceTransformers/PyTorch.
- Apple Silicon MPS for embedding when available.
- FAISS CPU `IndexFlatIP` over normalized vectors for Top-10 retrieval.
- `.semantic-cache/row_vectors.npz` for restartable vector cache.

The Python backend is mandatory. The previous Transformers.js browser fallback was removed because loading E5-large inside Chrome could create duplicate model instances and severe memory pressure on long cleanup sessions.

The frontend owns backend reconnection. If the user opens the page before `npm run semantic:server`, the UI shows a backend error, polls `/health`, and then reruns the CSV-to-backend configure/reuse handshake after the backend is reachable. A stale `localStorage.semanticBackendUrl` is cleared when the default backend at `127.0.0.1:8765` is reachable. A `ready 0 / 0` backend response is not treated as a complete index when the frontend has loaded rows.

## Build Vector Index Flow

1. Start Vite:

   ```bash
   npm run dev -- --port 5174
   ```

2. Start the backend:

   ```bash
   npm run semantic:server
   ```

3. Open `http://localhost:5174/`.
4. Wait for the UI to show the backend connection and row count. If the backend was started late, the UI should recover without a browser refresh.
5. Click **Build vector index** if vectors are incomplete.
6. Watch `indexed / total` increase.
7. Search and row-click Top-10 become enabled only at `62,978 / 62,978`.

Backend progress can also be checked with:

```bash
curl http://127.0.0.1:8765/status
```

## Cleanup Operations

Every mutation updates the row store immediately and then syncs changed rows to the semantic engine.

- **Edit:** changes the row tag/question, records an operation, invalidates that row fingerprint, and upserts a fresh vector.
- **Delete:** marks the row removed in the UI, records an operation, removes it from the backend corpus, and refreshes active Top-10 if needed.
- **Add:** creates a new active row, records an operation, upserts it into the backend, and makes it eligible for future Top-10 results.
- **Undo/redo:** replays the operation log and syncs affected rows back to the backend.
- **Rollback-point create:** stores a label pointing to the current operation id and confirms it immediately.
- **Rollback-point restore:** moves the row state forward or backward through the operation log to the rollback-point id, then reconfigures the backend.

Rollback points are labels over an operation log, not full copies of the 62k-row dataset. The clean CSV file is saved separately with **Save clean CSV**.

## State Ownership

Frontend state lives in Zustand with Immer in `src/store.js`.

Important slices:

- `rows`: current row table keyed by `rowId`, plus active/removed row-id indexes used by the virtual feed and backend sync.
- `selection`: checkbox selection and clicked semantic row.
- `search`: active query/row intent and current Top-10 results.
- `embeddings`: semantic engine status, vector counts, model progress, and cache counters.
- `revisions`: durable operation log and rollback-point labels.
- `flags`: export/dirty state.

The bridge in `src/workerBridge.js` talks only to the Python backend. If the backend is not available, semantic build/search actions show an error instead of starting a browser model.

The backend build worker checks for missing or stale row vectors before loading E5. A fully cached index can move back to `ready` without instantiating another model. Model loading is also guarded by a lock so concurrent requests do not create duplicate SentenceTransformer instances in one backend process.

## Verification Checklist

Before committing:

- `npm run build`
- `uv run --python 3.13 python -m py_compile backend/semantic_server.py`
- Open the app in a browser and confirm the feed renders.
- Load the page while `/health` is unavailable, then restore backend access and confirm the UI recovers to the cached vector count.
- Click **Build vector index** when vectors are incomplete and confirm backend `/status` moves into `indexing`.
- After the full vector index is ready, verify:
  - Text search returns 10 rows with scores.
  - Clicking a row returns 10 similar rows with scores.
  - Deleting a Top-10 row removes it and recomputes/refills the result list.
  - Editing a row records an operation and upserts the changed vector.
  - Adding a row records an operation and upserts the new vector.
  - Save status changes after edits and returns current after clean CSV save.

## Known Constraints

- Full Top-10 correctness requires all active vectors to be ready.
- FAISS is in-memory and rebuilt after corpus changes.
- `.semantic-cache/` is local runtime data and must stay ignored by git.
- Search is global over active rows.
