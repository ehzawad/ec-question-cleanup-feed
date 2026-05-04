# EC Question Cleanup Feed

Single-screen cleanup workbench for Bangladesh Election Commission chatbot rows from `question_tag.csv`.

The app renders one virtualized feed card per CSV row. Rows can be edited, deleted, restored, bulk-deleted, added, saved as CSV, and bookmarked with named rollback points. Semantic search uses multilingual E5 embeddings; clicking a row copies that row's question into the same Top-10 search path as a typed query.

## Run

```bash
npm install
npm run dev -- --port 5173
```

Open `http://localhost:5173/`.

Run the Apple Silicon semantic backend in a second terminal before opening the UI:

```bash
npm run semantic:server
```

The frontend connects to `http://127.0.0.1:8765` automatically. If the page opens before the backend is running, it keeps retrying and configures the loaded CSV rows as soon as the backend comes online. The Python backend is required; the browser E5 fallback was removed because E5-large can consume enough Chrome memory to make the tab unresponsive.

After the page loads, click **Build vector index** if the vector count is not complete. The button creates cached vectors for all active rows, and a fully cached index returns to ready without loading another E5 model instance. Search and row-click similarity stay disabled until the vector count reaches `62,978 / 62,978`, so the Top-10 panel only shows full-dataset results.

If port `5173` is busy:

```bash
npm run dev -- --port 5174
```

## Build

```bash
npm run build
```

## Data

- `question_tag.csv` is the active source file.
- It uses `question,tag` columns.
- The feed keeps row-level data, so repeated tags remain separate cards.
- Search is global over active rows.

## Semantic Models

- Python backend: `intfloat/multilingual-e5-large-instruct`, loaded by SentenceTransformers/PyTorch on MPS when available.
- The old browser fallback used an ONNX model in Chrome and has been removed to avoid duplicate model loads and memory pressure.
- Backend row vectors are cached under `.semantic-cache/row_vectors.npz`; closing the browser tab does not delete that cache. The cache path is repo-relative unless `SEMANTIC_CACHE_DIR` overrides it.

## Cleanup Features

- Scrollable virtualized feed for the full `question_tag.csv` dataset.
- Semantic result dock that shows Top-10 only after the full vector index is complete and after a typed query or row click.
- Per-row edit, delete, restore, click-to-inspect, and add-sibling actions. Row-click inspection becomes active only after the full vector index is ready.
- Checkbox plus shift-click range selection for bulk delete.
- Undo/redo with a persisted operation log. Cleanup edits are autosaved in the browser.
- Named rollback points that restore the dataset to a prior cleanup step.
- Save clean CSV and removed-row CSV downloads.
- Python 3.13 Apple Silicon backend with PyTorch MPS embeddings and FAISS Top-10 search.
- CSV loading and parsing run asynchronously so the feed can paint without blocking the tab.

## Dev Logging

Use the backend status endpoint:

```bash
curl http://127.0.0.1:8765/status
```

The status payload reports the current phase, indexed row count, cache hits, backend engine, and any model-load errors.
