# EC Question Cleanup Feed

Single-screen cleanup workbench for Bangladesh Election Commission chatbot rows from `question_tag.csv`.

The app renders one virtualized feed card per CSV row. Rows can be searched by question text, reordered with a tag-first selector, edited, deleted, restored, bulk-deleted, added, saved as CSV, and bookmarked with named rollback points. Semantic search uses multilingual E5 embeddings; clicking a row copies that row's question into the same Top-10 search path as a typed query.

## Run

```bash
npm install
npm run dev -- --port 5173
```

Open `http://localhost:5173/`.

Run the CUDA semantic backend in a second terminal before opening the UI:

```bash
npm run semantic:server
```

The npm script launches `/home/synesis/venv-election-commission/bin/python` by default and requests `SEMANTIC_DEVICE=cuda` plus `SEMANTIC_FAISS_DEVICE=gpu`. Override `SEMANTIC_PYTHON`, `SEMANTIC_DEVICE`, or `SEMANTIC_FAISS_DEVICE` only when you need a different runtime.

If that venv needs to be rebuilt, install the pinned CUDA runtime dependencies with:

```bash
/home/synesis/venv-election-commission/bin/python -m pip install -r requirements-linux-cuda.txt
```

The frontend connects to the semantic backend automatically. On localhost it uses `http://127.0.0.1:8765`; when opened from a LAN IP such as `http://172.16.213.6:5173`, it uses `http://172.16.213.6:8765`. If an old `localStorage.semanticBackendUrl` points at a dead backend, the app falls back to that page-host default and clears the stale setting. The Python backend is required; the browser E5 fallback was removed because E5-large can consume enough Chrome memory to make the tab unresponsive.

After the page loads, click **Build vector index** if the vector count is not complete. The button creates cached vectors for all active rows, and a fully cached index returns to ready without loading another E5 model instance. Search and row-click similarity stay disabled until the vector count reaches `62,978 / 62,978`, so the Top-10 panel only shows full-dataset results.

If port `5173` is busy:

```bash
npm run dev -- --port 5174
```

## LAN Sharing

Start the frontend and backend on LAN-visible interfaces:

```bash
npm run dev:lan
npm run semantic:server:lan
```

Find your machine's LAN IP:

```bash
hostname -I
```

Then a colleague on the same network can open:

```text
http://<your-lan-ip>:5173/
```

The backend allows browser origins from localhost and private LAN addresses by default. If your machine firewall blocks access, allow inbound TCP ports `5173` and `8765`.

## Build

```bash
npm run build
```

## Data

- `question_tag.csv` is the active source file.
- It uses `question,tag` columns.
- The feed keeps row-level data, so repeated tags remain separate cards.
- The feed-local search filters visible rows by question text only. This is separate from E5 semantic search.
- The tag-first selector keeps the full feed available but moves rows from the selected tag to the top.
- Semantic search is global over active rows.

## Semantic Models

- Python backend: `intfloat/multilingual-e5-large-instruct`, loaded by SentenceTransformers/PyTorch on CUDA by default.
- FAISS backend: GPU `IndexFlatIP` wrapped in `IndexIDMap2` when `faiss-gpu-cu12` exposes a CUDA GPU. It falls back to CPU FAISS only if GPU FAISS is unavailable or explicitly disabled with `SEMANTIC_FAISS_DEVICE=cpu`.
- The old browser fallback used an ONNX model in Chrome and has been removed to avoid duplicate model loads and memory pressure.
- Backend row vectors are cached under `.semantic-cache/row_vectors.npz`; closing the browser tab does not delete that cache. The cache path is repo-relative unless `SEMANTIC_CACHE_DIR` overrides it.

## Cleanup Features

- Scrollable virtualized feed for the full `question_tag.csv` dataset.
- Feed-local question search and tag-first ordering for faster row browsing without running semantic search.
- A single copyable CSV row chip on each original row card.
- Semantic result dock that shows Top-10 only after the full vector index is complete and after a typed query or row click.
- Per-row edit, delete, restore, click-to-inspect, and add-sibling actions. Row-click inspection becomes active only after the full vector index is ready.
- Checkbox plus shift-click range selection for bulk delete.
- Undo/redo with a persisted operation log. Cleanup edits are autosaved in the browser.
- Named rollback points that restore the dataset to a prior cleanup step.
- Save clean CSV and removed-row CSV downloads.
- Python semantic backend using `/home/synesis/venv-election-commission/bin/python`, CUDA PyTorch embeddings, and FAISS GPU Top-10 search.
- CSV loading and parsing run asynchronously so the feed can paint without blocking the tab.

## Dev Logging

Use the backend status endpoint:

```bash
curl http://127.0.0.1:8765/status
```

The status payload reports the current phase, indexed row count, cache hits, backend engine, and any model-load errors.
It also reports `pythonExecutable`, PyTorch CUDA availability, FAISS version, `faissDevice`, and `faissGpuCount` so you can confirm the GPU runtime is active.
