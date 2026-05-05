# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "fastapi>=0.115.0",
#   "uvicorn[standard]>=0.30.0",
#   "torch>=2.6.0",
#   "sentence-transformers>=3.4.0",
#   "transformers>=4.48.0",
#   "faiss-gpu-cu12>=1.8.0.2; platform_system == 'Linux'",
#   "faiss-cpu>=1.9.0; platform_system == 'Darwin'",
#   "numpy>=1.26.0,<3",
#   "safetensors>=0.4.5",
#   "tokenizers>=0.21.0",
# ]
# ///

"""Fast local semantic-search backend for the cleanup feed.

The browser E5 fallback has been removed; this server is the single semantic
engine. It runs the PyTorch-compatible `intfloat/multilingual-e5-large-instruct`
model through SentenceTransformers, prefers CUDA when available, falls back to
MPS on Apple Silicon or CPU, and uses FAISS GPU search when the installed FAISS
package exposes a CUDA backend.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("SEMANTIC_MODEL", "intfloat/multilingual-e5-large-instruct")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".semantic-cache"
CACHE_DIR = Path(os.environ.get("SEMANTIC_CACHE_DIR", str(DEFAULT_CACHE_DIR))).resolve()
HOST = os.environ.get("SEMANTIC_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEMANTIC_PORT", "8765"))
BATCH_SIZE = int(os.environ.get("SEMANTIC_BATCH_SIZE", "64"))
MAX_SEQ_LENGTH = int(os.environ.get("SEMANTIC_MAX_SEQ_LENGTH", "256"))
SAVE_EVERY = int(os.environ.get("SEMANTIC_SAVE_EVERY", "1024"))
E5_INSTRUCTION = (
    "You are an expert in matching Bangladeshi National Identity Card (NID) and voter registration queries. "
    "Your task is to identify the most semantically relevant question from the provided document, considering "
    "context, intent, and specific details."
)


def mps_available() -> bool:
    """Return whether PyTorch can use Apple Silicon MPS on this machine."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device() -> str:
    """Choose the fastest available PyTorch device with an env override."""
    requested = os.environ.get("SEMANTIC_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        if requested == "mps" and not mps_available():
            print("[semantic] SEMANTIC_DEVICE=mps requested but MPS is unavailable; using CPU")
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            print("[semantic] SEMANTIC_DEVICE=cuda requested but CUDA is unavailable; using CPU")
            return "cpu"
        return requested
    if requested != "auto":
        print(f"[semantic] SEMANTIC_DEVICE={requested} is invalid; using auto")
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def faiss_gpu_count() -> int:
    """Return the number of FAISS-visible GPUs without assuming GPU FAISS."""
    get_num_gpus = getattr(faiss, "get_num_gpus", None)
    if not callable(get_num_gpus):
        return 0
    try:
        return int(get_num_gpus())
    except Exception as exc:
        print(f"[semantic] FAISS GPU probe failed: {exc}", flush=True)
        return 0


def faiss_gpu_available() -> bool:
    """Return whether this FAISS install can build GPU indices."""
    return (
        faiss_gpu_count() > 0
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "index_cpu_to_gpu")
    )


def select_faiss_device() -> str:
    """Choose FAISS GPU when available, with an env override for CPU fallback."""
    requested = os.environ.get("SEMANTIC_FAISS_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "gpu", "cuda"}:
        if requested == "cpu":
            return "cpu"
        if faiss_gpu_available():
            return "gpu"
        print(f"[semantic] SEMANTIC_FAISS_DEVICE={requested} requested but FAISS GPU is unavailable; using CPU")
        return "cpu"
    if requested != "auto":
        print(f"[semantic] SEMANTIC_FAISS_DEVICE={requested} is invalid; using auto")
    return "gpu" if faiss_gpu_available() else "cpu"


def runtime_summary() -> dict[str, Any]:
    """Expose the Python/Torch/FAISS runtime so the UI and logs show the real backend."""
    return {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
        "torchCudaAvailable": torch.cuda.is_available(),
        "faissVersion": getattr(faiss, "__version__", "unknown"),
        "faissGpuCount": faiss_gpu_count(),
    }


def format_document(text: str) -> str:
    """Apply the E5 instruction format for stored dataset rows."""
    return f"Instruct: {E5_INSTRUCTION}\n{text}"


def format_query(text: str) -> str:
    """Apply the E5 instruction format for ad-hoc search queries."""
    return f"Instruct: {E5_INSTRUCTION}\n{text}"


class SemanticState:
    """Owns row documents, cached vectors, model state, and the FAISS index."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.docs: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.vector_fingerprints: dict[str, str] = {}
        self.index: faiss.IndexIDMap2 | None = None
        self.index_id_to_row_id: dict[int, str] = {}
        self.model: SentenceTransformer | None = None
        self.device = select_device()
        self.faiss_device = select_faiss_device()
        self.faiss_gpu_resources: Any | None = None
        self.status = "idle"
        self.phase = "idle"
        self.phase_done = 0
        self.phase_total = 0
        self.model_progress = 0
        self.message = "Semantic backend ready"
        self.cache_hits = 0
        self.cache_writes = 0
        self.cache_purges = 0
        self.error = ""
        self.build_thread: threading.Thread | None = None
        self.model_load_lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / "row_vectors.npz"

    def configure(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Replace the active row set and load any matching cached vectors."""
        with self.lock:
            self.docs = {
                str(doc["rowId"]): {
                    "rowId": str(doc["rowId"]),
                    "tag": str(doc.get("tag", "")),
                    "question": str(doc.get("question", "")),
                    "text": str(doc.get("text", "")),
                    "fingerprint": str(doc.get("fingerprint", "")),
                }
                for doc in docs
                if doc.get("rowId") and doc.get("text")
            }
            self._load_cache_locked()
            self._rebuild_index_locked()
            building = bool(self.build_thread and self.build_thread.is_alive())
            self.status = "indexing" if building else "ready" if self.indexed_count_locked() == len(self.docs) and self.docs else "configured"
            self.phase = self.phase if building and self.phase in {"queued", "loading", "embed"} else "configured"
            self.phase_done = self.indexed_count_locked()
            self.phase_total = len(self.docs)
            if not building:
                self.message = f"Configured {len(self.docs):,} rows on {self.device.upper()}"
            self.error = ""
            return self.configured_payload_locked()

    def start_build(self) -> dict[str, Any]:
        """Start background embedding/indexing if a build is not already active."""
        with self.lock:
            if self.build_thread and self.build_thread.is_alive():
                self.status = "indexing"
                return self.status_payload_locked()
            self.status = "indexing"
            self.phase = "queued"
            self.phase_done = self.indexed_count_locked()
            self.phase_total = len(self.docs)
            self.message = "Vector index build queued"
            self.error = ""
            self.build_thread = threading.Thread(target=self._build_worker, name="semantic-build", daemon=True)
            self.build_thread.start()
            return self.status_payload_locked()

    def upsert(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert or update changed rows, invalidating stale vectors by fingerprint."""
        with self.lock:
            for doc in rows:
                row_id = str(doc.get("rowId", ""))
                if not row_id:
                    continue
                fingerprint = str(doc.get("fingerprint", ""))
                self.docs[row_id] = {
                    "rowId": row_id,
                    "tag": str(doc.get("tag", "")),
                    "question": str(doc.get("question", "")),
                    "text": str(doc.get("text", "")),
                    "fingerprint": fingerprint,
                }
                if self.vector_fingerprints.get(row_id) != fingerprint:
                    self.vectors.pop(row_id, None)
                    self.vector_fingerprints.pop(row_id, None)
            self.status = "configured"
            self.phase = "configured"
            self.phase_done = self.indexed_count_locked()
            self.phase_total = len(self.docs)
            self.message = f"Queued {len(rows):,} changed row vectors"
            self._rebuild_index_locked()
        return self.start_build()

    def remove(self, row_ids: list[str]) -> dict[str, Any]:
        """Drop removed rows from the searchable corpus and persisted vector cache."""
        with self.lock:
            removed = 0
            for row_id in row_ids:
                row_id = str(row_id)
                if row_id in self.docs:
                    removed += 1
                self.docs.pop(row_id, None)
                self.vectors.pop(row_id, None)
                self.vector_fingerprints.pop(row_id, None)
            self.cache_purges += removed
            self._rebuild_index_locked()
            self._save_cache_locked()
            self.status = "ready" if self.indexed_count_locked() == len(self.docs) else "configured"
            self.phase = "remove"
            self.phase_done = removed
            self.phase_total = removed
            self.message = f"Removed {removed:,} rows from vector index"
            return self.status_payload_locked()

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Embed a free-text query and return the top semantic matches."""
        self._require_ready()
        query_vector = self._embed_queries([query])[0].astype("float32")
        return self._search_vector(query_vector, top_k=top_k)

    def status_payload_locked(self) -> dict[str, Any]:
        """Return a UI-friendly progress payload while the caller holds the lock."""
        indexed = self.indexed_count_locked()
        total = len(self.docs)
        return {
            "type": "status",
            "status": self.status,
            "phase": self.phase,
            "phaseDone": self.phase_done,
            "phaseTotal": self.phase_total,
            "indexed": indexed,
            "total": total,
            "evidenceCount": total,
            "modelProgress": self.model_progress,
            "progress": self.model_progress if self.phase == "loading" else None,
            "message": self.message,
            "cacheHits": self.cache_hits,
            "cacheWrites": self.cache_writes,
            "cachePurges": self.cache_purges,
            "engine": f"python-{self.device}-faiss-{self.faiss_device}",
            "faissDevice": self.faiss_device,
            "error": self.error,
            **runtime_summary(),
        }

    def configured_payload_locked(self) -> dict[str, Any]:
        """Return the same progress shape as status with a configure event type."""
        payload = self.status_payload_locked()
        payload["type"] = "configured"
        return payload

    def indexed_count_locked(self) -> int:
        """Count vectors that still match the active row fingerprints."""
        count = 0
        for row_id, doc in self.docs.items():
            if row_id in self.vectors and self.vector_fingerprints.get(row_id) == doc.get("fingerprint"):
                count += 1
        return count

    def _require_ready(self) -> None:
        """Reject semantic queries until every active row has a fresh vector."""
        with self.lock:
            ready = bool(self.docs) and self.indexed_count_locked() == len(self.docs) and self.index is not None
            if not ready:
                raise HTTPException(status_code=409, detail="Vector index is not complete yet")

    def _build_worker(self) -> None:
        """Embed missing rows in batches, persist cache, then rebuild FAISS."""
        try:
            while True:
                with self.lock:
                    pending = [
                        doc for row_id, doc in self.docs.items()
                        if row_id not in self.vectors or self.vector_fingerprints.get(row_id) != doc.get("fingerprint")
                    ]
                    self.phase_total = len(pending)
                    if not pending:
                        self._rebuild_index_locked()
                        self._save_cache_locked()
                        self.status = "ready"
                        self.phase = "ready"
                        self.phase_done = self.indexed_count_locked()
                        self.phase_total = len(self.docs)
                        self.message = f"Vector index ready with {self.phase_done:,} rows"
                        return
                    self.status = "loading" if self.model is None else "indexing"
                    self.phase = "loading" if self.model is None else "embed"
                    self.phase_done = 0
                    self.message = (
                        f"Loading {MODEL_ID} on {self.device.upper()}"
                        if self.model is None
                        else f"Embedding {len(pending):,} changed rows on {self.device.upper()}"
                    )

                self._ensure_model()

                with self.lock:
                    pending = [
                        doc for row_id, doc in self.docs.items()
                        if row_id not in self.vectors or self.vector_fingerprints.get(row_id) != doc.get("fingerprint")
                    ]
                    if not pending:
                        continue
                    self.status = "indexing"
                    self.phase = "embed"
                    self.phase_done = 0
                    self.phase_total = len(pending)
                    self.message = f"Embedding {len(pending):,} changed rows on {self.device.upper()}"

                for start in range(0, len(pending), BATCH_SIZE):
                    chunk = pending[start:start + BATCH_SIZE]
                    vectors = self._embed_documents([doc["text"] for doc in chunk])
                    with self.lock:
                        for doc, vector in zip(chunk, vectors, strict=False):
                            current = self.docs.get(doc["rowId"])
                            if not current or current.get("fingerprint") != doc.get("fingerprint"):
                                continue
                            self.vectors[doc["rowId"]] = vector.astype("float32")
                            self.vector_fingerprints[doc["rowId"]] = doc["fingerprint"]
                            self.cache_writes += 1
                        self.phase_done = min(start + len(chunk), len(pending))
                        self.phase_total = len(pending)
                        self.message = f"Embedded {self.phase_done:,} / {len(pending):,} changed rows"
                        if self.phase_done % SAVE_EVERY == 0:
                            self._save_cache_locked()
                with self.lock:
                    self._rebuild_index_locked()
                    self._save_cache_locked()
        except Exception as exc:
            with self.lock:
                self.status = "error"
                self.phase = "error"
                self.error = str(exc)
                self.message = f"Semantic backend failed: {exc}"
            print(f"[semantic] build failed: {exc}", flush=True)

    def _ensure_model(self) -> None:
        """Load SentenceTransformers once; MPS/CUDA choice is made at startup."""
        with self.model_load_lock:
            with self.lock:
                if self.model is not None:
                    return
                self.status = "loading"
                self.phase = "loading"
                self.phase_done = 0
                self.phase_total = 100
                self.model_progress = 0
                self.message = f"Loading {MODEL_ID} on {self.device.upper()}"
            print(f"[semantic] loading {MODEL_ID} on {self.device.upper()}", flush=True)
            try:
                model = SentenceTransformer(MODEL_ID, device=self.device)
                actual_device = self.device
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                allow_fallback = os.environ.get("SEMANTIC_ALLOW_CPU_FALLBACK", "").lower() in {"1", "true", "yes"}
                is_cuda_oom = self.device == "cuda" and "out of memory" in str(exc).lower()
                if not (allow_fallback and is_cuda_oom):
                    raise
                print("[semantic] CUDA OOM while loading model; falling back to CPU", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                model = SentenceTransformer(MODEL_ID, device="cpu")
                actual_device = "cpu"
            model.max_seq_length = MAX_SEQ_LENGTH
            with self.lock:
                self.model = model
                self.device = actual_device
                self.model_progress = 100
                self.message = f"Loaded {MODEL_ID} on {self.device.upper()}"

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode dataset rows with document-side E5 instructions."""
        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded")
        payload = [format_document(text) for text in texts]
        if hasattr(model, "encode_document"):
            vectors = model.encode_document(
                payload,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vectors = model.encode(
                payload,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def _embed_queries(self, texts: list[str]) -> np.ndarray:
        """Encode search text with query-side E5 instructions."""
        self._ensure_model()
        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded")
        payload = [format_query(text) for text in texts]
        if hasattr(model, "encode_query"):
            vectors = model.encode_query(
                payload,
                batch_size=min(BATCH_SIZE, 16),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vectors = model.encode(
                payload,
                batch_size=min(BATCH_SIZE, 16),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def _search_vector(self, vector: np.ndarray, top_k: int = 10, exclude: set[str] | None = None) -> dict[str, Any]:
        """Run cosine/IP search against FAISS and map internal IDs back to rows."""
        exclude = exclude or set()
        with self.lock:
            index = self.index
            id_to_row = dict(self.index_id_to_row_id)
            docs = dict(self.docs)
        if index is None or not id_to_row:
            return {"results": [], **self.status_payload()}
        limit = min(max(top_k + len(exclude) + 8, top_k), index.ntotal)
        scores, ids = index.search(vector.reshape(1, -1).astype("float32"), limit)
        results = []
        for score, internal_id in zip(scores[0], ids[0], strict=False):
            if internal_id < 0:
                continue
            row_id = id_to_row.get(int(internal_id))
            if not row_id or row_id in exclude:
                continue
            doc = docs.get(row_id)
            if not doc:
                continue
            results.append({
                "rowId": row_id,
                "tag": doc.get("tag", ""),
                "question": doc.get("question", ""),
                "score": float(score),
            })
            if len(results) >= top_k:
                break
        return {"results": results, **self.status_payload()}

    def status_payload(self) -> dict[str, Any]:
        """Return the current backend status with locking handled internally."""
        with self.lock:
            return self.status_payload_locked()

    def _load_cache_locked(self) -> None:
        """Load cached vectors whose row fingerprints match the active dataset."""
        if not self.cache_path.exists():
            return
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            row_ids = data["row_ids"].astype(str)
            fingerprints = data["fingerprints"].astype(str)
            vectors = data["vectors"].astype("float32")
            loaded = 0
            for row_id, fingerprint, vector in zip(row_ids, fingerprints, vectors, strict=False):
                doc = self.docs.get(str(row_id))
                if not doc or doc.get("fingerprint") != str(fingerprint):
                    continue
                self.vectors[str(row_id)] = vector
                self.vector_fingerprints[str(row_id)] = str(fingerprint)
                loaded += 1
            self.cache_hits = loaded
        except Exception as exc:
            print(f"[semantic] failed to load cache: {exc}", flush=True)

    def _save_cache_locked(self) -> None:
        """Persist active vectors atomically so a restart can resume quickly."""
        active = [
            (row_id, self.vector_fingerprints[row_id], self.vectors[row_id])
            for row_id in self.docs
            if row_id in self.vectors and row_id in self.vector_fingerprints
        ]
        if not active:
            return
        row_ids = np.asarray([item[0] for item in active], dtype="U128")
        fingerprints = np.asarray([item[1] for item in active], dtype="U64")
        vectors = np.stack([item[2] for item in active]).astype("float32")
        tmp_path = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, row_ids=row_ids, fingerprints=fingerprints, vectors=vectors)
        tmp_path.replace(self.cache_path)

    def _make_faiss_index(self, dim: int) -> faiss.IndexIDMap2:
        """Create a FAISS ID-mapped inner-product index on GPU when available."""
        base = faiss.IndexFlatIP(dim)
        if self.faiss_device == "gpu":
            try:
                self.faiss_gpu_resources = faiss.StandardGpuResources()
                gpu_base = faiss.index_cpu_to_gpu(self.faiss_gpu_resources, 0, base)
                return faiss.IndexIDMap2(gpu_base)
            except Exception as exc:
                print(f"[semantic] failed to create FAISS GPU index; falling back to CPU: {exc}", flush=True)
                self.faiss_device = "cpu"
                self.faiss_gpu_resources = None
        return faiss.IndexIDMap2(base)

    def _rebuild_index_locked(self) -> None:
        """Rebuild the FAISS inner-product index from normalized vectors."""
        active_row_ids = [
            row_id for row_id, doc in self.docs.items()
            if row_id in self.vectors and self.vector_fingerprints.get(row_id) == doc.get("fingerprint")
        ]
        if not active_row_ids:
            self.index = None
            self.index_id_to_row_id = {}
            return
        matrix = np.stack([self.vectors[row_id] for row_id in active_row_ids]).astype("float32")
        dim = int(matrix.shape[1])
        index = self._make_faiss_index(dim)
        ids = np.arange(len(active_row_ids), dtype=np.int64)
        index.add_with_ids(matrix, ids)
        self.index = index
        self.index_id_to_row_id = {int(index_id): row_id for index_id, row_id in enumerate(active_row_ids)}


state = SemanticState()
app = FastAPI(title="EC Cleanup Semantic Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    payload = state.status_payload()
    payload["ok"] = True
    return payload


@app.get("/status")
def status() -> dict[str, Any]:
    return state.status_payload()


@app.post("/configure")
def configure(payload: dict[str, Any]) -> dict[str, Any]:
    return state.configure(payload.get("docs", []))


@app.post("/build")
def build() -> dict[str, Any]:
    return state.start_build()


@app.post("/upsert")
def upsert(payload: dict[str, Any]) -> dict[str, Any]:
    return state.upsert(payload.get("rows", []))


@app.post("/remove")
def remove(payload: dict[str, Any]) -> dict[str, Any]:
    return state.remove(payload.get("rowIds", []))


@app.post("/search")
def search(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        return {"results": [], **state.status_payload()}
    return state.search(query, top_k=int(payload.get("topK", 10)))


if __name__ == "__main__":
    print(
        json.dumps({
            "event": "semantic-server-start",
            "host": HOST,
            "port": PORT,
            "model": MODEL_ID,
            "device": state.device,
            "faissDevice": state.faiss_device,
            "cache": str(CACHE_DIR),
            **runtime_summary(),
        }),
        flush=True,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
