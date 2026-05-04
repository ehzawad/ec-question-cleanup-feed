import { enableMapSet } from 'immer';
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { get as idbGet, set as idbSet } from 'idb-keyval';
import { datasetHash, fingerprintRow, rowToWorkerDoc } from './rowCsv.js';

enableMapSet();

const REVISION_KEY = 'ec-cleanup-feed:revisions:v1';
const HISTORY_LIMIT = 200;
const REPLAY_CHUNK_SIZE = 250;
let persistTimer = null;

const initialState = {
  baseHash: '',
  rows: {
    byId: {},
    allIds: [],
    activeIds: [],
    removedIds: []
  },
  selection: {
    multi: new Set(),
    shiftAnchorRowId: null
  },
  filters: {
    showRemoved: false
  },
  search: {
    // Similarity results are intentionally empty while browsing. They are only
    // populated after the full index is ready and the user searches or clicks a row.
    mode: 'idle',
    textQuery: '',
    activeRequestId: 0,
    results: [],
    topKCache: [],
    partial: false,
    indexedFraction: 0,
    status: 'idle',
    message: '',
    error: ''
  },
  embeddings: {
    // Progress from the mandatory Python/MPS backend.
    indexed: 0,
    total: 0,
    status: 'idle',
    phase: 'idle',
    phaseDone: 0,
    phaseTotal: 0,
    modelProgress: 0,
    message: '',
    engine: 'python-mps-faiss',
    cacheHits: 0,
    cacheWrites: 0,
    cachePurges: 0,
    quotaWarning: ''
  },
  history: {
    maxDepth: HISTORY_LIMIT
  },
  revisions: {
    // The durable cleanup timeline is an op log. Checkpoints are labels that
    // point to an op id, not full copies of the 62k-row dataset.
    nextOpId: 1,
    currentOpId: 0,
    log: [],
    labels: []
  },
  flags: {
    dirty: false,
    lastExportedOpId: 0
  },
  status: {
    boot: 'loading',
    activeRows: 0,
    removedRows: 0
  },
  ui: {
    editingRowId: null,
    composer: {
      open: false,
      afterRowId: null,
      tag: '',
      question: ''
    }
  },
  worker: {
    runId: 1
  }
};

export const useCleanupStore = create(immer((set, get) => ({
  ...initialState,

  async hydrate(baseRows) {
    const hash = datasetHash(baseRows);
    get().startHydrate();
    get().appendHydrateRows(baseRows);
    await get().finishHydrate(hash);
  },

  startHydrate() {
    set((state) => {
      state.baseHash = '';
      state.rows = rowsToState([]);
      state.status.boot = 'loading';
      state.revisions.nextOpId = 1;
      state.revisions.currentOpId = 0;
      state.revisions.log = [];
      state.revisions.labels = [];
      state.flags.dirty = false;
      state.flags.lastExportedOpId = 0;
      state.selection.multi.clear();
      state.selection.shiftAnchorRowId = null;
      state.search.mode = 'idle';
      state.search.textQuery = '';
      state.search.results = [];
      state.search.topKCache = [];
      updateCounts(state);
    });
  },

  appendHydrateRows(rowsChunk) {
    if (!rowsChunk?.length) return;
    set((state) => {
      appendRowsIntoState(state.rows, rowsChunk);
      updateCounts(state);
    });
  },

  async finishHydrate(hash) {
    let saved = null;
    try {
      saved = await idbGet(REVISION_KEY);
    } catch {
      saved = null;
    }

    const hasSavedTimeline = saved?.baseHash === hash && Array.isArray(saved.log) && saved.log.length;
    set((state) => {
      state.baseHash = hash;
      state.status.boot = hasSavedTimeline ? 'replaying' : 'ready';
      state.revisions.nextOpId = 1;
      state.revisions.currentOpId = 0;
      state.revisions.log = [];
      state.revisions.labels = [];
      state.flags.dirty = false;
      state.flags.lastExportedOpId = 0;
      updateCounts(state);
    });

    if (hasSavedTimeline) {
      await get().replaySavedTimeline(saved);
    } else if (saved?.baseHash === hash && Array.isArray(saved.labels)) {
      set((state) => {
        state.revisions.labels = saved.labels;
      });
    }
  },

  async replaySavedTimeline(saved) {
    const log = Array.isArray(saved.log) ? saved.log : [];
    const targetOpId = Number(saved.currentOpId) || lastOpId(log);
    const entries = log
      .filter((entry) => entry.id <= targetOpId)
      .sort((a, b) => a.id - b.id);

    set((state) => {
      state.revisions.log = log;
      state.revisions.labels = Array.isArray(saved.labels) ? saved.labels : [];
      state.revisions.currentOpId = 0;
      state.revisions.nextOpId = Math.max(lastOpId(log) + 1, Number(saved.nextOpId) || 1);
      state.flags.lastExportedOpId = Number(saved.lastExportedOpId) || 0;
      state.flags.dirty = targetOpId !== state.flags.lastExportedOpId;
      state.status.boot = 'replaying';
    });

    for (let start = 0; start < entries.length; start += REPLAY_CHUNK_SIZE) {
      const chunk = entries.slice(start, start + REPLAY_CHUNK_SIZE);
      set((state) => {
        for (const op of chunk) {
          applyOp(state.rows, op, 'forward');
          state.revisions.currentOpId = op.id;
        }
        updateCounts(state);
      });
      await yieldToBrowser();
    }

    set((state) => {
      state.revisions.currentOpId = targetOpId;
      state.flags.dirty = state.revisions.currentOpId !== state.flags.lastExportedOpId;
      state.status.boot = 'ready';
      updateCounts(state);
    });
  },

  getActiveDocs() {
    const { rows } = get();
    return rows.activeIds
      .map((rowId) => rows.byId[rowId])
      .filter(Boolean)
      .map(rowToWorkerDoc);
  },

  removedRowIds() {
    const { rows } = get();
    return [...rows.removedIds];
  },

  workerRunId() {
    return get().worker.runId;
  },

  markSemanticSyncPending(message = 'Syncing changed rows to vector index') {
    set((state) => {
      const total = Math.max(Number(state.embeddings.total || 0), Number(state.status.activeRows || 0));
      state.embeddings.status = 'indexing';
      state.embeddings.phase = 'sync';
      state.embeddings.total = total;
      state.embeddings.indexed = Math.min(Number(state.embeddings.indexed || 0), Math.max(0, total - 1));
      state.embeddings.modelProgress = Math.max(Number(state.embeddings.modelProgress || 0), 100);
      state.embeddings.message = message;
      if (state.search.mode === 'text' && state.search.textQuery.trim()) {
        state.search.status = 'indexing';
        state.search.message = 'Waiting for vector index sync';
      }
    });
  },

  setSearchText(textQuery) {
    set((state) => {
      state.search.textQuery = textQuery;
      if (textQuery.trim()) {
        state.search.mode = 'text';
      }
    });
  },

  setSearchMode({ mode, textQuery = '', reason = 'manual' }) {
    const requestId = get().search.activeRequestId + 1;
    set((state) => {
      state.search.mode = mode;
      state.search.textQuery = textQuery;
      state.search.activeRequestId = requestId;
      state.search.results = [];
      state.search.partial = false;
      state.search.indexedFraction = 0;
      state.search.status = mode === 'idle' ? 'idle' : 'querying';
      state.search.message = reason === 'manual' ? 'Running semantic search' : 'Refreshing semantic search';
      state.search.error = '';
    });
    return requestId;
  },

  clearSearch() {
    set((state) => {
      state.search.mode = 'idle';
      state.search.textQuery = '';
      state.search.results = [];
      state.search.topKCache = [];
      state.search.partial = false;
      state.search.indexedFraction = 0;
      state.search.status = 'idle';
      state.search.message = '';
      state.search.error = '';
    });
  },

  setShowRemoved(showRemoved) {
    set((state) => {
      state.filters.showRemoved = Boolean(showRemoved);
    });
  },

  toggleSelection(rowId, isRange, visibleIds) {
    set((state) => {
      const row = state.rows.byId[rowId];
      if (!row) return;
      if (isRange && state.selection.shiftAnchorRowId) {
        const start = visibleIds.indexOf(state.selection.shiftAnchorRowId);
        const end = visibleIds.indexOf(rowId);
        if (start >= 0 && end >= 0) {
          const [from, to] = start < end ? [start, end] : [end, start];
          for (const id of visibleIds.slice(from, to + 1)) {
            if (state.rows.byId[id]?.status === 'active') state.selection.multi.add(id);
          }
          return;
        }
      }
      if (state.selection.multi.has(rowId)) state.selection.multi.delete(rowId);
      else if (row.status === 'active') state.selection.multi.add(rowId);
      state.selection.shiftAnchorRowId = rowId;
    });
  },

  clearSelection() {
    set((state) => {
      state.selection.multi.clear();
      state.selection.shiftAnchorRowId = null;
    });
  },

  setEditingRow(rowId) {
    set((state) => {
      state.ui.editingRowId = rowId;
    });
  },

  setComposer(nextComposer) {
    set((state) => {
      state.ui.composer = {
        ...state.ui.composer,
        ...nextComposer
      };
    });
  },

  editRow(rowId, patch) {
    let changed = null;
    set((state) => {
      const row = state.rows.byId[rowId];
      if (!row) return;
      const next = {
        tag: patch.tag ?? row.tag,
        question: patch.question ?? row.question
      };
      if (next.tag === row.tag && next.question === row.question) {
        state.ui.editingRowId = null;
        return;
      }
      const op = nextOp(state, 'edit', {
        rowId,
        before: rowSnapshot(row),
        after: {
          ...rowSnapshot(row),
          tag: next.tag,
          question: next.question,
          fingerprint: patch.fingerprint || fingerprintRow(next.tag, next.question),
          updatedAt: Date.now()
        }
      });
      applyOp(state.rows, op, 'forward');
      recordOp(state, op);
      state.ui.editingRowId = null;
      changed = { ...state.rows.byId[rowId] };
      updateCounts(state);
    });
    persistSoon(get());
    return changed;
  },

  removeRows(rowIds, reason = 'manual cleanup') {
    const ids = [...new Set(rowIds)].filter(Boolean);
    const changed = [];
    set((state) => {
      const before = ids
        .map((rowId) => state.rows.byId[rowId])
        .filter((row) => row?.status !== 'removed')
        .map(rowSnapshot);
      if (!before.length) return;
      const op = nextOp(state, 'remove', {
        rowIds: before.map((row) => row.rowId),
        before,
        reason
      });
      applyOp(state.rows, op, 'forward');
      recordOp(state, op);
      for (const rowId of op.payload.rowIds) {
        state.selection.multi.delete(rowId);
        changed.push({ ...state.rows.byId[rowId] });
      }
      const removedSet = new Set(op.payload.rowIds);
      state.search.results = state.search.results.filter((result) => !removedSet.has(result.rowId));
      state.search.topKCache = state.search.topKCache.filter((result) => !removedSet.has(result.rowId));
      updateCounts(state);
    });
    if (changed.length) persistSoon(get());
    return changed;
  },

  restoreRows(rowIds) {
    const ids = [...new Set(rowIds)].filter(Boolean);
    const changed = [];
    set((state) => {
      const before = ids
        .map((rowId) => state.rows.byId[rowId])
        .filter((row) => row?.status === 'removed')
        .map(rowSnapshot);
      if (!before.length) return;
      const op = nextOp(state, 'restore', {
        rowIds: before.map((row) => row.rowId),
        before
      });
      applyOp(state.rows, op, 'forward');
      recordOp(state, op);
      for (const rowId of op.payload.rowIds) changed.push({ ...state.rows.byId[rowId] });
      updateCounts(state);
    });
    if (changed.length) persistSoon(get());
    return changed;
  },

  addRow({ tag, question, afterRowId = null }) {
    const cleanTag = String(tag ?? '').trim();
    const cleanQuestion = String(question ?? '').trim();
    if (!cleanTag || !cleanQuestion) return null;
    let added = null;
    set((state) => {
      const originalIndex = nextOriginalIndex(state.rows);
      const row = {
        rowId: `added-${Date.now()}-${Math.round(Math.random() * 100000)}`,
        originalIndex,
        originalQuestion: '',
        originalTag: '',
        question: cleanQuestion,
        tag: cleanTag,
        status: 'active',
        isAdded: true,
        removedReason: '',
        fingerprint: fingerprintRow(cleanTag, cleanQuestion),
        updatedAt: Date.now()
      };
      const op = nextOp(state, 'add', { row, afterRowId });
      applyOp(state.rows, op, 'forward');
      recordOp(state, op);
      state.ui.composer = { open: false, afterRowId: null, tag: '', question: '' };
      added = { ...row };
      updateCounts(state);
    });
    if (added) persistSoon(get());
    return added;
  },

  undo() {
    const changed = [];
    set((state) => {
      const op = [...state.revisions.log]
        .filter((entry) => entry.id <= state.revisions.currentOpId)
        .sort((a, b) => b.id - a.id)[0];
      if (!op) return;
      applyOp(state.rows, op, 'inverse');
      if (op.kind === 'add') {
        changed.push({ ...op.payload.row, status: 'removed' });
      } else {
        changed.push(...changedRowsAfter(state.rows, changedRowsForOp(state.rows, op)));
      }
      state.revisions.currentOpId = previousOpId(state.revisions.log, op.id);
      state.flags.dirty = state.revisions.currentOpId !== state.flags.lastExportedOpId;
      state.selection.multi.clear();
      updateCounts(state);
    });
    if (changed.length) persistSoon(get());
    return changed;
  },

  redo() {
    const changed = [];
    set((state) => {
      const op = state.revisions.log
        .filter((entry) => entry.id > state.revisions.currentOpId)
        .sort((a, b) => a.id - b.id)[0];
      if (!op) return;
      applyOp(state.rows, op, 'forward');
      state.revisions.currentOpId = op.id;
      changed.push(...changedRowsForOp(state.rows, op));
      state.flags.dirty = state.revisions.currentOpId !== state.flags.lastExportedOpId;
      state.selection.multi.clear();
      updateCounts(state);
    });
    if (changed.length) persistSoon(get());
    return changedRowsAfter(get().rows, changed);
  },

  saveSnapshot(name) {
    set((state) => {
      state.revisions.labels.push({
        id: `label-${Date.now()}`,
        name,
        opId: state.revisions.currentOpId,
        ts: Date.now()
      });
    });
    persistNow(get());
  },

  restoreSnapshot(labelId) {
    let changed = [];
    set((state) => {
      const label = state.revisions.labels.find((entry) => entry.id === labelId);
      if (!label) return;
      const currentOpId = state.revisions.currentOpId;
      const targetOpId = Number(label.opId) || 0;
      if (targetOpId === currentOpId) return;
      const changedIds = new Set();
      const ops = targetOpId < currentOpId
        ? state.revisions.log
          .filter((entry) => entry.id > targetOpId && entry.id <= currentOpId)
          .sort((a, b) => b.id - a.id)
        : state.revisions.log
          .filter((entry) => entry.id > currentOpId && entry.id <= targetOpId)
          .sort((a, b) => a.id - b.id);
      const direction = targetOpId < currentOpId ? 'inverse' : 'forward';
      for (const op of ops) {
        for (const rowId of changedRowsForOp(state.rows, op)) changedIds.add(rowId);
        applyOp(state.rows, op, direction);
      }
      state.revisions.currentOpId = targetOpId;
      state.selection.multi.clear();
      state.search.mode = 'idle';
      state.search.results = [];
      state.search.status = 'idle';
      state.worker.runId += 1;
      changed = [...changedIds].map((rowId) => (
        state.rows.byId[rowId]
          ? { ...state.rows.byId[rowId] }
          : { rowId, status: 'removed' }
      ));
      state.flags.dirty = state.revisions.currentOpId !== state.flags.lastExportedOpId;
      updateCounts(state);
    });
    if (changed.length) persistSoon(get());
    return changed;
  },

  markExported() {
    set((state) => {
      state.flags.lastExportedOpId = state.revisions.currentOpId;
      state.flags.dirty = false;
    });
    persistNow(get());
  },

  handleWorkerMessage(message) {
    // Backend status/search events share one message shape, so React components
    // do not need to know whether the server is configuring, indexing, or searching.
    set((state) => {
      if (message.runId && message.runId !== state.worker.runId) return;
      if (message.type === 'configured') {
        state.embeddings.indexed = message.indexed ?? state.embeddings.indexed;
        state.embeddings.total = message.total ?? message.evidenceCount ?? state.embeddings.total;
        state.embeddings.status = message.status ?? 'configured';
        state.embeddings.phase = message.phase ?? 'configured';
        state.embeddings.phaseDone = message.phaseDone ?? state.embeddings.phaseDone;
        state.embeddings.phaseTotal = message.phaseTotal ?? state.embeddings.phaseTotal;
        state.embeddings.modelProgress = message.modelProgress ?? state.embeddings.modelProgress;
        state.embeddings.engine = message.engine ?? state.embeddings.engine;
        state.embeddings.cacheHits = message.cacheHits ?? state.embeddings.cacheHits;
        state.embeddings.cacheWrites = message.cacheWrites ?? state.embeddings.cacheWrites;
        state.embeddings.cachePurges = message.cachePurges ?? state.embeddings.cachePurges;
        state.embeddings.message = message.message ?? `Configured ${state.embeddings.total.toLocaleString()} rows`;
        state.search.message = `Configured ${state.embeddings.total.toLocaleString()} rows`;
        return;
      }
      if (message.type === 'status') {
        state.search.status = message.status ?? state.search.status;
        state.search.message = message.message ?? state.search.message;
        state.embeddings.indexed = message.indexed ?? state.embeddings.indexed;
        state.embeddings.total = message.total ?? message.evidenceCount ?? state.embeddings.total;
        state.embeddings.status = message.status ?? state.embeddings.status;
        state.embeddings.phase = message.phase ?? state.embeddings.phase;
        state.embeddings.phaseDone = message.phaseDone ?? state.embeddings.phaseDone;
        state.embeddings.phaseTotal = message.phaseTotal ?? state.embeddings.phaseTotal;
        state.embeddings.message = message.message ?? state.embeddings.message;
        state.embeddings.engine = message.engine ?? state.embeddings.engine;
        state.embeddings.modelProgress = message.modelProgress ?? state.embeddings.modelProgress;
        if (message.phase === 'loading' && Number.isFinite(Number(message.progress))) {
          state.embeddings.modelProgress = Math.max(0, Math.min(100, Math.round(Number(message.progress))));
        }
        if (message.phase === 'embed' || (message.status === 'ready' && state.embeddings.phase !== 'loading')) {
          state.embeddings.modelProgress = 100;
        }
        state.embeddings.cacheHits = message.cacheHits ?? state.embeddings.cacheHits;
        state.embeddings.cacheWrites = message.cacheWrites ?? state.embeddings.cacheWrites;
        state.embeddings.cachePurges = message.cachePurges ?? state.embeddings.cachePurges;
        return;
      }
      if (message.type === 'results') {
        if (message.requestId !== state.search.activeRequestId) return;
        state.search.partial = Boolean(message.partial);
        state.search.indexedFraction = message.indexedFraction ?? state.search.indexedFraction;
        state.embeddings.indexed = message.indexed ?? state.embeddings.indexed;
        state.embeddings.total = message.total ?? state.embeddings.total;
        if (message.partial) {
          state.search.status = 'indexing';
          state.search.message = 'Waiting for full vector index';
          state.search.results = [];
          state.search.topKCache = [];
          return;
        }
        state.search.status = 'ready';
        state.search.message = `Top ${Math.min(10, message.results?.length ?? 0)} matches ready`;
        state.search.results = (message.results ?? []).slice(0, 10);
        state.search.topKCache = message.results ?? [];
        state.search.error = '';
        return;
      }
      if (message.type === 'embedRowsDone') {
        state.embeddings.indexed = message.indexed ?? state.embeddings.indexed;
        state.embeddings.total = message.total ?? state.embeddings.total;
        state.embeddings.status = 'indexing';
        state.embeddings.phase = 'embed';
        state.embeddings.modelProgress = 100;
        return;
      }
      if (message.type === 'quotaWarning') {
        state.embeddings.quotaWarning = message.message ?? 'Browser storage quota is nearly full.';
        return;
      }
      if (message.type === 'error') {
        if (message.requestId && message.requestId !== state.search.activeRequestId) return;
        state.search.status = 'error';
        state.search.error = message.message ?? 'Semantic backend failed';
        state.search.message = 'Semantic backend is unavailable';
      }
    });
  },

  setWorkerError(message) {
    set((state) => {
      state.search.status = 'error';
      state.search.error = message;
      state.search.message = 'Semantic backend is unavailable';
    });
  }
})));

function rowsToState(rows) {
  const byId = {};
  const allIds = [];
  const activeIds = [];
  const removedIds = [];
  appendRowsIntoState({ byId, allIds, activeIds, removedIds }, rows);
  return { byId, allIds, activeIds, removedIds };
}

function appendRowsIntoState(rowsState, rows) {
  for (const row of rows) {
    if (!row?.rowId || rowsState.byId[row.rowId]) continue;
    rowsState.byId[row.rowId] = { ...row };
    rowsState.allIds.push(row.rowId);
    if (row.status === 'removed') rowsState.removedIds.push(row.rowId);
    else rowsState.activeIds.push(row.rowId);
  }
}

function rowSnapshot(row) {
  return {
    rowId: row.rowId,
    originalIndex: row.originalIndex,
    originalQuestion: row.originalQuestion,
    originalTag: row.originalTag,
    question: row.question,
    tag: row.tag,
    status: row.status,
    isAdded: row.isAdded,
    removedReason: row.removedReason,
    fingerprint: row.fingerprint,
    updatedAt: row.updatedAt
  };
}

function nextOp(state, kind, payload) {
  return {
    id: state.revisions.nextOpId,
    ts: Date.now(),
    kind,
    payload
  };
}

function recordOp(state, op) {
  state.revisions.log = state.revisions.log.filter((entry) => entry.id <= state.revisions.currentOpId);
  state.revisions.log.push(op);
  state.revisions.currentOpId = op.id;
  state.revisions.nextOpId = op.id + 1;
  if (state.revisions.log.length > 5000) {
    state.revisions.log = state.revisions.log.slice(-5000);
  }
  state.flags.dirty = true;
}

function applyOp(rowsState, op, direction) {
  if (op.kind === 'edit') {
    const next = direction === 'forward' ? op.payload.after : op.payload.before;
    const row = rowsState.byId[op.payload.rowId];
    if (!row) return;
    Object.assign(row, next, { updatedAt: Date.now() });
    return;
  }
  if (op.kind === 'add') {
    if (direction === 'forward') {
      const row = { ...op.payload.row };
      rowsState.byId[row.rowId] = row;
      if (!rowsState.allIds.includes(row.rowId)) {
        insertAfter(rowsState.allIds, row.rowId, op.payload.afterRowId);
      }
      removeId(rowsState.removedIds, row.rowId);
      if (row.status === 'removed') insertByAllOrder(rowsState.removedIds, rowsState.allIds, row.rowId);
      else insertByAllOrder(rowsState.activeIds, rowsState.allIds, row.rowId);
    } else {
      delete rowsState.byId[op.payload.row.rowId];
      removeId(rowsState.allIds, op.payload.row.rowId);
      removeId(rowsState.activeIds, op.payload.row.rowId);
      removeId(rowsState.removedIds, op.payload.row.rowId);
    }
    return;
  }
  if (op.kind === 'remove') {
    for (const snapshot of op.payload.before) {
      const row = rowsState.byId[snapshot.rowId];
      if (!row) continue;
      if (direction === 'forward') {
        row.status = 'removed';
        row.removedReason = op.payload.reason ?? 'manual cleanup';
        row.updatedAt = Date.now();
        removeId(rowsState.activeIds, snapshot.rowId);
        insertByAllOrder(rowsState.removedIds, rowsState.allIds, snapshot.rowId);
      } else {
        Object.assign(row, snapshot, { updatedAt: Date.now() });
        removeId(rowsState.removedIds, snapshot.rowId);
        insertByAllOrder(rowsState.activeIds, rowsState.allIds, snapshot.rowId);
      }
    }
    return;
  }
  if (op.kind === 'restore') {
    for (const snapshot of op.payload.before) {
      const row = rowsState.byId[snapshot.rowId];
      if (!row) continue;
      if (direction === 'forward') {
        row.status = 'active';
        row.removedReason = '';
        row.updatedAt = Date.now();
        removeId(rowsState.removedIds, snapshot.rowId);
        insertByAllOrder(rowsState.activeIds, rowsState.allIds, snapshot.rowId);
      } else {
        Object.assign(row, snapshot, { updatedAt: Date.now() });
        removeId(rowsState.activeIds, snapshot.rowId);
        insertByAllOrder(rowsState.removedIds, rowsState.allIds, snapshot.rowId);
      }
    }
  }
}

function changedRowsForOp(rowsState, op) {
  if (op.kind === 'edit') return [op.payload.rowId];
  if (op.kind === 'add') return [op.payload.row.rowId];
  if (op.kind === 'remove' || op.kind === 'restore') return op.payload.before.map((row) => row.rowId);
  return [];
}

function changedRowsAfter(rowsState, rowIds) {
  return rowIds.map((rowId) => rowsState.byId[rowId]).filter(Boolean).map((row) => ({ ...row }));
}

function previousOpId(log, currentId) {
  return log
    .filter((entry) => entry.id < currentId)
    .sort((a, b) => b.id - a.id)[0]?.id ?? 0;
}

function lastOpId(log) {
  return log.reduce((max, op) => Math.max(max, Number(op.id) || 0), 0);
}

function updateCounts(state) {
  state.status.activeRows = state.rows.activeIds.length;
  state.status.removedRows = state.rows.removedIds.length;
}

function persistSoon(state) {
  window.clearTimeout(persistTimer);
  persistTimer = window.setTimeout(() => {
    persistNow(state);
  }, 400);
}

function persistNow(state) {
  window.clearTimeout(persistTimer);
  idbSet(REVISION_KEY, {
    baseHash: state.baseHash,
    currentOpId: state.revisions.currentOpId,
    nextOpId: state.revisions.nextOpId,
    log: state.revisions.log,
    labels: state.revisions.labels,
    lastExportedOpId: state.flags.lastExportedOpId
  }).catch(() => {});
}

function removeId(array, rowId) {
  const index = array.indexOf(rowId);
  if (index >= 0) array.splice(index, 1);
}

function insertAfter(array, rowId, afterRowId = null) {
  if (array.includes(rowId)) return;
  const afterIndex = afterRowId ? array.indexOf(afterRowId) : -1;
  if (afterIndex >= 0) array.splice(afterIndex + 1, 0, rowId);
  else array.push(rowId);
}

function insertByAllOrder(array, allIds, rowId) {
  if (array.includes(rowId)) return;
  const orderById = new Map(allIds.map((id, index) => [id, index]));
  const rowOrder = orderById.get(rowId);
  if (rowOrder == null) {
    array.push(rowId);
    return;
  }
  const lastOrder = orderById.get(array[array.length - 1]);
  if (lastOrder == null || lastOrder < rowOrder) {
    array.push(rowId);
    return;
  }
  const insertAt = array.findIndex((id) => (orderById.get(id) ?? Number.POSITIVE_INFINITY) > rowOrder);
  if (insertAt >= 0) array.splice(insertAt, 0, rowId);
  else array.push(rowId);
}

function nextOriginalIndex(rowsState) {
  let maxIndex = -1;
  for (const rowId of rowsState.allIds) {
    const index = Number(rowsState.byId[rowId]?.originalIndex ?? -1);
    if (index > maxIndex) maxIndex = index;
  }
  return maxIndex + 1;
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });
}
