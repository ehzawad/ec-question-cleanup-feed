// Routes all semantic work to the local Python/MPS backend. The browser E5
// fallback was intentionally removed because E5-large in Chrome can consume
// enough memory to make the cleanup tab unresponsive.
let onMessageRef = null;
let onErrorRef = null;
let backendModePromise = null;
let routeQueue = Promise.resolve();
let statusPollTimer = null;
let backendRetryTimer = null;
let latestBackendStatus = null;

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8765';
let backendUrl = localStorage.getItem('semanticBackendUrl') || DEFAULT_BACKEND_URL;

export function initWorkerBridge(onMessage, onError) {
  onMessageRef = onMessage;
  onErrorRef = onError;
  ensureBackendMode();
}

export function postConfigure(docs, runId, expectedRows = docs?.length || 0) {
  post({ type: 'configure', docs, runId, expectedRows });
}

export function postBuildIndex(runId) {
  post({ type: 'buildIndex', runId });
}

export function postSearch({ query, requestId, runId, excludeRowIds = [] }) {
  post({ type: 'search', query, requestId, runId, excludeRowIds });
}

export function postUpsertRows(rows, runId) {
  post({ type: 'embedRows', rows, runId });
}

export function postRemoveRows(rowIds, runId) {
  post({ type: 'removeRows', rowIds, runId });
}

export function postPurgeRows(rowIds, runId) {
  post({ type: 'purgeRows', rowIds, runId });
}

export async function tryReuseBackendIndex(activeRows, runId) {
  if (!(await ensureBackendMode())) return false;
  try {
    const payload = await fetchJson('/status', { method: 'GET' });
    latestBackendStatus = payload;
    if (canReuseBackendPayload(payload, activeRows)) {
      onMessageRef?.({ ...payload, runId, type: 'configured' });
      if (payload.status !== 'ready') startStatusPolling();
      return true;
    }
  } catch {
    backendModePromise = null;
  }
  return false;
}

export function isBackendAvailable() {
  return ensureBackendMode();
}

function post(message) {
  // Keep configure/build/search mutations ordered; stale search results are
  // filtered later by runId/requestId in the Zustand store.
  routeQueue = routeQueue
    .then(() => routeMessage(message))
    .catch((error) => {
      onErrorRef?.(error?.message || 'Semantic backend failed');
    });
}

async function routeMessage(message) {
  if (await ensureBackendMode()) {
    await postBackend(message);
    return;
  }
  onErrorRef?.('Start the Python semantic backend with npm run semantic:server before building or searching.');
}

function ensureBackendMode() {
  // Health is checked once per page load. Users can override the endpoint with
  // localStorage.semanticBackendUrl for experiments or a non-default port.
  if (!backendModePromise) {
    backendModePromise = probeBackendHealth()
      .then((payload) => {
        handleBackendConnected(payload);
        return true;
      })
      .catch(() => {
        backendModePromise = null;
        startBackendRetryPolling();
        return false;
      });
  }
  return backendModePromise;
}

function handleBackendConnected(payload) {
  latestBackendStatus = payload;
  stopBackendRetryPolling();
  onMessageRef?.({
    ...payload,
    type: 'status',
    status: payload.status || 'ready',
    message: `Apple Silicon semantic backend connected (${payload.engine || 'python-mps-faiss'})`
  });
  startStatusPolling();
}

function startBackendRetryPolling() {
  if (backendRetryTimer) return;
  backendRetryTimer = window.setInterval(async () => {
    try {
      const payload = await probeBackendHealth();
      backendModePromise = Promise.resolve(true);
      handleBackendConnected(payload);
    } catch {
      backendModePromise = null;
    }
  }, 1500);
}

function stopBackendRetryPolling() {
  if (!backendRetryTimer) return;
  window.clearInterval(backendRetryTimer);
  backendRetryTimer = null;
}

async function postBackend(message) {
  // Translate the worker-style messages into HTTP calls without leaking backend
  // details into the React components.
  if (message.type === 'configure') {
    if (canReuseBackendPayload(latestBackendStatus, message.expectedRows)) {
      onMessageRef?.({ ...latestBackendStatus, runId: message.runId, type: 'configured' });
      return;
    }
    const payload = await fetchJson('/configure', {
      method: 'POST',
      body: { docs: message.docs || [] }
    });
    latestBackendStatus = payload;
    onMessageRef?.({ ...payload, runId: message.runId, type: 'configured' });
    return;
  }
  if (message.type === 'buildIndex') {
    const payload = await fetchJson('/build', { method: 'POST', body: {} });
    latestBackendStatus = payload;
    onMessageRef?.({ ...payload, runId: message.runId, type: 'status' });
    startStatusPolling();
    return;
  }
  if (message.type === 'search') {
    const payload = await fetchJson('/search', {
      method: 'POST',
      body: { query: message.query, topK: 10 }
    });
    postBackendResults(message, payload);
    return;
  }
  if (message.type === 'embedRows') {
    const payload = await fetchJson('/upsert', {
      method: 'POST',
      body: { rows: message.rows || [] }
    });
    latestBackendStatus = payload;
    onMessageRef?.({ ...payload, runId: message.runId, type: 'status' });
    startStatusPolling();
    return;
  }
  if (message.type === 'removeRows') {
    const payload = await fetchJson('/remove', {
      method: 'POST',
      body: { rowIds: message.rowIds || [] }
    });
    latestBackendStatus = payload;
    onMessageRef?.({ ...payload, runId: message.runId, type: 'status' });
    return;
  }
}

function postBackendResults(message, payload) {
  // Backend search is full-index-only, so a partial result here means the UI
  // should keep the Top-10 hidden until indexing finishes.
  onMessageRef?.({
    type: 'results',
    runId: message.runId,
    requestId: message.requestId,
    results: payload.results || [],
    indexed: payload.indexed,
    total: payload.total,
    indexedFraction: payload.total ? payload.indexed / payload.total : 1,
    partial: payload.indexed < payload.total
  });
  onMessageRef?.({ ...payload, runId: message.runId, type: 'status' });
}

function startStatusPolling() {
  // Polling keeps progress visible while the backend embeds rows on its own
  // thread after /build or /upsert.
  if (statusPollTimer) return;
  statusPollTimer = window.setInterval(async () => {
    try {
      const payload = await fetchJson('/status', { method: 'GET' });
      latestBackendStatus = payload;
      onMessageRef?.({ ...payload, type: 'status' });
      if (
        payload.status === 'error'
        || (
          payload.status === 'ready'
          && Number(payload.total || 0) > 0
          && Number(payload.indexed || 0) >= Number(payload.total || 0)
        )
      ) {
        window.clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
    } catch {
      window.clearInterval(statusPollTimer);
      statusPollTimer = null;
      backendModePromise = null;
    }
  }, 1000);
}

async function probeBackendHealth() {
  try {
    return await fetchJson('/health', { method: 'GET' });
  } catch (error) {
    if (backendUrl === DEFAULT_BACKEND_URL) throw error;
    const payload = await fetchJson('/health', { method: 'GET' }, DEFAULT_BACKEND_URL);
    backendUrl = DEFAULT_BACKEND_URL;
    localStorage.removeItem('semanticBackendUrl');
    return payload;
  }
}

async function fetchJson(path, { method, body } = {}, baseUrl = backendUrl) {
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined
  });
  if (!response.ok) {
    let detail = '';
    try {
      const payload = await response.json();
      detail = payload.detail || payload.message || '';
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `Semantic backend returned ${response.status}`);
  }
  return response.json();
}

function canReuseBackendPayload(payload, expectedRows) {
  const total = Number(payload?.total || payload?.evidenceCount || 0);
  if (!expectedRows || !total || total !== expectedRows) return false;
  return payload.status !== 'idle';
}
