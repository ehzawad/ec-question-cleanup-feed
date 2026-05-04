import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useVirtualizer } from '@tanstack/react-virtual';
import './styles.css';
import csvUrl from '../question_tag.csv?url';
import {
  exportCsv,
  fingerprintRow,
  rowToWorkerDoc
} from './rowCsv.js';
import { useCleanupStore } from './store.js';
import {
  initWorkerBridge,
  postBuildIndex,
  postConfigure,
  postSearch,
  postUpsertRows,
  postRemoveRows,
  isBackendAvailable,
  tryReuseBackendIndex
} from './workerBridge.js';

function App() {
  const parentRef = useRef(null);
  const searchInputRef = useRef(null);
  const [pendingScrollRowId, setPendingScrollRowId] = useState(null);
  const [pendingSemanticRefresh, setPendingSemanticRefresh] = useState(null);
  const [queryDraft, setQueryDraft] = useState('');
  const [snapshotName, setSnapshotName] = useState('');
  const [saveNotice, setSaveNotice] = useState('');
  const [bootMessage, setBootMessage] = useState('Loading question CSV...');

  const status = useCleanupStore((state) => state.status);
  const search = useCleanupStore((state) => state.search);
  const embeddings = useCleanupStore((state) => state.embeddings);
  const rowsById = useCleanupStore((state) => state.rows.byId);
  const allIds = useCleanupStore((state) => state.rows.allIds);
  const activeIds = useCleanupStore((state) => state.rows.activeIds);
  const selectedIds = useCleanupStore((state) => state.selection.multi);
  const shiftAnchorRowId = useCleanupStore((state) => state.selection.shiftAnchorRowId);
  const showRemoved = useCleanupStore((state) => state.filters.showRemoved);
  const editingRowId = useCleanupStore((state) => state.ui.editingRowId);
  const composer = useCleanupStore((state) => state.ui.composer);
  const snapshots = useCleanupStore((state) => state.revisions.labels);
  const dirty = useCleanupStore((state) => state.flags.dirty);
  const lastExportedOpId = useCleanupStore((state) => state.flags.lastExportedOpId);
  const currentOpId = useCleanupStore((state) => state.revisions.currentOpId);
  const revisionLog = useCleanupStore((state) => state.revisions.log);

  const startHydrate = useCleanupStore((state) => state.startHydrate);
  const appendHydrateRows = useCleanupStore((state) => state.appendHydrateRows);
  const finishHydrate = useCleanupStore((state) => state.finishHydrate);
  const handleWorkerMessage = useCleanupStore((state) => state.handleWorkerMessage);
  const setSearchMode = useCleanupStore((state) => state.setSearchMode);
  const clearSearch = useCleanupStore((state) => state.clearSearch);
  const setShowRemoved = useCleanupStore((state) => state.setShowRemoved);
  const toggleSelection = useCleanupStore((state) => state.toggleSelection);
  const clearSelection = useCleanupStore((state) => state.clearSelection);
  const removeRows = useCleanupStore((state) => state.removeRows);
  const restoreRows = useCleanupStore((state) => state.restoreRows);
  const editRow = useCleanupStore((state) => state.editRow);
  const addRow = useCleanupStore((state) => state.addRow);
  const setEditingRow = useCleanupStore((state) => state.setEditingRow);
  const setComposer = useCleanupStore((state) => state.setComposer);
  const undo = useCleanupStore((state) => state.undo);
  const redo = useCleanupStore((state) => state.redo);
  const saveSnapshot = useCleanupStore((state) => state.saveSnapshot);
  const restoreSnapshot = useCleanupStore((state) => state.restoreSnapshot);
  const markExported = useCleanupStore((state) => state.markExported);
  const markSemanticSyncPending = useCleanupStore((state) => state.markSemanticSyncPending);

  useEffect(() => {
    let mounted = true;
    initWorkerBridge((message) => {
      useCleanupStore.getState().handleWorkerMessage(message);
    }, (message) => {
      useCleanupStore.getState().setWorkerError(message);
    });
    const csvWorker = new Worker(new URL('./csvLoaderWorker.js', import.meta.url), { type: 'module' });
    let loadedRows = 0;
    startHydrate();
    csvWorker.onmessage = (event) => {
      if (!mounted) return;
      if (event.data?.type === 'error') {
        useCleanupStore.getState().setWorkerError(event.data.message);
        setBootMessage(event.data.message);
        return;
      }
      if (event.data?.type === 'rowsChunk') {
        loadedRows += event.data.rows?.length || 0;
        appendHydrateRows(event.data.rows || []);
        setBootMessage(`Loading question CSV... ${loadedRows.toLocaleString()} rows`);
        return;
      }
      if (event.data?.type !== 'done') return;
      setBootMessage(`Preparing ${Number(event.data.total || loadedRows).toLocaleString()} rows...`);
      finishHydrate(event.data.hash).then(() => {
        if (!mounted) return;
        setBootMessage('');
        const state = useCleanupStore.getState();
        tryReuseBackendIndex(state.status.activeRows, state.workerRunId()).then((reused) => {
          if (!mounted || reused) return;
          isBackendAvailable().then((available) => {
            if (!mounted) return;
            if (!available) {
              useCleanupStore.getState().setWorkerError('Start the Python semantic backend with npm run semantic:server.');
              return;
            }
            const nextState = useCleanupStore.getState();
            postConfigure(nextState.getActiveDocs(), nextState.workerRunId(), nextState.status.activeRows);
          });
        });
      });
    };
    csvWorker.onerror = (event) => {
      if (!mounted) return;
      const message = event.message || 'CSV loader failed';
      useCleanupStore.getState().setWorkerError(message);
      setBootMessage(message);
    };
    fetch(csvUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`CSV request failed with ${response.status}`);
        return response.text();
      })
      .then((text) => {
        if (!mounted) return;
        csvWorker.postMessage({ type: 'parseCsv', text });
      })
      .catch((error) => {
        if (!mounted) return;
        const message = error?.message || 'Failed to load CSV';
        useCleanupStore.getState().setWorkerError(message);
        setBootMessage(message);
      });
    return () => {
      mounted = false;
      csvWorker.terminate();
    };
  }, [appendHydrateRows, finishHydrate, handleWorkerMessage, startHydrate]);

  useEffect(() => {
    setQueryDraft(search.textQuery);
  }, [search.textQuery]);

  useEffect(() => {
    const onKeyDown = (event) => {
      const isMacUndo = event.metaKey && !event.altKey && event.key.toLowerCase() === 'z';
      const isCtrlUndo = event.ctrlKey && !event.altKey && event.key.toLowerCase() === 'z';
      if (isMacUndo || isCtrlUndo) {
        event.preventDefault();
        if (event.shiftKey) {
          const changedRows = redo();
          syncRowsToWorker(changedRows);
        } else {
          const changedRows = undo();
          syncRowsToWorker(changedRows);
        }
        return;
      }
      if (event.key === '/' && !isTypingTarget(event.target)) {
        event.preventDefault();
        searchInputRef.current?.focus();
        return;
      }
      if (event.key === 'Escape') {
        clearSearch();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [clearSearch, redo, undo]);

  const visibleIds = showRemoved ? allIds : activeIds;

  const rowVirtualizer = useVirtualizer({
    count: visibleIds.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 156,
    overscan: 10
  });
  const virtualItems = rowVirtualizer.getVirtualItems();
  const virtualRows = virtualItems.length
    ? virtualItems
    : visibleIds.slice(0, 25).map((_, index) => ({
        index,
        key: `initial-${index}`,
        start: index * 156,
        size: 156
      }));
  const virtualHeight = Math.max(rowVirtualizer.getTotalSize(), visibleIds.length * 156);

  useEffect(() => {
    if (!pendingScrollRowId) return;
    const index = visibleIds.indexOf(pendingScrollRowId);
    if (index >= 0) {
      rowVirtualizer.scrollToIndex(index, { align: 'center' });
      setPendingScrollRowId(null);
    }
  }, [pendingScrollRowId, rowVirtualizer, visibleIds]);

  const activeRows = status.activeRows;
  const removedRows = status.removedRows;
  const selectedCount = selectedIds.size;
  const exportedClean = !dirty || lastExportedOpId === currentOpId;
  const pendingExportCount = Math.max(0, currentOpId - lastExportedOpId);
  const saveStateLabel = currentOpId === 0
    ? 'No cleanup changes'
    : exportedClean
      ? 'Clean CSV saved'
      : `${pendingExportCount.toLocaleString()} changes autosaved locally`;
  const indexInfo = semanticIndexInfo(embeddings, activeRows);
  const canRunSemantic = indexInfo.ready;
  const topResultIds = useMemo(
    () => new Set(search.results.map((result) => result.rowId)),
    [search.results]
  );

  useEffect(() => {
    if (!pendingSemanticRefresh || !canRunSemantic) return;
    const current = useCleanupStore.getState().search;
    if (current.mode === 'text' && current.textQuery.trim()) {
      runTextSearch(current.textQuery, pendingSemanticRefresh.reason);
    }
    setPendingSemanticRefresh(null);
  }, [canRunSemantic, pendingSemanticRefresh, search.mode, search.textQuery]);

  function runTextSearch(query, reason = 'manual') {
    // Top-10 is a full-index feature. Until every active row has a vector, the
    // feed stays browse-only so the user never sees misleading partial matches.
    const value = query.trim();
    if (!value) {
      clearSearch();
      return;
    }
    if (!semanticIndexInfo(useCleanupStore.getState().embeddings, useCleanupStore.getState().status.activeRows).ready) return;
    const requestId = setSearchMode({ mode: 'text', textQuery: value, reason });
    postSearch({
      query: value,
      requestId,
      runId: useCleanupStore.getState().workerRunId(),
      excludeRowIds: useCleanupStore.getState().removedRowIds()
    });
  }

  function runRowQuestionSearch(rowId) {
    const row = rowsById[rowId];
    if (!row || row.status === 'removed') return;
    setQueryDraft(row.question);
    runTextSearch(row.question, 'row-click');
  }

  function syncRowsToWorker(rows) {
    if (!rows?.length) return;
    const activeDocs = rows
      .filter((row) => row.status === 'active')
      .map(rowToWorkerDoc);
    const removedIds = rows
      .filter((row) => row.status === 'removed')
      .map((row) => row.rowId);
    markSemanticSyncPending();
    if (activeDocs.length) {
      postUpsertRows(activeDocs, useCleanupStore.getState().workerRunId());
    }
    if (removedIds.length) {
      postRemoveRows(removedIds, useCleanupStore.getState().workerRunId());
    }
    queueSemanticRefresh('undo-redo');
  }

  function queueSemanticRefresh(reason = 'mutation') {
    setPendingSemanticRefresh({
      reason,
      opId: useCleanupStore.getState().revisions.currentOpId,
      runId: useCleanupStore.getState().workerRunId()
    });
  }

  function handleEdit(rowId, patch) {
    const changed = editRow(rowId, patch);
    if (!changed) return;
    markSemanticSyncPending();
    postUpsertRows([rowToWorkerDoc(changed)], useCleanupStore.getState().workerRunId());
    queueSemanticRefresh('edit');
  }

  function handleRemove(rowIds, reason = 'remove') {
    const changed = removeRows(rowIds, reason);
    if (!changed.length) return;
    markSemanticSyncPending();
    postRemoveRows(changed.map((row) => row.rowId), useCleanupStore.getState().workerRunId());
    queueSemanticRefresh(reason);
  }

  function handleRestore(rowIds) {
    const changed = restoreRows(rowIds);
    if (!changed.length) return;
    markSemanticSyncPending();
    postUpsertRows(changed.map(rowToWorkerDoc), useCleanupStore.getState().workerRunId());
    queueSemanticRefresh('restore');
  }

  function handleAdd(payload) {
    const row = addRow(payload);
    if (!row) return;
    markSemanticSyncPending();
    postUpsertRows([rowToWorkerDoc(row)], useCleanupStore.getState().workerRunId());
    queueSemanticRefresh('add');
  }

  function handleBuildIndex() {
    isBackendAvailable().then((available) => {
      if (!available) {
        useCleanupStore.getState().setWorkerError('Start the Python semantic backend with npm run semantic:server.');
        return;
      }
      const state = useCleanupStore.getState();
      postConfigure(state.getActiveDocs(), state.workerRunId(), state.status.activeRows);
      postBuildIndex(state.workerRunId());
    });
  }

  function exportCleanCsv() {
    const rows = allIds
      .map((rowId) => rowsById[rowId])
      .filter((row) => row?.status === 'active')
      .sort((a, b) => a.originalIndex - b.originalIndex)
      .map((row) => ({ question: row.question, tag: row.tag }));
    downloadText('cleaned-question-tag.csv', exportCsv(rows, ['question', 'tag']), 'text/csv');
    markExported();
    setSaveNotice(`Clean CSV saved at step ${currentOpId}`);
  }

  function exportRemovedCsv() {
    const rows = allIds
      .map((rowId) => rowsById[rowId])
      .filter((row) => row?.status === 'removed')
      .sort((a, b) => a.originalIndex - b.originalIndex)
      .map((row) => ({
        rowId: row.rowId,
        question: row.question,
        tag: row.tag,
        removedReason: row.removedReason || 'manual cleanup'
      }));
    downloadText('removed-question-tag.csv', exportCsv(rows, ['rowId', 'question', 'tag', 'removedReason']), 'text/csv');
    setSaveNotice(`Removed-row CSV saved with ${rows.length.toLocaleString()} rows`);
  }

  function jumpToRow(rowId) {
    const row = rowsById[rowId];
    if (!row) return;
    if (row.status === 'removed' && !showRemoved) setShowRemoved(true);
    setPendingScrollRowId(rowId);
  }

  function saveNamedSnapshot() {
    const label = snapshotName.trim() || `rollback-step-${currentOpId || 0}`;
    saveSnapshot(label);
    setSnapshotName('');
    setSaveNotice(`Rollback point created: ${label}`);
  }

  function restoreNamedSnapshot(id) {
    const changed = restoreSnapshot(id);
    if (!changed.length) return;
    const state = useCleanupStore.getState();
    markSemanticSyncPending('Reconfiguring vector index for rollback point');
    postConfigure(state.getActiveDocs(), state.workerRunId(), state.status.activeRows);
    queueSemanticRefresh('rollback-point');
  }

  return (
    <main className="feed-app">
      <section className="feed-topbar">
        <div className="title-block">
          <p className="eyebrow">Question Tag Cleanup</p>
          <h1>Cleanup Feed</h1>
        </div>
        <div className="status-strip">
          <span
            className={exportedClean ? 'status-pill ok' : 'status-pill dirty'}
            title="Cleanup edits are autosaved in this browser. Use Save clean CSV to create the cleaned file."
          >
            {saveStateLabel}
          </span>
          <span className="status-pill">Rows {activeRows.toLocaleString()} active / {removedRows.toLocaleString()} removed</span>
          <span className={`status-pill ${indexInfo.ready ? 'ok' : 'dirty'}`}>
            E5 {indexInfo.modelReady ? `${indexInfo.vectorPercent}% vectors` : `${indexInfo.modelPercent}% model`} · {indexInfo.indexed.toLocaleString()} / {indexInfo.total.toLocaleString()}
          </span>
          <span className="status-pill">Cache {embeddings.cacheHits.toLocaleString()} hits</span>
        </div>
      </section>

      <section className="search-panel">
        <label className="search-box">
          <span>Search question with E5</span>
          <div>
            <input
              ref={searchInputRef}
              type="search"
              value={queryDraft}
              placeholder="ভোটার আইডি ফি কত, password reset, address correction..."
              onChange={(event) => {
                setQueryDraft(event.target.value);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') runTextSearch(queryDraft);
              }}
            />
            <button
              type="button"
              onClick={() => runTextSearch(queryDraft)}
              disabled={!canRunSemantic || !queryDraft.trim()}
              title={canRunSemantic ? 'Search the full vector index' : 'Build the full vector index before searching'}
            >
              Search
            </button>
            <button type="button" onClick={clearSearch}>Clear</button>
          </div>
        </label>

        <div className="feed-tools">
          <label className="checkbox-tool">
            <input type="checkbox" checked={showRemoved} onChange={(event) => setShowRemoved(event.target.checked)} />
            Show removed ({removedRows.toLocaleString()})
          </label>
          <button type="button" onClick={() => {
            const changedRows = undo();
            syncRowsToWorker(changedRows);
          }}>Undo</button>
          <button type="button" onClick={() => {
            const changedRows = redo();
            syncRowsToWorker(changedRows);
          }}>Redo</button>
        </div>
      </section>

      <section className="snapshot-row">
        <input
          value={snapshotName}
          onChange={(event) => setSnapshotName(event.target.value)}
          placeholder="Optional rollback point name"
        />
        <button type="button" onClick={saveNamedSnapshot}>Create rollback point</button>
        <select defaultValue="" onChange={(event) => {
          if (!event.target.value) return;
          restoreNamedSnapshot(event.target.value);
          event.target.value = '';
        }}>
          <option value="">Go back to rollback point...</option>
          {snapshots.map((snapshot) => (
            <option key={snapshot.id} value={snapshot.id}>
              {snapshot.name} · step {snapshot.opId}
            </option>
          ))}
        </select>
        <button type="button" className="primary-save" onClick={exportCleanCsv}>Save clean CSV</button>
        <button type="button" onClick={exportRemovedCsv}>Save removed CSV</button>
        {saveNotice ? <span className="save-notice">{saveNotice}</span> : null}
      </section>

      {embeddings.quotaWarning ? <div className="warning-banner">{embeddings.quotaWarning}</div> : null}
      {bootMessage ? <div className="loading-banner">{bootMessage}</div> : null}

      <TopResultsDock
        search={search}
        rowsById={rowsById}
        embeddings={embeddings}
        activeRows={activeRows}
        onBuildIndex={handleBuildIndex}
        onSelectResult={(rowId) => {
          jumpToRow(rowId);
          runRowQuestionSearch(rowId);
        }}
        onRemove={(rowId) => handleRemove([rowId], 'top-result-remove')}
      />

      <OperationsHistory
        log={revisionLog}
        currentOpId={currentOpId}
        lastExportedOpId={lastExportedOpId}
        rowsById={rowsById}
        onJump={jumpToRow}
      />

      {selectedCount ? (
        <section className="bulk-bar">
          <strong>{selectedCount.toLocaleString()} selected</strong>
          <button type="button" onClick={() => handleRemove([...selectedIds], 'bulk-remove')}>Delete selected</button>
          <button type="button" onClick={clearSelection}>Cancel</button>
        </section>
      ) : null}

      <section className="feed-shell">
        <div className="feed-count">
          <div className="feed-count-main">
            <span>Showing {visibleIds.length.toLocaleString()} rows in CSV order</span>
            {search.mode === 'text' && search.textQuery ? <span>Query: {search.textQuery}</span> : null}
          </div>
          <button
            className="feed-add-button"
            type="button"
            onClick={() => setComposer({ open: true, afterRowId: null, tag: '', question: '' })}
          >
            Add row
          </button>
        </div>
        <div ref={parentRef} className="feed-scroll" role="list" aria-label="Question tag cleanup feed">
          <div
            className="feed-virtual-spacer"
            style={{ height: `${virtualHeight}px` }}
          >
            {virtualRows.map((virtualRow) => {
              const rowId = visibleIds[virtualRow.index];
              const row = rowsById[rowId];
              if (!row) return null;
              return (
                <div
                  key={rowId}
                  ref={rowVirtualizer.measureElement}
                  data-index={virtualRow.index}
                  className="feed-virtual-row"
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                  <FeedCard
                    row={row}
                    canInspect={canRunSemantic}
                    isSelected={selectedIds.has(row.rowId)}
                    isTopResult={topResultIds.has(row.rowId)}
                    isEditing={editingRowId === row.rowId}
                    shiftAnchorRowId={shiftAnchorRowId}
                    onToggle={(event) => toggleSelection(row.rowId, event.shiftKey, visibleIds)}
                    onEdit={handleEdit}
                    onStartEdit={() => setEditingRow(row.rowId)}
                    onCancelEdit={() => setEditingRow(null)}
                    onRemove={() => handleRemove([row.rowId])}
                    onRestore={() => handleRestore([row.rowId])}
                    onInspect={() => runRowQuestionSearch(row.rowId)}
                    onAddSibling={() => setComposer({ open: true, afterRowId: row.rowId, tag: row.tag, question: '' })}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {composer.open ? (
        <>
          <div className="composer-backdrop" onClick={() => setComposer({ open: false, afterRowId: null, tag: '', question: '' })} />
          <AddComposer
            composer={composer}
            onChange={setComposer}
            onCancel={() => setComposer({ open: false, afterRowId: null, tag: '', question: '' })}
            onAdd={() => handleAdd({
              tag: composer.tag,
              question: composer.question,
              afterRowId: composer.afterRowId
            })}
          />
        </>
      ) : null}
    </main>
  );
}

function ProgressLines({ indexInfo, compact = false }) {
  return (
    <div className={compact ? 'progress-lines compact' : 'progress-lines'}>
      <div className="progress-line">
        <div className="progress-line-label">
          <span>Model</span>
          <strong>{indexInfo.modelReady ? 'ready' : `${indexInfo.modelPercent}%`}</strong>
        </div>
        <div className="progress-track">
          <span style={{ width: `${indexInfo.modelPercent}%` }} />
        </div>
      </div>
      <div className="progress-line">
        <div className="progress-line-label">
          <span>Row vectors</span>
          <strong>{indexInfo.vectorPercent}%</strong>
        </div>
        <div className="progress-track">
          <span style={{ width: `${indexInfo.vectorPercent}%` }} />
        </div>
      </div>
    </div>
  );
}

function TopResultsDock({ search, rowsById, embeddings, activeRows, onBuildIndex, onSelectResult, onRemove }) {
  const indexInfo = semanticIndexInfo(embeddings, activeRows);
  const isReady = indexInfo.ready;
  const activeQuery = search.mode === 'text' && search.textQuery.trim();
  const hasIntent = Boolean(activeQuery);
  const results = search.results
    .map((result) => ({ ...result, row: rowsById[result.rowId] }))
    .filter((result) => result.row)
    .slice(0, 10);
  const title = !isReady
    ? 'Semantic search setup'
    : activeQuery
        ? 'Top 10 semantic matches'
        : 'Semantic search ready';
  const topMessage = !isReady
    ? `Build the vector index first · ${indexInfo.vectorPercent}% vectors`
    : activeQuery
        ? search.textQuery
        : 'Search a question above, or click any feed row to use that question.';
  return (
    <section className="top-results">
      <div className="section-head">
        <div>
          <p className="eyebrow">{title}</p>
          <h2>{topMessage}</h2>
        </div>
        <div className="result-meta">
          <span className={`status-pill ${isReady ? 'ok' : 'dirty'}`}>
            {isReady ? 'Full index ready' : `${indexInfo.indexed.toLocaleString()} / ${indexInfo.total.toLocaleString()} vectors`}
          </span>
          {search.status === 'querying' ? <span className="status-pill">Computing Top-10</span> : null}
        </div>
      </div>

      {!isReady ? (
        <div className="index-setup">
          <ProgressLines indexInfo={indexInfo} compact />
          <div className="index-action-row">
            <p className="index-help">
              {(embeddings.message || 'Build the vector index before showing similarity scores.')}
            </p>
            <button type="button" className="primary-action" onClick={onBuildIndex}>
              {indexInfo.indexed ? 'Resume vector index' : 'Build vector index'}
            </button>
          </div>
        </div>
      ) : null}

      {search.error ? <p className="warning-text">{search.error}</p> : null}

      {isReady && hasIntent && results.length ? (
        <ol className="result-list" role="listbox">
          {results.map((result, index) => (
            <li key={`${result.rowId}-${index}`} className={result.row.status === 'removed' ? 'removed' : ''}>
              <button type="button" className="result-main" onClick={() => onSelectResult(result.rowId)}>
                <strong>{index + 1}. {result.row.tag}</strong>
                <span lang={hasBangla(result.row.question) ? 'bn' : 'en'}>{result.row.question}</span>
              </button>
              <span className="score">{Math.round((result.score ?? 0) * 100)}%</span>
              <button type="button" onClick={() => onRemove(result.rowId)} aria-label={`Delete ${result.rowId}`}>Delete</button>
            </li>
          ))}
        </ol>
      ) : isReady ? (
        <p className="empty-state">
          {hasIntent ? 'Computing full Top-10 from the completed vector index.' : 'No similarity scores are shown while browsing. Search a question or click any row to generate Top-10.'}
        </p>
      ) : null}
    </section>
  );
}

function OperationsHistory({ log, currentOpId, lastExportedOpId, rowsById, onJump }) {
  const activeOps = log
    .filter((op) => op.id <= currentOpId)
    .slice(-6)
    .reverse();

  return (
    <section className="operations-panel">
      <div className="section-head">
        <div>
          <p className="eyebrow">Cleanup Operations</p>
          <h2>{activeOps.length ? `${activeOps.length} recent changes` : 'No cleanup changes yet'}</h2>
        </div>
        <span className="status-pill">{currentOpId > lastExportedOpId ? `${currentOpId - lastExportedOpId} not saved to CSV` : 'Clean CSV saved'}</span>
      </div>
      {activeOps.length ? (
        <ol className="operation-list">
          {activeOps.map((op) => {
            const rowId = firstRowIdForOp(op);
            return (
              <li key={op.id} className={op.id > lastExportedOpId ? 'dirty-op' : ''}>
                <span>#{op.id}</span>
                <strong>{describeOperation(op, rowsById)}</strong>
                <em>{formatTime(op.ts)}</em>
                {rowId ? <button type="button" onClick={() => onJump(rowId)}>Jump</button> : null}
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="empty-state">Edits, adds, deletes, and restores will appear here.</p>
      )}
    </section>
  );
}

function FeedCard({
  row,
  canInspect,
  isSelected,
  isTopResult,
  isEditing,
  onToggle,
  onEdit,
  onStartEdit,
  onCancelEdit,
  onRemove,
  onRestore,
  onInspect,
  onAddSibling
}) {
  const [tagDraft, setTagDraft] = useState(row.tag);
  const [questionDraft, setQuestionDraft] = useState(row.question);

  useEffect(() => {
    if (!isEditing) {
      setTagDraft(row.tag);
      setQuestionDraft(row.question);
    }
  }, [isEditing, row.question, row.tag]);

  function save() {
    onEdit(row.rowId, {
      tag: tagDraft.trim(),
      question: questionDraft.trim(),
      fingerprint: fingerprintRow(tagDraft.trim(), questionDraft.trim())
    });
  }

  function handleInspectClick(event) {
    if (!canInspect || row.status !== 'active' || isEditing) return;
    if (event.target.closest('button, input, textarea, select, a')) return;
    if (window.getSelection()?.toString().trim()) return;
    onInspect();
  }

  return (
    <article
      role="listitem"
      className={[
        'feed-card',
        row.status === 'removed' ? 'removed' : '',
        isTopResult ? 'top-hit' : '',
        row.isAdded ? 'added' : ''
      ].filter(Boolean).join(' ')}
    >
      <div className="card-select">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          aria-label={`Select ${row.rowId}`}
        />
      </div>
      <div
        className={canInspect && row.status === 'active' && !isEditing ? 'card-body inspectable' : 'card-body'}
        role={canInspect && row.status === 'active' && !isEditing ? 'button' : undefined}
        tabIndex={canInspect && row.status === 'active' && !isEditing ? 0 : undefined}
        onClick={handleInspectClick}
        onKeyDown={(event) => {
          if (!canInspect || row.status !== 'active' || isEditing) return;
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onInspect();
          }
        }}
      >
        <div className="card-meta">
          <span>{row.rowId}</span>
          <span>#{row.originalIndex + 1}</span>
          {row.isAdded ? <span>added</span> : null}
          {row.status === 'removed' ? <span>removed</span> : null}
          {isTopResult ? <span>in top 10</span> : null}
        </div>
        {isEditing ? (
          <div className="inline-editor">
            <label>
              Tag
              <input value={tagDraft} onChange={(event) => setTagDraft(event.target.value)} />
            </label>
            <label>
              Question
              <textarea value={questionDraft} onChange={(event) => setQuestionDraft(event.target.value)} rows={3} />
            </label>
            <div>
              <button type="button" onClick={save}>Save</button>
              <button type="button" onClick={onCancelEdit}>Cancel</button>
            </div>
          </div>
        ) : (
          <>
            <div className="tag-line">
              <span>Tag</span>
              <strong>{row.tag || '(empty tag)'}</strong>
            </div>
            <p className="question-text" lang={hasBangla(row.question) ? 'bn' : 'en'}>
              {row.question || '(empty question)'}
            </p>
          </>
        )}
      </div>
      <div className="card-rail" aria-label={`Actions for ${row.rowId}`}>
        <button type="button" onClick={onStartEdit}>Edit</button>
        {row.status === 'removed'
          ? <button type="button" onClick={onRestore}>Restore</button>
          : <button type="button" onClick={onRemove}>Delete</button>}
        <button type="button" onClick={onAddSibling}>Add</button>
      </div>
    </article>
  );
}

function AddComposer({ composer, onChange, onCancel, onAdd }) {
  return (
    <section className="add-composer">
      <div className="section-head">
        <div>
          <p className="eyebrow">Add Row</p>
          <h2>{composer.afterRowId ? `After ${composer.afterRowId}` : 'Append to feed'}</h2>
        </div>
        <button type="button" onClick={onCancel}>Close</button>
      </div>
      <label>
        Tag
        <input
          value={composer.tag}
          onChange={(event) => onChange({ ...composer, tag: event.target.value })}
          placeholder="new_tag_name"
        />
      </label>
      <label>
        Question
        <textarea
          value={composer.question}
          onChange={(event) => onChange({ ...composer, question: event.target.value })}
          rows={3}
          placeholder="Example user question"
        />
      </label>
      <button type="button" onClick={onAdd}>Add to working set</button>
    </section>
  );
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function hasBangla(value) {
  return /[\u0980-\u09FF]/.test(value ?? '');
}

function isTypingTarget(target) {
  const tag = target?.tagName?.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable;
}

function firstRowIdForOp(op) {
  if (op.kind === 'edit') return op.payload.rowId;
  if (op.kind === 'add') return op.payload.row?.rowId;
  if (op.kind === 'remove' || op.kind === 'restore') return op.payload.before?.[0]?.rowId;
  return null;
}

function describeOperation(op, rowsById) {
  if (op.kind === 'edit') {
    const row = rowsById[op.payload.rowId] || op.payload.after || op.payload.before;
    return `Edited ${op.payload.rowId} · ${row?.tag || 'untagged'}`;
  }
  if (op.kind === 'add') {
    return `Added ${op.payload.row?.rowId || 'row'} · ${op.payload.row?.tag || 'untagged'}`;
  }
  if (op.kind === 'remove') {
    const count = op.payload.rowIds?.length || op.payload.before?.length || 0;
    const first = op.payload.before?.[0];
    return `Deleted ${count.toLocaleString()} row${count === 1 ? '' : 's'}${first?.tag ? ` · ${first.tag}` : ''}`;
  }
  if (op.kind === 'restore') {
    const count = op.payload.rowIds?.length || op.payload.before?.length || 0;
    const first = op.payload.before?.[0];
    return `Restored ${count.toLocaleString()} row${count === 1 ? '' : 's'}${first?.tag ? ` · ${first.tag}` : ''}`;
  }
  return op.kind;
}

function formatTime(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function semanticIndexInfo(embeddings, activeRows) {
  const total = Math.max(Number(embeddings.total || 0), Number(activeRows || 0));
  const indexed = total ? Math.min(embeddings.indexed, total) : embeddings.indexed;
  const vectorPercent = total ? Math.round((indexed / total) * 100) : 0;
  const phaseProgress = embeddings.phaseTotal
    ? Math.round((Number(embeddings.phaseDone || 0) / Number(embeddings.phaseTotal || 1)) * 100)
    : 0;
  const modelPercent = Math.max(
    0,
    Math.min(
      100,
      Math.round(Number(embeddings.modelProgress || (embeddings.status === 'ready' || indexed > 0 ? 100 : phaseProgress)))
    )
  );
  const modelReady = modelPercent >= 100 || embeddings.phase === 'embed' || indexed > 0 || (embeddings.status === 'ready' && embeddings.phase === 'cache');
  return {
    total,
    indexed,
    vectorPercent,
    phasePercent: Math.max(0, Math.min(100, phaseProgress)),
    modelPercent: modelReady ? 100 : modelPercent,
    modelReady,
    ready: total > 0 && indexed >= total
  };
}

createRoot(document.querySelector('#app')).render(<App />);
