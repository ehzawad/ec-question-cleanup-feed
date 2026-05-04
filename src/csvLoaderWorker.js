import { fingerprintRow } from './rowCsv.js';

const CHUNK_SIZE = 5000;

self.onmessage = (event) => {
  if (event.data?.type !== 'parseCsv') return;
  try {
    streamCsvRows(String(event.data.text ?? ''));
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error?.message || 'Failed to parse CSV'
    });
  }
};

function streamCsvRows(text) {
  let headers = null;
  let questionIndex = -1;
  let tagIndex = -1;
  let legacyTagIndex = -1;
  let legacyQuestionIndex = -1;
  let sourceIndex = 0;
  let hash = 0x811c9dc5;
  let emitted = 0;
  let chunk = [];
  let record = [];
  let cell = '';
  let inQuotes = false;
  const loadedAt = Date.now();

  const flushChunk = () => {
    if (!chunk.length) return;
    self.postMessage({ type: 'rowsChunk', rows: chunk });
    chunk = [];
  };

  const flushRecord = () => {
    record.push(cell);
    cell = '';
    if (!headers) {
      headers = record.map((header) => stripBom(header).trim());
      questionIndex = headers.indexOf('question');
      tagIndex = headers.indexOf('tag');
      legacyTagIndex = headers.indexOf('tag name');
      legacyQuestionIndex = headers.indexOf('one example');
      record = [];
      return;
    }

    const question = (record[questionIndex >= 0 ? questionIndex : legacyQuestionIndex] ?? '').trim();
    const tag = (record[tagIndex >= 0 ? tagIndex : legacyTagIndex] ?? '').trim();
    const rowId = `r-${sourceIndex}`;
    const originalIndex = sourceIndex;
    sourceIndex += 1;
    record = [];
    if (!tag && !question) return;

    const row = {
      rowId,
      originalIndex,
      originalQuestion: question,
      originalTag: tag,
      question,
      tag,
      status: 'active',
      isAdded: false,
      removedReason: '',
      fingerprint: fingerprintRow(tag, question),
      updatedAt: loadedAt
    };
    hash = hashRow(hash, row);
    emitted += 1;
    chunk.push(row);
    if (chunk.length >= CHUNK_SIZE) flushChunk();
  };

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        cell += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ',') {
      record.push(cell);
      cell = '';
    } else if (char === '\n') {
      flushRecord();
    } else if (char !== '\r') {
      cell += char;
    }
  }

  if (cell || record.length) flushRecord();
  flushChunk();
  self.postMessage({
    type: 'done',
    hash: `${emitted}:${hash.toString(16).padStart(8, '0')}`,
    total: emitted
  });
}

function stripBom(value) {
  return String(value ?? '').replace(/^\uFEFF/, '');
}

function hashRow(seed, row) {
  let hash = seed;
  hash = fnv1aStep(hash, row.rowId);
  hash = fnv1aStep(hash, row.originalIndex);
  hash = fnv1aStep(hash, row.originalTag);
  hash = fnv1aStep(hash, row.originalQuestion);
  hash = fnv1aStep(hash, '\u001e');
  return hash >>> 0;
}

function fnv1aStep(seed, value) {
  let hash = seed >>> 0;
  const text = String(value ?? '');
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}
