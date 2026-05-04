export function parseCsvRows(text) {
  const records = parseCsv(text);
  if (!records.length) return [];
  const headers = records[0].map((header) => stripBom(header).trim());
  const questionIndex = headers.indexOf('question');
  const tagIndex = headers.indexOf('tag');
  const legacyTagIndex = headers.indexOf('tag name');
  const legacyQuestionIndex = headers.indexOf('one example');
  const rows = [];

  for (let index = 1; index < records.length; index += 1) {
    const record = records[index];
    const question = (record[questionIndex >= 0 ? questionIndex : legacyQuestionIndex] ?? '').trim();
    const tag = (record[tagIndex >= 0 ? tagIndex : legacyTagIndex] ?? '').trim();
    if (!tag && !question) continue;
    const rowId = `r-${index - 1}`;
    rows.push({
      rowId,
      originalIndex: index - 1,
      originalQuestion: question,
      originalTag: tag,
      question,
      tag,
      status: 'active',
      isAdded: false,
      removedReason: '',
      fingerprint: fingerprintRow(tag, question),
      updatedAt: Date.now()
    });
  }

  return rows;
}

export function exportCsv(rows, columns) {
  return [
    columns.map(escapeCsvCell).join(','),
    ...rows.map((row) => columns.map((column) => escapeCsvCell(row[column])).join(','))
  ].join('\n');
}

export function rowToWorkerDoc(row) {
  return {
    rowId: row.rowId,
    tag: row.tag,
    question: row.question,
    text: `tag: ${row.tag}\nquestion: ${row.question}`,
    fingerprint: row.fingerprint || fingerprintRow(row.tag, row.question)
  };
}

export function fingerprintRow(tag, question) {
  return fnv1a(`${tag ?? ''}\u0000${question ?? ''}`);
}

export function datasetHash(rows) {
  let hash = 0x811c9dc5;
  for (const row of rows) {
    hash = fnv1aStep(hash, row.rowId);
    hash = fnv1aStep(hash, row.originalIndex);
    hash = fnv1aStep(hash, row.originalTag);
    hash = fnv1aStep(hash, row.originalQuestion);
    hash = fnv1aStep(hash, '\u001e');
  }
  return `${rows.length}:${hash.toString(16).padStart(8, '0')}`;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;

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
      row.push(cell);
      cell = '';
    } else if (char === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else if (char !== '\r') {
      cell += char;
    }
  }

  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }

  return rows;
}

function escapeCsvCell(value) {
  const text = String(value ?? '');
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function stripBom(value) {
  return String(value ?? '').replace(/^\uFEFF/, '');
}

function fnv1a(value) {
  return fnv1aStep(0x811c9dc5, value).toString(16).padStart(8, '0');
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
