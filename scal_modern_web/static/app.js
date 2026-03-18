/* ===== Constants ===== */
const DEFAULT_PROMPTS = {
  default: `---\nAttached is one page of a document that you must process.\nJust return the plain text representation of this document as if you were reading it naturally. Convert equations to LaTeX and tables to HTML.\nIf there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\nReturn your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.\n---`,
  table: `Extract table data from the image as JSON array. If no table: {"no_table": true}. Include all rows and columns.`,
};

/* ===== State ===== */
const state = {
  currentDoc: null,
  currentLogKind: 'status',
  lastHits: [],
  stopRequested: false,
};

/* ===== Helpers ===== */
function esc(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function chatAdd(role, text) {
  const box = document.getElementById('chatBox');
  const d = document.createElement('div');
  d.className = `msg ${role}`;
  d.innerHTML = `<div class="role">${role.toUpperCase()}</div><div>${esc(text).replaceAll('\n', '<br>')}</div>`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
  return data;
}

function formData(obj) {
  const fd = new FormData();
  Object.entries(obj).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') fd.append(k, v);
  });
  return fd;
}

/* ===== Folder browse via backend ===== */
async function browseFolder(inputId) {
  try {
    const data = await api('/api/browse/folder', { method: 'POST' });
    if (data.path) {
      document.getElementById(inputId).value = data.path;
    }
  } catch (e) {
    console.warn('Browse failed:', e.message);
  }
}

async function browseFile(inputId, accept) {
  try {
    const data = await api('/api/browse/file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accept }),
    });
    if (data.path) {
      document.getElementById(inputId).value = data.path;
      onPdfPathChanged(data.path);
    }
  } catch (e) {
    console.warn('Browse failed:', e.message);
  }
}

/* ===== Docs ===== */
async function refreshDocs() {
  const root = document.getElementById('dataRoot').value.trim();
  if (!root) return;
  try {
    const data = await api(`/api/docs?root=${encodeURIComponent(root)}`);
    const select = document.getElementById('docSelect');
    select.innerHTML = '';
    data.documents.forEach((d) => {
      const o = document.createElement('option');
      o.value = d;
      o.textContent = d;
      select.appendChild(o);
    });
    if (data.documents.length) {
      state.currentDoc = data.documents[0];
      select.value = state.currentDoc;
      renderCoverage(data.coverage[state.currentDoc]);
    } else {
      document.getElementById('coverageInfo').textContent = 'No documents found in folder.';
    }
  } catch (e) {
    document.getElementById('coverageInfo').textContent = `Error: ${e.message}`;
  }
}

function renderCoverage(cov) {
  if (!cov) { document.getElementById('coverageInfo').textContent = ''; return; }
  const missing = cov.missing_pages && cov.missing_pages.length
    ? `Missing pages: [${cov.missing_pages.join(', ')}]`
    : 'All pages extracted';
  document.getElementById('coverageInfo').textContent =
    `PDF pages: ${cov.pdf_pages}  |  Extracted: ${cov.extracted_pages}\n${missing}`;
}

/* ===== Index ===== */
async function buildIndex() {
  if (!state.currentDoc) { chatAdd('assistant', 'Select a document first.'); return; }
  try {
    await api('/api/index/build', {
      method: 'POST',
      body: formData({ doc_name: state.currentDoc }),
    });
  } catch (e) {
    chatAdd('assistant', `Index error: ${e.message}`);
  }
}

/* ===== Models ===== */
async function loadModels(kind) {
  const path = kind === 'vlm' ? '/api/models/load-vlm' : '/api/models/load-llm';
  const body = kind === 'llm' ? formData({ model_name: 'Qwen/Qwen2.5-3B-Instruct' }) : formData({});
  try {
    await api(path, { method: 'POST', body });
  } catch (e) {
    chatAdd('assistant', `Model load error: ${e.message}`);
  }
}

/* ===== State polling ===== */
async function refreshState() {
  try {
    const data = await api('/api/state');
    document.getElementById('modelState').textContent =
      `VLM: ${data.models.vlm_loaded ? 'loaded' : 'not loaded'} | LLM: ${data.models.llm_loaded ? 'loaded' : 'not loaded'}`;

    const idx = data.progress.index;
    const ex  = data.progress.extract;

    const bI = document.getElementById('barIndex');
    bI.style.width = `${idx.percent}%`;
    bI.textContent = `${idx.percent}% ${idx.stage}`;
    document.getElementById('indexDetail').textContent = idx.detail || '';

    const bE = document.getElementById('barExtract');
    bE.style.width = `${ex.percent}%`;
    bE.textContent = `${ex.percent}% ${ex.stage}`;
    document.getElementById('extractDetail').textContent = ex.detail || '';

    // Re-enable stop button logic
    const stopBtn = document.getElementById('stopExtractBtn');
    if (!ex.running && stopBtn) stopBtn.disabled = true;
  } catch (_) {}
}

/* ===== Logs ===== */
async function refreshLogs() {
  try {
    const data = await api(`/api/logs?kind=${encodeURIComponent(state.currentLogKind)}&limit=200`);
    const out = (data.items || []).map((x) => `[${x.time}] ${x.message}`).join('\n');
    document.getElementById('logsView').textContent = out || '(empty)';
  } catch (_) {}
}

async function clearLogs() {
  await api('/api/logs/clear', { method: 'POST', body: formData({ kind: 'all' }) });
  await refreshLogs();
}

/* ===== Prompt presets ===== */
function applyPromptPreset() {
  const val = document.getElementById('promptPreset').value;
  if (val === 'custom') return; // Don't overwrite user text
  document.getElementById('promptText').value = DEFAULT_PROMPTS[val] || '';
}

function resetPrompt() {
  const val = document.getElementById('promptPreset').value;
  if (val === 'custom') {
    document.getElementById('promptText').value = '';
  } else {
    document.getElementById('promptText').value = DEFAULT_PROMPTS[val] || '';
  }
}

/* ===== Rendering ===== */
function renderTables(tables) {
  const root = document.getElementById('tablesView');
  root.innerHTML = '';
  if (!tables || !tables.length) {
    root.innerHTML = '<div class="mono">No parsed HTML tables in current retrieval.</div>';
    return;
  }
  tables.slice(0, 3).forEach((t, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';

    const meta = document.createElement('div');
    meta.className = 'table-meta';
    meta.textContent = `[${i + 1}] ${t.file_name || ''} | page ${t.page_number} | ${t.table_id || ''}`;
    wrap.appendChild(meta);

    const table = document.createElement('table');
    const cols = t.columns || [];
    const rows = t.rows  || [];
    if (cols.length) {
      const trh = document.createElement('tr');
      cols.forEach((c) => {
        const th = document.createElement('th');
        th.textContent = c;
        trh.appendChild(th);
      });
      table.appendChild(trh);
    }
    rows.slice(0, 30).forEach((r) => {
      const tr = document.createElement('tr');
      cols.forEach((c) => {
        const td = document.createElement('td');
        td.textContent = r[c] ?? '';
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    wrap.appendChild(table);
    root.appendChild(wrap);
  });
}

function renderReasoning(reasoning) {
  const txt = (reasoning || [])
    .map((r) => `#${r.rank} score=${r.score}  file=${r.file_name}  page=${r.page_number}  table=${r.table_id}\n${r.snippet}`)
    .join('\n\n');
  document.getElementById('reasoningLog').textContent = txt || '(no reasoning)';
}

/* ===== Chat ===== */
async function askChat() {
  const question = document.getElementById('chatInput').value.trim();
  if (!question || !state.currentDoc) {
    if (!state.currentDoc) chatAdd('assistant', 'Select a document and build its index first.');
    return;
  }
  chatAdd('user', question);
  document.getElementById('chatInput').value = '';

  const payload = {
    doc_name: state.currentDoc,
    question,
    prompt_template: document.getElementById('promptText').value,
    filter_extraction_type: document.getElementById('fType').value || null,
    top_k: 8,
  };

  try {
    const data = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.lastHits = data.raw_hits || [];
    chatAdd('assistant', data.answer || '(no answer)');
    renderReasoning(data.reasoning || []);
    renderTables(data.tables || []);
  } catch (e) {
    chatAdd('assistant', `Error: ${e.message}`);
  }
}

function clearChat() {
  document.getElementById('chatBox').innerHTML = '';
  document.getElementById('reasoningLog').textContent = '(cleared)';
  document.getElementById('tablesView').innerHTML = '<div class="mono">No tables in current retrieval.</div>';
}

/* ===== Export ===== */
async function exportExcel() {
  if (!state.lastHits.length) { chatAdd('assistant', 'No results to export. Ask a question first.'); return; }
  const data = await api('/api/export/excel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hits: state.lastHits }),
  });
  chatAdd('assistant', `Excel exported: ${data.path}`);
}

async function exportWord() {
  if (!state.lastHits.length) { chatAdd('assistant', 'No results to export. Ask a question first.'); return; }
  const data = await api('/api/export/word', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hits: state.lastHits }),
  });
  chatAdd('assistant', `Word exported: ${data.path}`);
}

/* ===== PDF Extraction ===== */
function onPdfPathChanged(path) {
  document.getElementById('pdfPageInfo').textContent = path ? `Selected: ${path}` : '';
  document.getElementById('extractCheckOut').textContent = '';
}

async function checkPdfMissing() {
  const path = document.getElementById('pdfFilePath').value.trim();
  const fileInput = document.getElementById('pdfFile');
  const docName = state.currentDoc || '';

  if (!path && !fileInput.files.length) {
    document.getElementById('extractCheckOut').textContent = 'Select a PDF first.';
    return;
  }

  const fd = new FormData();
  fd.append('doc_name', docName);

  if (fileInput.files.length) {
    fd.append('file', fileInput.files[0]);
  } else {
    // path-based check via dedicated endpoint
    fd.append('pdf_path', path);
  }

  try {
    const data = await api('/api/extract/check', { method: 'POST', body: fd });
    const lines = [
      `Total PDF pages: ${data.total_pdf_pages}`,
      `Already extracted: ${data.already_extracted ? 'Yes (complete)' : 'No'}`,
      `Missing pages: ${data.missing_pages.length ? data.missing_pages.join(', ') : 'none'}`,
    ];
    document.getElementById('extractCheckOut').textContent = lines.join('\n');
    // Auto-fill pageTo with last page
    if (data.total_pdf_pages) {
      document.getElementById('pageTo').value = data.total_pdf_pages;
    }
  } catch (e) {
    document.getElementById('extractCheckOut').textContent = `Error: ${e.message}`;
  }
}

async function startExtraction() {
  const path = document.getElementById('pdfFilePath').value.trim();
  const fileInput = document.getElementById('pdfFile');
  const prompt = document.getElementById('promptText').value;
  const pageFrom = parseInt(document.getElementById('pageFrom').value) || 1;
  const pageTo   = parseInt(document.getElementById('pageTo').value)   || 0;
  const docName  = state.currentDoc || '';

  if (!path && !fileInput.files.length) {
    document.getElementById('extractCheckOut').textContent = 'Select a PDF first.';
    return;
  }

  state.stopRequested = false;
  document.getElementById('stopExtractBtn').disabled = false;
  document.getElementById('startExtractBtn').disabled = true;

  const fd = new FormData();
  fd.append('prompt', prompt);
  fd.append('target_doc_name', docName);
  fd.append('page_from', pageFrom);
  fd.append('page_to', pageTo);

  if (fileInput.files.length) {
    fd.append('file', fileInput.files[0]);
  } else {
    fd.append('pdf_path', path);
  }

  try {
    const data = await api('/api/extract/start', { method: 'POST', body: fd });
    document.getElementById('extractCheckOut').textContent = data.message || 'Extraction started.';
  } catch (e) {
    document.getElementById('extractCheckOut').textContent = `Error: ${e.message}`;
  }
  document.getElementById('startExtractBtn').disabled = false;
}

async function stopExtraction() {
  state.stopRequested = true;
  document.getElementById('stopExtractBtn').disabled = true;
  try {
    await api('/api/extract/stop', { method: 'POST' });
  } catch (_) {}
  document.getElementById('extractCheckOut').textContent += '\nStop requested.';
}

/* ===== Init ===== */
async function init() {
  // Default data root
  document.getElementById('dataRoot').value =
    'C:\\Users\\Mining\\Downloads\\Fine Tunining Datasets-20260318T052420Z-1-001\\Fine Tunining Datasets\\train';

  // Init prompt
  document.getElementById('promptText').value = DEFAULT_PROMPTS.default;

  // Browse buttons (server-side native dialog via backend)
  document.getElementById('browseDataRootBtn').onclick = () => browseFolder('dataRoot');
  document.getElementById('browseOutputBtn').onclick   = () => browseFolder('outputDir');
  document.getElementById('browsePdfBtn').onclick = () => {
    // Prefer native dialog via backend; fallback to hidden file input
    browseFile('pdfFilePath', '.pdf').catch(() => {
      document.getElementById('pdfFile').click();
    });
  };

  // Hidden file input fallback
  document.getElementById('pdfFile').onchange = (e) => {
    const f = e.target.files[0];
    if (f) {
      document.getElementById('pdfFilePath').value = f.name;
      onPdfPathChanged(f.name);
    }
  };

  // Docs
  document.getElementById('refreshDocsBtn').onclick = refreshDocs;
  document.getElementById('docSelect').onchange = async (e) => {
    state.currentDoc = e.target.value;
    const root = document.getElementById('dataRoot').value.trim();
    const data = await api(`/api/docs?root=${encodeURIComponent(root)}`).catch(() => null);
    if (data) renderCoverage(data.coverage[state.currentDoc]);
  };
  document.getElementById('buildIndexBtn').onclick = buildIndex;

  // Models
  document.getElementById('loadVlmBtn').onclick = () => loadModels('vlm');
  document.getElementById('loadLlmBtn').onclick = () => loadModels('llm');

  // Prompt preset
  document.getElementById('promptPreset').onchange = applyPromptPreset;
  document.getElementById('resetPromptBtn').onclick = resetPrompt;

  // Mark textarea as custom when user edits directly
  document.getElementById('promptText').oninput = () => {
    document.getElementById('promptPreset').value = 'custom';
  };

  // Chat
  document.getElementById('sendBtn').onclick = askChat;
  document.getElementById('clearChatBtn').onclick = clearChat;
  document.getElementById('exportExcelBtn').onclick = exportExcel;
  document.getElementById('exportWordBtn').onclick = exportWord;

  // chatInput enter key (Shift+Enter = newline)
  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askChat(); }
  });

  // Extraction
  document.getElementById('checkPdfBtn').onclick = checkPdfMissing;
  document.getElementById('startExtractBtn').onclick = startExtraction;
  document.getElementById('stopExtractBtn').onclick  = stopExtraction;

  // Logs
  document.querySelectorAll('.log-tabs button[data-kind]').forEach((b) => {
    b.onclick = async () => {
      document.querySelectorAll('.log-tabs button[data-kind]').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.currentLogKind = b.dataset.kind;
      await refreshLogs();
    };
  });
  document.getElementById('clearLogsBtn').onclick = clearLogs;

  // Initial load
  await refreshDocs();
  await refreshState();
  await refreshLogs();

  // Poll every 1.5s
  setInterval(async () => {
    await refreshState();
    await refreshLogs();
  }, 1500);
}

init();
