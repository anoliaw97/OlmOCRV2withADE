'use strict';

/* ══════════════════════════════════════════
   Constants
══════════════════════════════════════════ */
const DEFAULT_PROMPTS = {
  default: `---\nAttached is one page of a document that you must process.\nJust return the plain text representation of this document as if you were reading it naturally. Convert equations to LaTeX and tables to HTML.\nIf there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\nReturn your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.\n---`,
  table: `Extract table data from the image as JSON array. If no table: {"no_table": true}. Include all rows and columns.`,
};

/* ══════════════════════════════════════════
   App State
══════════════════════════════════════════ */
const S = {
  currentDoc: null,
  currentLogKind: 'status',
  lastHits: [],
  currentPage: null,        // { doc, page, files }
  coverageMap: {},          // doc -> coverage object
  docsMap: {},              // doc -> pages map from /api/docs
};

/* ══════════════════════════════════════════
   Utilities
══════════════════════════════════════════ */
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function $(id) { return document.getElementById(id); }

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) throw new Error((await res.text()) || `HTTP ${res.status}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
  return data;
}

function fd(obj) {
  const f = new FormData();
  for (const [k,v] of Object.entries(obj)) { if (v != null && v !== '') f.append(k, v); }
  return f;
}

function show(el, visible = true) {
  if (typeof el === 'string') el = $(el);
  if (!el) return;
  el.classList.toggle('hidden', !visible);
}

/* ══════════════════════════════════════════
   Tab switching (main tabs)
══════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      $('tab-' + btn.dataset.tab).classList.remove('hidden');
      if (btn.dataset.tab === 'logs') refreshLogs();
    };
  });
  // Viewer sub-tabs
  document.querySelectorAll('.vtab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.vtab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.vtab-pane').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      $('vtab-' + btn.dataset.vtab).classList.remove('hidden');
    };
  });
}

/* ══════════════════════════════════════════
   Browse helpers
   - Folders: use server-side tkinter dialog
     (runs in executor, non-blocking)
   - PDF files: use hidden <input type=file>
     for instant local pick
══════════════════════════════════════════ */
async function browseFolder(inputId, btnEl) {
  if (btnEl) { btnEl.textContent = '⏳'; btnEl.disabled = true; }
  try {
    const d = await apiFetch('/api/browse/folder', { method: 'POST' });
    if (d.path) $(inputId).value = d.path;
  } catch (e) {
    console.warn('Folder browse failed:', e.message);
  } finally {
    if (btnEl) { btnEl.textContent = '📁'; btnEl.disabled = false; }
  }
}

function initBrowse() {
  // Data root folder browse — server dialog
  $('browseDataRootBtn').onclick = (e) => browseFolder('dataRoot', e.currentTarget);

  // Output folder browse — server dialog
  $('browseOutputBtn').onclick = (e) => browseFolder('outputDir', e.currentTarget);

  // PDF file pick — instant browser native (no OS path restriction for files)
  $('pdfFilePick').onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $('pdfFilePath').value = f.name;
    $('pdfPageInfo').textContent = `Selected: ${f.name}`;
    show('pdfPageInfo', true);
    show('extractCheckOut', false);
  };
  $('browsePdfBtn').onclick = () => $('pdfFilePick').click();
}

/* ══════════════════════════════════════════
   Documents
══════════════════════════════════════════ */
async function refreshDocs() {
  const root = $('dataRoot').value.trim();
  if (!root) return;
  try {
    const data = await apiFetch(`/api/docs?root=${encodeURIComponent(root)}`);
    S.coverageMap = data.coverage || {};
    S.docsMap     = data.pages_map || {};

    const sel = $('docSelect');
    sel.innerHTML = '';
    (data.documents || []).forEach(d => {
      const o = document.createElement('option');
      o.value = d; o.textContent = d;
      sel.appendChild(o);
    });
    if (data.documents.length) {
      S.currentDoc = data.documents[0];
      sel.value = S.currentDoc;
      renderCoverage(S.coverageMap[S.currentDoc]);
      buildPageList(S.currentDoc, data.pages_map?.[S.currentDoc]);
    } else {
      $('coverageInfo').textContent = 'No documents found.';
    }
  } catch (e) {
    $('coverageInfo').textContent = `Error: ${e.message}`;
  }
}

function renderCoverage(cov) {
  if (!cov) { $('coverageInfo').textContent = '—'; return; }
  const miss = cov.missing_pages?.length
    ? `⚠ Missing: [${cov.missing_pages.join(', ')}]`
    : '✓ Fully extracted';
  $('coverageInfo').textContent = `PDF pages: ${cov.pdf_pages}  ·  Extracted: ${cov.extracted_pages}\n${miss}`;
}

/* ══════════════════════════════════════════
   Index
══════════════════════════════════════════ */
async function buildIndex() {
  const scope = document.querySelector('input[name="indexScope"]:checked')?.value;
  if (scope === 'all') {
    try {
      const r = await apiFetch('/api/index/build-all', {
        method: 'POST',
        body: fd({ data_root: $('dataRoot').value.trim() }),
      });
      addChatMsg('system', r.message || 'All-docs index build started.');
    } catch (e) { addChatMsg('system', `Index error: ${e.message}`); }
  } else {
    if (!S.currentDoc) { addChatMsg('system', 'Select a document first.'); return; }
    try {
      const r = await apiFetch('/api/index/build', {
        method: 'POST',
        body: fd({ doc_name: S.currentDoc }),
      });
      addChatMsg('system', r.message || 'Index build started.');
    } catch (e) { addChatMsg('system', `Index error: ${e.message}`); }
  }
}

/* ══════════════════════════════════════════
   Models — load in background, show spinner
══════════════════════════════════════════ */
async function loadModel(kind) {
  const btn = kind === 'vlm' ? $('loadVlmBtn') : $('loadLlmBtn');
  const dot = kind === 'vlm' ? $('vlmDot') : $('llmDot');
  const lbl = kind === 'vlm' ? $('vlmLabel') : $('llmLabel');
  btn.disabled = true;
  dot.className = 'dot dot-busy';
  lbl.textContent = kind.toUpperCase() + ' loading…';
  show('loadingSpinner', true);

  try {
    const path = kind === 'vlm' ? '/api/models/load-vlm' : '/api/models/load-llm';
    const body = kind === 'llm' ? fd({ model_name: 'Qwen/Qwen2.5-3B-Instruct' }) : fd({});
    const r = await apiFetch(path, { method: 'POST', body });
    if (r.ok) {
      dot.className = 'dot dot-on';
      lbl.textContent = kind.toUpperCase() + ' ✓';
    } else {
      dot.className = 'dot dot-off';
      lbl.textContent = kind.toUpperCase() + ' ✗';
      addChatMsg('system', `${kind.toUpperCase()} load failed: ${r.error || 'unknown error'}`);
    }
  } catch (e) {
    dot.className = 'dot dot-off';
    lbl.textContent = kind.toUpperCase() + ' ✗';
    addChatMsg('system', `${kind.toUpperCase()} load error: ${e.message}`);
  } finally {
    btn.disabled = false;
    show('loadingSpinner', false);
  }
}

/* ══════════════════════════════════════════
   State polling (every 1.5 s)
══════════════════════════════════════════ */
async function pollState() {
  try {
    const d = await apiFetch('/api/state');

    // Model pills
    const vlmOn = d.models.vlm_loaded;
    const llmOn = d.models.llm_loaded;
    $('vlmDot').className = 'dot ' + (vlmOn ? 'dot-on' : 'dot-off');
    $('vlmLabel').textContent = 'VLM' + (vlmOn ? ' ✓' : '');
    $('llmDot').className = 'dot ' + (llmOn ? 'dot-on' : 'dot-off');
    $('llmLabel').textContent = 'LLM' + (llmOn ? ' ✓' : '');

    // Index bar
    const ix = d.progress.index;
    const bI = $('barIndex');
    bI.style.width = ix.percent + '%';
    bI.textContent = ix.percent + '% ' + ix.stage;
    $('indexDetail').textContent = ix.detail || '';

    // Extract bar
    const ex = d.progress.extract;
    const bE = $('barExtract');
    bE.style.width = ex.percent + '%';
    bE.textContent = ex.percent + '% ' + ex.stage;
    $('extractDetail').textContent = ex.detail || '';

    if (!ex.running) $('stopExtractBtn').disabled = true;
  } catch (_) {}
}

/* ══════════════════════════════════════════
   Logs
══════════════════════════════════════════ */
async function refreshLogs() {
  try {
    const d = await apiFetch(`/api/logs?kind=${encodeURIComponent(S.currentLogKind)}&limit=500`);
    const box = $('logsView');
    box.innerHTML = (d.items || []).map(x => {
      const cls = `log-${x.kind}`;
      return `<div class="log-line ${cls}"><span class="log-time">[${x.time}]</span> ${esc(x.message)}</div>`;
    }).join('');
    if ($('autoScrollLogs').checked) box.scrollTop = box.scrollHeight;
  } catch (_) {}
}

async function clearLogs() {
  await apiFetch('/api/logs/clear', { method: 'POST', body: fd({ kind: 'all' }) });
  await refreshLogs();
}

function initLogTabs() {
  document.querySelectorAll('.lkbtn').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.lkbtn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      S.currentLogKind = b.dataset.kind;
      refreshLogs();
    };
  });
}

/* ══════════════════════════════════════════
   Prompt presets
══════════════════════════════════════════ */
function applyPreset() {
  const v = $('promptPreset').value;
  if (v !== 'custom') $('promptText').value = DEFAULT_PROMPTS[v] || '';
}
function resetPrompt() {
  const v = $('promptPreset').value;
  $('promptText').value = v === 'custom' ? '' : (DEFAULT_PROMPTS[v] || '');
}

/* ══════════════════════════════════════════
   Chat
══════════════════════════════════════════ */
function addChatMsg(role, text) {
  const box = $('chatBox');
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap msg-${role}`;
  const roleLabel = { user:'YOU', assistant:'ASSISTANT', system:'SYSTEM' }[role] || role.toUpperCase();
  wrap.innerHTML = `<div class="msg-role">${roleLabel}</div><div class="msg-body">${esc(text).replace(/\n/g,'<br>')}</div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

async function askChat() {
  const q = $('chatInput').value.trim();
  if (!q) return;
  if (!S.currentDoc) { addChatMsg('system', 'Select a document and build its index first.'); return; }

  addChatMsg('user', q);
  $('chatInput').value = '';

  const sendBtn = $('sendBtn');
  sendBtn.disabled = true;
  sendBtn.textContent = '…';

  try {
    const d = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_name: S.currentDoc,
        question: q,
        prompt_template: $('promptText').value,
        filter_extraction_type: $('fType').value || null,
        top_k: 8,
      }),
    });
    S.lastHits = d.raw_hits || [];
    addChatMsg('assistant', d.answer || '(no answer)');
    renderReasoning(d.reasoning || []);
    renderRetrievedTables(d.tables || []);
  } catch (e) {
    addChatMsg('system', `Error: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send ↵';
  }
}

function clearChat() {
  $('chatBox').innerHTML = '';
  $('reasoningList').innerHTML = '';
  $('tablesView').innerHTML = '<div class="placeholder-text">Tables from the last chat answer will appear here</div>';
}

/* ══════════════════════════════════════════
   Reasoning + Tables panels
══════════════════════════════════════════ */
function renderReasoning(items) {
  const box = $('reasoningList');
  box.innerHTML = items.map(r =>
    `<div class="reasoning-item">
      <span class="ri-rank">#${r.rank}</span>
      <span class="ri-score"> score=${r.score}</span>
      <span class="ri-source"> ${esc(r.file_name||'')} p${r.page_number}</span>
      <div class="ri-snippet">${esc(r.snippet)}</div>
    </div>`
  ).join('');
}

function renderRetrievedTables(tables) {
  const box = $('tablesView');
  if (!tables.length) {
    box.innerHTML = '<div class="placeholder-text">No parsed tables in last retrieval.</div>';
    return;
  }
  box.innerHTML = tables.slice(0,5).map((t,i) => {
    const cols = t.columns || [];
    const rows = (t.rows || []).slice(0, 30);
    const thead = cols.map(c => `<th>${esc(c)}</th>`).join('');
    const tbody = rows.map(r => '<tr>' + cols.map(c => `<td>${esc(r[c]??'')}</td>`).join('') + '</tr>').join('');
    return `<div class="rtable-wrap">
      <div class="rtable-meta">[${i+1}] ${esc(t.file_name||'')} · p${t.page_number} · ${esc(t.table_id||'')}</div>
      <table class="rtable"><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>
    </div>`;
  }).join('');
}

/* ══════════════════════════════════════════
   Export  (with output folder picker)
══════════════════════════════════════════ */
async function doExport(type, btn) {
  if (!S.lastHits.length) { addChatMsg('system', 'No results to export. Ask a question first.'); return; }

  // Ask user for output folder via server-side dialog
  if (btn) { btn.textContent = '⏳'; btn.disabled = true; }
  let outputDir = '';
  try {
    const d = await apiFetch('/api/browse/folder', { method: 'POST' });
    outputDir = d.path || '';          // empty = use default EXPORT_DIR
  } catch (_) {}
  if (btn) { btn.textContent = type === 'excel' ? '⬇ Excel' : '⬇ Word'; btn.disabled = false; }

  try {
    const d = await apiFetch(`/api/export/${type}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hits: S.lastHits, output_dir: outputDir }),
    });
    addChatMsg('system', `✓ Exported to: ${d.path}`);
  } catch (e) { addChatMsg('system', `Export error: ${e.message}`); }
}

/* ══════════════════════════════════════════
   Document Viewer
══════════════════════════════════════════ */
function buildPageList(docName, pagesMap) {
  const box = $('pageList');
  box.innerHTML = '';
  if (!pagesMap) return;
  const pages = Object.keys(pagesMap).map(Number).sort((a,b)=>a-b);
  pages.forEach(pg => {
    const files = pagesMap[pg];
    const exts = Object.keys(files).filter(e => e !== 'pdf').join('/');
    const hasPdf = 'pdf' in files;
    const el = document.createElement('div');
    el.className = 'page-item';
    el.innerHTML = `<div class="pi-name">Page ${pg}</div><div class="pi-ext">${hasPdf?'📄 PDF':''}${exts?' + '+exts.toUpperCase():''}</div>`;
    el.onclick = () => selectPage(docName, pg, files, el);
    box.appendChild(el);
  });
}

async function selectPage(docName, pg, files, el) {
  // deactivate all
  document.querySelectorAll('.page-item').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  S.currentPage = { doc: docName, page: pg, files };

  // PDF preview
  const pdfWrap = $('pdfPreviewWrap');
  pdfWrap.innerHTML = '';
  if (files.pdf) {
    const url = `/api/page/pdf?doc=${encodeURIComponent(docName)}&page=${pg}`;
    const obj = document.createElement('object');
    obj.data = url;
    obj.type = 'application/pdf';
    obj.style.cssText = 'width:100%;height:100%;border:none';
    obj.innerHTML = `<div class="preview-placeholder">PDF preview not supported in this browser.<br>
      <a href="${url}" target="_blank" style="color:#60a5fa">Open PDF ↗</a></div>`;
    pdfWrap.appendChild(obj);
  } else {
    pdfWrap.innerHTML = '<div class="preview-placeholder">No PDF file for this page.</div>';
  }

  // Raw content + auto-render HTML tables
  const jsonEl = $('jsonView');
  jsonEl.textContent = 'Loading…';
  $('htmlTablesView').innerHTML = '<div class="placeholder-text">Loading…</div>';
  try {
    const d = await apiFetch(`/api/page/raw?doc=${encodeURIComponent(docName)}&page=${pg}`);
    const content = d.content || '';
    jsonEl.textContent = content;
    renderHtmlTablesFromContent(content, pg);
  } catch (e) {
    jsonEl.textContent = `Error: ${e.message}`;
    $('htmlTablesView').innerHTML = `<div class="placeholder-text">Error: ${esc(e.message)}</div>`;
  }
}

/* Extract and render raw <table> HTML blocks directly in the browser */
function renderHtmlTablesFromContent(content, pg) {
  const box = $('htmlTablesView');

  // Pull all <table>…</table> blocks (including any imperfect HTML from OCR)
  const tableRx = /<table[\s\S]*?<\/table>/gi;
  const matches = content.match(tableRx);

  if (!matches || !matches.length) {
    box.innerHTML = '<div class="placeholder-text">No HTML tables found on this page.</div>';
    return;
  }

  // Inject the raw HTML — browser will handle tolerant parsing
  box.innerHTML = matches.map((html, i) => {
    // Apply our table styles by wrapping in a styled container
    // Sanitise: strip <script> tags just in case
    const safe = html.replace(/<script[\s\S]*?<\/script>/gi, '');
    return `<div class="viewer-table-block">
      <div class="viewer-table-label">Table ${i + 1} — page ${pg}</div>
      <div class="viewer-table-scroll">${safe}</div>
    </div>`;
  }).join('');

  // Auto-switch to Rendered Tables tab
  document.querySelectorAll('.vtab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.vtab-pane').forEach(p => p.classList.add('hidden'));
  document.querySelector('.vtab-btn[data-vtab="html"]').classList.add('active');
  $('vtab-html').classList.remove('hidden');
}



/* ══════════════════════════════════════════
   PDF Extraction
══════════════════════════════════════════ */
async function checkPdf() {
  const pdfInput = $('pdfFilePick');
  const pathVal  = $('pdfFilePath').value.trim();
  const docName  = S.currentDoc || '';

  if (!pdfInput.files.length && !pathVal) {
    show('extractCheckOut', true);
    $('extractCheckOut').textContent = 'Select a PDF file first.';
    return;
  }
  try {
    const f = new FormData();
    f.append('doc_name', docName);
    if (pdfInput.files.length) f.append('file', pdfInput.files[0]);
    else f.append('pdf_path', pathVal);
    const d = await apiFetch('/api/extract/check', { method: 'POST', body: f });
    $('extractCheckOut').textContent = [
      `Total PDF pages: ${d.total_pdf_pages}`,
      `Already extracted: ${d.already_extracted ? 'Yes ✓' : 'No'}`,
      `Missing pages: ${d.missing_pages.length ? d.missing_pages.join(', ') : 'none'}`,
    ].join('\n');
    show('extractCheckOut', true);
    if (d.total_pdf_pages) $('pageTo').value = d.total_pdf_pages;
  } catch (e) {
    $('extractCheckOut').textContent = `Error: ${e.message}`;
    show('extractCheckOut', true);
  }
}

async function startExtract() {
  const pdfInput = $('pdfFilePick');
  const pathVal  = $('pdfFilePath').value.trim();
  if (!pdfInput.files.length && !pathVal) {
    $('extractCheckOut').textContent = 'Select a PDF file first.';
    show('extractCheckOut', true);
    return;
  }
  $('stopExtractBtn').disabled = false;
  $('startExtractBtn').disabled = true;
  try {
    const f = new FormData();
    f.append('prompt', $('promptText').value);
    f.append('target_doc_name', S.currentDoc || '');
    f.append('output_dir', $('outputDir').value.trim());
    f.append('page_from', $('pageFrom').value || 1);
    f.append('page_to', $('pageTo').value || 0);
    if (pdfInput.files.length) f.append('file', pdfInput.files[0]);
    else f.append('pdf_path', pathVal);
    const d = await apiFetch('/api/extract/start', { method: 'POST', body: f });
    $('extractCheckOut').textContent = d.message || 'Extraction started.';
    show('extractCheckOut', true);
  } catch (e) {
    $('extractCheckOut').textContent = `Error: ${e.message}`;
    show('extractCheckOut', true);
    $('startExtractBtn').disabled = false;
  }
}

async function stopExtract() {
  $('stopExtractBtn').disabled = true;
  try { await apiFetch('/api/extract/stop', { method: 'POST' }); } catch (_) {}
  const el = $('extractCheckOut');
  el.textContent = (el.textContent || '') + '\nStop signal sent.';
  show('extractCheckOut', true);
}

/* ══════════════════════════════════════════
   Init
══════════════════════════════════════════ */
async function init() {
  initTabs();
  initLogTabs();
  initBrowse();

  // Prompt
  $('promptText').value = DEFAULT_PROMPTS.default;
  $('promptPreset').onchange = applyPreset;
  $('resetPromptBtn').onclick = resetPrompt;
  $('promptText').oninput = () => { $('promptPreset').value = 'custom'; };

  // Docs
  $('refreshDocsBtn').onclick = refreshDocs;
  $('docSelect').onchange = (e) => {
    S.currentDoc = e.target.value;
    renderCoverage(S.coverageMap[S.currentDoc]);
    buildPageList(S.currentDoc, S.docsMap?.[S.currentDoc]);
  };

  // Index
  $('buildIndexBtn').onclick = buildIndex;

  // Models — these are slow (CUDA load); apiFetch blocks until done
  $('loadVlmBtn').onclick = () => loadModel('vlm');
  $('loadLlmBtn').onclick = () => loadModel('llm');

  // Chat
  $('sendBtn').onclick = askChat;
  $('clearChatBtn').onclick = clearChat;
  $('exportExcelBtn').onclick = (e) => doExport('excel', e.currentTarget);
  $('exportWordBtn').onclick  = (e) => doExport('word',  e.currentTarget);
  $('chatInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askChat(); }
  });

  // Extraction
  $('checkPdfBtn').onclick    = checkPdf;
  $('startExtractBtn').onclick = startExtract;
  $('stopExtractBtn').onclick  = stopExtract;

  // Logs
  $('clearLogsBtn').onclick = clearLogs;

  // Boot
  await refreshDocs();
  await pollState();
  await refreshLogs();

  setInterval(() => { pollState(); }, 1500);
  setInterval(() => {
    const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab;
    if (activeTab === 'logs') refreshLogs();
  }, 2000);
}

init();
