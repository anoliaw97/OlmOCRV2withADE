const state = {
  currentDoc: null,
  currentLogKind: 'status',
  lastHits: [],
};

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
    if (v !== undefined && v !== null) fd.append(k, v);
  });
  return fd;
}

async function refreshDocs() {
  const root = document.getElementById('dataRoot').value;
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
  }
}

function renderCoverage(cov) {
  if (!cov) return;
  document.getElementById('coverageInfo').textContent =
    `PDF pages: ${cov.pdf_pages}\nExtracted pages: ${cov.extracted_pages}\nMissing pages: ${JSON.stringify(cov.missing_pages)}`;
}

async function buildIndex() {
  if (!state.currentDoc) return;
  await api('/api/index/build', {
    method: 'POST',
    body: formData({ doc_name: state.currentDoc }),
  });
}

async function loadModels(kind) {
  const path = kind === 'vlm' ? '/api/models/load-vlm' : '/api/models/load-llm';
  const body = kind === 'llm' ? formData({ model_name: 'Qwen/Qwen2.5-3B-Instruct' }) : formData({});
  await api(path, { method: 'POST', body });
}

async function refreshState() {
  const data = await api('/api/state');
  document.getElementById('modelState').textContent =
    `VLM: ${data.models.vlm_loaded ? 'loaded' : 'not loaded'} | LLM: ${data.models.llm_loaded ? 'loaded' : 'not loaded'}`;

  const idx = data.progress.index;
  const ex = data.progress.extract;

  const bI = document.getElementById('barIndex');
  bI.style.width = `${idx.percent}%`;
  bI.textContent = `${idx.percent}% ${idx.stage}`;
  document.getElementById('indexDetail').textContent = `${idx.detail || ''}`;

  const bE = document.getElementById('barExtract');
  bE.style.width = `${ex.percent}%`;
  bE.textContent = `${ex.percent}% ${ex.stage}`;
  document.getElementById('extractDetail').textContent = `${ex.detail || ''}`;
}

async function refreshLogs() {
  const data = await api(`/api/logs?kind=${encodeURIComponent(state.currentLogKind)}&limit=200`);
  const out = (data.items || []).map((x) => `[${x.time}] ${x.message}`).join('\n');
  document.getElementById('logsView').textContent = out || '(empty)';
}

async function clearLogs() {
  await api('/api/logs/clear', { method: 'POST', body: formData({ kind: 'all' }) });
  await refreshLogs();
}

async function refreshPrompts() {
  const data = await api('/api/prompts');
  const sel = document.getElementById('promptSelect');
  sel.innerHTML = '';
  (data.prompts || []).forEach((p) => {
    const o = document.createElement('option');
    o.value = p.name;
    o.textContent = p.name;
    o.dataset.text = p.text;
    sel.appendChild(o);
  });
  if (sel.options.length) {
    document.getElementById('promptText').value = sel.options[0].dataset.text || '';
  }
}

async function savePrompt() {
  const name = document.getElementById('promptName').value.trim();
  const text = document.getElementById('promptText').value.trim();
  if (!name || !text) return;
  await api('/api/prompts/save', { method: 'POST', body: formData({ name, text }) });
  await refreshPrompts();
}

function renderTables(tables) {
  const root = document.getElementById('tablesView');
  root.innerHTML = '';
  if (!tables || !tables.length) {
    root.innerHTML = '<div class="mono">No parsed HTML table in current retrieval.</div>';
    return;
  }
  tables.slice(0, 3).forEach((t, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    const title = document.createElement('div');
    title.className = 'mono';
    title.textContent = `[${i + 1}] ${t.file_name} | page ${t.page_number} | ${t.table_id}`;
    wrap.appendChild(title);

    const table = document.createElement('table');
    const cols = t.columns || [];
    const rows = t.rows || [];
    if (cols.length) {
      const trh = document.createElement('tr');
      cols.forEach((c) => {
        const th = document.createElement('th');
        th.textContent = c;
        trh.appendChild(th);
      });
      table.appendChild(trh);
    }
    rows.slice(0, 20).forEach((r) => {
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
    .map((r) => `#${r.rank} score=${r.score} file=${r.file_name} page=${r.page_number} table=${r.table_id}\n${r.snippet}`)
    .join('\n\n');
  document.getElementById('reasoningLog').textContent = txt || '(no reasoning)';
}

async function askChat() {
  const question = document.getElementById('chatInput').value.trim();
  if (!question || !state.currentDoc) return;

  chatAdd('user', question);
  document.getElementById('chatInput').value = '';

  const payload = {
    doc_name: state.currentDoc,
    question,
    prompt_template: document.getElementById('promptText').value,
    filter_file_name: document.getElementById('fFile').value,
    filter_report_name: document.getElementById('fReport').value,
    filter_page_number: document.getElementById('fPage').value,
    filter_table_id: document.getElementById('fTable').value,
    filter_extraction_type: document.getElementById('fType').value,
    filter_sample_id: document.getElementById('fSample').value,
    top_k: 8,
  };

  const data = await api('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  state.lastHits = data.raw_hits || [];
  chatAdd('assistant', data.answer || '(no answer)');
  renderReasoning(data.reasoning || []);
  renderTables(data.tables || []);
}

function clearChat() {
  document.getElementById('chatBox').innerHTML = '';
  document.getElementById('reasoningLog').textContent = '(cleared)';
}

async function exportExcel() {
  if (!state.lastHits.length) return;
  const data = await api('/api/export/excel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hits: state.lastHits }),
  });
  chatAdd('assistant', `Excel exported: ${data.path}`);
}

async function exportWord() {
  if (!state.lastHits.length) return;
  const data = await api('/api/export/word', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hits: state.lastHits }),
  });
  chatAdd('assistant', `Word exported: ${data.path}`);
}

async function checkPdfMissing() {
  const f = document.getElementById('pdfFile').files[0];
  if (!f || !state.currentDoc) return;
  const fd = new FormData();
  fd.append('file', f);
  fd.append('doc_name', state.currentDoc);
  const data = await api('/api/extract/check', { method: 'POST', body: fd });
  document.getElementById('extractCheckOut').textContent = JSON.stringify(data, null, 2);
}

async function startExtraction() {
  const f = document.getElementById('pdfFile').files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f);
  fd.append('prompt', document.getElementById('extractPrompt').value || '');
  fd.append('target_doc_name', state.currentDoc || '');
  const data = await api('/api/extract/start', { method: 'POST', body: fd });
  document.getElementById('extractCheckOut').textContent = JSON.stringify(data, null, 2);
}

async function init() {
  document.getElementById('dataRoot').value =
    'C:\\Users\\Mining\\Downloads\\Fine Tunining Datasets-20260318T052420Z-1-001\\Fine Tunining Datasets\\train';

  document.getElementById('refreshDocsBtn').onclick = async () => {
    await refreshDocs();
  };
  document.getElementById('docSelect').onchange = async (e) => {
    state.currentDoc = e.target.value;
    const d = await api(`/api/docs?root=${encodeURIComponent(document.getElementById('dataRoot').value)}`);
    renderCoverage(d.coverage[state.currentDoc]);
  };
  document.getElementById('buildIndexBtn').onclick = buildIndex;
  document.getElementById('loadVlmBtn').onclick = () => loadModels('vlm');
  document.getElementById('loadLlmBtn').onclick = () => loadModels('llm');

  document.getElementById('savePromptBtn').onclick = savePrompt;
  document.getElementById('promptSelect').onchange = (e) => {
    const opt = e.target.selectedOptions[0];
    document.getElementById('promptText').value = opt?.dataset?.text || '';
  };

  document.getElementById('sendBtn').onclick = askChat;
  document.getElementById('clearChatBtn').onclick = clearChat;
  document.getElementById('exportExcelBtn').onclick = exportExcel;
  document.getElementById('exportWordBtn').onclick = exportWord;

  document.getElementById('checkPdfBtn').onclick = checkPdfMissing;
  document.getElementById('startExtractBtn').onclick = startExtraction;

  document.querySelectorAll('.log-tabs button[data-kind]').forEach((b) => {
    b.onclick = async () => {
      document.querySelectorAll('.log-tabs button[data-kind]').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.currentLogKind = b.dataset.kind;
      await refreshLogs();
    };
  });
  document.getElementById('clearLogsBtn').onclick = clearLogs;

  await refreshDocs();
  await refreshPrompts();
  await refreshState();
  await refreshLogs();
  setInterval(async () => {
    await refreshState();
    await refreshLogs();
  }, 1500);
}

init();
