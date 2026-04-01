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
  sessionId: null,
  suggestions: [],
  advancedMode: false,
  experimentOptions: { retrieval_configs: [], prompt_types: [], models: [] },
  benchmarkState: { running: false },
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
  // Main tabs (Chat / Viewer only)
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      $('tab-' + btn.dataset.tab).classList.remove('hidden');
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

  // PDF file pick — instant browser native (friendly dropzone + button)
  const drop = $('pdfDropzone');
  const pick = $('pdfFilePick');

  function setPickedFile(f) {
    if (!f) return;
    $('pdfFilePath').value = f.name;
    $('pdfSelectedName').textContent = f.name;
    $('pdfPageInfo').textContent = `Selected: ${f.name}`;
    show('pdfPageInfo', true);
    show('extractCheckOut', false);
  }

  $('pdfFilePick').onchange = (e) => {
    const f = e.target.files[0];
    setPickedFile(f);
  };
  $('browsePdfBtn').onclick = () => pick.click();
  $('clearPdfBtn').onclick = () => {
    pick.value = '';
    $('pdfFilePath').value = '';
    $('pdfSelectedName').textContent = 'No file selected';
    $('pdfPageInfo').textContent = '';
    show('pdfPageInfo', false);
  };

  drop.onclick = () => pick.click();
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('drag-over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f || !f.name.toLowerCase().endsWith('.pdf')) return;
    const dt = new DataTransfer();
    dt.items.add(f);
    pick.files = dt.files;
    setPickedFile(f);
  });
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
  const modelName = $('llmModelSelect')?.value || 'Qwen/Qwen2.5-14B-Instruct';
  lbl.textContent = kind === 'llm' ? `LLM loading… (${modelName.split('/').pop()})` : 'VLM loading…';
  show('loadingSpinner', kind === 'llm');

  try {
    const path = kind === 'vlm' ? '/api/models/load-vlm' : '/api/models/switch-llm';
    const body = kind === 'llm' ? fd({ model_name: modelName }) : fd({});
    const r = await apiFetch(path, { method: 'POST', body });
    if (!r.ok) {
      dot.className = 'dot dot-off';
      lbl.textContent = kind.toUpperCase() + ' ✗';
      addChatMsg('system', `${kind.toUpperCase()} load failed: ${r.message || r.error || 'unknown error'}`);
    } else if (kind === 'llm') {
      addChatMsg('system', r.message || `LLM switch started for ${modelName}.`);
    }
  } catch (e) {
    dot.className = 'dot dot-off';
    lbl.textContent = kind.toUpperCase() + ' ✗';
    addChatMsg('system', `${kind.toUpperCase()} load error: ${e.message}`);
  } finally {
    btn.disabled = false;
    if (kind !== 'llm') show('loadingSpinner', false);
  }
}

async function unloadLlm() {
  const btn = $('unloadLlmBtn');
  if (btn) btn.disabled = true;
  try {
    const d = await apiFetch('/api/models/unload-llm', { method: 'POST', body: fd({}) });
    addChatMsg('system', d.message || 'LLM unload requested.');
  } catch (e) {
    addChatMsg('system', `LLM unload error: ${e.message}`);
  } finally {
    if (btn) btn.disabled = false;
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
    const llmLoading = !!d.models.llm_loading;
    const llmTarget = d.models.llm_target_model || '';
    const llmErr = d.models.llm_last_error || '';
    $('vlmDot').className = 'dot ' + (vlmOn ? 'dot-on' : 'dot-off');
    $('vlmLabel').textContent = 'VLM' + (vlmOn ? ' ✓' : '');
    $('llmDot').className = 'dot ' + (llmLoading ? 'dot-busy' : (llmOn ? 'dot-on' : 'dot-off'));
    if (llmLoading) {
      const target = llmTarget ? ` (${llmTarget.split('/').pop()})` : '';
      $('llmLabel').textContent = 'LLM loading…' + target;
    } else {
      $('llmLabel').textContent = 'LLM' + (llmOn ? ' ✓' : '');
    }
    $('loadLlmBtn').disabled = llmLoading;
    $('unloadLlmBtn').disabled = llmLoading;

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

    // Model load bar
    const md = d.progress.model || { percent: 0, stage: 'idle', detail: '' };
    const bM = $('barModel');
    if (bM) {
      bM.style.width = md.percent + '%';
      bM.textContent = md.percent + '% ' + md.stage;
    }
    const mdDetail = $('modelDetail');
    if (mdDetail) mdDetail.textContent = llmErr ? `${md.detail || ''} | Last error: ${llmErr}` : (md.detail || '');
    show('loadingSpinner', llmLoading);

    // Experiment bar
    const ep = d.experiment || { percent: 0, stage: 'idle', detail: '' };
    const bX = $('barExperiment');
    if (bX) {
      bX.style.width = (ep.percent || 0) + '%';
      bX.textContent = `${ep.percent || 0}% ${ep.stage || 'idle'}`;
    }
    const exDetail = $('experimentDetail');
    if (exDetail) exDetail.textContent = ep.detail || '';
    if (S.advancedMode && $('expStatusBox')) {
      $('expStatusBox').textContent = JSON.stringify(ep, null, 2);
    }

    const bm = d.benchmark || { percent: 0, stage: 'idle', detail: '' };
    if (S.advancedMode && $('benchStatusBox')) {
      $('benchStatusBox').textContent = JSON.stringify(bm, null, 2) + (llmErr ? `\n\nModel error: ${llmErr}` : '');
    }

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
   Model/session/suggestion helpers
══════════════════════════════════════════ */
async function initModelOptions() {
  try {
    const d = await apiFetch('/api/models/options');
    const sel = $('llmModelSelect');
    if (!sel) return;
    sel.innerHTML = '';
    (d.models || []).forEach((m) => {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = m.label;
      if (m.recommended) o.textContent += ' ★';
      sel.appendChild(o);
    });
    if (d.default) sel.value = d.default;
  } catch (_) {}
}

async function refreshSessions() {
  try {
    const d = await apiFetch('/api/chat/sessions');
    const sel = $('sessionSelect');
    if (!sel) return;
    sel.innerHTML = '';
    (d.sessions || []).forEach((s) => {
      const o = document.createElement('option');
      o.value = s.id;
      o.textContent = `${s.title} (${s.message_count})`;
      sel.appendChild(o);
    });
    if (!S.sessionId && sel.options.length) {
      S.sessionId = sel.options[0].value;
      sel.value = S.sessionId;
      await loadSessionMessages(S.sessionId);
    } else if (S.sessionId) {
      sel.value = S.sessionId;
    }
  } catch (_) {}
}

async function createNewSession() {
  try {
    const d = await apiFetch('/api/chat/session/new', { method: 'POST', body: fd({ title: '' }) });
    const s = d.session;
    S.sessionId = s.id;
    $('chatBox').innerHTML = '';
    await refreshSessions();
    addChatMsg('system', 'Started a new chat session.');
  } catch (e) {
    addChatMsg('system', `Session error: ${e.message}`);
  }
}

async function loadSessionMessages(sessionId) {
  if (!sessionId) return;
  try {
    const d = await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`);
    $('chatBox').innerHTML = '';
    (d.session?.messages || []).forEach((m) => {
      const role = m.role === 'assistant' ? 'assistant' : (m.role === 'user' ? 'user' : 'system');
      addChatMsg(role, m.content || '');
    });
  } catch (_) {}
}

function checkedValues(name) {
  return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map((x) => x.value);
}

function renderExperimentChecks() {
  const cfgBox = $('expConfigChecks');
  const prmBox = $('expPromptChecks');
  const mdlBox = $('expModelChecks');
  const bmdlBox = $('benchModelChecks');
  if (!cfgBox || !prmBox || !mdlBox) return;
  const opts = S.experimentOptions || { retrieval_configs: [], prompt_types: [], models: [] };
  cfgBox.innerHTML = (opts.retrieval_configs || []).map((c) =>
    `<label class="radio-label"><input type="checkbox" name="expCfg" value="${esc(c)}" checked /> ${esc(c)}</label>`
  ).join('');
  prmBox.innerHTML = (opts.prompt_types || []).map((p) =>
    `<label class="radio-label"><input type="checkbox" name="expPrompt" value="${esc(p)}" checked /> ${esc(p)}</label>`
  ).join('');
  mdlBox.innerHTML = (opts.models || []).map((m) =>
    `<label class="radio-label"><input type="checkbox" name="expModel" value="${esc(m.name)}" checked /> ${esc(m.label || m.name)}</label>`
  ).join('');
  if (bmdlBox) {
    bmdlBox.innerHTML = (opts.models || []).map((m) =>
      `<label class="radio-label"><input type="checkbox" name="benchModel" value="${esc(m.name)}" checked /> ${esc(m.label || m.name)}</label>`
    ).join('');
  }
}

function setAdvancedModeUi(enabled) {
  S.advancedMode = !!enabled;
  const tabBtn = $('experimentsTabBtn');
  const hint = $('advancedModeHint');
  if (tabBtn) tabBtn.classList.toggle('hidden', !enabled);
  if (hint) hint.textContent = enabled
    ? 'Advanced mode is ON. Experiments panel enabled.'
    : 'Advanced mode is OFF.';
  if (!enabled && $('tab-experiments') && !$('tab-experiments').classList.contains('hidden')) {
    document.querySelector('.tab-btn[data-tab="chat"]')?.click();
  }
}

async function initAdvancedMode() {
  try {
    const d = await apiFetch('/api/settings');
    const enabled = !!d.advanced_mode;
    $('advancedModeToggle').checked = enabled;
    setAdvancedModeUi(enabled);
  } catch (_) {
    $('advancedModeToggle').checked = false;
    setAdvancedModeUi(false);
  }

  $('advancedModeToggle').onchange = async (e) => {
    const enabled = !!e.target.checked;
    try {
      const d = await apiFetch('/api/settings/advanced-mode', { method: 'POST', body: fd({ enabled }) });
      setAdvancedModeUi(!!d.advanced_mode);
    } catch (err) {
      addChatMsg('system', `Advanced mode toggle failed: ${err.message}`);
      e.target.checked = S.advancedMode;
    }
  };

  try {
    const opts = await apiFetch('/api/experiments/options');
    S.experimentOptions = opts;
    renderExperimentChecks();
  } catch (_) {}
}

async function runExperiment() {
  const payload = {
    mode: $('expMode').value,
    data_root: $('dataRoot').value.trim(),
    benchmark_path: $('expBenchmarkPath').value.trim(),
    output_root: $('expOutputRoot').value.trim(),
    top_k: Number($('expTopK').value || 3),
    run_id: $('expRunId').value.trim(),
    selected_retrieval_config: $('expSelectedConfig').value.trim(),
    retrieval_configs: checkedValues('expCfg'),
    prompt_types: checkedValues('expPrompt'),
    model_names: checkedValues('expModel'),
  };
  if (!payload.benchmark_path) {
    $('expStatusBox').textContent = 'Benchmark path is required.';
    return;
  }
  try {
    const d = await apiFetch('/api/experiments/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    $('expStatusBox').textContent = d.message || 'Experiment started.';
  } catch (e) {
    $('expStatusBox').textContent = `Experiment start error: ${e.message}`;
  }
}

async function stopExperiment() {
  try {
    const d = await apiFetch('/api/experiments/stop', { method: 'POST', body: fd({}) });
    $('expStatusBox').textContent = d.message || 'Stop signal sent.';
  } catch (e) {
    $('expStatusBox').textContent = `Experiment stop error: ${e.message}`;
  }
}

async function runModelBenchmark() {
  const payload = {
    question: $('benchQuestion').value.trim(),
    simple_context: $('benchSimpleContext').value.trim(),
    detailed_context: $('benchDetailedContext').value.trim(),
    expected_keywords: $('benchExpectedKeywords').value.trim(),
    model_names: checkedValues('benchModel'),
    run_id: $('benchRunId').value.trim(),
    output_root: $('benchOutputRoot').value.trim(),
  };
  if (!payload.question) {
    $('benchStatusBox').textContent = 'Benchmark question is required.';
    return;
  }
  try {
    const d = await apiFetch('/api/benchmarks/models/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    $('benchStatusBox').textContent = d.message || 'Model benchmark started.';
  } catch (e) {
    $('benchStatusBox').textContent = `Benchmark start error: ${e.message}`;
  }
}

async function stopModelBenchmark() {
  try {
    const d = await apiFetch('/api/benchmarks/models/stop', { method: 'POST', body: fd({}) });
    $('benchStatusBox').textContent = d.message || 'Benchmark stop requested.';
  } catch (e) {
    $('benchStatusBox').textContent = `Benchmark stop error: ${e.message}`;
  }
}

async function loadChatSuggestions() {
  try {
    const q = S.currentDoc ? `?doc_name=${encodeURIComponent(S.currentDoc)}` : '';
    const d = await apiFetch(`/api/chat/suggestions${q}`);
    S.suggestions = d.suggestions || [];
    renderSuggestionChips();
  } catch (_) {}
}

function renderSuggestionChips() {
  const box = $('chatSuggestions');
  if (!box) return;
  const items = S.suggestions || [];
  box.innerHTML = items.map((s) =>
    `<button class="suggest-chip" data-sid="${esc(s.id)}" title="Click to apply suggestion">${esc(s.label)}</button>`
  ).join('');
  box.querySelectorAll('.suggest-chip').forEach((b) => {
    b.onclick = () => {
      const s = items.find((x) => x.id === b.dataset.sid);
      if (!s) return;
      $('chatInput').value = s.question || '';
      if (s.prompt_template) {
        $('promptText').value = s.prompt_template;
        $('promptPreset').value = 'custom';
      }
    };
  });
}

/* ══════════════════════════════════════════
   Chat
══════════════════════════════════════════ */
function addChatMsg(role, text, extraHtml = '', metaHtml = '') {
  const box = $('chatBox');
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap msg-${role}`;
  const roleLabel = { user:'YOU', assistant:'ASSISTANT', system:'SYSTEM' }[role] || role.toUpperCase();
  wrap.innerHTML = `<div class="msg-role">${roleLabel}</div><div class="msg-body">${esc(text).replace(/\n/g,'<br>')}${extraHtml}${metaHtml}</div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

function renderPerfBadges(m = {}) {
  const chips = [];
  if (m.response_mode) chips.push(`<span class="msg-badge">mode=${esc(m.response_mode)}</span>`);
  if (m.total_ms != null) chips.push(`<span class="msg-badge">total=${(Number(m.total_ms) / 1000).toFixed(2)}s</span>`);
  if (m.retrieval_ms != null) chips.push(`<span class="msg-badge">retrieval=${(Number(m.retrieval_ms) / 1000).toFixed(2)}s</span>`);
  if (m.generation_ms != null) chips.push(`<span class="msg-badge">generation=${(Number(m.generation_ms) / 1000).toFixed(2)}s</span>`);
  if (m.answer_tokens != null) chips.push(`<span class="msg-badge">tokens=${esc(m.answer_tokens)}</span>`);
  if (m.tokens_per_sec != null) chips.push(`<span class="msg-badge">tok/s=${esc(m.tokens_per_sec)}</span>`);
  if (!chips.length) return '';
  return `<div class="msg-meta">${chips.join('')}</div>`;
}

function addPendingAssistant() {
  const box = $('chatBox');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap msg-assistant msg-pending';
  wrap.innerHTML = `<div class="msg-role">ASSISTANT</div><div class="msg-body"><span class="typing-cursor">●</span> Thinking... <span class="msg-badge pending-elapsed">0.0s</span></div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

async function askChat() {
  const q = $('chatInput').value.trim();
  if (!q) return;
  const scope = $('chatScope')?.value || 'selected';
  const responseMode = $('responseMode')?.value || 'balanced';
  const topK = responseMode === 'fast' ? 5 : (responseMode === 'deep' ? 10 : 8);

  addChatMsg('user', q);
  $('chatInput').value = '';

  const sendBtn = $('sendBtn');
  sendBtn.disabled = true;
  sendBtn.textContent = '…';

  const pending = addPendingAssistant();
  const pendingElapsedEl = pending.querySelector('.pending-elapsed');
  const t0 = performance.now();
  const timer = setInterval(() => {
    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    if (pendingElapsedEl) pendingElapsedEl.textContent = `${sec}s`;
  }, 120);

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_name: scope === 'all' ? null : S.currentDoc,
        scope,
        session_id: S.sessionId,
        question: q,
        prompt_template: $('promptText').value,
        filter_extraction_type: $('fType').value || null,
        top_k: topK,
        response_mode: responseMode,
      }),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`Chat stream failed: HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let streamed = '';
    let donePayload = null;

    const body = pending.querySelector('.msg-body');
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      while (true) {
        const sep = buf.indexOf('\n\n');
        if (sep === -1) break;
        const rawEvent = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const lines = rawEvent.split('\n');
        const dataLines = lines.filter((ln) => ln.startsWith('data:')).map((ln) => ln.slice(5).trim());
        if (!dataLines.length) continue;
        const payload = dataLines.join('\n');
        let ev = null;
        try { ev = JSON.parse(payload); } catch (_) { ev = null; }
        if (!ev) continue;
        if (ev.type === 'token') {
          streamed += ev.text || '';
          if (body) body.innerHTML = esc(streamed).replace(/\n/g, '<br>');
        } else if (ev.type === 'done') {
          donePayload = ev;
        } else if (ev.type === 'error') {
          throw new Error(ev.message || 'stream error');
        }
      }
    }

    const d = donePayload || {};
    if (!d.answer) d.answer = streamed || '(no answer)';
    if (d.session_id) S.sessionId = d.session_id;
    S.lastHits = d.raw_hits || [];
    const tableHtml = buildInlineTables(d.tables || [], d.raw_hits || []);
    const perfHtml = renderPerfBadges(d.metrics || {});
    pending.className = 'msg-wrap msg-assistant';
    if (body) body.innerHTML = `${esc(d.answer).replace(/\n/g,'<br>')}${tableHtml}${perfHtml}`;
    const m = d.metrics || {};
    if ($('chatPerfHint') && m.total_ms != null) {
      $('chatPerfHint').textContent = `Last: ${(Number(m.total_ms)/1000).toFixed(2)}s · tok/s ${m.tokens_per_sec ?? 0}`;
    }
    renderReasoning(d.reasoning || []);
  } catch (e) {
    pending.className = 'msg-wrap msg-system';
    const role = pending.querySelector('.msg-role');
    if (role) role.textContent = 'SYSTEM';
    const body = pending.querySelector('.msg-body');
    if (body) body.innerHTML = esc(`Error: ${e.message}`).replace(/\n/g, '<br>');
  } finally {
    clearInterval(timer);
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send ↵';
  }
}

function clearChat() {
  $('chatBox').innerHTML = '';
  $('reasoningList').innerHTML = '';
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

/* Build inline table HTML to embed in a chat bubble.
   Prefers raw_html from the hit (preserves original OCR formatting including
   <sup>, merged cells, etc.), falls back to cols/rows JSON. */
function buildInlineTables(tables, rawHits) {
  // Collect raw_html blocks from hits that have them
  const htmlBlocks = [];
  (rawHits || []).forEach((h, hi) => {
    const m = h.meta || {};
    if (m.raw_html) {
      const safe = m.raw_html.replace(/<script[\s\S]*?<\/script>/gi, '');
      htmlBlocks.push({ html: safe, file: m.file_name || '', page: m.page_number, id: m.table_id || '' });
    }
  });

  // Also use parsed tables (for hits without raw_html)
  const seenIds = new Set(htmlBlocks.map(b => b.id));
  (tables || []).forEach((t) => {
    if (seenIds.has(t.table_id)) return; // already have raw HTML for this
    const cols = t.columns || [];
    const rows = (t.rows || []).slice(0, 50);
    if (!cols.length && !rows.length) return;
    const thead = `<tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>`;
    const tbody = rows.map(r => `<tr>${cols.map(c => `<td>${esc(r[c] ?? '')}</td>`).join('')}</tr>`).join('');
    const html = `<table>${thead}${tbody}</table>`;
    htmlBlocks.push({ html, file: t.file_name || '', page: t.page_number, id: t.table_id || '' });
  });

  if (!htmlBlocks.length) return '';

  return '<div class="chat-tables">' + htmlBlocks.map((b, i) =>
    `<div class="chat-table-block">
      <div class="chat-table-label">
        📊 Table ${i + 1}
        <span class="ct-source">${esc(b.file)} · p${b.page}${b.id ? ' · ' + esc(b.id) : ''}</span>
      </div>
      <div class="chat-table-scroll">${b.html}</div>
    </div>`
  ).join('') + '</div>';
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
    f.append('output_dir', $('outputDir').value.trim());
    if (pdfInput.files.length) f.append('file', pdfInput.files[0]);
    else f.append('pdf_path', pathVal);
    const d = await apiFetch('/api/extract/check', { method: 'POST', body: f });
    $('extractCheckOut').textContent = [
      `File: ${d.pdf_stem || ''}`,
      `Total PDF pages: ${d.total_pdf_pages}`,
      `Already extracted: ${d.already_extracted ? 'Yes ✓ (all pages found)' : 'No — new file or partial'}`,
      `Extracted pages: ${d.extracted_pages && d.extracted_pages.length ? d.extracted_pages.join(', ') : 'none'}`,
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
    loadChatSuggestions();
  };

  // Index
  $('buildIndexBtn').onclick = buildIndex;

  // Models — these are slow (CUDA load); apiFetch blocks until done
  $('loadVlmBtn').onclick = () => loadModel('vlm');
  $('loadLlmBtn').onclick = () => loadModel('llm');
  $('unloadLlmBtn').onclick = unloadLlm;

  // Scope changes suggestions context
  $('chatScope').onchange = () => loadChatSuggestions();

  // Chat
  $('sendBtn').onclick = askChat;
  $('clearChatBtn').onclick = clearChat;
  $('exportExcelBtn').onclick = (e) => doExport('excel', e.currentTarget);
  $('exportWordBtn').onclick  = (e) => doExport('word',  e.currentTarget);
  $('newSessionBtn').onclick = createNewSession;
  $('sessionSelect').onchange = async (e) => {
    S.sessionId = e.target.value;
    await loadSessionMessages(S.sessionId);
  };
  $('chatInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askChat(); }
  });

  // Extraction
  $('checkPdfBtn').onclick    = checkPdf;
  $('startExtractBtn').onclick = startExtract;
  $('stopExtractBtn').onclick  = stopExtract;

  // Logs
  $('clearLogsBtn').onclick = clearLogs;

  // Experiments (advanced)
  $('runExperimentBtn').onclick = runExperiment;
  $('stopExperimentBtn').onclick = stopExperiment;
  $('runBenchmarkBtn').onclick = runModelBenchmark;
  $('stopBenchmarkBtn').onclick = stopModelBenchmark;
  $('browseBenchmarkBtn').onclick = async () => {
    try {
      const d = await apiFetch('/api/browse/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accept: '.csv,.json' }),
      });
      if (d.path) $('expBenchmarkPath').value = d.path;
    } catch (_) {}
  };
  $('browseExpOutputBtn').onclick = async () => {
    try {
      const d = await apiFetch('/api/browse/folder', { method: 'POST' });
      if (d.path) $('expOutputRoot').value = d.path;
    } catch (_) {}
  };
  $('browseBenchOutputBtn').onclick = async () => {
    try {
      const d = await apiFetch('/api/browse/folder', { method: 'POST' });
      if (d.path) $('benchOutputRoot').value = d.path;
    } catch (_) {}
  };

  // Boot
  await initAdvancedMode();
  await initModelOptions();
  await refreshDocs();
  await loadChatSuggestions();
  await refreshSessions();
  if (!S.sessionId) {
    await createNewSession();
  }
  await pollState();
  await refreshLogs();

  // Logs are always visible in the right panel — poll every 2 s
  setInterval(() => { pollState(); refreshLogs(); }, 2000);
}

init();
