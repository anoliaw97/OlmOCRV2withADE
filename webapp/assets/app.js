'use strict';

const DEFAULT_SYSTEM_PROMPT =
  'You are a technical extraction assistant. Answer only from extracted JSON/MD/TXT context. If missing, say not found in extracted outputs.';

const S = {
  packages: [],
  currentPackageId: '',
  currentPreview: null,
  sessions: [],
  currentSessionId: '',
  chatRecords: [],
  currentLogKind: 'status',
  modelOptions: [],
  activeModelName: '',
  activeBackend: 'ollama',
  lastMetrics: null,
};

function $(id) {
  return document.getElementById(id);
}

function esc(text) {
  return String(text ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function nowTime() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

function setStatus(text) {
  $('buildInfo').textContent = String(text || 'build: web-simple');
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  const raw = await res.text();
  let data = {};
  if (raw.trim()) {
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Invalid JSON response from ${path}`);
    }
  }
  if (!res.ok) {
    throw new Error(data?.detail || `${res.status} ${res.statusText}`);
  }
  return data;
}

function addChatMsg(role, text, extraHtml = '') {
  const box = $('chatBox');
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  const roleName = role === 'assistant'
    ? (S.activeModelName || 'Assistant')
    : (role === 'user' ? 'YOU' : 'SYSTEM');

  msg.innerHTML =
    `<div class=\"msg-role\">${esc(roleName)} [${esc(nowTime())}]</div>` +
    `<div class=\"msg-body\">${esc(text).replace(/\n/g, '<br>')}${extraHtml}</div>`;
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

function resolveResponseIndicator(routeType) {
  const t = (routeType || '').toLowerCase();
  if (t === 'document') {
    return 'Using document context';
  }
  if (t === 'hybrid') {
    return 'Using hybrid context';
  }
  return 'General answer';
}

function clearChatView() {
  $('chatBox').innerHTML = '';
  S.chatRecords = [];
  $('chatPerfHint').textContent = 'Last: -';
  $('chatTablesView').innerHTML = '<div class="preview-placeholder">Tables extracted from assistant answers will appear here.</div>';
}

function updateModelPill(state) {
  const dot = $('modelDot');
  const label = $('modelLabel');
  if (state === 'busy') {
    dot.className = 'dot dot-busy';
    label.textContent = 'Model: loading...';
    return;
  }
  if (state === 'on' && S.activeModelName) {
    dot.className = 'dot dot-on';
    label.textContent = `Model: ${S.activeModelName}`;
    return;
  }
  dot.className = 'dot dot-off';
  label.textContent = 'Model: not loaded';
}

function renderPackageOptions() {
  const sel = $('packageSelect');
  sel.innerHTML = '';
  for (const pkg of S.packages) {
    const option = document.createElement('option');
    option.value = pkg.package_id;
    option.textContent = `${pkg.base_name} [${(pkg.tokens || []).join(', ') || 'EMPTY'}]`;
    sel.appendChild(option);
  }
  if (S.currentPackageId) {
    sel.value = S.currentPackageId;
  }
}

function packageById(packageId) {
  return S.packages.find((p) => p.package_id === packageId) || null;
}

function renderPackageCoverage() {
  const pkg = packageById(S.currentPackageId);
  if (!pkg) {
    $('coverageInfo').textContent = 'No package loaded.';
    return;
  }

  $('coverageInfo').textContent = [
    `Folder: ${pkg.folder}`,
    `JSON files: ${(pkg.json_paths || []).length}`,
    `Markdown files: ${(pkg.markdown_paths || []).length}`,
    `TXT files: ${(pkg.text_paths || []).length}`,
    `Full PDF: ${pkg.full_pdf_path || 'N/A'}`,
    `Grouped page PDFs: ${pkg.page_pdf_count || 0}${pkg.page_range ? ` (pages ${pkg.page_range})` : ''}`,
  ].join('\n');
}

function renderViewerTables(tables) {
  const box = $('viewerTablesView');
  box.innerHTML = '';

  if (!tables || !tables.length) {
    box.innerHTML = '<div class="preview-placeholder">No rendered tables found for selected package.</div>';
    return;
  }

  tables.forEach((table, idx) => {
    const section = document.createElement('div');
    section.className = 'viewer-table-block';

    const title = document.createElement('div');
    title.className = 'table-title';
    title.textContent = `Table ${idx + 1}: ${table.title || 'Untitled'}`;
    section.appendChild(title);

    if ((table.headers || []).length && (table.rows || []).length) {
      const html = [];
      html.push('<table><thead><tr>');
      for (const h of table.headers) {
        html.push(`<th>${esc(h)}</th>`);
      }
      html.push('</tr></thead><tbody>');
      for (const row of table.rows) {
        html.push('<tr>');
        for (const cell of row) {
          html.push(`<td>${esc(cell)}</td>`);
        }
        html.push('</tr>');
      }
      html.push('</tbody></table>');

      const wrapper = document.createElement('div');
      wrapper.className = 'html-tables-wrap';
      wrapper.innerHTML = html.join('');
      section.appendChild(wrapper);
    } else {
      const pre = document.createElement('pre');
      pre.className = 'json-view';
      pre.textContent = table.raw_text || '(table had no structured rows)';
      section.appendChild(pre);
    }

    box.appendChild(section);
  });
}

function renderRawPreview(preview) {
  const blocks = [];
  if (preview.markdown_text) {
    blocks.push('--- MARKDOWN ---\n' + preview.markdown_text);
  }
  if (preview.json_text) {
    blocks.push('--- JSON ---\n' + preview.json_text);
  }
  if (preview.text_text) {
    blocks.push('--- TXT ---\n' + preview.text_text);
  }
  $('rawView').textContent = blocks.length ? blocks.join('\n\n') : '(no extracted content)';
}

function resetPdfPreviewPlaceholder(message) {
  $('pdfPreviewWrap').innerHTML = `<div class="preview-placeholder">${esc(message)}</div>`;
}

async function renderPdfPreview() {
  if (!S.currentPackageId) {
    resetPdfPreviewPlaceholder('Select a package first.');
    return;
  }

  const page = Math.max(1, Number($('pdfPageInput').value || 1));
  const dpi = Math.max(72, Math.min(320, Number($('pdfDpiInput').value || 140)));

  const wrap = $('pdfPreviewWrap');
  wrap.innerHTML = '<div class="preview-placeholder">Rendering with Poppler...</div>';

  const url = `/api/loaders/preview/pdf-image?package_id=${encodeURIComponent(S.currentPackageId)}&page=${page}&dpi=${dpi}&_t=${Date.now()}`;
  const img = new Image();
  img.onload = () => {
    wrap.innerHTML = '';
    wrap.appendChild(img);
  };
  img.onerror = async () => {
    try {
      const test = await fetch(url);
      const maybeJson = await test.text();
      let detail = 'Failed to render PDF preview.';
      try {
        const parsed = JSON.parse(maybeJson);
        detail = parsed.detail || detail;
      } catch {
        if (maybeJson && maybeJson.length < 300) {
          detail = maybeJson;
        }
      }
      resetPdfPreviewPlaceholder(detail);
      addChatMsg('system', `PDF preview failed: ${detail}`);
    } catch (err) {
      resetPdfPreviewPlaceholder(`PDF preview failed: ${err.message}`);
      addChatMsg('system', `PDF preview failed: ${err.message}`);
    }
  };
  img.src = url;
}

async function selectPackage(packageId) {
  S.currentPackageId = packageId;
  renderPackageOptions();
  renderPackageCoverage();
  resetPdfPreviewPlaceholder('Select page and click Render PDF.');

  try {
    const preview = await apiFetch('/api/loaders/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ package_id: packageId }),
    });
    S.currentPreview = preview;
    renderRawPreview(preview);
    renderViewerTables(preview.tables || []);
  } catch (error) {
    addChatMsg('system', `Preview error: ${error.message}`);
  }
}

async function loadFolder() {
  const root = $('dataRootInput').value.trim();
  if (!root) {
    addChatMsg('system', 'Enter a folder path first.');
    return;
  }

  try {
    const data = await apiFetch('/api/loaders/folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: root }),
    });
    S.packages = data.packages || [];
    S.currentPackageId = S.packages.length ? S.packages[0].package_id : '';
    renderPackageOptions();
    renderPackageCoverage();
    if (S.currentPackageId) {
      await selectPackage(S.currentPackageId);
    }
    addChatMsg('system', `Loaded folder. Package count: ${S.packages.length}.`);
  } catch (error) {
    addChatMsg('system', `Load folder failed: ${error.message}`);
  }
}

async function loadDatabase() {
  try {
    const data = await apiFetch('/api/loaders/packages');
    const packages = data.packages || [];
    if (!packages.length) {
      await loadFolder();
      return;
    }

    S.packages = packages;
    if (!S.currentPackageId && packages.length) {
      S.currentPackageId = packages[0].package_id;
    }
    renderPackageOptions();
    renderPackageCoverage();
    if (S.currentPackageId) {
      await selectPackage(S.currentPackageId);
    }
    addChatMsg('system', `Loaded database state. Package count: ${packages.length}.`);
  } catch (error) {
    addChatMsg('system', `Load database failed: ${error.message}`);
  }
}

async function buildIndex() {
  try {
    const data = await apiFetch('/api/retrieval/index/build', { method: 'POST' });
    addChatMsg('system', `RAG index ready: ${data.indexed_chunks} chunk(s) across ${data.package_count} package(s).`);
  } catch (error) {
    addChatMsg('system', `RAG build failed: ${error.message}`);
  }
}

function applyModelSelectionFromDropdown() {
  $('modelInput').value = $('modelSelect').value || '';
}

async function refreshModels() {
  const backend = $('backendSelect').value;
  const scanPath = encodeURIComponent($('llamaScanPathInput').value.trim());
  const ollamaUrl = encodeURIComponent($('ollamaUrlInput').value.trim());

  $('loadingSpinner').classList.remove('hidden');
  try {
    const data = await apiFetch(
      `/api/system/models/options?backend=${encodeURIComponent(backend)}&scan_path=${scanPath}&ollama_url=${ollamaUrl}`,
    );

    S.modelOptions = data.models || [];
    const sel = $('modelSelect');
    sel.innerHTML = '';

    if (!S.modelOptions.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = data.message || 'No local models found.';
      sel.appendChild(option);
      $('modelInput').value = '';
      addChatMsg('system', data.message || 'No local models found.');
      return;
    }

    for (const model of S.modelOptions) {
      const option = document.createElement('option');
      option.value = model.path || model.name;
      option.textContent = model.label || model.name;
      sel.appendChild(option);
    }

    const chosen = data.default_model || sel.options[0].value;
    sel.value = chosen;
    $('modelInput').value = chosen;
    if (data.scan_path) {
      $('llamaScanPathInput').value = data.scan_path;
    }
    addChatMsg('system', data.message || `Models refreshed for ${backend}.`);
  } catch (error) {
    addChatMsg('system', `Model refresh failed: ${error.message}`);
  } finally {
    $('loadingSpinner').classList.add('hidden');
  }
}

function loadModel() {
  const backend = $('backendSelect').value;
  const selected = $('modelInput').value.trim();
  if (!selected) {
    addChatMsg('system', 'Choose a model first.');
    return;
  }

  S.activeBackend = backend;
  S.activeModelName = selected.includes('\\') || selected.includes('/')
    ? selected.split(/[/\\]/).pop()
    : selected;
  updateModelPill('on');
  addChatMsg('system', `Model loaded: ${S.activeModelName} via ${backend}.`);
}

function unloadModel() {
  S.activeModelName = '';
  updateModelPill('off');
  addChatMsg('system', 'Model unloaded (selection cleared).');
}

async function browseFolder(targetInputId, buttonEl) {
  const current = $(targetInputId).value.trim();
  if (buttonEl) {
    buttonEl.textContent = '⏳';
    buttonEl.disabled = true;
  }
  try {
    const query = current ? `?path=${encodeURIComponent(current)}` : '';
    const data = await apiFetch(`/api/system/browse/dialog${query}`, { method: 'POST' });
    const path = data.path || '';
    if (path) {
      $(targetInputId).value = path;
      addChatMsg('system', `Selected folder: ${path}`);
    } else {
      addChatMsg('system', 'Browse dialog canceled.');
    }
  } catch (error) {
    addChatMsg('system', `Browse dialog failed: ${error.message}`);
  } finally {
    if (buttonEl) {
      buttonEl.textContent = '📁';
      buttonEl.disabled = false;
    }
  }
}

async function refreshSessions(loadCurrent = false) {
  try {
    const data = await apiFetch('/api/chat/sessions');
    S.sessions = data.sessions || [];

    const sel = $('sessionSelect');
    sel.innerHTML = '';
    for (const session of S.sessions) {
      const option = document.createElement('option');
      option.value = session.session_id;
      option.textContent = `${session.title} (${session.message_count})`;
      sel.appendChild(option);
    }

    if (!S.sessions.length) {
      S.currentSessionId = '';
      return;
    }

    if (!S.currentSessionId || !S.sessions.some((s) => s.session_id === S.currentSessionId)) {
      S.currentSessionId = S.sessions[0].session_id;
    }

    sel.value = S.currentSessionId;
    if (loadCurrent) {
      await loadSession(S.currentSessionId);
    }
  } catch (error) {
    addChatMsg('system', `Session refresh failed: ${error.message}`);
  }
}

async function createSession() {
  try {
    const data = await apiFetch('/api/chat/session/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: '' }),
    });
    S.currentSessionId = data.session.session_id;
    clearChatView();
    await refreshSessions(false);
    addChatMsg('system', 'Started new chat session.');
  } catch (error) {
    addChatMsg('system', `Create session failed: ${error.message}`);
  }
}

async function deleteSession() {
  if (!S.currentSessionId) {
    return;
  }
  try {
    await apiFetch(`/api/chat/session/${encodeURIComponent(S.currentSessionId)}`, { method: 'DELETE' });
    S.currentSessionId = '';
    clearChatView();
    await refreshSessions(true);
    addChatMsg('system', 'Session deleted.');
  } catch (error) {
    addChatMsg('system', `Delete session failed: ${error.message}`);
  }
}

async function loadSession(sessionId) {
  if (!sessionId) {
    return;
  }
  try {
    const data = await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`);
    S.currentSessionId = data.session.session_id;
    clearChatView();

    let pendingQuestion = '';
    for (const msg of data.session.messages || []) {
      if ((msg.role || '').toLowerCase() === 'user') {
        pendingQuestion = msg.content || '';
        addChatMsg('user', msg.content || '');
        continue;
      }
      addChatMsg('assistant', msg.content || '');
      S.chatRecords.push({
        timestamp: new Date().toISOString().slice(0, 19),
        mode: $('chatModeSelect').value,
        runtime: msg.runtime || '',
        model: msg.model || '',
        question: pendingQuestion,
        answer: msg.content || '',
        citations: msg.citations || '',
      });
      pendingQuestion = '';
    }
  } catch (error) {
    addChatMsg('system', `Open session failed: ${error.message}`);
  }
}

function renderReasoning(items) {
  const box = $('reasoningList');
  box.innerHTML = '';
  if (!items || !items.length) {
    box.innerHTML = '<div class="reasoning-item">No reasoning details yet.</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'reasoning-item';
    row.textContent = String(item);
    box.appendChild(row);
  }
}

function buildMetricsBadges(metrics) {
  if (!metrics) {
    return '';
  }
  const badges = [
    `total=${Number(metrics.total_ms || 0).toFixed(2)}ms`,
    `retrieval=${Number(metrics.retrieval_ms || 0).toFixed(2)}ms`,
    `generation=${Number(metrics.generation_ms || 0).toFixed(2)}ms`,
    `chunks=${metrics.retrieval_chunks || 0}`,
  ];
  return `<div class="msg-meta">${badges.map((t) => `<span class=\"msg-badge\">${esc(t)}</span>`).join('')}</div>`;
}

function buildRouteBadge(routeType, routeConfidence, routeReason) {
  const type = routeType || 'general';
  const conf = Number(routeConfidence || 0).toFixed(2);
  const reason = routeReason || '-';
  const text = `route=${type} conf=${conf} reason=${reason}`;
  return `<div class="msg-meta"><span class=\"msg-badge\">${esc(text)}</span></div>`;
}

function buildExportBadge(data) {
  if (!data) {
    return '';
  }
  if (!data.action_type || !String(data.action_type).startsWith('export')) {
    return '';
  }
  const format = (data.export_format || '').toUpperCase();
  const path = data.export_file_path || '';
  const chips = [`action=${data.action_type}`, `format=${format || '-'}`];
  if (path) {
    chips.push(`file=${path}`);
  }
  return `<div class="msg-meta">${chips.map((t) => `<span class=\"msg-badge\">${esc(t)}</span>`).join('')}</div>`;
}

function splitPipeRow(line) {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  return trimmed.split('|').map((v) => v.trim());
}

function extractMarkdownTables(text) {
  const lines = String(text || '').split(/\r?\n/);
  const tables = [];
  let i = 0;

  while (i < lines.length - 1) {
    const head = lines[i];
    const sep = lines[i + 1];
    const isHead = head.includes('|');
    const isSep = /^\s*\|?\s*:?-{2,}:?(\s*\|\s*:?-{2,}:?)*\s*\|?\s*$/.test(sep || '');

    if (!isHead || !isSep) {
      i += 1;
      continue;
    }

    const block = [head, sep];
    i += 2;
    while (i < lines.length && lines[i].includes('|')) {
      block.push(lines[i]);
      i += 1;
    }

    const rows = block.slice(2).map(splitPipeRow).filter((r) => r.length > 0);
    const headers = splitPipeRow(block[0]);
    tables.push({ headers, rows });
  }

  return tables;
}

function extractHtmlTables(text) {
  const blocks = String(text || '').match(/<table[\s\S]*?<\/table>/gi) || [];
  return blocks.map((html) => html.replace(/<script[\s\S]*?<\/script>/gi, ''));
}

function renderChatGeneratedTables(answerText) {
  const box = $('chatTablesView');
  box.innerHTML = '';

  const htmlTables = extractHtmlTables(answerText);
  const mdTables = extractMarkdownTables(answerText);

  if (!htmlTables.length && !mdTables.length) {
    box.innerHTML = '<div class="preview-placeholder">No table detected in latest answer.</div>';
    return;
  }

  let idx = 1;
  for (const html of htmlTables) {
    const title = document.createElement('div');
    title.className = 'table-title';
    title.textContent = `Generated table ${idx}`;
    box.appendChild(title);

    const wrapper = document.createElement('div');
    wrapper.innerHTML = html;
    box.appendChild(wrapper);
    idx += 1;
  }

  for (const table of mdTables) {
    const title = document.createElement('div');
    title.className = 'table-title';
    title.textContent = `Generated table ${idx}`;
    box.appendChild(title);

    const html = [];
    html.push('<table><thead><tr>');
    for (const h of table.headers) {
      html.push(`<th>${esc(h)}</th>`);
    }
    html.push('</tr></thead><tbody>');
    for (const row of table.rows) {
      html.push('<tr>');
      for (const cell of row) {
        html.push(`<td>${esc(cell)}</td>`);
      }
      html.push('</tr>');
    }
    html.push('</tbody></table>');

    const wrapper = document.createElement('div');
    wrapper.innerHTML = html.join('');
    box.appendChild(wrapper);
    idx += 1;
  }
}

async function askChat() {
  const question = $('chatInput').value.trim();
  if (!question) {
    return;
  }

  if (!S.activeModelName) {
    addChatMsg('system', 'Load a local model first (Step 3) before chatting.');
    return;
  }

  if ($('backendSelect').value === 'llamacpp') {
    const cli = ($('llamaCliInput').value || '').trim().toLowerCase();
    if (!cli || cli === 'llama-cli') {
      addChatMsg(
        'system',
        'Set full llama.cpp executable path first in "llama-cli path" (example: C:\\llama.cpp\\build\\bin\\Release\\llama-cli.exe).',
      );
      return;
    }
  }

  const mode = $('chatModeSelect').value;
  if (mode === 'direct' && !S.currentPackageId) {
    addChatMsg('system', 'Direct mode needs a selected package.');
    return;
  }

  if (!S.currentSessionId) {
    await createSession();
  }

  addChatMsg('user', question);
  $('chatInput').value = '';
  $('sendBtn').disabled = true;
  $('sendBtn').textContent = '...';

  try {
    const payload = {
      question,
      mode,
      package_id: S.currentPackageId || null,
      session_id: S.currentSessionId,
      llm_settings: {
        backend: $('backendSelect').value,
        model: $('modelInput').value.trim(),
        system_prompt: DEFAULT_SYSTEM_PROMPT,
        max_tokens: Number($('maxTokensInput').value || 768),
        temperature: 0.2,
        ollama_url: $('ollamaUrlInput').value.trim(),
        llama_cli_path: $('llamaCliInput').value.trim(),
        context_limit: Number($('contextLimitInput').value || 24000),
      },
    };

    const data = await apiFetch('/api/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    S.currentSessionId = data.session_id || S.currentSessionId;
    S.lastMetrics = data.metrics || null;

    const metricsHtml = buildMetricsBadges(data.metrics);
    const routeHtml = buildRouteBadge(data.route_type, data.route_confidence, data.route_reason);
    const exportHtml = buildExportBadge(data);
    const indicator = resolveResponseIndicator(data.route_type);
    addChatMsg('system', indicator);
    addChatMsg('assistant', data.answer || '', metricsHtml + routeHtml + exportHtml);

    if ((data.citations || []).length) {
      const cites = data.citations.map((c) => `- ${c.source_file} (${c.source_type}, score=${Number(c.score).toFixed(2)})`);
      addChatMsg('system', `Sources:\n${cites.join('\n')}`);
    }

    renderReasoning(data.reasoning_chain || []);
    renderChatGeneratedTables(data.answer || '');

    const totalMs = Number(data.metrics?.total_ms || 0).toFixed(2);
    $('chatPerfHint').textContent = `Last: ${totalMs} ms`;

    const citationText = (data.citations || [])
      .map((c) => `${c.source_file}:${c.source_type}:${Number(c.score).toFixed(2)}`)
      .join('; ');

    S.chatRecords.push({
      timestamp: new Date().toISOString().slice(0, 19),
      mode: data.mode || mode,
      runtime: data.runtime || '',
      model: data.model || '',
      question,
      answer: data.answer || '',
      citations: citationText,
    });

    await refreshSessions(false);
  } catch (error) {
    addChatMsg('system', `Chat error: ${error.message}`);
  } finally {
    $('sendBtn').disabled = false;
    $('sendBtn').textContent = 'Send ↵';
  }
}

function buildExportPath(ext) {
  const base = ($('outputDirInput').value || '').trim() || ($('dataRootInput').value || '').trim() || '.';
  const stamp = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  return `${base}\\chat_export_${stamp}.${ext}`;
}

async function exportChat(kind) {
  if (!S.chatRecords.length) {
    addChatMsg('system', 'No chat records to export.');
    return;
  }

  const destination = buildExportPath(kind === 'excel' ? 'xlsx' : 'docx');
  try {
    const data = await apiFetch('/api/export/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ destination, records: S.chatRecords }),
    });
    addChatMsg('system', data.message || `Exported to ${destination}`);
  } catch (error) {
    addChatMsg('system', `Export failed: ${error.message}`);
  }
}

async function mlBuildDataset() {
  try {
    const outputCsv = $('mlDatasetPathInput').value.trim() || 'data/ml/structured_dataset.csv';
    const data = await apiFetch('/api/ml/dataset/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ output_csv: outputCsv }),
    });
    addChatMsg('system', data.message);
    await mlLoadDashboard();
  } catch (error) {
    addChatMsg('system', `ML dataset build failed: ${error.message}`);
  }
}

async function mlTrain() {
  try {
    const datasetCsv = $('mlDatasetPathInput').value.trim() || 'data/ml/structured_dataset.csv';
    const target = $('mlTargetSelect').value;
    const data = await apiFetch('/api/ml/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, dataset_csv: datasetCsv }),
    });
    addChatMsg('system', `${data.message} target=${data.target}, r2=${Number(data.metrics?.r2 || 0).toFixed(4)}`);
    await mlLoadDashboard();
  } catch (error) {
    addChatMsg('system', `ML train failed: ${error.message}`);
  }
}

async function mlPredict() {
  try {
    const target = $('mlTargetSelect').value;
    const features = JSON.parse($('mlPredictJsonInput').value || '{}');
    const data = await apiFetch('/api/ml/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, features }),
    });
    addChatMsg('system', `Prediction for ${target}: ${Number(data.prediction).toFixed(6)}`);
  } catch (error) {
    addChatMsg('system', `ML predict failed: ${error.message}`);
  }
}

async function mlRunPipeline() {
  try {
    const target = $('mlTargetSelect').value;
    const pipelinePath = $('mlPipelinePathInput').value.trim();
    const data = await apiFetch('/api/ml/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline_path: pipelinePath, default_target: target }),
    });
    addChatMsg('system', `${data.message}\n${(data.steps || []).join('\n')}`);
    await mlLoadDashboard();
  } catch (error) {
    addChatMsg('system', `ML pipeline failed: ${error.message}`);
  }
}

function renderMlScatter(points) {
  const canvas = $('mlScatterCanvas');
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0b1220';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!points || !points.length) {
    ctx.fillStyle = '#6b84a8';
    ctx.font = '14px Segoe UI';
    ctx.fillText('No chart data available.', 20, 30);
    return;
  }

  const pad = 30;
  const xs = points.map((p) => Number(p.x));
  const ys = points.map((p) => Number(p.y));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;

  ctx.strokeStyle = '#2b3c5d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, canvas.height - pad);
  ctx.lineTo(canvas.width - pad, canvas.height - pad);
  ctx.lineTo(canvas.width - pad, pad);
  ctx.stroke();

  for (const p of points) {
    const x = pad + ((Number(p.x) - xMin) / xSpan) * (canvas.width - pad * 2);
    const y = canvas.height - pad - ((Number(p.y) - yMin) / ySpan) * (canvas.height - pad * 2);
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = '#9eb3d8';
  ctx.font = '12px Segoe UI';
  ctx.fillText(`x range: ${xMin.toFixed(2)} .. ${xMax.toFixed(2)}`, 10, 16);
  ctx.fillText(`y range: ${yMin.toFixed(2)} .. ${yMax.toFixed(2)}`, 10, 32);
}

async function mlLoadDashboard() {
  try {
    const datasetCsv = $('mlDatasetPathInput').value.trim() || 'data/ml/structured_dataset.csv';
    const target = $('mlTargetSelect').value;
    const data = await apiFetch(
      `/api/ml/dashboard?dataset_csv=${encodeURIComponent(datasetCsv)}&target=${encodeURIComponent(target)}`,
    );

    const lines = [];
    lines.push(`Dataset: ${data.dataset_csv}`);
    lines.push(`Rows: ${data.row_count}`);
    lines.push(`Columns: ${(data.columns || []).join(', ')}`);
    lines.push('');
    lines.push('Stats:');
    for (const [name, stats] of Object.entries(data.stats || {})) {
      lines.push(
        `${name}: mean=${Number(stats.mean || 0).toFixed(4)}, min=${Number(stats.min || 0).toFixed(4)}, max=${Number(stats.max || 0).toFixed(4)}, std=${Number(stats.std || 0).toFixed(4)}`,
      );
    }
    lines.push('');
    lines.push('Feature importance:');
    for (const [name, score] of Object.entries(data.feature_importance || {})) {
      lines.push(`${name}: ${Number(score).toFixed(6)}`);
    }

    $('mlSummaryView').textContent = lines.join('\n');
    renderMlScatter(data.chart_points || []);
  } catch (error) {
    $('mlSummaryView').textContent = `ML dashboard error: ${error.message}`;
    renderMlScatter([]);
  }
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach((x) => x.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((x) => x.classList.add('hidden'));
      btn.classList.add('active');
      $(`tab-${btn.dataset.tab}`).classList.remove('hidden');
    };
  });

  document.querySelectorAll('.vtab-btn').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('.vtab-btn').forEach((x) => x.classList.remove('active'));
      document.querySelectorAll('.vtab-pane').forEach((x) => x.classList.add('hidden'));
      btn.classList.add('active');
      $(`vtab-${btn.dataset.vtab}`).classList.remove('hidden');
    };
  });
}

function initLogTabs() {
  document.querySelectorAll('.lkbtn').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('.lkbtn').forEach((x) => x.classList.remove('active'));
      btn.classList.add('active');
      S.currentLogKind = btn.dataset.kind;
      refreshLogs();
    };
  });
}

async function refreshLogs() {
  try {
    const data = await apiFetch(`/api/system/logs?kind=${encodeURIComponent(S.currentLogKind)}&limit=400`);
    const box = $('logsView');
    box.innerHTML = (data.items || []).map((item) => {
      return `<div class=\"log-line log-${esc(item.kind)}\"><span class=\"log-time\">[${esc(item.time)}]</span>${esc(item.message)}</div>`;
    }).join('');
    if ($('autoScrollLogs').checked) {
      box.scrollTop = box.scrollHeight;
    }
  } catch (_) {
    // silent polling failure
  }
}

async function clearLogs() {
  try {
    await apiFetch('/api/system/logs/clear?kind=all', { method: 'POST' });
    await refreshLogs();
  } catch (error) {
    addChatMsg('system', `Clear logs failed: ${error.message}`);
  }
}

async function pollState() {
  try {
    const data = await apiFetch('/api/system/state');
    const lines = [
      `Packages loaded: ${data.packages_loaded || 0}`,
      `Current package: ${data.current_package_id || '-'}`,
      `RAG index ready: ${data.rag_index_ready ? 'yes' : 'no'}`,
      `Sessions: ${data.sessions || 0}`,
      `Active model: ${S.activeModelName || '-'}`,
      `Last total ms: ${S.lastMetrics ? Number(S.lastMetrics.total_ms || 0).toFixed(2) : '-'}`,
    ];
    $('stateDetails').textContent = lines.join('\n');
  } catch (_) {
    // silent polling failure
  }
}

function wireEvents() {
  $('browseDataRootBtn').onclick = (e) => browseFolder('dataRootInput', e.currentTarget);
  $('browseOutputBtn').onclick = (e) => browseFolder('outputDirInput', e.currentTarget);

  $('loadFolderBtn').onclick = loadFolder;
  $('loadDatabaseBtn').onclick = loadDatabase;
  $('packageSelect').onchange = (e) => selectPackage(e.target.value);
  $('buildIndexBtn').onclick = buildIndex;

  $('backendSelect').onchange = refreshModels;
  $('modelSelect').onchange = applyModelSelectionFromDropdown;
  $('refreshModelsBtn').onclick = refreshModels;
  $('refreshModelsTopBtn').onclick = refreshModels;

  $('loadModelBtn').onclick = loadModel;
  $('loadModelTopBtn').onclick = loadModel;
  $('unloadModelBtn').onclick = unloadModel;
  $('unloadModelTopBtn').onclick = unloadModel;

  $('newSessionBtn').onclick = createSession;
  $('deleteSessionBtn').onclick = deleteSession;
  $('sessionSelect').onchange = (e) => loadSession(e.target.value);

  $('sendBtn').onclick = askChat;
  $('clearChatBtn').onclick = () => {
    clearChatView();
    addChatMsg('system', 'Chat view cleared. Session history stays saved.');
  };
  $('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askChat();
    }
  });

  $('exportExcelBtn').onclick = () => exportChat('excel');
  $('exportWordBtn').onclick = () => exportChat('word');

  $('renderPdfBtn').onclick = renderPdfPreview;
  $('clearLogsBtn').onclick = clearLogs;

  if ($('mlBuildDatasetBtn')) {
    $('mlBuildDatasetBtn').onclick = mlBuildDataset;
  }
  if ($('mlTrainBtn')) {
    $('mlTrainBtn').onclick = mlTrain;
  }
  if ($('mlPredictBtn')) {
    $('mlPredictBtn').onclick = mlPredict;
  }
  if ($('mlLoadDashboardBtn')) {
    $('mlLoadDashboardBtn').onclick = mlLoadDashboard;
  }
  if ($('mlRunPipelineBtn')) {
    $('mlRunPipelineBtn').onclick = mlRunPipeline;
  }
}

async function init() {
  initTabs();
  initLogTabs();
  wireEvents();
  updateModelPill('off');

  try {
    await apiFetch('/health');
    addChatMsg('system', 'Backend connected. Follow steps: load folder/database -> load RAG -> load model -> chat -> table -> export.');
  } catch (error) {
    addChatMsg('system', `Backend unavailable: ${error.message}`);
    return;
  }

  await refreshSessions(false);
  if (!S.currentSessionId) {
    await createSession();
  } else {
    await loadSession(S.currentSessionId);
  }

  await refreshModels();
  await loadDatabase();
  await pollState();
  await refreshLogs();
  await mlLoadDashboard();

  setInterval(() => {
    pollState();
    refreshLogs();
  }, 2000);
}

init();
