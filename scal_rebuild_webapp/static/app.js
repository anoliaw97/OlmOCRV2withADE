'use strict';

const S = {
  docs: [],
  coverage: {},
  currentDoc: null,
  sessions: [],
  sessionId: null,
  activeModelName: '',
  settings: {
    backend: 'inference_api',
    ui_mode: 'layman',
    data_root: '',
  },
  services: {
    legacy_ui_url: 'http://127.0.0.1:8090',
  },
};

function $(id) { return document.getElementById(id); }
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

async function apiFetch(path, opts = {}) {
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    throw new Error((await r.text()) || `HTTP ${r.status}`);
  }
  const d = await r.json();
  if (!r.ok) throw new Error(d?.detail || `HTTP ${r.status}`);
  return d;
}

function fd(obj) {
  const f = new FormData();
  for (const [k, v] of Object.entries(obj || {})) {
    if (v !== null && v !== undefined && v !== '') f.append(k, v);
  }
  return f;
}

function assistantLabel() {
  const raw = (S.activeModelName || '').trim();
  if (!raw) return 'ASSISTANT';
  const short = raw.includes('/') ? raw.split('/').pop() : raw;
  return short.toUpperCase();
}

function addMsg(role, text, extraHtml = '') {
  const box = $('chatBox');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const label = role === 'assistant'
    ? assistantLabel()
    : ({ user: 'YOU', system: 'SYSTEM' }[role] || role.toUpperCase());
  wrap.innerHTML = `<div class="role">${label}</div><div class="body">${esc(text).replace(/\n/g, '<br>')}${extraHtml}</div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

function addPendingAssistant() {
  const box = $('chatBox');
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.innerHTML = `<div class="role">${assistantLabel()}</div><div class="body">Thinking... <span class="badge pending-elapsed">0.0s</span></div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

function perfBadges(m = {}) {
  const chips = [];
  if (m.backend) chips.push(`<span class="badge">backend=${esc(m.backend)}</span>`);
  if (m.response_mode) chips.push(`<span class="badge">mode=${esc(m.response_mode)}</span>`);
  if (m.first_token_ms != null) chips.push(`<span class="badge">first=${(Number(m.first_token_ms) / 1000).toFixed(2)}s</span>`);
  if (m.total_ms != null) chips.push(`<span class="badge">total=${(Number(m.total_ms) / 1000).toFixed(2)}s</span>`);
  if (m.tokens_per_sec != null) chips.push(`<span class="badge">tok/s=${esc(m.tokens_per_sec)}</span>`);
  if (m.hits != null) chips.push(`<span class="badge">hits=${esc(m.hits)}</span>`);
  if (!chips.length) return '';
  return `<div class="meta">${chips.join('')}</div>`;
}

function setUiMode(mode) {
  const advanced = mode === 'advanced';
  document.querySelectorAll('.advanced-only').forEach((el) => {
    el.classList.toggle('hidden', !advanced);
  });
}

async function saveSettings(partial) {
  const d = await apiFetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  });
  S.settings = d.settings || S.settings;
  $('dataRoot').value = S.settings.data_root || $('dataRoot').value;
  $('uiModeSelect').value = S.settings.ui_mode || 'layman';
  $('backendSelect').value = S.settings.backend || 'inference_api';
  setUiMode(S.settings.ui_mode || 'layman');
  return d;
}

function renderDocs() {
  const box = $('docsList');
  if (!S.docs.length) {
    box.innerHTML = '<div class="small-text">No extracted documents found.</div>';
    return;
  }
  box.innerHTML = S.docs.map((d) => {
    const c = S.coverage[d] || {};
    const cls = d === S.currentDoc ? 'doc-item active' : 'doc-item';
    const missing = Array.isArray(c.missing_pages) ? c.missing_pages.length : 0;
    return `<div class="${cls}" data-doc="${esc(d)}"><div>${esc(d)}</div><div class="doc-meta">pages=${esc(c.extracted_pages || 0)} · missing=${esc(missing)}</div></div>`;
  }).join('');
  box.querySelectorAll('[data-doc]').forEach((el) => {
    el.onclick = () => {
      S.currentDoc = el.getAttribute('data-doc');
      renderDocs();
    };
  });
}

async function refreshDocs() {
  try {
    const root = $('dataRoot').value.trim();
    const d = await apiFetch(`/api/docs?root=${encodeURIComponent(root)}`);
    S.docs = d.documents || [];
    S.coverage = d.coverage || {};
    if (!S.currentDoc && S.docs.length) S.currentDoc = S.docs[0];
    renderDocs();
  } catch (e) {
    addMsg('system', `Docs refresh failed: ${e.message}`);
  }
}

function renderSessions() {
  const box = $('sessionsList');
  if (!S.sessions.length) {
    box.innerHTML = '<div class="small-text">No chats yet.</div>';
    return;
  }
  box.innerHTML = S.sessions.map((s) => {
    const cls = s.id === S.sessionId ? 'session-item active' : 'session-item';
    return `<div class="${cls}" data-sid="${esc(s.id)}"><div class="session-title">${esc(s.title || 'SCAL Chat')}</div><div class="session-meta">${esc(s.updated_at || '')}</div><div class="row"><button class="btn-muted session-del" data-del="${esc(s.id)}">Delete</button></div></div>`;
  }).join('');

  box.querySelectorAll('[data-sid]').forEach((el) => {
    el.onclick = async (ev) => {
      if (ev.target.classList.contains('session-del')) return;
      const sid = el.getAttribute('data-sid');
      S.sessionId = sid;
      await loadSession(sid);
      renderSessions();
    };
  });
  box.querySelectorAll('.session-del').forEach((b) => {
    b.onclick = async (ev) => {
      ev.stopPropagation();
      const sid = b.getAttribute('data-del');
      await deleteSession(sid);
    };
  });
}

async function refreshSessions() {
  try {
    const d = await apiFetch('/api/chat/sessions');
    S.sessions = d.sessions || [];
    if (!S.sessionId && S.sessions.length) S.sessionId = S.sessions[0].id;
    renderSessions();
  } catch (e) {
    addMsg('system', `Session list failed: ${e.message}`);
  }
}

async function loadSession(sessionId) {
  if (!sessionId) return;
  try {
    const d = await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`);
    const msgs = d.session?.messages || [];
    const box = $('chatBox');
    box.innerHTML = '';
    msgs.forEach((m) => {
      const role = m.role === 'assistant' ? 'assistant' : (m.role === 'user' ? 'user' : 'system');
      addMsg(role, m.content || '');
    });
  } catch (e) {
    addMsg('system', `Session load failed: ${e.message}`);
  }
}

async function newSession() {
  try {
    const d = await apiFetch('/api/chat/session/new', { method: 'POST', body: fd({ title: 'SCAL Chat' }) });
    S.sessionId = d.session?.id || null;
    await refreshSessions();
    $('chatBox').innerHTML = '';
    $('sourcesList').innerHTML = '';
  } catch (e) {
    addMsg('system', `New chat failed: ${e.message}`);
  }
}

async function deleteSession(sessionId) {
  if (!sessionId) return;
  try {
    await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    if (S.sessionId === sessionId) {
      S.sessionId = null;
      $('chatBox').innerHTML = '';
    }
    await refreshSessions();
    if (S.sessionId) await loadSession(S.sessionId);
    else if (S.sessions.length) {
      S.sessionId = S.sessions[0].id;
      await loadSession(S.sessionId);
    }
  } catch (e) {
    addMsg('system', `Delete chat failed: ${e.message}`);
  }
}

function renderSources(items = []) {
  const box = $('sourcesList');
  if (!items.length) {
    box.innerHTML = '<div class="small-text">No retrieval sources.</div>';
    return;
  }
  box.innerHTML = items.map((x) => {
    const title = `[${x.rank}] ${x.file_name || '-'} · p${x.page_number || '-'} · ${x.table_id || '-'}`;
    return `<div class="source-item"><div>${esc(title)}</div><div class="source-meta">score=${esc(x.score)} · ${esc(x.extraction_type || 'general')}</div><div class="source-meta">${esc(x.snippet || '')}</div></div>`;
  }).join('');
}

async function refreshModels() {
  try {
    const d = await apiFetch('/api/models/options');
    const sel = $('modelSelect');
    sel.innerHTML = '';
    (d.models || []).forEach((m) => {
      const op = document.createElement('option');
      op.value = m.name;
      op.textContent = m.label || m.name;
      sel.appendChild(op);
    });
    const active = d.active || '';
    const def = d.default || '';
    if (active) sel.value = active;
    else if (def) sel.value = def;
  } catch (e) {
    addMsg('system', `Model options failed: ${e.message}`);
  }
}

async function switchModel() {
  const modelName = $('modelSelect').value;
  if (!modelName) return;
  try {
    const d = await apiFetch('/api/models/switch', { method: 'POST', body: fd({ model_name: modelName }) });
    addMsg('system', d.message || `Switch requested: ${modelName}`);
  } catch (e) {
    addMsg('system', `Switch failed: ${e.message}`);
  }
}

async function pullModel() {
  const modelName = $('modelSelect').value;
  if (!modelName) return;
  try {
    const d = await apiFetch('/api/models/pull', { method: 'POST', body: fd({ model_name: modelName }) });
    addMsg('system', d.message || `Pull started: ${modelName}`);
  } catch (e) {
    addMsg('system', `Pull failed: ${e.message}`);
  }
}

async function unloadModel() {
  try {
    const d = await apiFetch('/api/models/unload', { method: 'POST', body: fd({}) });
    addMsg('system', d.message || 'Unload requested.');
  } catch (e) {
    addMsg('system', `Unload failed: ${e.message}`);
  }
}

async function browseDataRoot() {
  try {
    const d = await apiFetch('/api/browse/folder', { method: 'POST' });
    if (!d.path) return;
    $('dataRoot').value = d.path;
    await saveSettings({ data_root: d.path });
    await refreshDocs();
  } catch (e) {
    addMsg('system', `Browse failed: ${e.message}`);
  }
}

async function pollState() {
  try {
    const d = await apiFetch('/api/state');
    if (d.app) {
      $('buildChip').textContent = `build: ${d.app.build || 'dev'}`;
      $('buildChip').title = `Started: ${d.app.started_at || '-'}`;
    }
    if (d.settings) {
      S.settings = d.settings;
      $('uiModeSelect').value = S.settings.ui_mode || 'layman';
      $('backendSelect').value = S.settings.backend || 'inference_api';
      setUiMode(S.settings.ui_mode || 'layman');
    }
    if (d.services) S.services = d.services;

    const m = d.model || {};
    S.activeModelName = m.model_name || '';
    const stateText = m.loading
      ? `loading ${m.target_model || ''}`
      : (m.loaded ? (m.model_name || 'loaded') : 'idle');
    $('modelStatus').textContent = `Model: ${stateText}`;
    $('modelStatus').style.color = m.last_error ? 'var(--warn)' : (m.loaded ? 'var(--good)' : 'var(--muted)');
    $('modelStatus').title = m.last_error || '';
  } catch (_) {}
}

async function refreshLogs() {
  if (S.settings.ui_mode !== 'advanced') return;
  try {
    const d = await apiFetch('/api/logs?kind=status&limit=120');
    const box = $('logsList');
    const items = d.items || [];
    if (!items.length) {
      box.innerHTML = '<div class="small-text">No logs.</div>';
      return;
    }
    box.innerHTML = items.map((x) => `<div class="log-item"><div class="source-meta">${esc(x.time || '')}</div><div>${esc(x.msg || '')}</div></div>`).join('');
  } catch (_) {}
}

async function clearLogs() {
  try {
    await apiFetch('/api/logs/clear', { method: 'POST', body: fd({ kind: 'all' }) });
    await refreshLogs();
  } catch (e) {
    addMsg('system', `Clear logs failed: ${e.message}`);
  }
}

async function askChat() {
  const q = $('chatInput').value.trim();
  if (!q) return;

  const scope = (S.settings.ui_mode === 'advanced') ? ($('scopeSelect').value || 'all') : 'all';
  const fType = (S.settings.ui_mode === 'advanced') ? ($('fType').value || null) : null;
  const responseMode = (S.settings.ui_mode === 'advanced') ? ($('responseMode').value || 'fast') : 'fast';
  const topK = responseMode === 'fast' ? 5 : (responseMode === 'deep' ? 10 : 8);
  const docName = (scope === 'selected' && S.currentDoc) ? S.currentDoc : null;

  addMsg('user', q);
  $('chatInput').value = '';

  const pending = addPendingAssistant();
  const elapsed = pending.querySelector('.pending-elapsed');
  const t0 = performance.now();
  const timer = setInterval(() => {
    if (elapsed) elapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 120);

  const sendBtn = $('sendBtn');
  sendBtn.disabled = true;

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        session_id: S.sessionId,
        doc_name: docName,
        scope,
        filter_extraction_type: fType,
        response_mode: responseMode,
        prompt_template: '',
        top_k: topK,
      }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let streamed = '';
    let donePayload = null;
    const body = pending.querySelector('.body');

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      while (true) {
        const sep = buf.indexOf('\n\n');
        if (sep < 0) break;
        const eventRaw = buf.slice(0, sep);
        buf = buf.slice(sep + 2);

        const dataLines = eventRaw.split('\n').filter((ln) => ln.startsWith('data:')).map((ln) => ln.slice(5).trim());
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
          throw new Error(ev.message || 'stream failed');
        }
      }
    }

    const d = donePayload || {};
    if (!d.answer) d.answer = streamed || '(no answer)';
    if (d.session_id) S.sessionId = d.session_id;

    pending.className = 'msg assistant';
    if (body) body.innerHTML = `${esc(d.answer).replace(/\n/g, '<br>')}${perfBadges(d.metrics || {})}`;

    renderSources(d.reasoning || []);
    if (d.metrics && d.metrics.total_ms != null) {
      $('perfHint').textContent = `Last: ${(Number(d.metrics.total_ms) / 1000).toFixed(2)}s · tok/s ${d.metrics.tokens_per_sec ?? 0}`;
    }

    await refreshSessions();
    renderSessions();
  } catch (e) {
    pending.className = 'msg system';
    const role = pending.querySelector('.role');
    if (role) role.textContent = 'SYSTEM';
    const body = pending.querySelector('.body');
    if (body) body.innerHTML = esc(`Error: ${e.message}`).replace(/\n/g, '<br>');
  } finally {
    clearInterval(timer);
    sendBtn.disabled = false;
  }
}

function clearChatView() {
  $('chatBox').innerHTML = '';
  renderSources([]);
}

async function init() {
  $('sendBtn').onclick = askChat;
  $('clearBtn').onclick = clearChatView;
  $('newSessionBtn').onclick = newSession;
  $('refreshDocsBtn').onclick = refreshDocs;
  $('browseDataRootBtn').onclick = browseDataRoot;

  $('switchModelBtn').onclick = switchModel;
  $('pullModelBtn').onclick = pullModel;
  $('unloadModelBtn').onclick = unloadModel;
  $('clearLogsBtn').onclick = clearLogs;
  $('openLegacyBtn').onclick = () => {
    window.open(S.services.legacy_ui_url || 'http://127.0.0.1:8090', '_blank');
  };

  $('uiModeSelect').onchange = async (e) => {
    const mode = e.target.value;
    await saveSettings({ ui_mode: mode });
  };

  $('backendSelect').onchange = async (e) => {
    const backend = e.target.value;
    await saveSettings({ backend });
    await refreshModels();
    await pollState();
  };

  $('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askChat();
    }
  });

  try {
    const s = await apiFetch('/api/settings');
    S.settings = s.settings || S.settings;
  } catch (_) {}

  $('dataRoot').value = S.settings.data_root || '';
  $('uiModeSelect').value = S.settings.ui_mode || 'layman';
  $('backendSelect').value = S.settings.backend || 'inference_api';
  setUiMode(S.settings.ui_mode || 'layman');

  await refreshModels();
  await refreshDocs();
  await refreshSessions();
  if (!S.sessionId) {
    await newSession();
  } else {
    await loadSession(S.sessionId);
  }
  await pollState();
  await refreshLogs();

  setInterval(async () => {
    await pollState();
    await refreshLogs();
  }, 2000);
}

init();
