'use strict';

const S = {
  docs: [],
  currentDoc: null,
  sessionId: null,
  activeModelName: '',
  lastHits: [],
};

const DEFAULT_DATA_ROOT =
  'C:\\Users\\Mining\\Downloads\\Fine Tunining Datasets-20260318T052420Z-1-001\\Fine Tunining Datasets\\train';

function $(id) { return document.getElementById(id); }
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
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

function modelLabel() {
  const raw = (S.activeModelName || '').trim();
  if (!raw) return 'ASSISTANT';
  return raw.includes('/') ? raw.split('/').pop().toUpperCase() : raw.toUpperCase();
}

function perfBadges(m = {}) {
  const chips = [];
  if (m.response_mode) chips.push(`<span class="badge">mode=${esc(m.response_mode)}</span>`);
  if (m.first_token_ms != null) chips.push(`<span class="badge">first=${(Number(m.first_token_ms)/1000).toFixed(2)}s</span>`);
  if (m.total_ms != null) chips.push(`<span class="badge">total=${(Number(m.total_ms)/1000).toFixed(2)}s</span>`);
  if (m.tokens_per_sec != null) chips.push(`<span class="badge">tok/s=${esc(m.tokens_per_sec)}</span>`);
  if (m.hits != null) chips.push(`<span class="badge">hits=${esc(m.hits)}</span>`);
  return chips.length ? `<div class="meta">${chips.join('')}</div>` : '';
}

function addMsg(role, text, html = '') {
  const box = $('chatBox');
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  const label = role === 'assistant' ? modelLabel() : ({ user: 'YOU', system: 'SYSTEM' }[role] || role.toUpperCase());
  el.innerHTML = `<div class="role">${label}</div><div class="body">${esc(text).replace(/\n/g, '<br>')}${html}</div>`;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

function addPending() {
  const box = $('chatBox');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `<div class="role">${modelLabel()}</div><div class="body">Thinking... <span class="badge pending">0.0s</span></div>`;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

function renderSources(items = []) {
  const list = $('sourcesList');
  if (!items.length) {
    list.innerHTML = '<div class="small">No retrieval sources</div>';
    return;
  }
  list.innerHTML = items.map((x) => {
    const t = `[${x.rank}] ${x.file_name || '-'} · p${x.page_number || '-'} · ${x.table_id || '-'}`;
    return `<div class="list-item"><div>${esc(t)}</div><div class="small">score=${esc(x.score)} · ${esc(x.extraction_type || 'general')}</div></div>`;
  }).join('');
}

function renderDocs() {
  const wrap = $('docsList');
  if (!S.docs.length) {
    wrap.innerHTML = '<div class="small">No extracted docs found.</div>';
    return;
  }
  wrap.innerHTML = S.docs.map((d) => {
    const cls = d === S.currentDoc ? 'list-item active' : 'list-item';
    return `<button class="${cls}" data-doc="${esc(d)}">${esc(d)}</button>`;
  }).join('');
  wrap.querySelectorAll('[data-doc]').forEach((b) => {
    b.onclick = () => {
      S.currentDoc = b.dataset.doc;
      renderDocs();
    };
  });
}

async function refreshDocs() {
  try {
    const root = $('dataRoot').value.trim();
    const d = await apiFetch(`/api/docs?root=${encodeURIComponent(root)}`);
    S.docs = d.documents || [];
    if (!S.currentDoc && S.docs.length) S.currentDoc = S.docs[0];
    renderDocs();
  } catch (e) {
    addMsg('system', `Docs refresh failed: ${e.message}`);
  }
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
    if (d.active) sel.value = d.active;
    else if (d.default) sel.value = d.default;
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

async function unloadModel() {
  try {
    const d = await apiFetch('/api/models/unload', { method: 'POST', body: fd({}) });
    addMsg('system', d.message || 'Unload requested.');
  } catch (e) {
    addMsg('system', `Unload failed: ${e.message}`);
  }
}

async function pollState() {
  try {
    const d = await apiFetch('/api/state');
    if (d.app) {
      $('buildChip').textContent = `build: ${d.app.build || 'dev'}`;
      $('buildChip').title = `Started: ${d.app.started_at || '-'}`;
    }
    const m = d.model || {};
    S.activeModelName = m.model_name || '';
    const suffix = m.loading ? `loading ${m.target_model || ''}` : (m.loaded ? (m.model_name || 'loaded') : 'idle');
    $('modelStatus').textContent = `Model: ${suffix}`;
    $('modelStatus').style.color = m.last_error ? 'var(--warn)' : (m.loaded ? 'var(--ok)' : 'var(--muted)');
    $('modelStatus').title = m.last_error || '';
  } catch (_) {}
}

async function refreshSessions() {
  try {
    const d = await apiFetch('/api/chat/sessions');
    const sel = $('sessionSelect');
    sel.innerHTML = '';
    (d.sessions || []).forEach((s) => {
      const op = document.createElement('option');
      op.value = s.id;
      op.textContent = `${s.title || 'Session'} (${s.message_count || 0})`;
      sel.appendChild(op);
    });
    if (!S.sessionId && d.sessions?.length) S.sessionId = d.sessions[0].id;
    if (S.sessionId) sel.value = S.sessionId;
  } catch (e) {
    addMsg('system', `Session list failed: ${e.message}`);
  }
}

async function loadSession(sessionId) {
  if (!sessionId) return;
  try {
    const d = await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`);
    const box = $('chatBox');
    box.innerHTML = '';
    const msgs = d.session?.messages || [];
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
  } catch (e) {
    addMsg('system', `New session failed: ${e.message}`);
  }
}

function clearChat() {
  $('chatBox').innerHTML = '';
  renderSources([]);
}

async function askChat() {
  const q = $('chatInput').value.trim();
  if (!q) return;

  const scope = $('scopeSelect').value || 'all';
  const responseMode = $('responseMode').value || 'fast';
  const docName = scope === 'selected' ? (S.currentDoc || null) : null;

  addMsg('user', q);
  $('chatInput').value = '';
  const pending = addPending();
  const elapsed = pending.querySelector('.pending');
  const t0 = performance.now();
  const timer = setInterval(() => {
    if (elapsed) elapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 120);

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        session_id: S.sessionId,
        doc_name: docName,
        scope,
        filter_extraction_type: $('fType').value || null,
        response_mode: responseMode,
        prompt_template: '',
        top_k: responseMode === 'fast' ? 5 : (responseMode === 'deep' ? 10 : 8),
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
        const evt = buf.slice(0, sep);
        buf = buf.slice(sep + 2);

        const lines = evt.split('\n').filter((ln) => ln.startsWith('data:')).map((ln) => ln.slice(5).trim());
        if (!lines.length) continue;
        const payload = lines.join('\n');
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

    pending.className = 'msg assistant';
    if (body) body.innerHTML = `${esc(d.answer).replace(/\n/g, '<br>')}${perfBadges(d.metrics || {})}`;

    S.lastHits = d.raw_hits || [];
    renderSources(d.reasoning || []);
    if (d.metrics && d.metrics.total_ms != null) {
      $('perfHint').textContent = `Last: ${(Number(d.metrics.total_ms) / 1000).toFixed(2)}s · tok/s ${d.metrics.tokens_per_sec ?? 0}`;
    }
    await refreshSessions();
    if (S.sessionId) $('sessionSelect').value = S.sessionId;
  } catch (e) {
    pending.className = 'msg system';
    const role = pending.querySelector('.role');
    const body = pending.querySelector('.body');
    if (role) role.textContent = 'SYSTEM';
    if (body) body.innerHTML = esc(`Error: ${e.message}`).replace(/\n/g, '<br>');
  } finally {
    clearInterval(timer);
  }
}

async function init() {
  $('dataRoot').value = DEFAULT_DATA_ROOT;
  $('refreshDocsBtn').onclick = refreshDocs;
  $('switchModelBtn').onclick = switchModel;
  $('unloadModelBtn').onclick = unloadModel;
  $('sendBtn').onclick = askChat;
  $('clearBtn').onclick = clearChat;
  $('newSessionBtn').onclick = newSession;
  $('sessionSelect').onchange = async (e) => {
    S.sessionId = e.target.value;
    await loadSession(S.sessionId);
  };
  $('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askChat();
    }
  });

  await refreshModels();
  await refreshDocs();
  await refreshSessions();
  if (!S.sessionId) {
    await newSession();
  } else {
    await loadSession(S.sessionId);
  }
  await pollState();
  setInterval(pollState, 2000);
}

init();
