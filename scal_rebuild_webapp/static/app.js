const state = {
  app: {},
  settings: {
    backend: "llama_cpp",
    ui_mode: "layman",
    data_root: "",
    llama_server_exe: "",
    llama_model_dir: "D:\\models",
    llama_model_path: "D:\\models\\Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    llama_ctx_size: 16384,
    llama_auto_download: true,
  },
  model: {},
  progress: {},
  services: {},
  defaults: {},
  sessions: [],
  currentSessionId: "",
  currentDoc: "",
  docs: [],
  coverage: {},
  pagesMap: {},
  logKind: "status",
  previewTab: "pdf",
  lastSources: [],
  lastTables: [],
  streaming: false,
};

const modelNotice = {
  booted: false,
  lastStage: "",
  lastPercent: -1,
  lastLoaded: false,
  lastModelName: "",
};

const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

async function apiJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiForm(url, formObj) {
  const body = new URLSearchParams();
  Object.entries(formObj || {}).forEach(([k, v]) => body.append(k, String(v ?? "")));
  return apiJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });
}

function systemMsg(text) {
  addMessage("system", text || "");
}

function applyUiMode(mode) {
  const isAdvanced = mode === "advanced";
  document.querySelectorAll(".advanced-only").forEach((el) => {
    el.classList.toggle("hidden", !isAdvanced);
  });
}

function updateModelStatus() {
  const m = state.model || {};
  const ctx = Number(m.context_limit || 0);
  const ctxLabel = ctx > 0 ? `, ctx ${ctx}` : "";
  const status = m.loading
    ? `Model: loading ${m.target_model || ""}`
    : m.loaded
      ? `Model: ${m.model_name || "ready"} (${m.backend || state.settings.backend}${ctxLabel})`
      : `Model: idle (${state.settings.backend})`;
  $("modelStatus").textContent = status;
  const canPull = state.settings.backend === "ollama";
  $("pullModelBtn").style.display = canPull ? "" : "none";
}

function updateBackendUi() {
  const isLlama = (state.settings.backend || "") === "llama_cpp";
  $("llamaCppPanel").classList.toggle("hidden", !isLlama || state.settings.ui_mode !== "advanced");
  $("downloadDefaultLlamaBtn").disabled = !isLlama;
}

function applySettingsToInputs() {
  $("dataRoot").value = state.settings.data_root || "";
  $("llamaServerExe").value = state.settings.llama_server_exe || "";
  $("llamaModelDir").value = state.settings.llama_model_dir || "";
  $("llamaModelPath").value = state.settings.llama_model_path || "";
  $("llamaCtxInput").value = Number(state.settings.llama_ctx_size || 16384);
  $("llamaAutoDownloadChk").checked = !!state.settings.llama_auto_download;
}

function updateDefaultLlamaHint() {
  const d = state.defaults?.llama_cpp_model || {};
  const path = state.settings.llama_model_path || d.target_path || "";
  const label = d.label || "Default GGUF";
  $("llamaDefaultHint").textContent = `${label} | recommended: ${d.recommended_for || "-"} | path: ${path}`;
}

function maybeReportModelProgress() {
  const p = state.progress?.model || {};
  const m = state.model || {};
  const stage = String(p.stage || "");
  const detail = String(p.detail || "");
  const percent = Number(p.percent ?? -1);
  const loaded = !!m.loaded;
  const modelName = String(m.model_name || m.target_model || "model");

  if (!modelNotice.booted) {
    modelNotice.booted = true;
    modelNotice.lastStage = stage;
    modelNotice.lastPercent = percent;
    modelNotice.lastLoaded = loaded;
    modelNotice.lastModelName = modelName;
    return;
  }

  if ((stage === "loading" || stage === "downloading" || stage === "finalizing") && percent >= 0) {
    const bucket = Math.floor(percent / 10) * 10;
    const lastBucket = Math.floor((modelNotice.lastPercent >= 0 ? modelNotice.lastPercent : -1) / 10) * 10;
    if (stage !== modelNotice.lastStage || bucket !== lastBucket) {
      systemMsg(`Model loading ${modelName}: ${percent}% (${stage}${detail ? ` - ${detail}` : ""})`);
    }
  }

  if (!modelNotice.lastLoaded && loaded) {
    systemMsg(`Model loaded: ${m.model_name || "ready"} (${m.backend || state.settings.backend})`);
  }

  if (stage === "failed" && stage !== modelNotice.lastStage) {
    systemMsg(`Model load failed: ${m.last_error || detail || "unknown error"}`);
  }

  modelNotice.lastStage = stage;
  modelNotice.lastPercent = percent;
  modelNotice.lastLoaded = loaded;
  modelNotice.lastModelName = modelName;
}

async function saveSettings(patch) {
  const data = await apiJson("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch || {}),
  });
  state.settings = data.settings || state.settings;
  applyUiMode(state.settings.ui_mode);
  return data;
}

async function loadState() {
  const s = await apiJson("/api/state");
  state.app = s.app || {};
  state.settings = s.settings || state.settings;
  state.model = s.model || {};
  state.progress = s.progress || state.progress;
  state.services = s.services || {};
  state.defaults = s.defaults || state.defaults;

  $("buildChip").textContent = `build: ${state.app.build || "-"}`;
  $("uiModeSelect").value = state.settings.ui_mode || "layman";
  $("backendSelect").value = state.settings.backend || "llama_cpp";
  applySettingsToInputs();
  applyUiMode(state.settings.ui_mode || "layman");
  updateBackendUi();
  updateDefaultLlamaHint();
  updateModelStatus();
  maybeReportModelProgress();
}

function renderSessions() {
  const host = $("sessionsList");
  host.innerHTML = "";
  for (const s of state.sessions) {
    const row = document.createElement("div");
    row.className = `session-item ${s.id === state.currentSessionId ? "active" : ""}`;
    row.innerHTML = `
      <div class="row-spread">
        <div class="session-title" title="${esc(s.title || "")}">${esc(s.title || "Session")}</div>
        <div class="row">
          <button class="btn-muted" data-act="rename" data-id="${esc(s.id)}">Rename</button>
          <button class="btn-muted" data-act="delete" data-id="${esc(s.id)}">Delete</button>
        </div>
      </div>
      <div class="session-meta">${esc(fmtTime(s.updated_at))} · ${Number(s.message_count || 0)} msgs</div>
    `;
    row.addEventListener("click", async (e) => {
      const btn = e.target.closest("button");
      if (btn) return;
      await openSession(s.id);
    });
    host.appendChild(row);
  }

  host.querySelectorAll("button[data-act='rename']").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const sid = b.getAttribute("data-id") || "";
      const cur = state.sessions.find((x) => x.id === sid);
      const next = window.prompt("New session name", cur?.title || "SCAL Chat");
      if (next === null) return;
      await apiJson(`/api/chat/session/${encodeURIComponent(sid)}/title`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: next }),
      });
      await refreshSessions();
    });
  });

  host.querySelectorAll("button[data-act='delete']").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const sid = b.getAttribute("data-id") || "";
      if (!window.confirm("Delete this session?")) return;
      await apiJson(`/api/chat/session/${encodeURIComponent(sid)}`, { method: "DELETE" });
      if (state.currentSessionId === sid) {
        state.currentSessionId = "";
        $("chatBox").innerHTML = "";
      }
      await refreshSessions();
      if (!state.currentSessionId && state.sessions.length) {
        await openSession(state.sessions[0].id);
      }
    });
  });
}

async function refreshSessions() {
  const data = await apiJson("/api/chat/sessions");
  state.sessions = data.sessions || [];
  renderSessions();
}

async function downloadSessionTranscript() {
  const sid = state.currentSessionId || (state.sessions[0]?.id || "");
  if (!sid) {
    systemMsg("No chat session available to download.");
    return;
  }
  const res = await fetch(`/api/chat/session/${encodeURIComponent(sid)}/export`);
  if (!res.ok) {
    systemMsg(`Download failed: ${await res.text()}`);
    return;
  }
  const blob = await res.blob();
  const dispo = res.headers.get("Content-Disposition") || "";
  let fileName = `session_${sid}.txt`;
  const m = dispo.match(/filename="?([^";]+)"?/i);
  if (m && m[1]) fileName = m[1];
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function addMessage(role, content, extra = {}) {
  const chat = $("chatBox");
  const item = document.createElement("div");
  item.className = `msg ${role}`;
  item.innerHTML = `
    <div class="role">${esc(role)}</div>
    <div class="content"></div>
    <div class="meta"></div>
  `;
  item.querySelector(".content").innerHTML = esc(content || "").replace(/\n/g, "<br>");

  if (role === "assistant" && Array.isArray(extra.sources) && extra.sources.length) {
    const rb = document.createElement("details");
    rb.className = "reasoning-block";
    rb.open = false;
    const lines = extra.sources
      .map((s) => {
        const snip = esc(String(s.snippet || ""));
        return `<div class="reasoning-item">#${s.rank} score=${s.score} file=${esc(s.file_name || "?")} page=${esc(s.page_number || "?")}<br>${snip}</div>`;
      })
      .join("");
    rb.innerHTML = `<summary>Reasoning (${extra.sources.length})</summary>${lines}`;
    item.appendChild(rb);
  }

  if (extra.metrics && role === "assistant") {
    const m = extra.metrics;
    const meta = item.querySelector(".meta");
    const badges = [
      `backend:${m.backend || ""}`,
      `mode:${m.response_mode || ""}`,
      `hits:${m.hits ?? "-"}`,
      `tps:${m.tokens_per_sec ?? "-"}`,
      `ms:${m.total_ms ?? "-"}`,
    ];
    meta.innerHTML = badges.map((b) => `<span class="badge">${esc(b)}</span>`).join("");
  }

  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
  return item;
}

async function openSession(sessionId) {
  const data = await apiJson(`/api/chat/session/${encodeURIComponent(sessionId)}`);
  state.currentSessionId = sessionId;
  $("chatBox").innerHTML = "";
  const msgs = data.session?.messages || [];
  for (const m of msgs) {
    addMessage(m.role || "assistant", m.content || "", { sources: m.sources || [] });
  }
  renderSessions();
}

function renderDocs() {
  const host = $("docsList");
  host.innerHTML = "";
  for (const name of state.docs) {
    const cov = state.coverage[name] || {};
    const item = document.createElement("div");
    item.className = `doc-item ${name === state.currentDoc ? "active" : ""}`;
    item.innerHTML = `
      <div class="session-title" title="${esc(name)}">${esc(name)}</div>
      <div class="doc-meta">pdf:${cov.pdf_pages || 0} extracted:${cov.extracted_pages || 0} missing:${(cov.missing_pages || []).length}</div>
    `;
    item.addEventListener("click", () => {
      state.currentDoc = name;
      renderDocs();
      populatePreviewPages();
      loadPreview().catch((e) => systemMsg(`Preview error: ${e.message || e}`));
      if ($("scopeSelect").value === "selected") {
        systemMsg(`Selected doc: ${name}`);
      }
    });
    host.appendChild(item);
  }
  if (!state.currentDoc && state.docs.length) {
    state.currentDoc = state.docs[0];
    renderDocs();
    populatePreviewPages();
  }
}

async function refreshDocs() {
  const root = ($("dataRoot").value || "").trim();
  const q = root ? `?root=${encodeURIComponent(root)}` : "";
  const data = await apiJson(`/api/docs${q}`);
  state.docs = data.documents || [];
  state.coverage = data.coverage || {};
  state.pagesMap = data.pages_map || {};
  $("dataRoot").value = data.data_root || root;
  renderDocs();
  if (state.currentDoc && ($("previewPageSelect").value || "")) {
    await loadPreview();
  }
}

function populatePreviewPages() {
  const sel = $("previewPageSelect");
  sel.innerHTML = "";
  const doc = state.currentDoc;
  const pages = doc ? state.pagesMap[doc] || [] : [];
  for (const p of pages) {
    const opt = document.createElement("option");
    opt.value = String(p.page);
    const tags = [p.has_pdf ? "pdf" : "", p.has_json ? "json" : "", p.has_md ? "md" : ""].filter(Boolean).join("/");
    opt.textContent = `Page ${p.page} (${tags || "none"})`;
    sel.appendChild(opt);
  }
}

function switchPreviewTab(tab) {
  state.previewTab = tab;
  const ids = ["pdf", "json", "html"];
  for (const t of ids) {
    $(`tab${t[0].toUpperCase() + t.slice(1)}Btn`).classList.toggle("active-tab", t === tab);
    $(`preview${t[0].toUpperCase() + t.slice(1)}Pane`).classList.toggle("hidden", t !== tab);
  }
}

async function loadPreview() {
  const doc = state.currentDoc;
  const page = Number($("previewPageSelect").value || 0);
  if (!doc || !page) {
    systemMsg("Select a document/page for preview.");
    return;
  }
  const data = await apiJson(`/api/page/view?doc_name=${encodeURIComponent(doc)}&page=${encodeURIComponent(page)}`);
  const files = data.files || {};
  const pdfFrame = $("previewPdfFrame");
  const imageFrame = $("previewImageFrame");
  const status = $("previewPdfStatus");

  if (files.pdf_url) {
    pdfFrame.src = `${files.pdf_url}#view=FitH`;
    pdfFrame.classList.remove("hidden");
    imageFrame.classList.add("hidden");
    imageFrame.removeAttribute("src");
    status.classList.add("hidden");
    status.textContent = "";
  } else if (files.image_url) {
    pdfFrame.src = "about:blank";
    pdfFrame.classList.add("hidden");
    imageFrame.src = files.image_url;
    imageFrame.classList.remove("hidden");
    status.classList.remove("hidden");
    status.textContent = "PDF page file not found for this page. Showing extracted image preview instead.";
  } else {
    pdfFrame.src = "about:blank";
    pdfFrame.classList.remove("hidden");
    imageFrame.classList.add("hidden");
    imageFrame.removeAttribute("src");
    status.classList.remove("hidden");
    status.textContent = "No PDF or image preview available for this page.";
  }
  $("previewJsonPane").textContent = data.raw_json || data.raw_text || "(empty)";
  $("previewHtmlPane").innerHTML = (data.tables || []).join("\n") || "<div class='small-text'>No HTML table found on this page.</div>";

  if (state.previewTab === "pdf" && !files.pdf_url && !files.image_url) switchPreviewTab(files.json_url || files.md_url ? "json" : "html");
}

function renderSources(list) {
  state.lastSources = list || [];
  const host = $("sourcesList");
  host.innerHTML = "";
  for (const s of state.lastSources) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div>${esc(s.file_name || "?")} · page ${esc(s.page_number || "?")}</div>
      <div class="source-meta">score ${esc(s.score ?? "-")} · ${esc(s.extraction_type || "-")}</div>
    `;
    host.appendChild(item);
  }
}

async function exportTables(format) {
  if (!state.lastTables.length) {
    $("exportStatus").textContent = "Export: no tables in last response";
    return;
  }
  $("exportStatus").textContent = `Export: generating ${format}...`;
  const res = await fetch("/api/tables/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      format,
      title: "SCAL combined retrieved tables",
      tables: state.lastTables,
    }),
  });
  if (!res.ok) {
    $("exportStatus").textContent = `Export failed: ${await res.text()}`;
    return;
  }
  const blob = await res.blob();
  const dispo = res.headers.get("Content-Disposition") || "";
  let name = format === "excel" ? "scal_tables.xls" : "scal_tables.doc";
  const m = dispo.match(/filename=([^;]+)/i);
  if (m) name = m[1].replace(/^"|"$/g, "");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  $("exportStatus").textContent = `Export: downloaded ${name}`;
}

async function refreshLogs() {
  const data = await apiJson(`/api/logs?kind=${encodeURIComponent(state.logKind)}&limit=200`);
  const host = $("logsList");
  host.innerHTML = "";
  for (const x of data.items || []) {
    const item = document.createElement("div");
    item.className = "log-item";
    item.innerHTML = `<div class="source-meta">${esc(x.time || "")}</div><div>${esc(x.msg || "")}</div>`;
    host.appendChild(item);
  }
  host.scrollTop = host.scrollHeight;
}

async function refreshModels() {
  const data = await apiJson("/api/models/options");
  const select = $("modelSelect");
  select.innerHTML = "";
  for (const m of data.models || []) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.label || m.name;
    select.appendChild(o);
  }
  const pick = data.active || data.default || "";
  if (pick) {
    select.value = pick;
    $("modelInput").value = data.backend === "llama_cpp"
      ? (data.configured_model_path || state.settings.llama_model_path || pick)
      : pick;
  }
}

async function runChatStream(payload) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    throw new Error(await res.text() || `HTTP ${res.status}`);
  }

  const assistantItem = addMessage("assistant", "");
  const contentEl = assistantItem.querySelector(".content");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let doneEvent = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop() || "";
    for (const c of chunks) {
      const lines = c.split("\n").filter((x) => x.startsWith("data:"));
      if (!lines.length) continue;
      const raw = lines.map((x) => x.slice(5).trim()).join("\n");
      if (!raw) continue;
      let ev;
      try {
        ev = JSON.parse(raw);
      } catch {
        continue;
      }
      if (ev.type === "token") {
        contentEl.innerHTML += esc(ev.text || "").replace(/\n/g, "<br>");
        $("chatBox").scrollTop = $("chatBox").scrollHeight;
      } else if (ev.type === "done") {
        doneEvent = ev;
      } else if (ev.type === "error") {
        throw new Error(ev.message || "Stream failed");
      }
    }
  }

  if (doneEvent) {
    const finalContent = (doneEvent.answer || "").trim();
    const safeContent = finalContent || "[No response text returned. Check model logs.]";
    contentEl.innerHTML = esc(safeContent).replace(/\n/g, "<br>");
    if (Array.isArray(doneEvent.sources) && doneEvent.sources.length) {
      const rb = document.createElement("details");
      rb.className = "reasoning-block";
      rb.open = false;
      rb.innerHTML = `<summary>Reasoning (${doneEvent.sources.length})</summary>`;
      for (const s of doneEvent.sources) {
        const line = document.createElement("div");
        line.className = "reasoning-item";
        line.innerHTML = `#${esc(s.rank)} score=${esc(s.score)} file=${esc(s.file_name || "?")} page=${esc(s.page_number || "?")}<br>${esc(s.snippet || "")}`;
        rb.appendChild(line);
      }
      assistantItem.appendChild(rb);
    }
    if (doneEvent.metrics) {
      const m = doneEvent.metrics;
      assistantItem.querySelector(".meta").innerHTML = [
        `backend:${m.backend || ""}`,
        `mode:${m.response_mode || ""}`,
        `table:${m.table_mode ? "yes" : "no"}`,
        `hits:${m.hits ?? "-"}`,
        `used:${m.evidence_hits_used ?? "-"}`,
        `ctx:${m.context_limit ?? "-"}`,
        `prompt:${m.prompt_tokens ?? "-"}`,
        `compact:${m.session_compacted ? "yes" : "no"}`,
        `tps:${m.tokens_per_sec ?? "-"}`,
        `ms:${m.total_ms ?? "-"}`,
      ].map((x) => `<span class='badge'>${esc(x)}</span>`).join("");
      $("perfHint").textContent = `Last: ${m.total_ms || "-"} ms | hits ${m.hits || 0} used ${m.evidence_hits_used || 0} | ctx ${m.context_limit || "-"} | ${m.tokens_per_sec || "-"} tok/s`;
    }
    if (doneEvent.session_id) state.currentSessionId = doneEvent.session_id;
    renderSources(doneEvent.sources || []);
    state.lastTables = Array.isArray(doneEvent.tables) ? doneEvent.tables : [];
    if (state.lastTables.length) {
      $("exportStatus").textContent = `Export: ${state.lastTables.length} table(s) ready`;
    }
  }
}

async function sendChat() {
  if (state.streaming) return;
  const q = ($("chatInput").value || "").trim();
  if (!q) return;

  if (!state.currentSessionId) {
    const s = await apiForm("/api/chat/session/new", { title: "SCAL Chat" });
    state.currentSessionId = s.session?.id || "";
  }

  const payload = {
    question: q,
    session_id: state.currentSessionId || null,
    doc_name: state.currentDoc || null,
    scope: $("scopeSelect").value || "all",
    filter_extraction_type: $("fType").value || null,
    response_mode: $("responseMode").value || "fast",
    top_k: Number($("topKInput").value || 24),
    include_table_html: !!$("includeHtmlChk").checked,
    use_pdf_vision: !!$("useVisionChk").checked,
  };

  addMessage("user", q);
  $("chatInput").value = "";
  state.streaming = true;
  $("sendBtn").disabled = true;

  try {
    await runChatStream(payload);
    await refreshSessions();
    if (state.currentSessionId) renderSessions();
  } catch (e) {
    systemMsg(`Error: ${e.message || e}`);
  } finally {
    state.streaming = false;
    $("sendBtn").disabled = false;
  }
}

async function buildRag(scope) {
  const form = { scope };
  if (scope === "selected") form.doc_name = state.currentDoc || "";
  if (scope === "selected" && !form.doc_name) {
    systemMsg("Select a document first for selected RAG build.");
    return;
  }
  $("ragStatus").textContent = "RAG status: building...";
  const data = await apiForm("/api/rag/build", form);
  $("ragStatus").textContent = `RAG status: ${data.message || "done"}`;
  await refreshRagStatus();
}

async function refreshRagStatus() {
  try {
    const s = await apiJson("/api/rag/status");
    if (s.global_index_ready) {
      $("ragStatus").textContent = `RAG status: loaded ${s.global_chunks || 0} chunk(s)`;
    } else {
      $("ragStatus").textContent = "RAG status: no saved index loaded";
    }
  } catch {
    // ignore
  }
}

function bindEvents() {
  $("downloadSessionBtn").addEventListener("click", downloadSessionTranscript);

  $("newSessionBtn").addEventListener("click", async () => {
    const data = await apiForm("/api/chat/session/new", { title: "SCAL Chat" });
    state.currentSessionId = data.session?.id || "";
    $("chatBox").innerHTML = "";
    await refreshSessions();
    renderSessions();
  });

  $("sendBtn").addEventListener("click", sendChat);
  $("clearBtn").addEventListener("click", () => { $("chatBox").innerHTML = ""; });
  $("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  $("refreshDocsBtn").addEventListener("click", refreshDocs);
  $("browseDataRootBtn").addEventListener("click", async () => {
    const r = await apiJson("/api/browse/folder", { method: "POST" });
    if (r.path) {
      $("dataRoot").value = r.path;
      await saveSettings({ data_root: r.path });
      await refreshDocs();
    }
  });
  $("dataRoot").addEventListener("change", async () => {
    await saveSettings({ data_root: $("dataRoot").value.trim() });
    await refreshDocs();
  });

  $("browseLlamaServerBtn").addEventListener("click", async () => {
    const r = await apiJson("/api/browse/file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accept: ".exe" }),
    });
    if (r.path) $("llamaServerExe").value = r.path;
  });
  $("browseLlamaModelDirBtn").addEventListener("click", async () => {
    const r = await apiJson("/api/browse/folder", { method: "POST" });
    if (r.path) {
      $("llamaModelDir").value = r.path;
      const cur = ($("llamaModelPath").value || "").trim();
      if (!cur || cur === (state.settings.llama_model_path || "")) {
        $("llamaModelPath").value = `${r.path.replace(/[\\/]$/, "")}\\Qwen2.5-32B-Instruct-Q4_K_M.gguf`;
      }
    }
  });
  $("browseLlamaModelBtn").addEventListener("click", async () => {
    const r = await apiJson("/api/browse/file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accept: ".gguf" }),
    });
    if (r.path) {
      $("llamaModelPath").value = r.path;
      const parts = r.path.split(/[/\\]/);
      parts.pop();
      $("llamaModelDir").value = parts.join("\\");
    }
  });
  $("saveLlamaSettingsBtn").addEventListener("click", async () => {
    await saveSettings({
      llama_server_exe: $("llamaServerExe").value.trim(),
      llama_model_dir: $("llamaModelDir").value.trim(),
      llama_model_path: $("llamaModelPath").value.trim(),
      llama_ctx_size: Number($("llamaCtxInput").value || 16384),
      llama_auto_download: !!$("llamaAutoDownloadChk").checked,
    });
    applySettingsToInputs();
    updateDefaultLlamaHint();
    systemMsg("llama.cpp settings saved.");
  });
  $("downloadDefaultLlamaBtn").addEventListener("click", async () => {
    await saveSettings({
      llama_server_exe: $("llamaServerExe").value.trim(),
      llama_model_dir: $("llamaModelDir").value.trim(),
      llama_model_path: $("llamaModelPath").value.trim(),
      llama_ctx_size: Number($("llamaCtxInput").value || 16384),
      llama_auto_download: !!$("llamaAutoDownloadChk").checked,
    });
    const r = await apiForm("/api/llama_cpp/download-default", {});
    if (r.path) {
      $("llamaModelPath").value = r.path;
      const parts = r.path.split(/[/\\]/);
      parts.pop();
      $("llamaModelDir").value = parts.join("\\");
      state.settings.llama_model_path = r.path;
      state.settings.llama_model_dir = parts.join("\\");
    }
    updateDefaultLlamaHint();
    systemMsg(r.message || "Default GGUF downloaded.");
  });

  $("buildRagAllBtn").addEventListener("click", () => buildRag("all"));
  $("buildRagSelectedBtn").addEventListener("click", () => buildRag("selected"));

  $("uiModeSelect").addEventListener("change", async () => {
    await saveSettings({ ui_mode: $("uiModeSelect").value });
    applyUiMode(state.settings.ui_mode);
    updateBackendUi();
  });
  $("backendSelect").addEventListener("change", async () => {
    await saveSettings({ backend: $("backendSelect").value });
    await loadState();
    await refreshModels();
  });

  $("modelSelect").addEventListener("change", () => {
    const v = $("modelSelect").value || "";
    if (v) $("modelInput").value = v;
  });

  $("switchModelBtn").addEventListener("click", async () => {
    const modelName = state.settings.backend === "llama_cpp"
      ? (($("llamaModelPath").value || "").trim() || ($("modelInput").value || "").trim())
      : (($("modelInput").value || "").trim() || $("modelSelect").value || "");
    if (!modelName) {
      systemMsg(state.settings.backend === "llama_cpp" ? "Set a GGUF model path first." : "Enter or select a model name first.");
      return;
    }
    if (state.settings.backend === "llama_cpp") {
      await saveSettings({
        llama_server_exe: $("llamaServerExe").value.trim(),
        llama_model_dir: $("llamaModelDir").value.trim(),
        llama_model_path: modelName,
        llama_ctx_size: Number($("llamaCtxInput").value || 16384),
        llama_auto_download: !!$("llamaAutoDownloadChk").checked,
      });
    }
    const r = await apiForm("/api/models/switch", { model_name: modelName });
    systemMsg(r.message || "Switch complete");
    await loadState();
    await refreshModels();
  });

  $("pullModelBtn").addEventListener("click", async () => {
    let modelName = ($("modelInput").value || "").trim() || $("modelSelect").value || "";
    if (!modelName) modelName = window.prompt("Ollama model name (e.g. llama3.1:8b)", "") || "";
    if (!modelName) return;
    const r = await apiForm("/api/models/pull", { model_name: modelName });
    systemMsg(r.message || "Pull started");
    await loadState();
    await refreshModels();
  });

  $("unloadModelBtn").addEventListener("click", async () => {
    const r = await apiForm("/api/models/unload", {});
    systemMsg(r.message || "Unloaded");
    await loadState();
  });

  $("logStatusBtn").addEventListener("click", async () => {
    state.logKind = "status";
    setLogTab();
    await refreshLogs();
  });
  $("logDebugBtn").addEventListener("click", async () => {
    state.logKind = "debug";
    setLogTab();
    await refreshLogs();
  });
  $("logErrorBtn").addEventListener("click", async () => {
    state.logKind = "error";
    setLogTab();
    await refreshLogs();
  });
  $("clearLogsBtn").addEventListener("click", async () => {
    await apiForm("/api/logs/clear", { kind: state.logKind });
    await refreshLogs();
  });

  $("refreshPreviewBtn").addEventListener("click", loadPreview);
  $("previewPageSelect").addEventListener("change", loadPreview);
  $("popoutPdfBtn").addEventListener("click", () => {
    const src = !$("previewPdfFrame").classList.contains("hidden")
      ? ($("previewPdfFrame").src || "")
      : ($("previewImageFrame").src || "");
    if (!src || src === "about:blank") {
      systemMsg("Load a PDF page preview first.");
      return;
    }
    window.open(src, "_blank", "noopener,noreferrer");
  });
  $("tabPdfBtn").addEventListener("click", () => switchPreviewTab("pdf"));
  $("tabJsonBtn").addEventListener("click", () => switchPreviewTab("json"));
  $("tabHtmlBtn").addEventListener("click", () => switchPreviewTab("html"));

  $("exportExcelBtn").addEventListener("click", () => exportTables("excel"));
  $("exportWordBtn").addEventListener("click", () => exportTables("word"));

}

function setLogTab() {
  ["Status", "Debug", "Error"].forEach((k) => {
    const id = `log${k}Btn`;
    $(id).classList.toggle("active-tab", state.logKind === k.toLowerCase());
  });
}

async function boot() {
  bindEvents();
  switchPreviewTab("pdf");
  setLogTab();
  try {
    await loadState();
    await refreshModels();
    await refreshDocs();
    await refreshRagStatus();
    await refreshSessions();
    if (state.sessions.length) {
      await openSession(state.sessions[0].id);
    }
    await refreshLogs();
  } catch (e) {
    systemMsg(`Startup error: ${e.message || e}`);
  }

  window.setInterval(async () => {
    try {
      await loadState();
      if (state.settings.ui_mode === "advanced") await refreshLogs();
    } catch {
      // ignore polling failures
    }
  }, 3500);
}

boot();
