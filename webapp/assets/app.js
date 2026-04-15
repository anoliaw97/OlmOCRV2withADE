const state = {
  packages: [],
  currentPackageId: "",
  sessions: [],
  currentSessionId: "",
  chatRecords: [],
};

const el = {
  healthBadge: document.getElementById("healthBadge"),
  statusLine: document.getElementById("statusLine"),
  pathInput: document.getElementById("pathInput"),
  browsePanel: document.getElementById("browsePanel"),
  packageList: document.getElementById("packageList"),
  packageMeta: document.getElementById("packageMeta"),
  markdownFrame: document.getElementById("markdownFrame"),
  tablesTab: document.getElementById("tab-tables"),
  jsonText: document.getElementById("jsonText"),
  pdfPathText: document.getElementById("pdfPathText"),
  pdfOpenLink: document.getElementById("pdfOpenLink"),
  pagePdfList: document.getElementById("pagePdfList"),
  sessionSelect: document.getElementById("sessionSelect"),
  modeSelect: document.getElementById("modeSelect"),
  backendSelect: document.getElementById("backendSelect"),
  modelInput: document.getElementById("modelInput"),
  modelSelect: document.getElementById("modelSelect"),
  ollamaUrlInput: document.getElementById("ollamaUrlInput"),
  llamaScanPathInput: document.getElementById("llamaScanPathInput"),
  llamaCliInput: document.getElementById("llamaCliInput"),
  maxTokensInput: document.getElementById("maxTokensInput"),
  contextLimitInput: document.getElementById("contextLimitInput"),
  systemPromptInput: document.getElementById("systemPromptInput"),
  questionInput: document.getElementById("questionInput"),
  exportPathInput: document.getElementById("exportPathInput"),
  chatLog: document.getElementById("chatLog"),
  chatMetrics: document.getElementById("chatMetrics"),
};

function setStatus(text) {
  el.statusLine.textContent = text;
}

function nowTime() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(method, path, payload = null) {
  const options = { method, headers: { Accept: "application/json" } };
  if (payload !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(path, options);
  const raw = await response.text();
  const data = raw.trim() ? JSON.parse(raw) : {};
  if (!response.ok) {
    throw new Error(data.detail || `${response.status} ${response.statusText}`);
  }
  return data;
}

function addMessage(role, content, label = "", time = "") {
  const block = document.createElement("div");
  block.className = `msg ${role}`;
  const title = label || (role === "user" ? "You" : role === "assistant" ? "Assistant" : "System");
  block.innerHTML =
    `<div class=\"msg-header\"><strong>${escapeHtml(title)}${time ? ` [${escapeHtml(time)}]` : ""}</strong></div>` +
    `<div class=\"msg-body\">${escapeHtml(content).replaceAll("\n", "<br>")}</div>`;
  el.chatLog.appendChild(block);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function systemMessage(text) {
  addMessage("system", text, "System", nowTime());
}

function resetChatView() {
  el.chatLog.innerHTML = "";
  state.chatRecords = [];
  el.chatMetrics.textContent = "No response metrics yet.";
}

function renderMetrics(metrics = {}, reasoningChain = []) {
  const lines = [
    `Context limit: ${metrics.context_limit || 0} (${metrics.context_limit_source || "heuristic"})`,
    `Context used: ${metrics.context_chars || 0} chars${metrics.context_truncated ? " (truncated)" : ""}`,
    `Retrieval: ${metrics.retrieval_chunks || 0} chunks in ${Number(metrics.retrieval_ms || 0).toFixed(2)} ms`,
    `Generation: ${Number(metrics.generation_ms || 0).toFixed(2)} ms`,
    `Total: ${Number(metrics.total_ms || 0).toFixed(2)} ms`,
  ];
  if (reasoningChain.length) {
    lines.push("", "Reasoning chain:", ...reasoningChain.map((v) => `- ${v}`));
  }
  el.chatMetrics.textContent = lines.join("\n");
}

function renderPackageList() {
  el.packageList.innerHTML = "";
  if (!state.packages.length) {
    el.packageList.innerHTML = "<div style='padding:8px'>No packages loaded.</div>";
    return;
  }
  for (const pkg of state.packages) {
    const btn = document.createElement("button");
    btn.textContent = `${pkg.base_name} [${(pkg.tokens || []).join(", ") || "EMPTY"}]`;
    if (pkg.package_id === state.currentPackageId) {
      btn.classList.add("active");
    }
    btn.addEventListener("click", () => selectPackage(pkg.package_id));
    el.packageList.appendChild(btn);
  }
}

function renderPdfInfo(preview, pkg) {
  const fullPdf = preview.full_pdf_path || pkg.full_pdf_path || "";
  const previewPdf = preview.pdf_path || pkg.pdf_path || "";
  const pagePdfs = preview.page_pdf_paths || pkg.page_pdf_paths || [];

  if (fullPdf) {
    el.pdfPathText.textContent = `Full PDF available: ${fullPdf}`;
  } else if (previewPdf) {
    el.pdfPathText.textContent = `Grouped page PDFs detected. Preview target: ${previewPdf}`;
  } else {
    el.pdfPathText.textContent = "No PDF in selected package.";
  }

  const openPath = fullPdf || previewPdf;
  if (openPath) {
    el.pdfOpenLink.classList.remove("hidden");
    el.pdfOpenLink.href = `file:///${openPath.replaceAll("\\", "/")}`;
  } else {
    el.pdfOpenLink.classList.add("hidden");
    el.pdfOpenLink.removeAttribute("href");
  }

  el.pagePdfList.innerHTML = "";
  if (!pagePdfs.length) {
    el.pagePdfList.textContent = "No grouped page-PDF set detected.";
    return;
  }
  for (const path of pagePdfs.slice(0, 80)) {
    const row = document.createElement("div");
    row.className = "item";
    row.textContent = path;
    el.pagePdfList.appendChild(row);
  }
}

function renderTables(tables) {
  el.tablesTab.innerHTML = "";
  if (!tables.length) {
    el.tablesTab.textContent = "No tables detected in JSON/Markdown for this package.";
    return;
  }
  for (const table of tables) {
    const card = document.createElement("section");
    card.className = "table-card";
    const h = document.createElement("h4");
    h.textContent = table.title;
    card.appendChild(h);

    if ((table.headers || []).length && (table.rows || []).length) {
      const t = document.createElement("table");
      const thead = document.createElement("thead");
      const trh = document.createElement("tr");
      for (const col of table.headers) {
        const th = document.createElement("th");
        th.textContent = col;
        trh.appendChild(th);
      }
      thead.appendChild(trh);
      t.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const row of table.rows) {
        const tr = document.createElement("tr");
        for (const val of row) {
          const td = document.createElement("td");
          td.textContent = val;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      t.appendChild(tbody);
      card.appendChild(t);
    } else {
      const pre = document.createElement("pre");
      pre.textContent = table.raw_text || "No structured rows found.";
      card.appendChild(pre);
    }
    el.tablesTab.appendChild(card);
  }
}

async function selectPackage(packageId) {
  state.currentPackageId = packageId;
  renderPackageList();
  const pkg = state.packages.find((p) => p.package_id === packageId);
  if (!pkg) {
    return;
  }

  el.packageMeta.textContent = [
    `Folder: ${pkg.folder}`,
    `JSON files: ${(pkg.json_paths || []).length}`,
    `Markdown files: ${(pkg.markdown_paths || []).length}`,
    `TXT files: ${(pkg.text_paths || []).length}`,
    `Full PDF: ${pkg.full_pdf_path || "N/A"}`,
    `Grouped page PDFs: ${pkg.page_pdf_count || 0}${pkg.page_range ? ` (pages ${pkg.page_range})` : ""}`,
  ].join("\n");

  try {
    const preview = await api("POST", "/api/loaders/preview", { package_id: packageId });
    el.markdownFrame.srcdoc = preview.markdown_html || "";
    el.jsonText.textContent = preview.json_text || "No JSON content available.";
    renderTables(preview.tables || []);
    renderPdfInfo(preview, pkg);
    setStatus(`Selected package: ${pkg.base_name}`);
  } catch (error) {
    systemMessage(`Preview error: ${error.message}`);
    setStatus("Preview failed.");
  }
}

async function loadFolder() {
  const path = el.pathInput.value.trim();
  if (!path) {
    setStatus("Enter a folder path first.");
    return;
  }
  try {
    const res = await api("POST", "/api/loaders/folder", { path });
    state.packages = res.packages || [];
    state.currentPackageId = state.packages.length ? state.packages[0].package_id : "";
    renderPackageList();
    if (state.currentPackageId) {
      await selectPackage(state.currentPackageId);
    }
    setStatus(`Loaded ${state.packages.length} package(s) from ${path}`);
  } catch (error) {
    systemMessage(`Load folder failed: ${error.message}`);
    setStatus("Load folder failed.");
  }
}

async function loadFile() {
  const path = el.pathInput.value.trim();
  if (!path) {
    setStatus("Enter a file path first.");
    return;
  }
  try {
    const res = await api("POST", "/api/loaders/file", { path });
    state.packages = res.packages || [];
    state.currentPackageId = state.packages.length ? state.packages[0].package_id : "";
    renderPackageList();
    if (state.currentPackageId) {
      await selectPackage(state.currentPackageId);
    }
    setStatus(`Loaded package from ${path}`);
  } catch (error) {
    systemMessage(`Load file failed: ${error.message}`);
    setStatus("Load file failed.");
  }
}

async function buildIndex() {
  try {
    const res = await api("POST", "/api/retrieval/index/build", {});
    systemMessage(`Indexed chunks: ${res.indexed_chunks}.`);
    setStatus(`Indexed ${res.indexed_chunks} chunk(s) from ${res.package_count} package(s).`);
  } catch (error) {
    systemMessage(`Index build failed: ${error.message}`);
    setStatus("Index build failed.");
  }
}

function renderSessionOptions() {
  el.sessionSelect.innerHTML = "";
  for (const s of state.sessions) {
    const opt = document.createElement("option");
    opt.value = s.session_id;
    opt.textContent = `${s.title} (${s.message_count})`;
    el.sessionSelect.appendChild(opt);
  }
  if (state.currentSessionId) {
    el.sessionSelect.value = state.currentSessionId;
  }
}

async function refreshSessions(loadCurrent = false) {
  const res = await api("GET", "/api/chat/sessions");
  state.sessions = res.sessions || [];
  if (!state.sessions.length) {
    state.currentSessionId = "";
    renderSessionOptions();
    return;
  }
  if (!state.currentSessionId || !state.sessions.some((s) => s.session_id === state.currentSessionId)) {
    state.currentSessionId = state.sessions[0].session_id;
  }
  renderSessionOptions();
  if (loadCurrent && state.currentSessionId) {
    await loadSession(state.currentSessionId);
  }
}

async function createSession() {
  const res = await api("POST", "/api/chat/session/new", { title: "Workflow Chat" });
  state.currentSessionId = res.session.session_id;
  resetChatView();
  await refreshSessions(false);
  setStatus(`Session created: ${res.session.title}`);
}

async function deleteCurrentSession() {
  if (!state.currentSessionId) {
    return;
  }
  await api("DELETE", `/api/chat/session/${encodeURIComponent(state.currentSessionId)}`);
  state.currentSessionId = "";
  resetChatView();
  await refreshSessions(true);
  setStatus("Session deleted.");
}

async function loadSession(sessionId) {
  if (!sessionId) {
    return;
  }
  const res = await api("GET", `/api/chat/session/${encodeURIComponent(sessionId)}`);
  state.currentSessionId = res.session.session_id;
  renderSessionOptions();
  resetChatView();

  let pendingUser = "";
  for (const msg of res.session.messages || []) {
    const role = (msg.role || "assistant").toLowerCase();
    if (role === "user") {
      pendingUser = msg.content || "";
      addMessage("user", msg.content || "", "You", msg.time || "");
      continue;
    }
    addMessage("assistant", msg.content || "", msg.model || msg.runtime || "Assistant", msg.time || "");
    state.chatRecords.push({
      timestamp: new Date().toISOString().slice(0, 19),
      mode: el.modeSelect.value,
      runtime: msg.runtime || "",
      model: msg.model || "",
      question: pendingUser,
      answer: msg.content || "",
      citations: msg.citations || "",
    });
    pendingUser = "";
  }
  setStatus(`Loaded session: ${res.session.title}`);
}

async function refreshModels() {
  const backend = el.backendSelect.value;
  const scanPath = encodeURIComponent(el.llamaScanPathInput.value.trim());
  const ollamaUrl = encodeURIComponent(el.ollamaUrlInput.value.trim());

  const res = await api(
    "GET",
    `/api/system/models/options?backend=${encodeURIComponent(backend)}&scan_path=${scanPath}&ollama_url=${ollamaUrl}`,
  );

  el.modelSelect.innerHTML = "";
  if (!(res.models || []).length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = res.message || "No models found";
    el.modelSelect.appendChild(opt);
    setStatus(res.message || "No models discovered.");
    return;
  }

  for (const m of res.models) {
    const opt = document.createElement("option");
    opt.value = m.path || m.name;
    opt.textContent = m.label || m.name;
    el.modelSelect.appendChild(opt);
  }

  const chosen = res.default_model || el.modelSelect.options[0].value;
  el.modelSelect.value = chosen;
  el.modelInput.value = chosen;
  if (res.scan_path) {
    el.llamaScanPathInput.value = res.scan_path;
  }
  setStatus(res.message || "Models refreshed.");
}

async function askQuestion() {
  const question = el.questionInput.value.trim();
  if (!question) {
    setStatus("Enter a question first.");
    return;
  }
  if (el.modeSelect.value === "direct" && !state.currentPackageId) {
    systemMessage("Direct mode requires selecting a package first.");
    return;
  }

  if (!state.currentSessionId) {
    await createSession();
  }

  addMessage("user", question, "You", nowTime());
  el.questionInput.value = "";

  const payload = {
    question,
    mode: el.modeSelect.value,
    package_id: state.currentPackageId || null,
    session_id: state.currentSessionId,
    llm_settings: {
      backend: el.backendSelect.value,
      model: el.modelInput.value.trim(),
      system_prompt: el.systemPromptInput.value,
      max_tokens: Number(el.maxTokensInput.value || 512),
      temperature: 0.2,
      ollama_url: el.ollamaUrlInput.value.trim(),
      llama_cli_path: el.llamaCliInput.value.trim(),
      context_limit: Number(el.contextLimitInput.value || 24000),
    },
  };

  try {
    const res = await api("POST", "/api/chat/ask", payload);
    state.currentSessionId = res.session_id || state.currentSessionId;
    addMessage("assistant", res.answer || "", res.assistant_name || res.model || res.runtime || "Assistant", nowTime());

    if ((res.citations || []).length) {
      const lines = res.citations.map(
        (c) => `- ${c.source_file} (${c.source_type}, score=${Number(c.score).toFixed(2)})`,
      );
      systemMessage(`Sources:\n${lines.join("\n")}`);
    }

    renderMetrics(res.metrics || {}, res.reasoning_chain || []);

    const citationText = (res.citations || [])
      .map((c) => `${c.source_file}:${c.source_type}:${Number(c.score).toFixed(2)}`)
      .join("; ");
    state.chatRecords.push({
      timestamp: new Date().toISOString().slice(0, 19),
      mode: res.mode,
      runtime: res.runtime,
      model: res.model,
      question,
      answer: res.answer,
      citations: citationText,
    });

    await refreshSessions(false);
    setStatus(
      `Answered via ${res.assistant_name || res.model || res.runtime} in ${Number(res.metrics?.total_ms || 0).toFixed(2)} ms.`,
    );
  } catch (error) {
    systemMessage(`Chat error: ${error.message}`);
    setStatus("Chat failed.");
  }
}

async function exportChat() {
  if (!state.chatRecords.length) {
    systemMessage("No chat history to export yet.");
    return;
  }
  const destination = el.exportPathInput.value.trim();
  if (!destination) {
    systemMessage("Set an export destination path first.");
    return;
  }
  try {
    const res = await api("POST", "/api/export/chat", { destination, records: state.chatRecords });
    systemMessage(res.message);
    setStatus(res.message);
  } catch (error) {
    systemMessage(`Export failed: ${error.message}`);
    setStatus("Export failed.");
  }
}

function clearChatView() {
  resetChatView();
  systemMessage("Local chat view cleared (session data remains saved). ");
}

async function browse(path = "") {
  try {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const res = await api("GET", `/api/system/browse${query}`);
    if (!path && res.default_root) {
      el.pathInput.value = res.default_root;
    }

    el.browsePanel.classList.remove("hidden");
    el.browsePanel.innerHTML = "";

    const current = document.createElement("div");
    current.style.padding = "8px";
    current.style.borderBottom = "1px solid #ecf0eb";
    current.textContent = `Current: ${res.current_path}`;
    el.browsePanel.appendChild(current);

    if (res.parent_path) {
      const up = document.createElement("button");
      up.className = "browse-item";
      up.textContent = "..";
      up.addEventListener("click", () => browse(res.parent_path));
      el.browsePanel.appendChild(up);
    }

    for (const entry of res.entries || []) {
      const btn = document.createElement("button");
      btn.className = "browse-item";
      btn.textContent = `${entry.is_dir ? "[DIR]" : "[FILE]"} ${entry.name}`;
      btn.addEventListener("click", () => {
        el.pathInput.value = entry.path;
        if (entry.is_dir) {
          browse(entry.path);
        }
      });
      el.browsePanel.appendChild(btn);
    }

    setStatus(`Browsing: ${res.current_path}`);
  } catch (error) {
    systemMessage(`Browse failed: ${error.message}`);
  }
}

function wireTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panes = document.querySelectorAll(".tab-content");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.remove("active"));
      panes.forEach((x) => x.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(`tab-${tab.dataset.tab}`);
      if (target) {
        target.classList.add("active");
      }
    });
  });
}

function wireEvents() {
  document.getElementById("browseBtn").addEventListener("click", () => browse(el.pathInput.value.trim()));
  document.getElementById("loadFolderBtn").addEventListener("click", loadFolder);
  document.getElementById("loadFileBtn").addEventListener("click", loadFile);
  document.getElementById("buildIndexBtn").addEventListener("click", buildIndex);

  document.getElementById("newSessionBtn").addEventListener("click", createSession);
  document.getElementById("deleteSessionBtn").addEventListener("click", deleteCurrentSession);
  el.sessionSelect.addEventListener("change", () => loadSession(el.sessionSelect.value));

  el.backendSelect.addEventListener("change", refreshModels);
  el.modelSelect.addEventListener("change", () => {
    el.modelInput.value = el.modelSelect.value;
  });
  document.getElementById("refreshModelsBtn").addEventListener("click", refreshModels);

  document.getElementById("askBtn").addEventListener("click", askQuestion);
  document.getElementById("clearChatBtn").addEventListener("click", clearChatView);
  document.getElementById("exportBtn").addEventListener("click", exportChat);

  el.questionInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      askQuestion();
    }
  });
}

async function init() {
  wireTabs();
  wireEvents();
  try {
    await api("GET", "/health");
    el.healthBadge.textContent = "Backend: online";
    systemMessage("Web app connected. Local runtimes only: Ollama or llama.cpp.");
    setStatus("Backend connected.");
  } catch (error) {
    el.healthBadge.textContent = "Backend: offline";
    systemMessage(`Backend health check failed: ${error.message}`);
    setStatus("Backend not reachable.");
    return;
  }

  await browse("");
  await refreshSessions(false);
  if (!state.currentSessionId) {
    await createSession();
  } else {
    await loadSession(state.currentSessionId);
  }
  await refreshModels();
}

init();
