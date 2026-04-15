const state = {
  packages: [],
  currentPackageId: null,
  chatRecords: [],
  sessions: [],
  currentSessionId: "",
};

const healthBadge = document.getElementById("healthBadge");
const statusLine = document.getElementById("statusLine");
const pathInput = document.getElementById("pathInput");
const browsePanel = document.getElementById("browsePanel");
const packageList = document.getElementById("packageList");
const packageMeta = document.getElementById("packageMeta");
const markdownFrame = document.getElementById("markdownFrame");
const tablesTab = document.getElementById("tab-tables");
const jsonText = document.getElementById("jsonText");
const pdfPathText = document.getElementById("pdfPathText");
const pdfOpenLink = document.getElementById("pdfOpenLink");
const pagePdfList = document.getElementById("pagePdfList");

const sessionSelect = document.getElementById("sessionSelect");
const modeSelect = document.getElementById("modeSelect");
const backendSelect = document.getElementById("backendSelect");
const modelInput = document.getElementById("modelInput");
const ollamaUrlInput = document.getElementById("ollamaUrlInput");
const modelSelect = document.getElementById("modelSelect");
const llamaScanPathInput = document.getElementById("llamaScanPathInput");
const llamaCliInput = document.getElementById("llamaCliInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const contextLimitInput = document.getElementById("contextLimitInput");
const systemPromptInput = document.getElementById("systemPromptInput");
const questionInput = document.getElementById("questionInput");
const exportPathInput = document.getElementById("exportPathInput");
const chatLog = document.getElementById("chatLog");
const chatMetrics = document.getElementById("chatMetrics");

function setStatus(text) {
  statusLine.textContent = text;
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
  const text = await response.text();
  let data = {};
  if (text.trim()) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Invalid API response from ${path}`);
    }
  }

  if (!response.ok) {
    throw new Error(data.detail || `${response.status} ${response.statusText}`);
  }
  return data;
}

function addChatMessage(kind, text, label = "", timestamp = "") {
  const block = document.createElement("div");
  block.className = `msg ${kind}`;

  const defaultLabel = kind === "user" ? "You" : kind === "assistant" ? "Assistant" : "System";
  const header = `${label || defaultLabel}${timestamp ? ` [${timestamp}]` : ""}`;

  block.innerHTML =
    `<div class="msg-header"><strong>${escapeHtml(header)}</strong></div>` +
    `<div class="msg-body">${escapeHtml(text).replaceAll("\n", "<br>")}</div>`;

  chatLog.appendChild(block);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addSystemMessage(text) {
  addChatMessage("system", text, "System", nowTime());
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
    lines.push("", "Reasoning chain:", ...reasoningChain.map((item) => `- ${item}`));
  }
  chatMetrics.textContent = lines.join("\n");
}

function renderPackageList() {
  packageList.innerHTML = "";
  if (!state.packages.length) {
    packageList.innerHTML = "<div style='padding:8px'>No packages loaded.</div>";
    return;
  }

  for (const pkg of state.packages) {
    const btn = document.createElement("button");
    btn.textContent = `${pkg.base_name} [${(pkg.tokens || []).join(", ") || "EMPTY"}]`;
    if (pkg.package_id === state.currentPackageId) {
      btn.classList.add("active");
    }
    btn.addEventListener("click", () => selectPackage(pkg.package_id));
    packageList.appendChild(btn);
  }
}

function renderPdfInfo(preview, selected) {
  const fullPdfPath = preview.full_pdf_path || selected.full_pdf_path || "";
  const pdfPath = preview.pdf_path || selected.pdf_path || "";
  const pagePdfs = preview.page_pdf_paths || selected.page_pdf_paths || [];

  if (fullPdfPath) {
    pdfPathText.textContent = `Full PDF available: ${fullPdfPath}`;
  } else if (pdfPath) {
    pdfPathText.textContent = `PDF page set grouped under report: preview path ${pdfPath}`;
  } else {
    pdfPathText.textContent = "No PDF in selected package.";
  }

  const openTarget = fullPdfPath || pdfPath;
  if (openTarget) {
    pdfOpenLink.classList.remove("hidden");
    pdfOpenLink.href = `file:///${openTarget.replaceAll("\\", "/")}`;
  } else {
    pdfOpenLink.classList.add("hidden");
    pdfOpenLink.removeAttribute("href");
  }

  pagePdfList.innerHTML = "";
  if (!pagePdfs.length) {
    pagePdfList.textContent = "No grouped page-PDF set detected.";
    return;
  }
  for (const path of pagePdfs.slice(0, 60)) {
    const row = document.createElement("div");
    row.className = "item";
    row.textContent = path;
    pagePdfList.appendChild(row);
  }
}

function renderTables(tables) {
  tablesTab.innerHTML = "";
  if (!tables.length) {
    tablesTab.textContent = "No tables detected in JSON/Markdown for this package.";
    return;
  }

  for (const table of tables) {
    const card = document.createElement("section");
    card.className = "table-card";

    const header = document.createElement("h4");
    header.textContent = table.title;
    card.appendChild(header);

    if ((table.headers || []).length && (table.rows || []).length) {
      const htmlTable = document.createElement("table");
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      for (const h of table.headers) {
        const th = document.createElement("th");
        th.textContent = h;
        hr.appendChild(th);
      }
      thead.appendChild(hr);
      htmlTable.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const row of table.rows) {
        const tr = document.createElement("tr");
        for (const cell of row) {
          const td = document.createElement("td");
          td.textContent = cell;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      htmlTable.appendChild(tbody);
      card.appendChild(htmlTable);
    } else {
      const pre = document.createElement("pre");
      pre.textContent = table.raw_text || "No structured rows found.";
      card.appendChild(pre);
    }

    tablesTab.appendChild(card);
  }
}

async function selectPackage(packageId) {
  state.currentPackageId = packageId;
  renderPackageList();
  const selected = state.packages.find((item) => item.package_id === packageId);
  if (!selected) {
    return;
  }

  packageMeta.textContent = [
    `Folder: ${selected.folder}`,
    `JSON files: ${(selected.json_paths || []).length}`,
    `Markdown files: ${(selected.markdown_paths || []).length}`,
    `TXT files: ${(selected.text_paths || []).length}`,
    `Full PDF: ${selected.full_pdf_path || "N/A"}`,
    `Grouped page PDFs: ${selected.page_pdf_count || 0}${selected.page_range ? ` (pages ${selected.page_range})` : ""}`,
  ].join("\n");

  try {
    const preview = await api("POST", "/api/loaders/preview", { package_id: packageId });
    markdownFrame.srcdoc = preview.markdown_html || "";
    jsonText.textContent = preview.json_text || "No JSON content available.";
    renderTables(preview.tables || []);
    renderPdfInfo(preview, selected);
    setStatus(`Selected package: ${selected.base_name}`);
  } catch (error) {
    addSystemMessage(`Preview error: ${error.message}`);
    setStatus("Preview failed.");
  }
}

async function loadFolder() {
  const path = pathInput.value.trim();
  if (!path) {
    setStatus("Enter a folder path first.");
    return;
  }
  try {
    const result = await api("POST", "/api/loaders/folder", { path });
    state.packages = result.packages || [];
    state.currentPackageId = state.packages.length ? state.packages[0].package_id : null;
    renderPackageList();
    if (state.currentPackageId) {
      await selectPackage(state.currentPackageId);
    }
    setStatus(`Loaded ${state.packages.length} package(s) from ${path}`);
  } catch (error) {
    addSystemMessage(`Load folder failed: ${error.message}`);
    setStatus("Load folder failed.");
  }
}

async function loadFile() {
  const path = pathInput.value.trim();
  if (!path) {
    setStatus("Enter a file path first.");
    return;
  }
  try {
    const result = await api("POST", "/api/loaders/file", { path });
    state.packages = result.packages || [];
    state.currentPackageId = state.packages.length ? state.packages[0].package_id : null;
    renderPackageList();
    if (state.currentPackageId) {
      await selectPackage(state.currentPackageId);
    }
    setStatus(`Loaded package from ${path}`);
  } catch (error) {
    addSystemMessage(`Load file failed: ${error.message}`);
    setStatus("Load file failed.");
  }
}

async function buildIndex() {
  try {
    const result = await api("POST", "/api/retrieval/index/build", {});
    addSystemMessage(`Indexed chunks: ${result.indexed_chunks}.`);
    setStatus(`Indexed ${result.indexed_chunks} chunk(s) from ${result.package_count} package(s).`);
  } catch (error) {
    addSystemMessage(`Index build failed: ${error.message}`);
    setStatus("Index build failed.");
  }
}

async function loadSessions() {
  try {
    const result = await api("GET", "/api/chat/sessions");
    state.sessions = result.sessions || [];
    renderSessionOptions();
    if (!state.sessions.length) {
      await createSession("Workflow Chat");
      return;
    }
    if (!state.currentSessionId || !state.sessions.some((s) => s.session_id === state.currentSessionId)) {
      state.currentSessionId = state.sessions[0].session_id;
    }
    sessionSelect.value = state.currentSessionId;
    await loadSession(state.currentSessionId);
  } catch (error) {
    addSystemMessage(`Session load failed: ${error.message}`);
  }
}

function renderSessionOptions() {
  sessionSelect.innerHTML = "";
  for (const session of state.sessions) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = `${session.title} (${session.message_count})`;
    sessionSelect.appendChild(option);
  }
}

async function createSession(title = "") {
  try {
    const result = await api("POST", "/api/chat/session/new", { title });
    const created = result.session;
    state.currentSessionId = created.session_id;
    await loadSessions();
    setStatus(`Session created: ${created.title}`);
  } catch (error) {
    addSystemMessage(`Create session failed: ${error.message}`);
  }
}

async function deleteCurrentSession() {
  if (!state.currentSessionId) {
    return;
  }
  try {
    await api("DELETE", `/api/chat/session/${encodeURIComponent(state.currentSessionId)}`);
    chatLog.innerHTML = "";
    state.chatRecords = [];
    state.currentSessionId = "";
    chatMetrics.textContent = "No response metrics yet.";
    await loadSessions();
    setStatus("Session deleted.");
  } catch (error) {
    addSystemMessage(`Delete session failed: ${error.message}`);
  }
}

async function loadSession(sessionId) {
  if (!sessionId) {
    return;
  }
  try {
    const result = await api("GET", `/api/chat/session/${encodeURIComponent(sessionId)}`);
    const session = result.session;
    state.currentSessionId = session.session_id;
    sessionSelect.value = session.session_id;

    chatLog.innerHTML = "";
    state.chatRecords = [];

    let pendingUser = "";
    for (const msg of session.messages || []) {
      const role = (msg.role || "assistant").toLowerCase();
      if (role === "user") {
        pendingUser = msg.content || "";
        addChatMessage("user", msg.content || "", "You", msg.time || "");
        continue;
      }

      const label = msg.model || msg.runtime || "Assistant";
      addChatMessage("assistant", msg.content || "", label, msg.time || "");
      state.chatRecords.push({
        timestamp: new Date().toISOString().slice(0, 19),
        mode: modeSelect.value,
        runtime: msg.runtime || "",
        model: msg.model || "",
        question: pendingUser,
        answer: msg.content || "",
        citations: msg.citations || "",
      });
      pendingUser = "";
    }

    setStatus(`Loaded session: ${session.title}`);
  } catch (error) {
    addSystemMessage(`Session open failed: ${error.message}`);
  }
}

async function refreshModels() {
  const backend = backendSelect.value;
  const backendQuery = backend === "llamacpp" ? "llamacpp" : "ollama";
  const ollamaUrl = encodeURIComponent(ollamaUrlInput.value.trim());
  const scanPath = encodeURIComponent(llamaScanPathInput.value.trim());

  try {
    const result = await api(
      "GET",
      `/api/system/models/options?backend=${encodeURIComponent(backendQuery)}&ollama_url=${ollamaUrl}&scan_path=${scanPath}`,
    );

    modelSelect.innerHTML = "";
    const models = result.models || [];
    if (!models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = result.message || "No models discovered.";
      modelSelect.appendChild(opt);
      setStatus(result.message || "No models found.");
      return;
    }

    for (const model of models) {
      const opt = document.createElement("option");
      opt.value = model.path || model.name;
      opt.textContent = model.label || model.name;
      modelSelect.appendChild(opt);
    }

    const chosen = result.default_model || modelSelect.options[0].value;
    modelSelect.value = chosen;
    modelInput.value = chosen;
    if (result.scan_path) {
      llamaScanPathInput.value = result.scan_path;
    }

    setStatus(result.message || `Loaded ${models.length} model option(s).`);
  } catch (error) {
    addSystemMessage(`Model discovery failed: ${error.message}`);
    setStatus("Model discovery failed.");
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("Enter a question first.");
    return;
  }

  const mode = modeSelect.value;
  if (mode === "direct" && !state.currentPackageId) {
    addSystemMessage("Direct mode requires selecting a package first.");
    return;
  }

  addChatMessage("user", question, "You", nowTime());
  questionInput.value = "";

  const payload = {
    question,
    mode,
    package_id: state.currentPackageId,
    session_id: state.currentSessionId || null,
    llm_settings: {
      backend: backendSelect.value,
      model: modelInput.value.trim(),
      system_prompt: systemPromptInput.value,
      max_tokens: Number(maxTokensInput.value || 512),
      temperature: 0.2,
      ollama_url: ollamaUrlInput.value.trim(),
      llama_cli_path: llamaCliInput.value.trim(),
      context_limit: Number(contextLimitInput.value || 24000),
    },
  };

  try {
    const response = await api("POST", "/api/chat/ask", payload);
    state.currentSessionId = response.session_id || state.currentSessionId;
    await loadSessions();

    addChatMessage(
      "assistant",
      response.answer || "",
      response.assistant_name || response.model || response.runtime || "Assistant",
      nowTime(),
    );

    if ((response.citations || []).length) {
      const lines = response.citations.map(
        (item) => `- ${item.source_file} (${item.source_type}, score=${Number(item.score).toFixed(2)})`,
      );
      addSystemMessage(`Sources:\n${lines.join("\n")}`);
    }

    renderMetrics(response.metrics || {}, response.reasoning_chain || []);

    const citationText = (response.citations || [])
      .map((item) => `${item.source_file}:${item.source_type}:${Number(item.score).toFixed(2)}`)
      .join("; ");

    state.chatRecords.push({
      timestamp: new Date().toISOString().slice(0, 19),
      mode: response.mode,
      runtime: response.runtime,
      model: response.model,
      question,
      answer: response.answer,
      citations: citationText,
    });

    setStatus(
      `Answered via ${response.assistant_name || response.model || response.runtime} in ${Number(response.metrics?.total_ms || 0).toFixed(2)} ms.`,
    );
  } catch (error) {
    addSystemMessage(`Chat error: ${error.message}`);
    setStatus("Chat failed.");
  }
}

async function exportChat() {
  if (!state.chatRecords.length) {
    addSystemMessage("No chat history to export yet.");
    return;
  }

  const destination = exportPathInput.value.trim();
  if (!destination) {
    addSystemMessage("Set an export destination path first.");
    return;
  }

  try {
    const result = await api("POST", "/api/export/chat", { destination, records: state.chatRecords });
    addSystemMessage(result.message);
    setStatus(result.message);
  } catch (error) {
    addSystemMessage(`Export failed: ${error.message}`);
    setStatus("Export failed.");
  }
}

function clearChat() {
  chatLog.innerHTML = "";
  state.chatRecords = [];
  renderMetrics({}, []);
  addSystemMessage("Local chat view cleared (session history is preserved). ");
}

async function browse(path = "") {
  try {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const result = await api("GET", `/api/system/browse${query}`);

    if (!path && result.default_root) {
      pathInput.value = result.default_root;
    }

    browsePanel.classList.remove("hidden");
    browsePanel.innerHTML = "";

    const current = document.createElement("div");
    current.style.padding = "8px";
    current.style.borderBottom = "1px solid #ecf0eb";
    current.textContent = `Current: ${result.current_path}`;
    browsePanel.appendChild(current);

    if (result.parent_path) {
      const up = document.createElement("button");
      up.className = "browse-item";
      up.textContent = "..";
      up.addEventListener("click", () => browse(result.parent_path));
      browsePanel.appendChild(up);
    }

    for (const entry of result.entries || []) {
      const btn = document.createElement("button");
      btn.className = "browse-item";
      btn.textContent = `${entry.is_dir ? "[DIR]" : "[FILE]"} ${entry.name}`;
      btn.addEventListener("click", () => {
        pathInput.value = entry.path;
        if (entry.is_dir) {
          browse(entry.path);
        }
      });
      browsePanel.appendChild(btn);
    }

    setStatus(`Browsing: ${result.current_path}`);
  } catch (error) {
    addSystemMessage(`Browse failed: ${error.message}`);
  }
}

function wireTabs() {
  const tabs = document.querySelectorAll(".tab");
  const contents = document.querySelectorAll(".tab-content");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => item.classList.remove("active"));
      contents.forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(`tab-${tab.dataset.tab}`);
      if (target) {
        target.classList.add("active");
      }
    });
  });
}

function wireEvents() {
  document.getElementById("browseBtn").addEventListener("click", () => browse(pathInput.value.trim()));
  document.getElementById("loadFolderBtn").addEventListener("click", loadFolder);
  document.getElementById("loadFileBtn").addEventListener("click", loadFile);
  document.getElementById("buildIndexBtn").addEventListener("click", buildIndex);

  document.getElementById("newSessionBtn").addEventListener("click", () => createSession("Workflow Chat"));
  document.getElementById("deleteSessionBtn").addEventListener("click", deleteCurrentSession);
  sessionSelect.addEventListener("change", () => loadSession(sessionSelect.value));

  backendSelect.addEventListener("change", refreshModels);
  modelSelect.addEventListener("change", () => {
    modelInput.value = modelSelect.value;
  });
  document.getElementById("refreshModelsBtn").addEventListener("click", refreshModels);

  document.getElementById("askBtn").addEventListener("click", askQuestion);
  document.getElementById("clearChatBtn").addEventListener("click", clearChat);
  document.getElementById("exportBtn").addEventListener("click", exportChat);
}

async function init() {
  wireTabs();
  wireEvents();

  try {
    await api("GET", "/health");
    healthBadge.textContent = "Backend: online";
    setStatus("Backend connected.");
    addSystemMessage("Web app connected. Load extracted outputs to begin.");
  } catch (error) {
    healthBadge.textContent = "Backend: offline";
    addSystemMessage(`Backend health check failed: ${error.message}`);
    setStatus("Backend not reachable.");
    return;
  }

  await browse("");
  await loadSessions();
  await refreshModels();
}

init();
