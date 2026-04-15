const state = {
  packages: [],
  currentPackageId: null,
  chatRecords: [],
};

const healthBadge = document.getElementById("healthBadge");
const statusLine = document.getElementById("statusLine");
const pathInput = document.getElementById("pathInput");
const packageList = document.getElementById("packageList");
const packageMeta = document.getElementById("packageMeta");
const browsePanel = document.getElementById("browsePanel");
const markdownFrame = document.getElementById("markdownFrame");
const tablesTab = document.getElementById("tab-tables");
const jsonText = document.getElementById("jsonText");
const pdfPathText = document.getElementById("pdfPathText");
const pdfOpenLink = document.getElementById("pdfOpenLink");
const chatLog = document.getElementById("chatLog");

const modeSelect = document.getElementById("modeSelect");
const backendSelect = document.getElementById("backendSelect");
const modelInput = document.getElementById("modelInput");
const ollamaUrlInput = document.getElementById("ollamaUrlInput");
const llamaCliInput = document.getElementById("llamaCliInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const systemPromptInput = document.getElementById("systemPromptInput");
const questionInput = document.getElementById("questionInput");
const exportPathInput = document.getElementById("exportPathInput");

function setStatus(text) {
  statusLine.textContent = text;
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
    const detail = data.detail || `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
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

async function selectPackage(packageId) {
  state.currentPackageId = packageId;
  renderPackageList();
  setStatus(`Loading preview for package ${packageId}...`);

  const selected = state.packages.find((p) => p.package_id === packageId);
  if (!selected) {
    return;
  }

  packageMeta.textContent = [
    `Folder: ${selected.folder}`,
    `JSON: ${selected.json_path || "N/A"}`,
    `Markdown: ${selected.markdown_path || "N/A"}`,
    `TXT: ${selected.text_path || "N/A"}`,
    `PDF: ${selected.pdf_path || "N/A"}`,
  ].join("\n");

  try {
    const preview = await api("POST", "/api/loaders/preview", { package_id: packageId });
    markdownFrame.srcdoc = preview.markdown_html || "";
    jsonText.textContent = preview.json_text || "No JSON content available.";
    renderTables(preview.tables || []);
    renderPdfInfo(preview.pdf_path || "");
    setStatus(`Selected package: ${selected.base_name}`);
  } catch (error) {
    addSystemMessage(`Preview error: ${error.message}`);
    setStatus("Preview failed.");
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
      const t = document.createElement("table");
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      for (const h of table.headers) {
        const th = document.createElement("th");
        th.textContent = h;
        hr.appendChild(th);
      }
      thead.appendChild(hr);
      t.appendChild(thead);

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
      t.appendChild(tbody);
      card.appendChild(t);
    } else {
      const fallback = document.createElement("pre");
      fallback.textContent = table.raw_text || "No structured rows found.";
      card.appendChild(fallback);
    }

    tablesTab.appendChild(card);
  }
}

function renderPdfInfo(pdfPath) {
  if (!pdfPath) {
    pdfPathText.textContent = "No PDF in selected package.";
    pdfOpenLink.classList.add("hidden");
    pdfOpenLink.removeAttribute("href");
    return;
  }

  pdfPathText.textContent = `PDF available at: ${pdfPath}`;
  pdfOpenLink.classList.remove("hidden");
  pdfOpenLink.href = `file:///${pdfPath.replaceAll("\\", "/")}`;
}

function addChatMessage(kind, text) {
  const block = document.createElement("div");
  block.className = `msg ${kind}`;
  block.innerHTML = `<strong>${kind.toUpperCase()}</strong><br>${escapeHtml(text).replaceAll("\n", "<br>")}`;
  chatLog.appendChild(block);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addSystemMessage(text) {
  addChatMessage("system", text);
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

  addChatMessage("user", question);
  questionInput.value = "";

  const payload = {
    question,
    mode,
    package_id: state.currentPackageId,
    llm_settings: {
      backend: backendSelect.value,
      model: modelInput.value.trim(),
      system_prompt: systemPromptInput.value,
      max_tokens: Number(maxTokensInput.value || 512),
      temperature: 0.2,
      ollama_url: ollamaUrlInput.value.trim(),
      llama_cli_path: llamaCliInput.value.trim(),
    },
  };

  try {
    const response = await api("POST", "/api/chat/ask", payload);
    addChatMessage("assistant", response.answer || "");
    if ((response.citations || []).length) {
      const lines = response.citations.map((c) => `- ${c.source_file} (${c.source_type}, score=${Number(c.score).toFixed(2)})`);
      addSystemMessage(`Sources:\n${lines.join("\n")}`);
    }
    addSystemMessage(`Runtime: ${response.runtime} | Model: ${response.model || "not set"}`);

    const citationText = (response.citations || [])
      .map((c) => `${c.source_file}:${c.source_type}:${Number(c.score).toFixed(2)}`)
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
    setStatus("Question answered.");
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
    const response = await api("POST", "/api/export/chat", {
      destination,
      records: state.chatRecords,
    });
    addSystemMessage(response.message);
    setStatus(response.message);
  } catch (error) {
    addSystemMessage(`Export failed: ${error.message}`);
    setStatus("Export failed.");
  }
}

function clearChat() {
  state.chatRecords = [];
  chatLog.innerHTML = "";
  addSystemMessage("Chat cleared.");
  setStatus("Chat cleared.");
}

async function browse(path = "") {
  try {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const result = await api("GET", `/api/system/browse${query}`);
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
    setStatus("Browse failed.");
  }
}

function wireTabs() {
  const tabs = document.querySelectorAll(".tab");
  const contents = document.querySelectorAll(".tab-content");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      contents.forEach((c) => c.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(`tab-${tab.dataset.tab}`);
      if (target) {
        target.classList.add("active");
      }
    });
  });
}

async function init() {
  wireTabs();

  document.getElementById("loadFolderBtn").addEventListener("click", loadFolder);
  document.getElementById("loadFileBtn").addEventListener("click", loadFile);
  document.getElementById("buildIndexBtn").addEventListener("click", buildIndex);
  document.getElementById("askBtn").addEventListener("click", askQuestion);
  document.getElementById("clearChatBtn").addEventListener("click", clearChat);
  document.getElementById("exportBtn").addEventListener("click", exportChat);
  document.getElementById("browseBtn").addEventListener("click", () => browse(pathInput.value.trim()));

  try {
    await api("GET", "/health");
    healthBadge.textContent = "Backend: online";
    addSystemMessage("Web app connected. Load a folder or file to start.");
    setStatus("Backend connected.");
  } catch (error) {
    healthBadge.textContent = "Backend: offline";
    addSystemMessage(`Backend health check failed: ${error.message}`);
    setStatus("Backend not reachable.");
  }
}

init();
