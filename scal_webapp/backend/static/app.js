async function runExtraction() {
  const file = document.getElementById("pdfFile").files[0];
  if (!file) {
    alert("Upload a PDF first.");
    return;
  }

  const mode = document.getElementById("mode").value;
  const settings = {
    mode,
    use_case: "full_extraction",
    page_range: document.getElementById("pageRange")?.value || null,
    extraction_types: (document.getElementById("types")?.value || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean),
    prompt_profile: document.getElementById("promptProfile")?.value || "default",
    prompt_text: document.getElementById("defaultPrompt")?.value || null,
    model_name: document.getElementById("modelName")?.value || "offline_heuristic",
    normalize: document.getElementById("normalize")?.checked ?? true,
    build_index: document.getElementById("buildIndex")?.checked ?? true,
  };

  const form = new FormData();
  form.append("file", file);
  form.append("settings_json", JSON.stringify(settings));

  const resp = await fetch("/api/extraction/run", { method: "POST", body: form });
  const data = await resp.json();
  document.getElementById("resultOut").textContent = JSON.stringify(data, null, 2);
  await refreshReports();
  await refreshLogs();
}

async function importJson() {
  const file = document.getElementById("jsonFile").files[0];
  if (!file) {
    alert("Select a JSON file first.");
    return;
  }
  const form = new FormData();
  form.append("json_file", file);
  const resp = await fetch("/api/extraction/import-json", { method: "POST", body: form });
  const data = await resp.json();
  document.getElementById("resultOut").textContent = JSON.stringify(data, null, 2);
  await refreshReports();
  await refreshLogs();
}

async function loadLlms() {
  const resp = await fetch("/api/models/load-llm", { method: "POST" });
  const data = await resp.json();
  document.getElementById("resultOut").textContent = JSON.stringify(data, null, 2);
  await refreshModelStatus();
}

async function loadVlm() {
  const resp = await fetch("/api/models/load-vlm", { method: "POST" });
  const data = await resp.json();
  document.getElementById("resultOut").textContent = JSON.stringify(data, null, 2);
  await refreshModelStatus();
}

async function refreshModelStatus() {
  const resp = await fetch("/api/models/status");
  const data = await resp.json();
  document.getElementById("modelStatus").textContent =
    `VLM: ${data.vlm_loaded ? "loaded" : "not loaded"} | LLM: ${data.llm_loaded ? "loaded" : "not loaded"}`;
  const promptEl = document.getElementById("defaultPrompt");
  if (promptEl && !promptEl.value) {
    promptEl.value = data.default_prompt || "";
  }
}

async function refreshSavedPrompts() {
  const resp = await fetch("/api/chat/prompts");
  const data = await resp.json();
  const prompts = data.prompts || [];
  const select = document.getElementById("savedPromptSelect");
  select.innerHTML = "";
  prompts.forEach((p, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = p.name;
    opt.dataset.promptText = p.text;
    select.appendChild(opt);
  });
}

function loadSavedPromptToTextarea() {
  const select = document.getElementById("savedPromptSelect");
  const idx = Number(select.value || 0);
  const opt = select.options[idx];
  if (!opt) return;
  document.getElementById("useCasePrompt").value = opt.dataset.promptText || "";
}

async function saveCurrentPrompt() {
  const name = (document.getElementById("savePromptName").value || "").trim();
  const text = (document.getElementById("useCasePrompt").value || "").trim();
  if (!name || !text) {
    alert("Prompt name and prompt text are required.");
    return;
  }
  const resp = await fetch("/api/chat/prompts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, text }),
  });
  const data = await resp.json();
  document.getElementById("resultOut").textContent = JSON.stringify(data, null, 2);
  await refreshSavedPrompts();
}

async function refreshReports() {
  const resp = await fetch("/api/extraction/reports");
  const rows = await resp.json();
  const el = document.getElementById("reports");
  if (!rows.length) {
    el.innerHTML = "No reports yet.";
    return;
  }
  el.innerHTML = rows
    .map(
      (r) =>
        `<div class="report-item"><b>${r.file_name}</b> [${r.status}] 
        <a href="/api/extraction/report/${r.id}/export/json" target="_blank">JSON</a> |
        <a href="/api/extraction/report/${r.id}/export/xlsx" target="_blank">Excel</a> |
        <a href="/api/extraction/report/${r.id}/export/docx" target="_blank">Word</a></div>`
    )
    .join("");
}

async function askQuestion() {
  const payload = {
    question: document.getElementById("question").value,
    use_case_prompt: document.getElementById("useCasePrompt").value || null,
    file_name: document.getElementById("filterFile").value || null,
    extraction_type: document.getElementById("filterType").value || null,
    sample_id: document.getElementById("filterSample").value || null,
  };
  const resp = await fetch("/api/chat/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  document.getElementById("chatOut").textContent = JSON.stringify(data, null, 2);
}

async function refreshLogs() {
  const resp = await fetch("/api/chat/logs");
  const rows = await resp.json();
  document.getElementById("logs").textContent = JSON.stringify(rows, null, 2);
}

async function clearLogs() {
  await fetch("/api/chat/logs", { method: "DELETE" });
  await refreshLogs();
}

function toggleMode() {
  const mode = document.getElementById("mode").value;
  document.getElementById("operatorPanel").classList.toggle("hidden", mode !== "operator");
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      document.getElementById(target).classList.add("active");
    });
  });
}

document.getElementById("runExtraction").addEventListener("click", runExtraction);
document.getElementById("importJson").addEventListener("click", importJson);
document.getElementById("loadLlmBtn").addEventListener("click", loadLlms);
document.getElementById("loadLlmTopBtn").addEventListener("click", loadLlms);
document.getElementById("loadVlmBtn").addEventListener("click", loadVlm);
document.getElementById("refreshReports").addEventListener("click", refreshReports);
document.getElementById("askBtn").addEventListener("click", askQuestion);
document.getElementById("loadSavedPromptBtn").addEventListener("click", loadSavedPromptToTextarea);
document.getElementById("savePromptBtn").addEventListener("click", saveCurrentPrompt);
document.getElementById("refreshLogs").addEventListener("click", refreshLogs);
document.getElementById("clearLogs").addEventListener("click", clearLogs);
document.getElementById("mode").addEventListener("change", toggleMode);

initTabs();
toggleMode();
refreshReports();
refreshLogs();
refreshModelStatus();
refreshSavedPrompts();
