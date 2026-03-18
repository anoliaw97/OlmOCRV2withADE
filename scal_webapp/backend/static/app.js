async function runExtraction() {
  const file = document.getElementById("pdfFile").files[0];
  if (!file) {
    alert("Upload a PDF first.");
    return;
  }

  const mode = document.getElementById("mode").value;
  const settings = {
    mode,
    use_case: document.getElementById("useCase").value,
    page_range: document.getElementById("pageRange")?.value || null,
    extraction_types: (document.getElementById("types")?.value || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean),
    prompt_profile: document.getElementById("promptProfile")?.value || "default",
    model_name: document.getElementById("modelName")?.value || "offline_heuristic",
    normalize: document.getElementById("normalize")?.checked ?? true,
    build_index: document.getElementById("buildIndex")?.checked ?? true,
  };

  const form = new FormData();
  form.append("file", file);
  form.append("settings_json", JSON.stringify(settings));

  const resp = await fetch("/api/extraction/run", { method: "POST", body: form });
  const data = await resp.json();
  alert(JSON.stringify(data, null, 2));
  await refreshReports();
  await refreshLogs();
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
        `<div><b>${r.file_name}</b> [${r.status}] 
        <a href="/api/extraction/report/${r.id}/export/json" target="_blank">JSON</a> |
        <a href="/api/extraction/report/${r.id}/export/xlsx" target="_blank">Excel</a> |
        <a href="/api/extraction/report/${r.id}/export/docx" target="_blank">Word</a></div>`
    )
    .join("");
}

async function askQuestion() {
  const payload = {
    question: document.getElementById("question").value,
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
  document.getElementById("debugCard").classList.toggle("hidden", mode !== "operator");
}

document.getElementById("runExtraction").addEventListener("click", runExtraction);
document.getElementById("refreshReports").addEventListener("click", refreshReports);
document.getElementById("askBtn").addEventListener("click", askQuestion);
document.getElementById("refreshLogs").addEventListener("click", refreshLogs);
document.getElementById("clearLogs").addEventListener("click", clearLogs);
document.getElementById("mode").addEventListener("change", toggleMode);

toggleMode();
refreshReports();
refreshLogs();
