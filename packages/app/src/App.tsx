import { createSignal, For, onMount } from "solid-js"
import { invoke } from "@tauri-apps/api/core"
import type {
  AppInfo,
  LlamaRunResult,
  OcrIngestionResult,
  PipelineMode,
  PipelineResult,
} from "@ocrpetro/core"

type Mode = PipelineMode

type Entry = {
  role: "user" | "assistant"
  text: string
  ts: string
  mode?: Mode
  latency?: number
}

function nowLabel() {
  return new Date().toLocaleTimeString()
}

export default function App() {
  const [info, setInfo] = createSignal<AppInfo | null>(null)
  const [mode, setMode] = createSignal<Mode>("balanced")
  const [prompt, setPrompt] = createSignal("")
  const [busy, setBusy] = createSignal(false)
  const [error, setError] = createSignal("")
  const [timeline, setTimeline] = createSignal<Entry[]>([])

  const [ocrPath, setOcrPath] = createSignal("")
  const [ocrLanguage, setOcrLanguage] = createSignal("eng")
  const [tesseractPath, setTesseractPath] = createSignal("")
  const [ocrText, setOcrText] = createSignal("")
  const [ocrMeta, setOcrMeta] = createSignal("")

  const [modelPath, setModelPath] = createSignal("")
  const [llamaPath, setLlamaPath] = createSignal("")
  const [systemPrompt, setSystemPrompt] = createSignal("You are a concise OCR analysis assistant.")

  onMount(async () => {
    try {
      const res = await invoke<AppInfo>("app_info")
      setInfo(res)
    } catch (e) {
      setError(String(e))
    }
  })

  async function runPipeline() {
    const q = prompt().trim()
    if (!q || busy()) return

    setError("")
    setTimeline((prev) => [...prev, { role: "user", text: q, ts: nowLabel() }])
    setPrompt("")
    setBusy(true)

    try {
      const out = await invoke<PipelineResult>("run_pipeline", {
        question: q,
        mode: mode(),
      })
      setTimeline((prev) => [
        ...prev,
        {
          role: "assistant",
          text: out.answer,
          ts: nowLabel(),
          mode: out.mode,
          latency: out.latency_ms,
        },
      ])
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function ingestOcr() {
    const file = ocrPath().trim()
    if (!file || busy()) return

    setError("")
    setBusy(true)
    try {
      const out = await invoke<OcrIngestionResult>("ingest_ocr", {
        file_path: file,
        language: ocrLanguage().trim() || undefined,
        tesseract_path: tesseractPath().trim() || undefined,
      })

      setOcrText(out.text)
      setOcrMeta(`${out.extractor} | ${out.char_count} chars | ${out.elapsed_ms} ms`)
      setTimeline((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `OCR ready from ${out.file_path}\n${out.preview}`,
          ts: nowLabel(),
        },
      ])
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function runLlamaCpp() {
    const q = prompt().trim()
    if (!q || busy()) return

    setError("")
    setTimeline((prev) => [...prev, { role: "user", text: q, ts: nowLabel() }])
    setPrompt("")
    setBusy(true)

    try {
      const out = await invoke<LlamaRunResult>("run_llama_cpp", {
        prompt: q,
        model_path: modelPath().trim(),
        system_prompt: systemPrompt().trim() || undefined,
        context: ocrText().trim() || undefined,
        max_tokens: 384,
        temperature: 0.2,
        llama_cli_path: llamaPath().trim() || undefined,
      })

      setTimeline((prev) => [
        ...prev,
        {
          role: "assistant",
          text: out.answer,
          ts: nowLabel(),
          latency: out.latency_ms,
        },
      ])
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div class="shell">
      <aside class="left">
        <div class="panel-title">Workspace</div>
        <div class="card">
          <div class="small">Project</div>
          <div class="value">OCRLLMVLMMLPETROAFRO</div>
          <div class="small gap">Stack</div>
          <div class="value mono">Tauri + Solid + TS</div>
        </div>

        <div class="card">
          <div class="small">Run Mode</div>
          <select value={mode()} onInput={(e) => setMode(e.currentTarget.value as Mode)}>
            <option value="fast">Fast</option>
            <option value="balanced">Balanced</option>
            <option value="deep">Deep</option>
          </select>
        </div>

        <div class="card">
          <div class="small">OCR Input Path</div>
          <textarea
            class="short"
            placeholder="C:\\data\\scan.png"
            value={ocrPath()}
            onInput={(e) => setOcrPath(e.currentTarget.value)}
          />
          <div class="small gap">OCR Language</div>
          <input value={ocrLanguage()} onInput={(e) => setOcrLanguage(e.currentTarget.value)} />
          <div class="small gap">Tesseract Path (optional)</div>
          <input
            value={tesseractPath()}
            onInput={(e) => setTesseractPath(e.currentTarget.value)}
            placeholder="C:\\tools\\tesseract.exe"
          />
          <button onClick={() => void ingestOcr()} disabled={busy()}>
            {busy() ? "Working..." : "Ingest OCR"}
          </button>
          {ocrMeta() && <div class="small gap mono">{ocrMeta()}</div>}
        </div>
      </aside>

      <main class="center">
        <header class="topline">
          <h1>PetroAfro AI Desktop</h1>
          <div class="chips">
            <span class="chip">desktop</span>
            <span class="chip">ocr + llama.cpp</span>
          </div>
        </header>

        <section class="timeline" aria-live="polite">
          <For each={timeline()}>
            {(entry) => (
              <article class={`bubble ${entry.role}`}>
                <div class="meta">
                  <span>{entry.role.toUpperCase()}</span>
                  <span>{entry.ts}</span>
                  {entry.mode && <span>mode:{entry.mode}</span>}
                  {entry.latency != null && <span>{entry.latency} ms</span>}
                </div>
                <p>{entry.text}</p>
              </article>
            )}
          </For>
          {!timeline().length && (
            <div class="empty">
              Start by ingesting OCR content, then prompt llama.cpp with extraction or reasoning tasks.
            </div>
          )}
        </section>

        <section class="composer">
          <textarea
            placeholder="Ask the desktop pipeline..."
            value={prompt()}
            onInput={(e) => setPrompt(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                void runLlamaCpp()
              }
            }}
          />
          <div class="button-row">
            <button onClick={() => void runLlamaCpp()} disabled={busy()}>
              {busy() ? "Running..." : "Run llama.cpp"}
            </button>
            <button class="secondary" onClick={() => void runPipeline()} disabled={busy()}>
              {busy() ? "Running..." : "Run mock pipeline"}
            </button>
          </div>
        </section>
      </main>

      <aside class="right">
        <div class="panel-title">Runtime</div>
        <div class="card">
          <div class="small">App Info</div>
          <div class="value">{info()?.app ?? "loading"}</div>
          <div class="small gap">Version</div>
          <div class="value mono">{info()?.version ?? "-"}</div>
          <div class="small gap">Engine</div>
          <div class="value mono">{info()?.stack ?? "-"}</div>
        </div>

        <div class="card">
          <div class="small">llama.cpp Model Path</div>
          <textarea
            class="short"
            value={modelPath()}
            onInput={(e) => setModelPath(e.currentTarget.value)}
            placeholder="C:\\models\\model.gguf"
          />
          <div class="small gap">llama-cli Path (optional)</div>
          <textarea
            class="short"
            value={llamaPath()}
            onInput={(e) => setLlamaPath(e.currentTarget.value)}
            placeholder="C:\\tools\\llama-cli.exe"
          />
          <div class="small gap">System Prompt</div>
          <textarea value={systemPrompt()} onInput={(e) => setSystemPrompt(e.currentTarget.value)} />
        </div>

        {error() && (
          <div class="card error">
            <div class="small">Error</div>
            <div class="value">{error()}</div>
          </div>
        )}
      </aside>
    </div>
  )
}
