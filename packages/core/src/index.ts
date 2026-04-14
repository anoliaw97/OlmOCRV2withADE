export type PipelineMode = "fast" | "balanced" | "deep"

export type PipelineResult = {
  answer: string
  mode: PipelineMode
  latency_ms: number
}

export type AppInfo = {
  app: string
  version: string
  stack: string
}

export type OcrIngestionResult = {
  file_path: string
  extractor: string
  text: string
  preview: string
  char_count: number
  elapsed_ms: number
}

export type LlamaRunResult = {
  answer: string
  model_path: string
  runtime: string
  latency_ms: number
}
