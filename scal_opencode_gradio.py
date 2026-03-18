from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Mining\Downloads\Fine Tunining Datasets-20260318T052420Z-1-001\Fine Tunining Datasets\train"
)
PROMPT_STORE = Path("scal_saved_prompts.json")
INDEX_STORE = Path("scal_gradio_index")


DEFAULT_PROMPTS = [
    {
        "name": "Extract certain columns only",
        "text": (
            "You are a document analysis assistant.\n"
            "Use retrieved extracted evidence only.\n"
            "Return ONLY Sample ID, Porosity, Depth as valid JSON.\n"
            "Missing fields must be NULL. No extra text."
        ),
    },
    {
        "name": "Extract table based on keyword",
        "text": (
            "Find the table containing keyword '[keyword]' in retrieved evidence.\n"
            "Return full headers/rows as JSON array in original order.\n"
            "If missing: {\"no_table\": true}."
        ),
    },
    {
        "name": "Graph extraction",
        "text": (
            "From retrieved graph/table evidence, extract structured numeric values as JSON.\n"
            "No commentary. Numeric values only."
        ),
    },
]


def load_prompt_store() -> list[dict[str, str]]:
    if not PROMPT_STORE.exists():
        PROMPT_STORE.write_text(json.dumps(DEFAULT_PROMPTS, indent=2), encoding="utf-8")
    try:
        data = json.loads(PROMPT_STORE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return DEFAULT_PROMPTS


def save_prompt_store(prompts: list[dict[str, str]]) -> None:
    PROMPT_STORE.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_name(file_name: str) -> tuple[str | None, int | None, str]:
    m = re.match(r"^(.*)_page(\d+)\.(pdf|md|json)$", file_name, flags=re.IGNORECASE)
    if not m:
        return None, None, Path(file_name).suffix.lower().lstrip(".")
    return m.group(1), int(m.group(2)), m.group(3).lower()


def scan_dataset(root: Path) -> dict[str, dict[int, dict[str, Path]]]:
    docs: dict[str, dict[int, dict[str, Path]]] = {}
    if not root.exists():
        return docs
    for p in root.glob("*"):
        if not p.is_file():
            continue
        stem, page, ext = parse_name(p.name)
        if stem is None or page is None:
            continue
        docs.setdefault(stem, {}).setdefault(page, {})[ext] = p
    return docs


def doc_coverage(pages: dict[int, dict[str, Path]]) -> dict[str, Any]:
    if not pages:
        return {"pdf_pages": 0, "extracted_pages": 0, "missing_extraction_pages": []}
    all_pages = sorted(pages.keys())
    pdf_pages = [p for p in all_pages if "pdf" in pages[p]]
    extracted = [p for p in all_pages if "md" in pages[p] or "json" in pages[p]]
    missing = [p for p in pdf_pages if p not in extracted]
    return {
        "pdf_pages": len(pdf_pages),
        "extracted_pages": len(extracted),
        "missing_extraction_pages": missing,
    }


def _flatten_json(obj: Any) -> str:
    if isinstance(obj, dict):
        if "raw_response" in obj:
            return str(obj["raw_response"])
        return "\n".join(_flatten_json(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(_flatten_json(x) for x in obj)
    return str(obj)


def load_chunks(root: Path, doc_name: str, pages: dict[int, dict[str, Path]]) -> list[dict]:
    chunks = []
    for pg in sorted(pages.keys()):
        f = pages[pg]
        text = ""
        source = ""
        if "json" in f:
            source = f["json"].name
            try:
                obj = json.loads(f["json"].read_text(encoding="utf-8", errors="ignore"))
                text = _flatten_json(obj)
            except Exception:
                text = f["json"].read_text(encoding="utf-8", errors="ignore")
        elif "md" in f:
            source = f["md"].name
            text = f["md"].read_text(encoding="utf-8", errors="ignore")

        if text.strip():
            chunks.append({"text": text, "meta": {"doc": doc_name, "page": pg, "source": source}})
    return chunks


def build_index(namespace: str, chunks: list[dict]) -> str:
    INDEX_STORE.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts) if texts else vec.fit_transform([""])
    joblib.dump(vec, INDEX_STORE / f"{namespace}_vec.joblib")
    joblib.dump(mat, INDEX_STORE / f"{namespace}_mat.joblib")
    joblib.dump(texts, INDEX_STORE / f"{namespace}_texts.joblib")
    joblib.dump(metas, INDEX_STORE / f"{namespace}_metas.joblib")
    return f"Index built: {len(chunks)} chunks"


def load_index(namespace: str):
    try:
        vec = joblib.load(INDEX_STORE / f"{namespace}_vec.joblib")
        mat = joblib.load(INDEX_STORE / f"{namespace}_mat.joblib")
        texts = joblib.load(INDEX_STORE / f"{namespace}_texts.joblib")
        metas = joblib.load(INDEX_STORE / f"{namespace}_metas.joblib")
        return vec, mat, texts, metas
    except Exception:
        return None


def search(idx_obj, query: str, top_k: int = 6):
    vec, mat, texts, metas = idx_obj
    qv = vec.transform([query])
    sims = linear_kernel(qv, mat).flatten()
    order = sims.argsort()[::-1]
    out = []
    for i in order:
        score = float(sims[i])
        if score <= 0:
            continue
        out.append({"score": score, "text": texts[i], "meta": metas[i]})
        if len(out) >= top_k:
            break
    return out


def ask_llm(model_name: str, system_prompt: str, prompt: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda", dtype=torch.long)
    with torch.no_grad():
        out = mdl.generate(inp, max_new_tokens=700, temperature=0.2, do_sample=True, top_p=0.9)
    return tok.decode(out[0][inp.shape[1] :], skip_special_tokens=True).strip()


def summarize_coverage(data_root: str, doc_name: str):
    docs = scan_dataset(Path(data_root))
    pages = docs.get(doc_name, {})
    cov = doc_coverage(pages)
    msg = (
        f"PDF pages: {cov['pdf_pages']}\n"
        f"Extracted pages: {cov['extracted_pages']}\n"
        f"Missing extraction pages: {cov['missing_extraction_pages']}"
    )
    return msg


def check_uploaded_pdf_missing(uploaded_pdf_path: str, data_root: str, doc_name: str):
    if not uploaded_pdf_path:
        return "Upload a PDF first"
    docs = scan_dataset(Path(data_root))
    pages = docs.get(doc_name, {})
    extracted_pages = {p for p, flags in pages.items() if "md" in flags or "json" in flags}
    reader = PdfReader(uploaded_pdf_path)
    total = len(reader.pages)
    missing = [p for p in range(1, total + 1) if p not in extracted_pages]
    if not missing:
        return f"Already extracted fully. Total pages: {total}"
    return f"Missing extracted pages ({len(missing)}): {missing}"


def load_prompts_ui():
    prompts = load_prompt_store()
    names = [p["name"] for p in prompts]
    default_name = names[0] if names else ""
    default_text = prompts[0]["text"] if prompts else ""
    return gr.update(choices=names, value=default_name), default_text


def select_prompt(name: str):
    prompts = load_prompt_store()
    for p in prompts:
        if p["name"] == name:
            return p["text"]
    return ""


def save_prompt(name: str, text: str):
    if not name.strip() or not text.strip():
        return gr.update(), "Name and text required"
    prompts = load_prompt_store()
    hit = False
    for p in prompts:
        if p["name"] == name:
            p["text"] = text
            hit = True
            break
    if not hit:
        prompts.append({"name": name, "text": text})
    save_prompt_store(prompts)
    names = [p["name"] for p in prompts]
    return gr.update(choices=names, value=name), "Prompt saved"


def build_doc_index(data_root: str, doc_name: str):
    docs = scan_dataset(Path(data_root))
    if doc_name not in docs:
        return "Document not found"
    chunks = load_chunks(Path(data_root), doc_name, docs[doc_name])
    if not chunks:
        return "No extracted .md/.json chunks for selected document"
    namespace = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)
    return build_index(namespace, chunks)


def chat_fn(message: str, history: list, data_root: str, doc_name: str, prompt_text: str, model_name: str):
    namespace = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)
    idx = load_index(namespace)
    if idx is None:
        return "No index found. Build index first from sidebar."
    hits = search(idx, message, top_k=6)
    if not hits:
        return "No relevant extracted chunks found."

    context = []
    for i, h in enumerate(hits, start=1):
        txt = h["text"]
        if len(txt) > 600:
            txt = txt[:600] + "..."
        context.append(f"[{i}] page={h['meta'].get('page')} source={h['meta'].get('source')}\n{txt}")
    ctx = "\n\n".join(context)

    system = "You are a SCAL assistant. Use only retrieved extracted evidence. Cite chunks as [1],[2]."
    user_prompt = f"Task prompt:\n{prompt_text}\n\nQuestion:\n{message}\n\nRetrieved evidence:\n{ctx}"

    try:
        ans = ask_llm(model_name, system, user_prompt)
    except Exception as e:
        ans = f"Local LLM failed: {e}\n\nFallback evidence:\n{ctx[:1600]}"
    return ans


def build_ui():
    with gr.Blocks(title="SCAL OpenCode Gradio (Offline)") as demo:
        gr.Markdown("# SCAL OpenCode-style Interface (Gradio, Offline)")
        gr.Markdown("Chat RAG uses only already extracted .md/.json data. It does not query raw PDF directly.")

        with gr.Row():
            with gr.Column(scale=1):
                data_root = gr.Textbox(label="Extracted data folder", value=str(DEFAULT_DATA_ROOT))
                doc_name = gr.Textbox(label="Document name (prefix before _pageN)", value="A3.1 SPECIAL CORE ANALYSIS")
                coverage_btn = gr.Button("Check Coverage")
                coverage_out = gr.Textbox(label="Coverage", lines=6)

                upload_pdf = gr.File(label="Upload same PDF to verify missing extracted pages", file_types=[".pdf"])
                check_pdf_btn = gr.Button("Check Uploaded PDF vs Extraction")
                check_pdf_out = gr.Textbox(label="Uploaded PDF Check", lines=4)

                build_idx_btn = gr.Button("Build / Rebuild Vector Index")
                build_idx_out = gr.Textbox(label="Index Status", lines=2)

                llm_model = gr.Textbox(label="Local LLM model", value="Qwen/Qwen2.5-3B-Instruct")

                gr.Markdown("### Saved Chat Prompts")
                prompt_dropdown = gr.Dropdown(label="Prompt template", choices=[])
                prompt_text = gr.Textbox(label="Prompt text", lines=10)
                save_prompt_name = gr.Textbox(label="Save prompt as", value="Custom Prompt")
                save_prompt_btn = gr.Button("Save Prompt")
                save_prompt_status = gr.Textbox(label="Prompt Save Status", lines=1)

            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    fn=lambda message, history, dr, dn, pt, lm: chat_fn(message, history, dr, dn, pt, lm),
                    additional_inputs=[data_root, doc_name, prompt_text, llm_model],
                    type="messages",
                    title="OpenCode-style Chat",
                    description="Ask questions against indexed extracted data",
                )

        coverage_btn.click(fn=summarize_coverage, inputs=[data_root, doc_name], outputs=[coverage_out])
        check_pdf_btn.click(fn=check_uploaded_pdf_missing, inputs=[upload_pdf, data_root, doc_name], outputs=[check_pdf_out])
        build_idx_btn.click(fn=build_doc_index, inputs=[data_root, doc_name], outputs=[build_idx_out])
        prompt_dropdown.change(fn=select_prompt, inputs=[prompt_dropdown], outputs=[prompt_text])
        save_prompt_btn.click(fn=save_prompt, inputs=[save_prompt_name, prompt_text], outputs=[prompt_dropdown, save_prompt_status])

        demo.load(fn=load_prompts_ui, inputs=None, outputs=[prompt_dropdown, prompt_text])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
