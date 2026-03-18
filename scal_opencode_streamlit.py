from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Mining\Downloads\Fine Tunining Datasets-20260318T052420Z-1-001\Fine Tunining Datasets\train"
)
PROMPT_STORE = Path("scal_saved_prompts.json")
INDEX_STORE = Path("scal_streamlit_index")


DEFAULT_PROMPTS = [
    {
        "name": "Extract certain columns only",
        "text": """You are a document analysis assistant.

I will provide retrieved extracted evidence from report JSON/markdown chunks.

Your task is to:
- Return ONLY these columns: Sample ID, Porosity, Depth
- Ignore all other columns
- Preserve original row order
- If missing column in row, return NULL
- Output strictly valid JSON
- No explanations, no summary, no extra text""",
    },
    {
        "name": "Extract table based on keyword",
        "text": """I need you to extract a specific table from retrieved evidence.

Steps:
1. Scan retrieved tables
2. Identify table containing keyword "[keyword]"
3. Extract complete table with headers, rows, columns
4. Return ALL rows in exact order with exact column names as JSON array
5. If no table present: {"no_table": true}""",
    },
    {
        "name": "Graph extraction",
        "text": """Perform structured visual data extraction from retrieved evidence.

Steps:
1. Identify chart type
2. Identify axis scale intervals
3. Determine legend-to-series mapping
4. Extract exact values if labeled
5. If not labeled, calculate approximate values using axis scaling

Return:
- Clean structured JSON table
- Numeric values only
- No commentary""",
    },
]


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


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
    stem = m.group(1)
    page = int(m.group(2))
    ext = m.group(3).lower()
    return stem, page, ext


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


def document_coverage(pages: dict[int, dict[str, Path]]) -> dict[str, Any]:
    if not pages:
        return {
            "total_page_slots": 0,
            "pdf_pages": 0,
            "extracted_pages": 0,
            "missing_extraction_pages": [],
            "gaps_in_sequence": [],
        }
    all_pages = sorted(pages.keys())
    pdf_pages = [p for p in all_pages if "pdf" in pages[p]]
    extracted_pages = [p for p in all_pages if "md" in pages[p] or "json" in pages[p]]
    missing_extraction = [p for p in pdf_pages if p not in extracted_pages]

    min_p, max_p = min(all_pages), max(all_pages)
    gaps = [p for p in range(min_p, max_p + 1) if p not in all_pages]
    return {
        "total_page_slots": len(all_pages),
        "pdf_pages": len(pdf_pages),
        "extracted_pages": len(extracted_pages),
        "missing_extraction_pages": missing_extraction,
        "gaps_in_sequence": gaps,
    }


def _flatten_json_to_text(obj: Any) -> str:
    if isinstance(obj, dict):
        if "raw_response" in obj:
            return str(obj["raw_response"])
        if "rows" in obj and isinstance(obj.get("rows"), list):
            return json.dumps(obj, ensure_ascii=False)
        parts = []
        for v in obj.values():
            parts.append(_flatten_json_to_text(v))
        return "\n".join([p for p in parts if p])
    if isinstance(obj, list):
        return "\n".join(_flatten_json_to_text(x) for x in obj)
    return str(obj)


def load_chunks_for_doc(root: Path, doc_name: str, pages: dict[int, dict[str, Path]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_num in sorted(pages.keys()):
        ex = pages[page_num]
        if "json" in ex:
            try:
                obj = json.loads(ex["json"].read_text(encoding="utf-8", errors="ignore"))
                text = _flatten_json_to_text(obj)
            except Exception:
                text = ex["json"].read_text(encoding="utf-8", errors="ignore")
            chunks.append(Chunk(text=text, metadata={"doc": doc_name, "page": page_num, "source": ex["json"].name}))
        elif "md" in ex:
            text = ex["md"].read_text(encoding="utf-8", errors="ignore")
            chunks.append(Chunk(text=text, metadata={"doc": doc_name, "page": page_num, "source": ex["md"].name}))
    return chunks


def build_index(chunks: list[Chunk], namespace: str) -> tuple[TfidfVectorizer, Any, list[dict], list[str]]:
    INDEX_STORE.mkdir(parents=True, exist_ok=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    mat = vec.fit_transform(texts) if texts else vec.fit_transform([""])

    joblib.dump(vec, INDEX_STORE / f"{namespace}_vec.joblib")
    joblib.dump(mat, INDEX_STORE / f"{namespace}_mat.joblib")
    joblib.dump(metas, INDEX_STORE / f"{namespace}_meta.joblib")
    joblib.dump(texts, INDEX_STORE / f"{namespace}_text.joblib")
    return vec, mat, metas, texts


def load_index(namespace: str):
    vec_p = INDEX_STORE / f"{namespace}_vec.joblib"
    mat_p = INDEX_STORE / f"{namespace}_mat.joblib"
    meta_p = INDEX_STORE / f"{namespace}_meta.joblib"
    text_p = INDEX_STORE / f"{namespace}_text.joblib"
    if not all(p.exists() for p in [vec_p, mat_p, meta_p, text_p]):
        return None
    return joblib.load(vec_p), joblib.load(mat_p), joblib.load(meta_p), joblib.load(text_p)


def search_index(query: str, idx_obj, top_k: int = 6):
    vec, mat, metas, texts = idx_obj
    q = vec.transform([query])
    sims = linear_kernel(q, mat).flatten()
    order = sims.argsort()[::-1]
    out = []
    for i in order[: top_k * 2]:
        score = float(sims[i])
        if score <= 0:
            continue
        out.append({"score": score, "text": texts[i], "meta": metas[i]})
        if len(out) >= top_k:
            break
    return out


@st.cache_resource(show_spinner=False)
def get_local_llm(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for local LLM")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
    return tokenizer, model


def ask_local_llm(model_name: str, system_prompt: str, user_prompt: str) -> str:
    import torch

    tokenizer, model = get_local_llm(model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda", dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=700, temperature=0.2, do_sample=True, top_p=0.9)
    ans = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
    return ans.strip()


def check_pdf_coverage_against_extracted(uploaded_pdf, doc_pages: dict[int, dict[str, Path]]) -> dict[str, Any]:
    reader = PdfReader(uploaded_pdf)
    total_pdf_pages = len(reader.pages)
    extracted_pages = {p for p, f in doc_pages.items() if "md" in f or "json" in f}
    missing = [p for p in range(1, total_pdf_pages + 1) if p not in extracted_pages]
    return {
        "total_pdf_pages": total_pdf_pages,
        "extracted_count": len(extracted_pages),
        "missing_pages": missing,
        "already_extracted": len(missing) == 0,
    }


def main():
    st.set_page_config(page_title="SCAL OpenCode Web", layout="wide")
    st.title("SCAL OpenCode-style Interface (Offline)")
    st.caption("RAG source of truth: already extracted JSON/MD only. No raw PDF querying in chat.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Workspace")
        data_root_str = st.text_input("Extracted data folder", str(DEFAULT_DATA_ROOT))
        data_root = Path(data_root_str)

        docs = scan_dataset(data_root)
        doc_names = sorted(docs.keys())
        if not doc_names:
            st.warning("No *_pageN.(md|json|pdf) files found.")
            return

        doc_name = st.selectbox("Document", doc_names)
        pages = docs[doc_name]
        cov = document_coverage(pages)

        st.subheader("Extraction Coverage")
        st.metric("PDF pages found", cov["pdf_pages"])
        st.metric("Extracted pages", cov["extracted_pages"])
        st.metric("Missing extraction pages", len(cov["missing_extraction_pages"]))
        if cov["missing_extraction_pages"]:
            st.error(f"Missing extracted pages: {cov['missing_extraction_pages']}")
        else:
            st.success("All detected PDF pages have extraction output.")

        namespace = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)
        if st.button("Build / Rebuild Vector Index", use_container_width=True):
            chunks = load_chunks_for_doc(data_root, doc_name, pages)
            if not chunks:
                st.error("No extracted chunks (.md/.json) found for this document.")
            else:
                build_index(chunks, namespace)
                st.success(f"Index built with {len(chunks)} chunks.")

        st.divider()
        st.subheader("Local LLM")
        llm_model = st.text_input("Model", "Qwen/Qwen2.5-3B-Instruct")
        if st.button("Load LLM", use_container_width=True):
            try:
                get_local_llm(llm_model)
                st.success("LLM loaded")
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.subheader("Saved Chat Prompts")
        prompts = load_prompt_store()
        prompt_names = [p["name"] for p in prompts]
        sel_prompt_name = st.selectbox("Prompt template", prompt_names)
        sel_prompt = next(p for p in prompts if p["name"] == sel_prompt_name)
        prompt_text = st.text_area("Prompt text", sel_prompt["text"], height=220)
        new_prompt_name = st.text_input("Save as", sel_prompt_name)
        if st.button("Save Prompt", use_container_width=True):
            replaced = False
            for p in prompts:
                if p["name"] == new_prompt_name:
                    p["text"] = prompt_text
                    replaced = True
                    break
            if not replaced:
                prompts.append({"name": new_prompt_name, "text": prompt_text})
            save_prompt_store(prompts)
            st.success("Prompt saved")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.subheader("Completeness Check (Uploaded PDF vs Existing Extraction)")
        up_pdf = st.file_uploader("Upload PDF to check missing extraction pages", type=["pdf"], key="pdf_check")
        if up_pdf is not None:
            result = check_pdf_coverage_against_extracted(up_pdf, pages)
            if result["already_extracted"]:
                st.success("This file appears already extracted (no missing pages).")
            else:
                st.warning(f"Missing extracted pages: {result['missing_pages']}")
            st.json(result)

        st.subheader("Page Status")
        rows = []
        for pg in sorted(pages.keys()):
            flags = pages[pg]
            rows.append(
                {
                    "page": pg,
                    "pdf": "yes" if "pdf" in flags else "-",
                    "md": "yes" if "md" in flags else "-",
                    "json": "yes" if "json" in flags else "-",
                    "source_files": ", ".join(sorted([f.name for f in flags.values()])),
                }
            )
        st.dataframe(rows, use_container_width=True, height=420)

    with col_b:
        st.subheader("Chat (OpenCode-style)")

        idx_obj = load_index(namespace)
        if idx_obj is None:
            st.info("No vector index found for this document. Build index from sidebar first.")

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about extracted SCAL data...")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                if idx_obj is None:
                    msg = "No index loaded. Please build index first."
                    st.markdown(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                else:
                    hits = search_index(user_q, idx_obj, top_k=6)
                    if not hits:
                        msg = "No relevant extracted chunks found."
                        st.markdown(msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": msg})
                    else:
                        context = []
                        for i, h in enumerate(hits, start=1):
                            meta = h["meta"]
                            txt = h["text"]
                            if len(txt) > 700:
                                txt = txt[:700] + "..."
                            context.append(f"[{i}] page={meta.get('page')} source={meta.get('source')}\n{txt}")
                        context_text = "\n\n".join(context)

                        system_prompt = (
                            "You are a SCAL extraction assistant. Answer ONLY from retrieved extracted evidence. "
                            "Do not use outside knowledge. Include citations like [1], [2]."
                        )
                        user_prompt = (
                            f"Task prompt:\n{prompt_text}\n\n"
                            f"Question:\n{user_q}\n\n"
                            f"Retrieved evidence:\n{context_text}"
                        )
                        try:
                            ans = ask_local_llm(llm_model, system_prompt, user_prompt)
                        except Exception as e:
                            ans = f"Local LLM failed: {e}\n\nFallback evidence preview:\n" + "\n\n".join(context[:3])

                        st.markdown(ans)
                        with st.expander("Retrieved chunks"):
                            st.json(hits)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    main()
