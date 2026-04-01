from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# Executor for blocking tkinter dialogs (1 thread — dialogs must be serial)
_DIALOG_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tkdialog")

# Executor for GPU inference (1 thread — serialises LLM calls, never blocks event loop)
# Extraction runs in its own plain daemon thread (see extraction_job), not here.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_infer")

# Executor for CPU-bound RAG search (TF-IDF matrix multiply) so chat stays async
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_search")

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from bs4 import BeautifulSoup

    BS4_OK = True
except Exception:
    BS4_OK = False


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    r"C:\Users\Mining\Downloads\Fine Tunining Datasets-20260318T052420Z-1-001\Fine Tunining Datasets\train"
)
INDEX_DIR = ROOT / "scal_modern_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = ROOT / "scal_modern_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = ROOT / "scal_modern_sessions.json"

LLM_MODEL_OPTIONS = [
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "label": "Qwen3-30B-A3B-Instruct (Best overall RAG)",
        "recommended": True,
        "notes": "MoE long-context model (up to ~262K) with strong synthesis for document RAG",
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "label": "DeepSeek-R1-Distill-Qwen-32B (Best reasoning)",
        "recommended": False,
        "notes": "Strong analytical/multi-step reasoning for complex query answering",
    },
    {
        "name": "zai-org/GLM-4-32B-0414",
        "label": "GLM-4-32B-0414 (Open-source alternative)",
        "recommended": False,
        "notes": "Open-weight GLM model with strong coding/tool capabilities",
    },
    {
        "name": "moonshotai/Kimi-K2-Instruct",
        "label": "Kimi-K2-Instruct (Very high VRAM)",
        "recommended": False,
        "notes": "Open-weight MoE model; excellent quality but very heavy for single-GPU local inference",
    },
    {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "label": "Qwen2.5-14B-Instruct (Stable fallback)",
        "recommended": False,
        "notes": "Stable default for lower VRAM pressure and fast iteration",
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5-7B-Instruct (Faster fallback)",
        "recommended": False,
        "notes": "Lower VRAM and faster responses",
    },
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B-Instruct (Alternative)",
        "recommended": False,
        "notes": "Good general instruction model",
    },
]

USE_CASE_PROMPT_SUGGESTIONS = [
    {
        "id": "scal_summary",
        "label": "Summarize SCAL report highlights",
        "question": "Summarize the key SCAL findings for this report with supporting evidence.",
        "prompt_template": "Return concise technical summary grouped by: porosity/permeability, capillary pressure, relative permeability, key sample IDs, and anomalies. Cite sources [1],[2].",
    },
    {
        "id": "extract_por_perm",
        "label": "Extract porosity and permeability table",
        "question": "Extract porosity and permeability values by sample as a table.",
        "prompt_template": "From evidence, return structured table columns: Sample ID, Depth, Porosity, Air Permeability, Grain Density, Notes. Keep units exactly.",
    },
    {
        "id": "extract_cap_pressure",
        "label": "Extract capillary pressure points",
        "question": "Extract capillary pressure vs saturation data points by sample.",
        "prompt_template": "Return JSON array grouped by core/sample with fields: capillary_pressure_psi, liquid_saturation_pct_pv, source_table.",
    },
    {
        "id": "qc_missing_values",
        "label": "QC missing values",
        "question": "Check for missing values, malformed rows, and inconsistent units.",
        "prompt_template": "Perform quality checks on extracted rows and return issue list with severity, row identifier, and suggested correction.",
    },
]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.docs: dict[str, dict[int, dict[str, Path]]] = {}
        self.current_doc: str | None = None

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.index_texts: list[str] = []
        self.index_meta: list[dict[str, Any]] = []
        self.index_namespace: str | None = None

        self.progress = {
            "index": {"running": False, "percent": 0, "stage": "idle", "detail": ""},
            "extract": {"running": False, "percent": 0, "stage": "idle", "detail": ""},
        }

        self.logs = {
            "status": deque(maxlen=500),
            "debug": deque(maxlen=500),
            "error": deque(maxlen=500),
        }

        self.llm_loaded = False
        self.vlm_loaded = False
        self.llm_model_name = "Qwen/Qwen2.5-14B-Instruct"
        self._llm_tok = None
        self._llm_model = None
        self._llm_lock = threading.Lock()

        self.vlm = None
        self.extract_stop = threading.Event()  # set to request stop


R = Runtime()


def log(kind: str, message: str):
    if kind not in R.logs:
        kind = "debug"
    R.logs[kind].append({"time": now(), "kind": kind, "message": message})


def _load_sessions() -> dict[str, Any]:
    if not SESSION_FILE.exists():
        return {"sessions": []}
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sessions"), list):
            return data
    except Exception:
        pass
    return {"sessions": []}


def _save_sessions(data: dict[str, Any]):
    SESSION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_sessions() -> list[dict[str, Any]]:
    data = _load_sessions()
    items = []
    for s in data.get("sessions", []):
        items.append(
            {
                "id": s.get("id", ""),
                "title": s.get("title", "Untitled Session"),
                "created_at": s.get("created_at", ""),
                "updated_at": s.get("updated_at", ""),
                "message_count": len(s.get("messages", [])),
            }
        )
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


def get_session(session_id: str) -> dict[str, Any] | None:
    data = _load_sessions()
    for s in data.get("sessions", []):
        if s.get("id") == session_id:
            return s
    return None


def create_session(title: str = "") -> dict[str, Any]:
    data = _load_sessions()
    sid = str(uuid.uuid4())
    ts = datetime.now().isoformat(timespec="seconds")
    obj = {
        "id": sid,
        "title": title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": ts,
        "updated_at": ts,
        "messages": [],
    }
    data["sessions"].append(obj)
    _save_sessions(data)
    return obj


def append_session_messages(session_id: str, new_messages: list[dict[str, Any]]):
    data = _load_sessions()
    found = None
    for s in data.get("sessions", []):
        if s.get("id") == session_id:
            found = s
            break
    if found is None:
        found = create_session("Chat Session")
        data = _load_sessions()
        for s in data.get("sessions", []):
            if s.get("id") == found.get("id"):
                found = s
                break
    found.setdefault("messages", []).extend(new_messages)
    found["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if len(found["messages"]) > 300:
        found["messages"] = found["messages"][-300:]
    _save_sessions(data)


def parse_name(file_name: str) -> tuple[str | None, int | None, str]:
    m = re.match(r"^(.*)_page(\d+)\.(pdf|md|json)$", file_name, flags=re.IGNORECASE)
    if not m:
        return None, None, Path(file_name).suffix.lower().lstrip(".")
    return m.group(1), int(m.group(2)), m.group(3).lower()


def scan_docs(root: Path) -> dict[str, dict[int, dict[str, Path]]]:
    """Scan folder for extracted page files AND plain PDFs.

    Recognised patterns (case-insensitive):
      - <stem>_page<N>.pdf / .md / .json  → grouped under <stem>
      - <stem>.pdf (plain, no page number) → added as page 1 under <stem>
    """
    docs: dict[str, dict[int, dict[str, Path]]] = {}
    if not root.exists():
        return docs
    for p in root.iterdir():
        if not p.is_file():
            continue
        stem, page, ext = parse_name(p.name)
        if stem is not None and page is not None:
            # Normal _pageN file
            docs.setdefault(stem, {}).setdefault(page, {})[ext] = p
        elif p.suffix.lower() == ".pdf":
            # Plain PDF — treat as its own single-page document
            doc_stem = p.stem
            docs.setdefault(doc_stem, {}).setdefault(1, {})["pdf"] = p
    return docs


def coverage_for_doc(doc_name: str) -> dict[str, Any]:
    pages = R.docs.get(doc_name, {})
    if not pages:
        return {"pdf_pages": 0, "extracted_pages": 0, "missing_pages": []}
    all_pages = sorted(pages.keys())
    pdf_pages = [p for p in all_pages if "pdf" in pages[p]]
    ext_pages = [p for p in all_pages if "md" in pages[p] or "json" in pages[p]]
    missing = [p for p in pdf_pages if p not in ext_pages]
    return {"pdf_pages": len(pdf_pages), "extracted_pages": len(ext_pages), "missing_pages": missing}


def flatten_json(obj: Any) -> str:
    if isinstance(obj, dict):
        if "raw_response" in obj:
            return str(obj["raw_response"])
        if "rows" in obj and isinstance(obj["rows"], list):
            return json.dumps(obj, ensure_ascii=False)
        return "\n".join(flatten_json(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(flatten_json(x) for x in obj)
    return str(obj)


def extract_html_tables(text: str) -> list[str]:
    return re.findall(r"<table[\s\S]*?</table>", text, flags=re.IGNORECASE)


def parse_html_table(html: str) -> tuple[list[str], list[dict[str, Any]]]:
    if BS4_OK:
        soup = BeautifulSoup(html, "html.parser")
        headers = [th.get_text(" ", strip=True) for th in soup.find_all("th")]
        rows = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            if headers and cells == headers:
                continue
            if not headers:
                headers = [f"col_{i+1}" for i in range(len(cells))]
            if len(cells) < len(headers):
                cells += [None] * (len(headers) - len(cells))
            rows.append({headers[i]: cells[i] for i in range(min(len(headers), len(cells)))})
        return headers, rows
    return [], []


def infer_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["capillary", "pc", "sw"]):
        return "capillary_pressure"
    if any(k in t for k in ["relative permeability", "krw", "kro", "krg"]):
        return "relative_permeability"
    if any(k in t for k in ["porosity", "permeability", "md"]):
        return "porosity_permeability"
    return "general"


def is_casual_chat(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    simple = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "nice",
        "cool",
        "how are you",
        "who are you",
        "what can you do",
    }
    if q in simple:
        return True
    # Short small-talk style messages should not trigger RAG retrieval
    tokens = re.findall(r"[a-zA-Z0-9']+", q)
    if len(tokens) <= 3 and any(w in q for w in ["hi", "hello", "hey", "thanks", "yo"]):
        return True
    return False


def chunks_for_doc(doc_name: str) -> list[dict[str, Any]]:
    pages = R.docs.get(doc_name, {})
    chunks = []
    tcount = 0
    for pg in sorted(pages.keys()):
        files = pages[pg]
        raw, source = "", ""
        if "json" in files:
            source = files["json"].name
            try:
                raw = flatten_json(json.loads(files["json"].read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                raw = files["json"].read_text(encoding="utf-8", errors="ignore")
        elif "md" in files:
            source = files["md"].name
            raw = files["md"].read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            continue
        tables = extract_html_tables(raw)
        if tables:
            for h in tables:
                tcount += 1
                cols, rows = parse_html_table(h)
                txt = json.dumps(rows, ensure_ascii=False) if rows else h
                sample = ""
                if rows:
                    sample = str(rows[0].get("Sample ID") or rows[0].get("sample_id") or "")
                chunks.append(
                    {
                        "text": txt,
                        "meta": {
                            "file_name": source,
                            "report_name": doc_name,
                            "page_number": pg,
                            "table_id": f"T{pg:03d}_{tcount:02d}",
                            "extraction_type": infer_type(txt),
                            "title": f"HTML table page {pg}",
                            "sample_id": sample,
                            "raw_html": h,
                            "parsed_columns": cols,
                            "parsed_rows": rows,
                            "units": {},
                        },
                    }
                )
        else:
            chunks.append(
                {
                    "text": raw,
                    "meta": {
                        "file_name": source,
                        "report_name": doc_name,
                        "page_number": pg,
                        "table_id": f"P{pg:03d}_FULL",
                        "extraction_type": infer_type(raw),
                        "title": f"Full page {pg}",
                        "sample_id": "",
                        "raw_html": "",
                        "parsed_columns": [],
                        "parsed_rows": [],
                        "units": {},
                    },
                }
            )
    return chunks


def ns(doc_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)


def save_index(namespace: str, vec, mat, texts, metas):
    joblib.dump(vec, INDEX_DIR / f"{namespace}_vec.joblib")
    joblib.dump(mat, INDEX_DIR / f"{namespace}_mat.joblib")
    joblib.dump(texts, INDEX_DIR / f"{namespace}_texts.joblib")
    joblib.dump(metas, INDEX_DIR / f"{namespace}_metas.joblib")


def load_index(namespace: str):
    paths = [
        INDEX_DIR / f"{namespace}_vec.joblib",
        INDEX_DIR / f"{namespace}_mat.joblib",
        INDEX_DIR / f"{namespace}_texts.joblib",
        INDEX_DIR / f"{namespace}_metas.joblib",
    ]
    if not all(p.exists() for p in paths):
        return None
    return joblib.load(paths[0]), joblib.load(paths[1]), joblib.load(paths[2]), joblib.load(paths[3])



def set_progress(kind: str, percent: int, stage: str, detail: str = ""):
    if kind not in R.progress:
        return
    R.progress[kind].update({"percent": max(0, min(100, int(percent))), "stage": stage, "detail": detail})


def build_index_job(doc_name: str):
    try:
        R.progress["index"]["running"] = True
        set_progress("index", 5, "scanning", f"Scanning extracted files for {doc_name}")
        log("status", f"Starting index build for {doc_name}")

        chunks = chunks_for_doc(doc_name)
        set_progress("index", 35, "parsing", f"Prepared {len(chunks)} chunks")
        if not chunks:
            log("error", "No extracted chunks found")
            set_progress("index", 100, "failed", "No chunks")
            return

        texts = [c["text"] for c in chunks]
        metas = [c["meta"] for c in chunks]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(texts)
        set_progress("index", 70, "vectorizing", f"Vectorized {len(texts)} chunks")

        namespace = ns(doc_name)
        save_index(namespace, vec, mat, texts, metas)

        R.vectorizer, R.matrix = vec, mat
        R.index_texts, R.index_meta = texts, metas
        R.index_namespace = namespace
        set_progress("index", 100, "completed", f"Index ready ({len(texts)} chunks)")
        log("status", f"Index build completed for {doc_name}")
    except Exception as e:
        log("error", f"Index build failed: {e}")
        log("debug", traceback.format_exc())
        set_progress("index", 100, "failed", str(e))
    finally:
        R.progress["index"]["running"] = False


def chunks_for_docs(doc_names: list[str]) -> list[dict[str, Any]]:
    out = []
    for name in doc_names:
        out.extend(chunks_for_doc(name))
    return out


def build_global_index_job(doc_names: list[str]):
    if not doc_names:
        return 0
    chunks = chunks_for_docs(doc_names)
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    save_index(ns("__ALL__"), vec, mat, texts, metas)
    log("status", f"Global all-doc index updated ({len(texts)} chunks)")
    return len(texts)


def ensure_index_loaded(doc_name: str) -> bool:
    if R.vectorizer is not None and R.matrix is not None and R.current_doc == doc_name:
        return True
    obj = load_index(ns(doc_name))
    if obj is None:
        return False
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = obj
    R.current_doc = doc_name
    return True


def ensure_all_index_loaded() -> bool:
    obj = load_index(ns("__ALL__"))
    if obj is None:
        return False
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = obj
    R.current_doc = "__ALL__"
    return True


def search(query: str, doc_name: str, filters: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    if doc_name == "__ALL__":
        if not ensure_all_index_loaded():
            # Fallback: aggregate top hits from each available doc index
            merged = []
            for d in sorted(R.docs.keys()):
                if not ensure_index_loaded(d):
                    continue
                qv = R.vectorizer.transform([query])
                sims = linear_kernel(qv, R.matrix).flatten()
                order = sims.argsort()[::-1]
                for i in order[: max(8, top_k)]:
                    score = float(sims[i])
                    if score <= 0:
                        continue
                    m = R.index_meta[i]
                    ok = True
                    for k, v in filters.items():
                        if v in (None, ""):
                            continue
                        if str(m.get(k, "")).lower() != str(v).lower():
                            ok = False
                            break
                    if ok:
                        merged.append({"score": score, "text": R.index_texts[i], "meta": m})
            merged.sort(key=lambda x: x["score"], reverse=True)
            return merged[:top_k]
    elif not ensure_index_loaded(doc_name):
        return []
    qv = R.vectorizer.transform([query])
    sims = linear_kernel(qv, R.matrix).flatten()
    order = sims.argsort()[::-1]
    out = []
    for i in order:
        score = float(sims[i])
        if score <= 0:
            continue
        m = R.index_meta[i]
        ok = True
        for k, v in filters.items():
            if v in (None, ""):
                continue
            if str(m.get(k, "")).lower() != str(v).lower():
                ok = False
                break
        if not ok:
            continue
        out.append({"score": score, "text": R.index_texts[i], "meta": m})
        if len(out) >= top_k:
            break
    return out


def load_llm(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with R._llm_lock:
        log("status", f"Loading LLM {model_name} …")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        R._llm_tok, R._llm_model = tok, mdl
        R.llm_loaded = True
        R.llm_model_name = model_name
        log("status", f"LLM ready: {model_name}")


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    import torch

    if not R.llm_loaded or R._llm_tok is None or R._llm_model is None:
        raise RuntimeError("LLM not loaded")
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    inp = R._llm_tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda", dtype=torch.long)
    with torch.no_grad():
        out = R._llm_model.generate(inp, max_new_tokens=700, temperature=0.2, do_sample=True, top_p=0.9)
    return R._llm_tok.decode(out[0][inp.shape[1] :], skip_special_tokens=True).strip()


def load_vlm():
    import sys

    sys.path.insert(0, str(ROOT))
    from scal_webapp.backend.services.web_olmocr_runtime import get_vlm

    v = get_vlm()
    v.load()
    R.vlm = v
    R.vlm_loaded = True


def extraction_job(
    pdf_path: Path,
    stem: str,
    prompt: str,
    doc_name: str | None,
    output_dir: Path | None,
    page_from: int,
    page_to: int,
    is_tmp: bool = False,
):
    R.extract_stop.clear()
    out_dir = output_dir or DATA_ROOT
    try:
        R.progress["extract"]["running"] = True
        set_progress("extract", 5, "starting", "Preparing extraction")
        log("status", f"Extraction started for {pdf_path.name}")

        if not R.vlm_loaded or R.vlm is None:
            raise RuntimeError("VLM not loaded — click Load VLM first")

        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)

        # Build candidate page list from user range, then subtract already-extracted.
        # Use the PDF's own stem for the lookup so a new/unrecognised file shows
        # all pages as missing rather than falsely reporting them as done.
        _from = max(1, page_from)
        _to = min(total, page_to) if page_to > 0 else total
        pages = list(range(_from, _to + 1))

        # Check by PDF stem in R.docs AND by scanning output_dir on disk
        existing: set[int] = set()
        if stem in R.docs:
            existing |= {p for p, flags in R.docs[stem].items()
                         if "md" in flags or "json" in flags}
        # Also scan output dir for _pageN files matching the stem
        for ext in ("md", "json"):
            for f in out_dir.glob(f"{stem}_page*.{ext}"):
                m = re.match(rf"^{re.escape(stem)}_page(\d+)\.(?:md|json)$",
                              f.name, flags=re.IGNORECASE)
                if m:
                    existing.add(int(m.group(1)))
        pages = [p for p in pages if p not in existing]

        if not pages:
            set_progress("extract", 100, "completed", "All pages in range already extracted")
            log("status", "No missing pages to extract in selected range")
            return

        log("status", f"Extracting {len(pages)} page(s) (range {_from}-{_to}) of {total} total")
        extracted_count = 0
        for i, p in enumerate(pages, start=1):
            if R.extract_stop.is_set():
                set_progress("extract", int((i - 1) / max(1, len(pages)) * 100), "stopped", "Stopped by user")
                log("status", f"Extraction stopped after {extracted_count} page(s)")
                return
            pct = 10 + int((i - 1) / max(1, len(pages)) * 80)
            set_progress("extract", pct, "extracting", f"Page {p}/{total}")
            res = R.vlm.extract_page(str(pdf_path), p, prompt=prompt)
            (out_dir / f"{stem}_page{p}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"{stem}_page{p}.md").write_text(str(res.get("raw_response", "")), encoding="utf-8")
            extracted_count += 1

        R.docs = scan_docs(out_dir)
        set_progress("extract", 100, "completed", f"Extracted {extracted_count} page(s)")
        log("status", f"Extraction completed ({extracted_count} page(s))")
    except Exception as e:
        log("error", f"Extraction failed: {e}")
        log("debug", traceback.format_exc())
        set_progress("extract", 100, "failed", str(e))
    finally:
        R.progress["extract"]["running"] = False
        if is_tmp:
            try:
                if pdf_path.exists():
                    pdf_path.unlink()
            except Exception:
                pass


def _resolve_export_dir(output_dir: str) -> Path:
    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return EXPORT_DIR


def export_excel(hits: list[dict[str, Any]], output_dir: str = "") -> Path:
    rows = []
    for h in hits:
        m = h["meta"]
        rows.append(
            {
                "file_name": m.get("file_name"),
                "report_name": m.get("report_name"),
                "page_number": m.get("page_number"),
                "table_id": m.get("table_id"),
                "extraction_type": m.get("extraction_type"),
                "title": m.get("title"),
                "sample_id": m.get("sample_id"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
        )
    out = _resolve_export_dir(output_dir)
    path = out / f"retrieved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def export_word(hits: list[dict[str, Any]], output_dir: str = "") -> Path:
    from docx import Document

    doc = Document()
    doc.add_heading("Retrieved SCAL Results", level=1)
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        doc.add_heading(f"Result {i}", level=2)
        doc.add_paragraph(
            f"File: {m.get('file_name')} | Report: {m.get('report_name')} | "
            f"Page: {m.get('page_number')} | Table: {m.get('table_id')}"
        )
        if m.get("raw_html") and m.get("parsed_rows"):
            rows = m.get("parsed_rows") or []
            cols = m.get("parsed_columns") or (list(rows[0].keys()) if rows else [])
            if cols:
                t = doc.add_table(rows=1, cols=len(cols))
                for ci, c in enumerate(cols):
                    t.rows[0].cells[ci].text = str(c)
                for r in rows:
                    rr = t.add_row().cells
                    for ci, c in enumerate(cols):
                        rr[ci].text = str(r.get(c, ""))
        else:
            doc.add_paragraph(str(h.get("text", ""))[:2000])
    out = _resolve_export_dir(output_dir)
    path = out / f"retrieved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(path)
    return path


app = FastAPI(title="SCAL Modern Local App")
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def ui_index():
    return FileResponse(static_dir / "index.html")


class ChatReq(BaseModel):
    doc_name: str | None = None
    scope: str = "selected"  # selected | all
    session_id: str | None = None
    question: str
    prompt_template: str = ""
    filter_extraction_type: str | None = None
    top_k: int = 8


# ── Folder / file browse (non-blocking, runs tkinter in executor) ─────────────

def _tkdialog_folder() -> str:
    import tkinter
    import tkinter.filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = tkinter.filedialog.askdirectory(parent=root) or ""
    root.destroy()
    return path


def _tkdialog_file(accept: str) -> str:
    import tkinter
    import tkinter.filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ftypes = [("PDF files", "*.pdf"), ("All files", "*.*")] if ".pdf" in accept else [("All files", "*.*")]
    path = tkinter.filedialog.askopenfilename(parent=root, filetypes=ftypes) or ""
    root.destroy()
    return path


@app.post("/api/browse/folder")
async def api_browse_folder():
    """Opens a native folder picker dialog (non-blocking)."""
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(_DIALOG_EXECUTOR, _tkdialog_folder)
    return {"path": path}


@app.post("/api/browse/file")
async def api_browse_file(payload: dict = {}):
    accept = (payload or {}).get("accept", "")
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(_DIALOG_EXECUTOR, _tkdialog_file, accept)
    return {"path": path}


# ── Documents & index ─────────────────────────────────────────────────────────

@app.get("/api/docs")
async def api_docs(root: str | None = None):
    loop = asyncio.get_event_loop()
    rr = Path(root) if root else DATA_ROOT

    def _scan():
        R.docs = scan_docs(rr)
        names = sorted(R.docs.keys())
        pages_map: dict[str, dict[int, dict[str, str]]] = {
            doc_name: {
                pg: {ext: str(path) for ext, path in files.items()}
                for pg, files in pages.items()
            }
            for doc_name, pages in R.docs.items()
        }
        return names, pages_map

    names, pages_map = await loop.run_in_executor(_SEARCH_EXECUTOR, _scan)
    return {
        "data_root": str(rr),
        "documents": names,
        "coverage": {n: coverage_for_doc(n) for n in names},
        "pages_map": pages_map,
    }


@app.get("/api/docs/debug")
async def api_docs_debug(root: str | None = None):
    """Returns raw file listing to help diagnose why documents aren't appearing."""
    loop = asyncio.get_event_loop()
    rr = Path(root) if root else DATA_ROOT
    if not rr.exists():
        return {"error": f"Path does not exist: {rr}", "path": str(rr)}

    def _list():
        out = []
        for p in rr.iterdir():
            if p.is_file():
                stem, page, ext = parse_name(p.name)
                out.append({"name": p.name, "parsed_stem": stem,
                             "parsed_page": page, "parsed_ext": ext,
                             "matched": stem is not None})
        return out

    files = await loop.run_in_executor(_SEARCH_EXECUTOR, _list)
    return {
        "path": str(rr),
        "total_files": len(files),
        "matched_files": sum(1 for f in files if f["matched"]),
        "files": files[:50],
    }


@app.post("/api/index/build")
def api_build_index(doc_name: str = Form(...)):
    if doc_name not in R.docs:
        raise HTTPException(status_code=404, detail="Document not found")
    if R.progress["index"]["running"]:
        return {"ok": False, "message": "Index build already running"}
    t = threading.Thread(target=build_index_job, args=(doc_name,), daemon=True)
    t.start()
    return {"ok": True, "message": f"Index build started for {doc_name}"}


def build_all_index_job(data_root: Path):
    """Index ALL documents in the folder sequentially."""
    docs = scan_docs(data_root)
    names = sorted(docs.keys())
    if not names:
        log("error", "No documents found in folder")
        return
    log("status", f"Building index for all {len(names)} document(s)")
    R.docs = docs
    total_chunks = 0
    for i, doc_name in enumerate(names):
        if R.progress["index"]["running"]:
            # Wait briefly if another build is still finishing
            import time; time.sleep(0.2)
        R.progress["index"]["running"] = True
        set_progress("index", int(i / len(names) * 90), "indexing", f"({i+1}/{len(names)}) {doc_name}")
        try:
            chunks = chunks_for_doc(doc_name)
            if not chunks:
                continue
            texts = [c["text"] for c in chunks]
            metas = [c["meta"] for c in chunks]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            mat = vec.fit_transform(texts)
            save_index(ns(doc_name), vec, mat, texts, metas)
            total_chunks += len(chunks)
            log("status", f"Indexed {doc_name}: {len(chunks)} chunks")
        except Exception as e:
            log("error", f"Index failed for {doc_name}: {e}")
        finally:
            R.progress["index"]["running"] = False
    try:
        set_progress("index", 95, "indexing", "Building global all-doc index")
        total_chunks = build_global_index_job(names)
    except Exception as e:
        log("error", f"Global all-doc index build failed: {e}")
    set_progress("index", 100, "completed", f"All docs indexed — {total_chunks} total chunks")
    log("status", f"All-docs index complete: {total_chunks} chunks")


@app.post("/api/index/build-all")
def api_build_all(data_root: str = Form("")):
    if R.progress["index"]["running"]:
        return {"ok": False, "message": "Index build already running"}
    rr = Path(data_root) if data_root else DATA_ROOT
    t = threading.Thread(target=build_all_index_job, args=(rr,), daemon=True)
    t.start()
    return {"ok": True, "message": f"All-docs index build started for {rr}"}


# ── State & logs ──────────────────────────────────────────────────────────────

@app.get("/api/state")
def api_state():
    return {
        "progress": R.progress,
        "models": {
            "llm_loaded": R.llm_loaded,
            "llm_model": R.llm_model_name,
            "vlm_loaded": R.vlm_loaded,
        },
    }


@app.get("/api/logs")
def api_logs(kind: str = "status", limit: int = 200):
    if kind not in R.logs:
        raise HTTPException(status_code=400, detail="Invalid log kind")
    items = list(R.logs[kind])[-limit:]
    return {"kind": kind, "items": items}


@app.post("/api/logs/clear")
def api_clear_logs(kind: str = Form("all")):
    if kind == "all":
        for k in R.logs:
            R.logs[k].clear()
    elif kind in R.logs:
        R.logs[kind].clear()
    else:
        raise HTTPException(status_code=400, detail="Invalid log kind")
    return {"ok": True}


# ── Models ────────────────────────────────────────────────────────────────────

def _load_llm_bg(model_name: str):
    """Run in a daemon thread so the HTTP response returns immediately."""
    try:
        load_llm(model_name)
    except Exception as e:
        log("error", f"LLM load failed: {e}")
        log("debug", traceback.format_exc())


def _load_vlm_bg():
    try:
        load_vlm()
    except Exception as e:
        log("error", f"VLM load failed: {e}")
        log("debug", traceback.format_exc())


@app.post("/api/models/load-llm")
def api_load_llm(model_name: str = Form("Qwen/Qwen2.5-14B-Instruct")):
    if R.llm_loaded:
        return {"ok": True, "message": "LLM already loaded"}
    if R._llm_lock.locked():
        return {"ok": False, "message": "LLM is already loading, check logs"}
    log("status", f"LLM load requested: {model_name}")
    threading.Thread(target=_load_llm_bg, args=(model_name,), daemon=True).start()
    return {"ok": True, "message": "LLM loading in background — watch logs / model pill"}


@app.get("/api/models/options")
def api_model_options():
    return {"models": LLM_MODEL_OPTIONS, "default": R.llm_model_name}


@app.post("/api/models/load-vlm")
def api_load_vlm():
    if R.vlm_loaded:
        return {"ok": True, "message": "VLM already loaded"}
    log("status", "VLM load requested")
    threading.Thread(target=_load_vlm_bg, daemon=True).start()
    return {"ok": True, "message": "VLM loading in background — watch logs / model pill"}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def api_chat(req: ChatReq):
    """
    Fully async chat handler:
    - RAG search runs in _SEARCH_EXECUTOR (CPU, non-blocking to event loop)
    - LLM inference runs in _INFERENCE_EXECUTOR (GPU, serialised, non-blocking)
    Both executors are independent of the extraction daemon thread, so PDF
    extraction and chat can run simultaneously without interfering.
    """
    loop = asyncio.get_event_loop()
    filters = {"extraction_type": req.filter_extraction_type}
    scope = (req.scope or "selected").lower()
    target_doc = "__ALL__" if scope == "all" else (req.doc_name or "")
    if scope != "all" and not target_doc:
        raise HTTPException(status_code=400, detail="Select a document or use scope=all")

    # Casual chat mode: avoid retrieving random PDF chunks for greetings/small talk
    if is_casual_chat(req.question):
        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                _INFERENCE_EXECUTOR,
                lambda: ask_llm(
                    "You are a friendly assistant in a SCAL document app. "
                    "For casual chat, respond naturally and briefly. "
                    "Do not cite PDFs unless user asks document questions.",
                    req.question,
                ),
            )
        except Exception:
            answer = (
                "Hi! I can chat casually, and when you're ready I can also help query your extracted SCAL PDFs. "
                "Try asking about porosity, permeability, capillary pressure, or specific samples."
            )

        sid = req.session_id
        if not sid:
            s = create_session("SCAL Chat Session")
            sid = s["id"]
        append_session_messages(
            sid,
            [
                {"role": "user", "content": req.question, "time": now()},
                {"role": "assistant", "content": answer, "time": now(), "sources": []},
            ],
        )

        return {
            "session_id": sid,
            "answer": answer,
            "reasoning": [],
            "sources": [],
            "tables": [],
            "raw_hits": [],
        }

    # ── 1. RAG search (CPU-bound TF-IDF) ──────────────────────────────────────
    hits: list[dict[str, Any]] = await loop.run_in_executor(
        _SEARCH_EXECUTOR,
        lambda: search(req.question, target_doc, filters, top_k=req.top_k),
    )

    # ── 2. Build reasoning list ────────────────────────────────────────────────
    reasoning = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        reasoning.append(
            {
                "rank": i,
                "score": round(h["score"], 4),
                "file_name": m.get("file_name"),
                "page_number": m.get("page_number"),
                "table_id": m.get("table_id"),
                "extraction_type": m.get("extraction_type"),
                "snippet": str(h.get("text", ""))[:300],
            }
        )

    # ── 3. LLM answer (GPU-bound, serialised) ─────────────────────────────────
    if not hits:
        answer = "No relevant chunks found. Build the index first, or broaden the extraction type filter."
    else:
        ctx = []
        for i, h in enumerate(hits, start=1):
            m = h["meta"]
            txt = h["text"]
            if len(txt) > 700:
                txt = txt[:700] + "..."
            ctx.append(
                f"[{i}] file={m.get('file_name')} page={m.get('page_number')} table={m.get('table_id')}\n{txt}"
            )
        context = "\n\n".join(ctx)
        system = "You are a SCAL assistant. Use only retrieved evidence and cite [1],[2]."
        user_prompt = f"Task prompt:\n{req.prompt_template}\n\nQuestion:\n{req.question}\n\nEvidence:\n{context}"
        try:
            answer = await loop.run_in_executor(
                _INFERENCE_EXECUTOR,
                lambda: ask_llm(system, user_prompt),
            )
        except Exception as e:
            answer = f"LLM not loaded or failed: {e}\n\nFallback evidence:\n{context[:1800]}"

    # ── 4. Collect tables from hits ────────────────────────────────────────────
    tables = []
    for h in hits:
        m = h["meta"]
        if m.get("parsed_rows") or m.get("raw_html"):
            tables.append(
                {
                    "file_name": m.get("file_name"),
                    "report_name": m.get("report_name"),
                    "page_number": m.get("page_number"),
                    "table_id": m.get("table_id"),
                    "raw_html": m.get("raw_html"),
                    "columns": m.get("parsed_columns"),
                    "rows": m.get("parsed_rows"),
                }
            )

    # Persist chat memory (session-based)
    sid = req.session_id
    if not sid:
        s = create_session("SCAL Chat Session")
        sid = s["id"]
    append_session_messages(
        sid,
        [
            {"role": "user", "content": req.question, "time": now()},
            {"role": "assistant", "content": answer, "time": now(), "sources": reasoning},
        ],
    )

    return {
        "session_id": sid,
        "answer": answer,
        "reasoning": reasoning,
        "sources": reasoning,
        "tables": tables,
        "raw_hits": hits,
    }


@app.get("/api/chat/sessions")
def api_chat_sessions():
    return {"sessions": list_sessions()}


@app.post("/api/chat/session/new")
def api_chat_session_new(title: str = Form("")):
    s = create_session(title)
    return {"session": {"id": s["id"], "title": s["title"], "messages": s["messages"]}}


@app.get("/api/chat/session/{session_id}")
def api_chat_session_get(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": s}


@app.get("/api/chat/suggestions")
def api_chat_suggestions(doc_name: str | None = None):
    # Static use-case prompts (user-facing labels + hidden prompt templates)
    out = list(USE_CASE_PROMPT_SUGGESTIONS)

    # Dynamic suggestions from current RAG metadata
    dynamic = []
    docs = [doc_name] if doc_name and doc_name in R.docs else list(R.docs.keys())[:20]
    type_counts: dict[str, int] = {}
    for d in docs:
        pages = R.docs.get(d, {})
        for pg in pages.values():
            md = pg.get("md")
            js = pg.get("json")
            txt = ""
            if md and md.exists():
                txt = md.read_text(encoding="utf-8", errors="ignore")
            elif js and js.exists():
                try:
                    txt = flatten_json(json.loads(js.read_text(encoding="utf-8", errors="ignore")))
                except Exception:
                    txt = js.read_text(encoding="utf-8", errors="ignore")
            t = infer_type(txt)
            type_counts[t] = type_counts.get(t, 0) + 1
    for t, n in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:4]:
        dynamic.append(
            {
                "id": f"dyn_{t}",
                "label": f"Explore {t.replace('_', ' ')} ({n} pages)",
                "question": f"Summarize key findings for {t.replace('_', ' ')} with sources.",
                "prompt_template": "Return concise technical bullets, include key values and cite sources [1],[2].",
            }
        )
    return {"suggestions": out + dynamic}


# ── Export ────────────────────────────────────────────────────────────────────

@app.post("/api/export/excel")
def api_export_excel(payload: dict):
    path = export_excel(payload.get("hits", []), output_dir=payload.get("output_dir", ""))
    return {"ok": True, "path": str(path)}


@app.post("/api/export/word")
def api_export_word(payload: dict):
    path = export_word(payload.get("hits", []), output_dir=payload.get("output_dir", ""))
    return {"ok": True, "path": str(path)}


# ── PDF Extraction ────────────────────────────────────────────────────────────

def _pdf_check(pdf_path: Path, doc_name: str, output_dir: str = "",
               stem_override: str = "") -> dict:
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    # Use the real filename stem (may differ from the temp path stem for uploads)
    pdf_stem = stem_override or pdf_path.stem

    # Search both the configured output_dir and DATA_ROOT for already-extracted pages
    search_roots = [DATA_ROOT]
    if output_dir:
        out = Path(output_dir)
        if out.exists() and out != DATA_ROOT:
            search_roots.insert(0, out)

    extracted: set[int] = set()

    # 1. Check R.docs by PDF stem (most reliable when already scanned)
    if pdf_stem in R.docs:
        extracted |= {p for p, flags in R.docs[pdf_stem].items()
                      if "md" in flags or "json" in flags}

    # 2. Also scan roots directly for _pageN.md / _pageN.json files matching the stem
    for root in search_roots:
        if not root.exists():
            continue
        for ext in ("md", "json"):
            for f in root.glob(f"{pdf_stem}_page*.{ext}"):
                m = re.match(rf"^{re.escape(pdf_stem)}_page(\d+)\.(?:md|json)$",
                              f.name, flags=re.IGNORECASE)
                if m:
                    extracted.add(int(m.group(1)))

    missing = [p for p in range(1, total + 1) if p not in extracted]
    return {
        "total_pdf_pages": total,
        "already_extracted": len(missing) == 0,
        "missing_pages": missing,
        "extracted_pages": sorted(extracted),
        "pdf_stem": pdf_stem,
    }


@app.post("/api/extract/check")
async def api_extract_check(
    file: UploadFile = File(None),
    pdf_path: str = Form(""),
    doc_name: str = Form(""),
    output_dir: str = Form(""),
):
    if file and file.filename:
        real_stem = Path(file.filename).stem
        data = await file.read()
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _pdf_check(tmp_path, doc_name, output_dir, stem_override=real_stem)
        finally:
            try: tmp_path.unlink()
            except Exception: pass
    elif pdf_path:
        p = Path(pdf_path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {pdf_path}")
        return _pdf_check(p, doc_name, output_dir)
    else:
        raise HTTPException(status_code=400, detail="Provide file upload or pdf_path")


@app.post("/api/extract/start")
async def api_extract_start(
    file: UploadFile = File(None),
    pdf_path: str = Form(""),
    prompt: str = Form(""),
    target_doc_name: str = Form(""),
    output_dir: str = Form(""),
    page_from: int = Form(1),
    page_to: int = Form(0),
):
    if R.progress["extract"]["running"]:
        return JSONResponse({"ok": False, "message": "Extraction already running"}, status_code=409)

    if file and file.filename:
        data = await file.read()
        stem = Path(file.filename).stem
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        is_tmp = True
    elif pdf_path:
        p = Path(pdf_path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {pdf_path}")
        stem = p.stem
        tmp_path = p
        is_tmp = False
    else:
        raise HTTPException(status_code=400, detail="Provide file upload or pdf_path")

    out = Path(output_dir) if output_dir else None
    t = threading.Thread(
        target=extraction_job,
        args=(tmp_path, stem, prompt, target_doc_name or None, out, page_from, page_to, is_tmp),
        daemon=True,
    )
    t.start()
    return {"ok": True, "message": f"Extraction started ({stem})"}


@app.post("/api/extract/stop")
def api_extract_stop():
    R.extract_stop.set()
    return {"ok": True, "message": "Stop signal sent"}


# ── Viewer endpoints ──────────────────────────────────────────────────────────

@app.get("/api/page/raw")
async def api_page_raw(doc: str, page: int):
    """Return the raw text content (JSON or MD) for a specific page."""
    pages = R.docs.get(doc, {})
    if page not in pages:
        raise HTTPException(status_code=404, detail="Page not found")
    files = pages[page]

    def _read():
        if "json" in files:
            content = files["json"].read_text(encoding="utf-8", errors="ignore")
            try:
                content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
            except Exception:
                pass
            return {"kind": "json", "content": content}
        elif "md" in files:
            content = files["md"].read_text(encoding="utf-8", errors="ignore")
            return {"kind": "md", "content": content}
        return {"kind": "none", "content": "(no extracted file for this page)"}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_SEARCH_EXECUTOR, _read)


@app.get("/api/page/pdf")
def api_page_pdf(doc: str, page: int):
    """Serve the raw page PDF file for preview (FileResponse is already async-friendly)."""
    pages = R.docs.get(doc, {})
    if page not in pages:
        raise HTTPException(status_code=404, detail="Page not found")
    files = pages[page]
    if "pdf" not in files:
        raise HTTPException(status_code=404, detail="No PDF for this page")
    return FileResponse(str(files["pdf"]), media_type="application/pdf")


@app.get("/api/page/parse")
async def api_page_parse(doc: str, page: int):
    """Parse HTML tables from the extracted content of a page."""
    pages = R.docs.get(doc, {})
    if page not in pages:
        raise HTTPException(status_code=404, detail="Page not found")
    files = pages[page]

    def _parse():
        raw = ""
        if "json" in files:
            try:
                raw = flatten_json(json.loads(files["json"].read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                raw = files["json"].read_text(encoding="utf-8", errors="ignore")
        elif "md" in files:
            raw = files["md"].read_text(encoding="utf-8", errors="ignore")
        html_tables = extract_html_tables(raw)
        result = []
        for h in html_tables:
            cols, rows = parse_html_table(h)
            result.append({"raw_html": h, "columns": cols, "rows": rows, "row_count": len(rows)})
        return result

    loop = asyncio.get_event_loop()
    tables = await loop.run_in_executor(_SEARCH_EXECUTOR, _parse)
    return {"page": page, "tables": tables, "table_count": len(tables)}
