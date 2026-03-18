from __future__ import annotations

import json
import re
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

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
PROMPT_FILE = ROOT / "scal_modern_prompts.json"

DEFAULT_PROMPTS = [
    {
        "name": "Extract certain columns only",
        "text": (
            "Return only Sample ID, Porosity, Depth from retrieved evidence as valid JSON. "
            "Preserve row order and use NULL for missing fields. No extra text."
        ),
    },
    {
        "name": "Extract table based on keyword",
        "text": (
            "Find table containing [keyword] in retrieved evidence and return full table as JSON. "
            "If not found return {\"no_table\": true}."
        ),
    },
    {
        "name": "Graph extraction",
        "text": "From retrieved graph/table evidence return numeric structured JSON only.",
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
        self.llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
        self._llm_tok = None
        self._llm_model = None
        self._llm_lock = threading.Lock()

        self.vlm = None


R = Runtime()


def log(kind: str, message: str):
    if kind not in R.logs:
        kind = "debug"
    R.logs[kind].append({"time": now(), "kind": kind, "message": message})


def parse_name(file_name: str) -> tuple[str | None, int | None, str]:
    m = re.match(r"^(.*)_page(\d+)\.(pdf|md|json)$", file_name, flags=re.IGNORECASE)
    if not m:
        return None, None, Path(file_name).suffix.lower().lstrip(".")
    return m.group(1), int(m.group(2)), m.group(3).lower()


def scan_docs(root: Path) -> dict[str, dict[int, dict[str, Path]]]:
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


def ensure_prompts():
    if not PROMPT_FILE.exists():
        PROMPT_FILE.write_text(json.dumps(DEFAULT_PROMPTS, indent=2), encoding="utf-8")


def load_prompts() -> list[dict[str, str]]:
    ensure_prompts()
    try:
        data = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return DEFAULT_PROMPTS


def save_prompts(prompts: list[dict[str, str]]):
    PROMPT_FILE.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")


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


def ensure_index_loaded(doc_name: str) -> bool:
    if R.vectorizer is not None and R.matrix is not None and R.current_doc == doc_name:
        return True
    obj = load_index(ns(doc_name))
    if obj is None:
        return False
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = obj
    R.current_doc = doc_name
    return True


def search(query: str, doc_name: str, filters: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    if not ensure_index_loaded(doc_name):
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
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
        R._llm_tok, R._llm_model = tok, mdl
        R.llm_loaded = True
        R.llm_model_name = model_name


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


def extraction_job(pdf_path: Path, stem: str, prompt: str, doc_name: str | None):
    try:
        R.progress["extract"]["running"] = True
        set_progress("extract", 5, "starting", "Preparing extraction")
        log("status", f"Extraction started for {pdf_path.name}")

        if not R.vlm_loaded or R.vlm is None:
            raise RuntimeError("VLM not loaded")

        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        pages = list(range(1, total + 1))
        if doc_name and doc_name in R.docs:
            existing = {p for p, flags in R.docs[doc_name].items() if "md" in flags or "json" in flags}
            pages = [p for p in pages if p not in existing]
        if not pages:
            set_progress("extract", 100, "completed", "Already fully extracted")
            log("status", "No missing pages to extract")
            return

        for i, p in enumerate(pages, start=1):
            pct = 10 + int((i - 1) / max(1, len(pages)) * 80)
            set_progress("extract", pct, "extracting", f"Page {p}/{total}")
            res = R.vlm.extract_page(str(pdf_path), p, prompt=prompt)
            (DATA_ROOT / f"{stem}_page{p}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            (DATA_ROOT / f"{stem}_page{p}.md").write_text(str(res.get("raw_response", "")), encoding="utf-8")

        R.docs = scan_docs(DATA_ROOT)
        set_progress("extract", 100, "completed", f"Extracted {len(pages)} pages")
        log("status", f"Extraction completed ({len(pages)} pages)")
    except Exception as e:
        log("error", f"Extraction failed: {e}")
        log("debug", traceback.format_exc())
        set_progress("extract", 100, "failed", str(e))
    finally:
        R.progress["extract"]["running"] = False
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass


def export_excel(hits: list[dict[str, Any]]) -> Path:
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
    path = EXPORT_DIR / f"retrieved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def export_word(hits: list[dict[str, Any]]) -> Path:
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
    path = EXPORT_DIR / f"retrieved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(path)
    return path


app = FastAPI(title="SCAL Modern Local App")
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def ui_index():
    return FileResponse(static_dir / "index.html")


class ChatReq(BaseModel):
    doc_name: str
    question: str
    prompt_template: str = ""
    filter_file_name: str | None = None
    filter_report_name: str | None = None
    filter_page_number: str | None = None
    filter_table_id: str | None = None
    filter_extraction_type: str | None = None
    filter_sample_id: str | None = None
    top_k: int = 8


@app.get("/api/docs")
def api_docs(root: str | None = None):
    rr = Path(root) if root else DATA_ROOT
    R.docs = scan_docs(rr)
    names = sorted(R.docs.keys())
    return {
        "data_root": str(rr),
        "documents": names,
        "coverage": {n: coverage_for_doc(n) for n in names},
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


@app.get("/api/prompts")
def api_get_prompts():
    return {"prompts": load_prompts()}


@app.post("/api/prompts/save")
def api_save_prompt(name: str = Form(...), text: str = Form(...)):
    prompts = load_prompts()
    found = False
    for p in prompts:
        if p.get("name") == name:
            p["text"] = text
            found = True
            break
    if not found:
        prompts.append({"name": name, "text": text})
    save_prompts(prompts)
    return {"ok": True, "prompts": prompts}


@app.post("/api/models/load-llm")
def api_load_llm(model_name: str = Form("Qwen/Qwen2.5-3B-Instruct")):
    try:
        load_llm(model_name)
        log("status", f"LLM loaded: {model_name}")
        return {"ok": True}
    except Exception as e:
        log("error", f"LLM load error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/models/load-vlm")
def api_load_vlm():
    try:
        load_vlm()
        log("status", "VLM loaded")
        return {"ok": True}
    except Exception as e:
        log("error", f"VLM load error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/chat")
def api_chat(req: ChatReq):
    filters = {
        "file_name": req.filter_file_name,
        "report_name": req.filter_report_name,
        "page_number": req.filter_page_number,
        "table_id": req.filter_table_id,
        "extraction_type": req.filter_extraction_type,
        "sample_id": req.filter_sample_id,
    }
    hits = search(req.question, req.doc_name, filters, top_k=req.top_k)
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

    if not hits:
        answer = "No relevant chunks found."
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
            answer = ask_llm(system, user_prompt)
        except Exception as e:
            answer = f"LLM unavailable or failed: {e}\n\nFallback evidence:\n{context[:1800]}"

    tables = []
    for h in hits:
        m = h["meta"]
        if m.get("parsed_rows"):
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

    return {
        "answer": answer,
        "reasoning": reasoning,
        "sources": [r for r in reasoning],
        "tables": tables,
        "raw_hits": hits,
    }


@app.post("/api/export/excel")
def api_export_excel(payload: dict):
    hits = payload.get("hits", [])
    path = export_excel(hits)
    return {"ok": True, "path": str(path)}


@app.post("/api/export/word")
def api_export_word(payload: dict):
    hits = payload.get("hits", [])
    path = export_word(hits)
    return {"ok": True, "path": str(path)}


@app.post("/api/extract/check")
async def api_extract_check(file: UploadFile = File(...), doc_name: str = Form("")):
    data = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        reader = PdfReader(str(tmp_path))
        total = len(reader.pages)
        extracted = set()
        if doc_name and doc_name in R.docs:
            extracted = {p for p, flags in R.docs[doc_name].items() if "md" in flags or "json" in flags}
        missing = [p for p in range(1, total + 1) if p not in extracted]
        return {
            "total_pdf_pages": total,
            "already_extracted": len(missing) == 0,
            "missing_pages": missing,
        }
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@app.post("/api/extract/start")
async def api_extract_start(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    target_doc_name: str = Form(""),
):
    data = await file.read()
    stem = Path(file.filename).stem
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    if R.progress["extract"]["running"]:
        return JSONResponse({"ok": False, "message": "Extraction already running"}, status_code=409)
    t = threading.Thread(target=extraction_job, args=(tmp_path, stem, prompt, target_doc_name or None), daemon=True)
    t.start()
    return {"ok": True, "message": "Extraction started"}
