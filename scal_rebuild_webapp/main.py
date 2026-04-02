from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from bs4 import BeautifulSoup

    BS4_OK = True
except Exception:
    BS4_OK = False


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "SCAL_DATA_ROOT",
        r"C:\Users\Mining\Downloads\Fine Tunining Datasets-20260318T052420Z-1-001\Fine Tunining Datasets\train",
    )
)
INDEX_DIR = ROOT / "scal_rebuild_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = ROOT / "scal_rebuild_sessions.json"
SETTINGS_FILE = ROOT / "scal_rebuild_settings.json"
INFERENCE_API_URL = os.environ.get("SCAL_INFERENCE_API_URL", "http://127.0.0.1:8010").rstrip("/")
OLLAMA_BASE_URL = os.environ.get("SCAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
LOCALAI_BASE_URL = os.environ.get("SCAL_LOCALAI_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
CLASSIC_UI_URL = os.environ.get("SCAL_CLASSIC_UI_URL", "http://127.0.0.1:8080").rstrip("/")


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip() or "dev"
    except Exception:
        return "dev"


APP_BUILD = os.environ.get("SCAL_BUILD", _git_commit_short())
APP_STARTED_AT = datetime.now().isoformat(timespec="seconds")


class Runtime:
    def __init__(self):
        self.docs: dict[str, dict[int, dict[str, Path]]] = {}
        self.current_doc = "__ALL__"
        self.data_root = DATA_ROOT
        self.vectorizer = None
        self.matrix = None
        self.index_texts: list[str] = []
        self.index_meta: list[dict[str, Any]] = []

        self.settings = {
            "backend": "inference_api",  # inference_api | ollama | localai
            "ui_mode": "layman",  # layman | advanced
            "data_root": str(DATA_ROOT),
        }

        self.model = {
            "loaded": False,
            "model_name": "",
            "target_model": "",
            "loading": False,
            "last_error": "",
            "backend": "inference_api",
        }
        self.progress = {
            "index": {"running": False, "percent": 0, "stage": "idle", "detail": ""},
            "model": {"running": False, "percent": 0, "stage": "idle", "detail": ""},
        }
        self.logs = {
            "status": deque(maxlen=600),
            "debug": deque(maxlen=400),
            "error": deque(maxlen=300),
        }
        self.lock = threading.Lock()


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(kind: str, message: str):
    if kind not in R.logs:
        kind = "status"
    R.logs[kind].append({"time": now(), "msg": str(message)})


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


def _load_settings() -> dict[str, Any]:
    defaults = {
        "backend": "inference_api",
        "ui_mode": "layman",
        "data_root": str(DATA_ROOT),
    }
    if not SETTINGS_FILE.exists():
        return defaults
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
        backend = str(data.get("backend", defaults["backend"]))
        ui_mode = str(data.get("ui_mode", defaults["ui_mode"]))
        data_root = str(data.get("data_root", defaults["data_root"]))
        if backend not in {"inference_api", "ollama", "localai"}:
            backend = defaults["backend"]
        if ui_mode not in {"layman", "advanced"}:
            ui_mode = defaults["ui_mode"]
        return {"backend": backend, "ui_mode": ui_mode, "data_root": data_root}
    except Exception:
        return defaults


def _save_settings(data: dict[str, Any]):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


R = Runtime()
R.settings = _load_settings()
R.data_root = Path(R.settings.get("data_root", str(DATA_ROOT)))
R.model["backend"] = R.settings.get("backend", "inference_api")

_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rebuild_search")
_DIALOG_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebuild_dialog")


def list_sessions() -> list[dict[str, Any]]:
    data = _load_sessions()
    out = []
    for s in data.get("sessions", []):
        out.append(
            {
                "id": s.get("id"),
                "title": s.get("title", "Session"),
                "updated_at": s.get("updated_at", ""),
                "message_count": len(s.get("messages", [])),
            }
        )
    out.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return out


def create_session(title: str = "") -> dict[str, Any]:
    data = _load_sessions()
    sid = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    t = title.strip() or "SCAL Chat"
    session = {
        "id": sid,
        "title": t,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    data.setdefault("sessions", []).append(session)
    _save_sessions(data)
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    for s in _load_sessions().get("sessions", []):
        if s.get("id") == session_id:
            return s
    return None


def session_to_text(session: dict[str, Any]) -> str:
    title = str(session.get("title") or "SCAL Chat")
    sid = str(session.get("id") or "")
    created = str(session.get("created_at") or "")
    updated = str(session.get("updated_at") or "")
    lines = [
        f"Session Title: {title}",
        f"Session ID: {sid}",
        f"Created: {created}",
        f"Updated: {updated}",
        "",
    ]

    for i, m in enumerate(session.get("messages", []), start=1):
        role = str(m.get("role") or "assistant").upper()
        t = str(m.get("time") or "")
        content = str(m.get("content") or "").strip()
        lines.append(f"[{i}] {role}" + (f" ({t})" if t else ""))
        lines.append(content)

        if role == "ASSISTANT":
            sources = m.get("sources") or []
            if isinstance(sources, list) and sources:
                lines.append("Sources:")
                for s in sources:
                    if not isinstance(s, dict):
                        continue
                    rank = s.get("rank", "?")
                    file_name = s.get("file_name", "?")
                    page = s.get("page_number", "?")
                    score = s.get("score", "?")
                    lines.append(f"- [{rank}] {file_name} page {page} score {score}")
        lines.append("")
    return "\n".join(lines)


def append_session_messages(session_id: str, new_messages: list[dict[str, Any]]):
    data = _load_sessions()
    found = None
    for s in data.get("sessions", []):
        if s.get("id") == session_id:
            found = s
            break
    if found is None:
        found = create_session("SCAL Chat")
        data = _load_sessions()
        for s in data.get("sessions", []):
            if s.get("id") == found.get("id"):
                found = s
                break
    found.setdefault("messages", []).extend(new_messages)
    if str(found.get("title") or "").strip() in {"", "SCAL Chat"}:
        for m in found.get("messages", []):
            if str(m.get("role") or "") == "user":
                first = str(m.get("content") or "").strip().replace("\n", " ")
                if first:
                    found["title"] = first[:52] + ("..." if len(first) > 52 else "")
                break
    found["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if len(found["messages"]) > 400:
        found["messages"] = found["messages"][-400:]
    _save_sessions(data)


def parse_name(file_name: str) -> tuple[str | None, int | None, str]:
    m = re.match(r"^(.*)_page(\d+)\.(pdf|md|json|png|jpg|jpeg|webp)$", file_name, flags=re.IGNORECASE)
    if not m:
        return None, None, Path(file_name).suffix.lower().lstrip(".")
    return m.group(1), int(m.group(2)), m.group(3).lower()


def scan_docs(root: Path) -> dict[str, dict[int, dict[str, Path]]]:
    docs: dict[str, dict[int, dict[str, Path]]] = {}
    if not root.exists():
        return docs
    for p in root.iterdir():
        if not p.is_file():
            continue
        stem, page, ext = parse_name(p.name)
        if stem is not None and page is not None:
            docs.setdefault(stem, {}).setdefault(page, {})[ext] = p
        elif p.suffix.lower() == ".pdf":
            docs.setdefault(p.stem, {}).setdefault(1, {})["pdf"] = p
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


def pages_map_for_doc(doc_name: str) -> list[dict[str, Any]]:
    pages = R.docs.get(doc_name, {})
    out: list[dict[str, Any]] = []
    for pg in sorted(pages.keys()):
        files = pages[pg]
        out.append(
            {
                "page": int(pg),
                "has_pdf": "pdf" in files,
                "has_json": "json" in files,
                "has_md": "md" in files,
                "has_image": any(k in files for k in ("png", "jpg", "jpeg", "webp")),
            }
        )
    return out


def page_content_for(doc_name: str, page: int) -> dict[str, Any] | None:
    files = R.docs.get(doc_name, {}).get(int(page))
    if not files:
        return None

    raw_text = ""
    source_type = ""
    source_name = ""
    raw_json = ""
    if "json" in files:
        source_type = "json"
        source_name = files["json"].name
        raw_json = files["json"].read_text(encoding="utf-8", errors="ignore")
        try:
            raw_text = flatten_json(json.loads(raw_json))
        except Exception:
            raw_text = raw_json
    elif "md" in files:
        source_type = "md"
        source_name = files["md"].name
        raw_text = files["md"].read_text(encoding="utf-8", errors="ignore")

    tables = extract_html_tables(raw_text)
    return {
        "raw_text": raw_text,
        "raw_json": raw_json,
        "source_type": source_type,
        "source_name": source_name,
        "tables": tables,
        "has_pdf": "pdf" in files,
        "has_json": "json" in files,
        "has_md": "md" in files,
        "has_image": any(k in files for k in ("png", "jpg", "jpeg", "webp")),
    }


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
    if not BS4_OK:
        return [], []
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


def infer_type(text: str) -> str:
    t = (text or "").lower()
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
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "how are you",
    }
    if q in simple:
        return True
    tokens = re.findall(r"[a-zA-Z0-9']+", q)
    return len(tokens) <= 3 and any(w in q for w in ["hi", "hello", "hey", "thanks", "yo"])


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
                rows_json = json.dumps(rows, ensure_ascii=False) if rows else "[]"
                txt = (
                    "TABLE_HTML_BEGIN\n"
                    f"{h}\n"
                    "TABLE_HTML_END\n"
                    "TABLE_ROWS_JSON_BEGIN\n"
                    f"{rows_json}\n"
                    "TABLE_ROWS_JSON_END"
                )
                chunks.append(
                    {
                        "text": txt,
                        "meta": {
                            "file_name": source,
                            "report_name": doc_name,
                            "page_number": pg,
                            "table_id": f"T{pg:03d}_{tcount:02d}",
                            "extraction_type": infer_type(txt),
                            "title": f"Table page {pg}",
                            "raw_html": h,
                            "parsed_columns": cols,
                            "parsed_rows": rows,
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
                        "raw_html": "",
                        "parsed_columns": [],
                        "parsed_rows": [],
                    },
                }
            )
    return chunks


def ns(doc_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)


def save_index(namespace: str, vec, mat, texts: list[str], metas: list[dict[str, Any]]):
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


def build_doc_index(doc_name: str) -> bool:
    chunks = chunks_for_doc(doc_name)
    if not chunks:
        return False
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    save_index(ns(doc_name), vec, mat, texts, metas)
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = vec, mat, texts, metas
    R.current_doc = doc_name
    return True


def build_global_index(doc_names: list[str]) -> int:
    chunks = []
    for name in doc_names:
        chunks.extend(chunks_for_doc(name))
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    save_index(ns("__ALL__"), vec, mat, texts, metas)
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = vec, mat, texts, metas
    R.current_doc = "__ALL__"
    return len(chunks)


def ensure_doc_index_loaded(doc_name: str) -> bool:
    if R.vectorizer is not None and R.matrix is not None and R.current_doc == doc_name:
        return True
    obj = load_index(ns(doc_name))
    if obj is None:
        log("status", f"No existing RAG index for {doc_name}. Run RAG Build.")
        return False
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = obj
    R.current_doc = doc_name
    return True


def ensure_global_index_loaded() -> bool:
    if R.vectorizer is not None and R.matrix is not None and R.current_doc == "__ALL__":
        return True
    obj = load_index(ns("__ALL__"))
    if obj is None:
        log("status", "No existing global RAG index. Run RAG Build.")
        return False
    R.vectorizer, R.matrix, R.index_texts, R.index_meta = obj
    R.current_doc = "__ALL__"
    return True


def search(query: str, doc_name: str, filters: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    if doc_name == "__ALL__":
        if not ensure_global_index_loaded():
            return []
    else:
        if not ensure_doc_index_loaded(doc_name):
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


def _json_request(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    url = f"{base_url}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Service unavailable at {base_url}: {e}")


def _inference_request(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    return _json_request(INFERENCE_API_URL, method, path, payload, timeout)


def _ollama_request(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    return _json_request(OLLAMA_BASE_URL, method, path, payload, timeout)


def _localai_request(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    return _json_request(LOCALAI_BASE_URL, method, path, payload, timeout)


def _localai_models_list(timeout: int = 8) -> list[str]:
    # LocalAI commonly exposes /v1/models; some setups expose /models.
    try:
        resp = _localai_request("GET", "/v1/models", timeout=timeout)
        if isinstance(resp, dict):
            out = [str(m.get("id") or "") for m in resp.get("data", []) if isinstance(m, dict)]
            out = [x for x in out if x]
            if out:
                return out
    except Exception:
        pass

    try:
        resp = _localai_request("GET", "/models", timeout=timeout)
        if isinstance(resp, list):
            out = []
            for m in resp:
                if isinstance(m, str):
                    out.append(m)
                elif isinstance(m, dict):
                    out.append(str(m.get("id") or m.get("name") or ""))
            return [x for x in out if x]
        if isinstance(resp, dict):
            out = [str(m.get("id") or m.get("name") or "") for m in resp.get("data", []) if isinstance(m, dict)]
            return [x for x in out if x]
    except Exception:
        pass
    return []


def _inference_stream_events(path: str, payload: dict[str, Any], timeout: int = 600):
    url = f"{INFERENCE_API_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if not data_line or data_line == "[DONE]":
                    continue
                try:
                    yield json.loads(data_line)
                except Exception:
                    continue
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Inference API HTTP {e.code}: {detail or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Inference API stream unavailable at {INFERENCE_API_URL}: {e}")


def _ollama_stream_generate(payload: dict[str, Any], timeout: int = 600):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/x-ndjson", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Ollama unavailable at {OLLAMA_BASE_URL}: {e}")


def _compat_stream_chat(base_url: str, payload: dict[str, Any], timeout: int = 1200):
    url = f"{base_url}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if not data_line:
                    continue
                if data_line == "[DONE]":
                    break
                try:
                    yield json.loads(data_line)
                except Exception:
                    continue
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Local compatible API HTTP {e.code}: {detail or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Local compatible API unavailable at {base_url}: {e}")


def _sync_model_state_inference() -> dict[str, Any]:
    st = _inference_request("GET", "/v1/health", timeout=5)
    R.model = {
        "loaded": bool(st.get("loaded", False)),
        "model_name": str(st.get("model_name") or ""),
        "target_model": str(st.get("target_model") or ""),
        "loading": bool(st.get("busy", False) or st.get("state") in {"loading", "unloading"}),
        "last_error": str(st.get("last_error") or ""),
        "backend": "inference_api",
    }
    p = st.get("progress") or {}
    R.progress["model"] = {
        "running": R.model["loading"],
        "percent": int(p.get("percent", 0) or 0),
        "stage": str(p.get("stage", st.get("state", "idle"))),
        "detail": str(p.get("detail", "")),
    }
    return st


def _sync_model_state_ollama() -> dict[str, Any]:
    tags = _ollama_request("GET", "/api/tags", timeout=8)
    names = [str(m.get("name") or "") for m in tags.get("models", []) if isinstance(m, dict)]
    current = str(R.model.get("model_name") or "")
    loaded = bool(current and current in names)
    R.model = {
        "loaded": loaded,
        "model_name": current,
        "target_model": "",
        "loading": False,
        "last_error": "",
        "backend": "ollama",
    }
    R.progress["model"] = {
        "running": False,
        "percent": 100 if loaded else 0,
        "stage": "loaded" if loaded else "idle",
        "detail": f"Ollama models available: {len(names)}",
    }
    return {"models": names}


def _sync_model_state_localai() -> dict[str, Any]:
    names = _localai_models_list(timeout=8)
    current = str(R.model.get("model_name") or "")
    loaded = bool(current and current in names)
    R.model = {
        "loaded": loaded,
        "model_name": current,
        "target_model": "",
        "loading": False,
        "last_error": "",
        "backend": "localai",
    }
    R.progress["model"] = {
        "running": False,
        "percent": 100 if loaded else 0,
        "stage": "loaded" if loaded else "idle",
        "detail": f"LocalAI models available: {len(names)}",
    }
    return {"models": names}


def sync_model_state() -> dict[str, Any]:
    backend = str(R.settings.get("backend", "inference_api"))
    try:
        if backend == "ollama":
            return _sync_model_state_ollama()
        if backend == "localai":
            return _sync_model_state_localai()
        return _sync_model_state_inference()
    except Exception as e:
        R.model["loaded"] = False
        R.model["loading"] = False
        R.model["last_error"] = str(e)
        R.model["backend"] = backend
        R.progress["model"].update({"running": False, "stage": "failed", "detail": str(e)})
        return {}


def _classic_ui_candidates() -> list[str]:
    candidates = [CLASSIC_UI_URL, "http://127.0.0.1:8090", "http://127.0.0.1:8080"]
    out = []
    for url in candidates:
        if url and url not in out:
            out.append(url)
    return out


def _first_image_for_page(files: dict[str, Path]) -> Path | None:
    for ext in ("png", "jpg", "jpeg", "webp"):
        if ext in files:
            return files[ext]
    return None


def _as_data_url(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _rows_to_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<table><tr><td>(no rows)</td></tr></table>"
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(str(k))
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    body = []
    for r in rows:
        tds = "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _export_tables_document(tables: list[dict[str, Any]], title: str, kind: str) -> tuple[Path, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "xls" if kind == "excel" else "doc"
    out_path = Path(tempfile.gettempdir()) / f"scal_tables_{ts}.{ext}"
    blocks = []
    for i, t in enumerate(tables, start=1):
        html = str(t.get("raw_html") or "").strip()
        if not html:
            html = _rows_to_html(list(t.get("rows") or []))
        meta = f"{t.get('file_name','?')} | page {t.get('page_number','?')} | {t.get('table_id','T'+str(i))}"
        blocks.append(f"<h3>Table {i}</h3><div>{escape(meta)}</div>{html}<hr>")
    if not blocks:
        blocks.append("<p>No tables available for export.</p>")
    body = "\n".join(blocks)
    doc = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:Segoe UI,Arial,sans-serif;font-size:11pt;}"
        "table{border-collapse:collapse;width:100%;margin:8px 0;}"
        "th,td{border:1px solid #999;padding:4px 6px;font-size:10pt;}"
        "h2,h3{margin:8px 0 4px 0;}hr{border:none;border-top:1px solid #ccc;margin:10px 0;}"
        "</style></head><body>"
        f"<h2>{escape(title)}</h2>{body}</body></html>"
    )
    out_path.write_text(doc, encoding="utf-8")
    media = "application/vnd.ms-excel" if kind == "excel" else "application/msword"
    return out_path, media


def llm_generation_settings(mode: str) -> dict[str, Any]:
    m = (mode or "balanced").strip().lower()
    if m == "fast":
        return {"max_new_tokens": 220, "temperature": 0.0, "top_p": 1.0, "do_sample": False}
    if m == "deep":
        return {"max_new_tokens": 700, "temperature": 0.3, "top_p": 0.95, "do_sample": True}
    return {"max_new_tokens": 420, "temperature": 0.2, "top_p": 0.9, "do_sample": True}


def approx_token_count(text: str) -> int:
    t = text or ""
    if not t:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", t, flags=re.UNICODE))


class ChatReq(BaseModel):
    question: str
    session_id: str | None = None
    doc_name: str | None = None
    scope: str = "all"  # all | selected
    filter_extraction_type: str | None = None
    response_mode: str = "fast"
    prompt_template: str = ""
    top_k: int = 24
    include_table_html: bool = True
    use_pdf_vision: bool = False


class SettingsReq(BaseModel):
    backend: str | None = None  # inference_api | ollama | localai
    ui_mode: str | None = None  # layman | advanced
    data_root: str | None = None


class SessionTitleReq(BaseModel):
    title: str = ""


class ExportTablesReq(BaseModel):
    format: str = "excel"  # excel | word
    title: str = "SCAL Combined Tables"
    tables: list[dict[str, Any]] = Field(default_factory=list)


app = FastAPI(title="SCAL Rebuild WebApp")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def index_page():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/api/state")
def api_state():
    sync_model_state()
    return {
        "app": {"build": APP_BUILD, "started_at": APP_STARTED_AT},
        "settings": R.settings,
        "model": R.model,
        "progress": R.progress,
        "doc_count": len(R.docs),
        "services": {
            "inference_api_url": INFERENCE_API_URL,
            "ollama_url": OLLAMA_BASE_URL,
            "localai_url": LOCALAI_BASE_URL,
            "legacy_ui_url": CLASSIC_UI_URL,
        },
    }


@app.get("/api/settings")
def api_settings_get():
    return {"settings": R.settings}


@app.post("/api/settings")
def api_settings_set(req: SettingsReq):
    data = dict(R.settings)
    if req.backend is not None:
        backend = str(req.backend).strip().lower()
        if backend not in {"inference_api", "ollama", "localai"}:
            raise HTTPException(status_code=400, detail="backend must be inference_api, ollama, or localai")
        data["backend"] = backend
    if req.ui_mode is not None:
        ui_mode = str(req.ui_mode).strip().lower()
        if ui_mode not in {"layman", "advanced"}:
            raise HTTPException(status_code=400, detail="ui_mode must be layman or advanced")
        data["ui_mode"] = ui_mode
    if req.data_root is not None:
        root = str(req.data_root).strip()
        if root:
            data["data_root"] = root

    R.settings = data
    R.data_root = Path(data.get("data_root", str(DATA_ROOT)))
    R.model["backend"] = data.get("backend", "inference_api")
    _save_settings(data)
    return {"ok": True, "settings": data}


@app.get("/api/logs")
def api_logs(kind: str = "status", limit: int = 200):
    if kind not in R.logs:
        raise HTTPException(status_code=400, detail="Invalid log kind")
    return {"kind": kind, "items": list(R.logs[kind])[-limit:]}


@app.post("/api/logs/clear")
def api_logs_clear(kind: str = Form("all")):
    if kind == "all":
        for k in R.logs:
            R.logs[k].clear()
    elif kind in R.logs:
        R.logs[kind].clear()
    else:
        raise HTTPException(status_code=400, detail="Invalid log kind")
    return {"ok": True}


@app.get("/api/legacy/health")
def api_legacy_health():
    errors = []
    for legacy_url in _classic_ui_candidates():
        req = urllib.request.Request(legacy_url, headers={"Accept": "text/html"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                code = int(getattr(resp, "status", 200) or 200)
                return {"ok": True, "url": legacy_url, "status_code": code, "message": "Classic app reachable"}
        except Exception as e:
            errors.append(f"{legacy_url} -> {e}")
    return {
        "ok": False,
        "url": CLASSIC_UI_URL,
        "status_code": 0,
        "message": " ; ".join(errors) if errors else "Classic app unreachable",
    }


def _tkdialog_folder() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="Select folder")
    finally:
        root.destroy()
    return path or ""


def _tkdialog_file(accept: str = "") -> str:
    import tkinter as tk
    from tkinter import filedialog

    ftypes = [("All files", "*.*")]
    if accept:
        pats = []
        for ext in str(accept).split(","):
            ext = ext.strip()
            if not ext:
                continue
            pats.append(f"*{ext}" if ext.startswith(".") else ext)
        if pats:
            ftypes = [("Accepted", " ".join(pats)), ("All files", "*.*")]

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(title="Select file", filetypes=ftypes)
    finally:
        root.destroy()
    return path or ""


@app.post("/api/browse/folder")
async def api_browse_folder():
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(_DIALOG_EXECUTOR, _tkdialog_folder)
    return {"path": path}


@app.post("/api/browse/file")
async def api_browse_file(payload: dict = {}):
    accept = str((payload or {}).get("accept") or "")
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(_DIALOG_EXECUTOR, _tkdialog_file, accept)
    return {"path": path}


@app.get("/api/docs")
async def api_docs(root: str | None = None):
    rr = Path(root) if root else R.data_root
    if root:
        R.settings["data_root"] = str(rr)
        _save_settings(R.settings)
    R.data_root = rr

    def _scan():
        R.docs = scan_docs(rr)
        names = sorted(R.docs.keys())
        return names

    names = await asyncio.get_event_loop().run_in_executor(_SEARCH_EXECUTOR, _scan)
    return {
        "data_root": str(rr),
        "documents": names,
        "coverage": {n: coverage_for_doc(n) for n in names},
        "pages_map": {n: pages_map_for_doc(n) for n in names},
    }


@app.get("/api/page/view")
def api_page_view(doc_name: str, page: int):
    doc = (doc_name or "").strip()
    if not doc:
        raise HTTPException(status_code=400, detail="doc_name required")
    if not R.docs:
        R.docs = scan_docs(R.data_root)
    info = page_content_for(doc, int(page))
    if not info:
        raise HTTPException(status_code=404, detail="Page not found")

    qdoc = urllib.parse.quote(doc, safe="")
    qpage = urllib.parse.quote(str(int(page)), safe="")
    files = {
        "pdf_url": f"/api/page/file?doc_name={qdoc}&page={qpage}&file_type=pdf" if info.get("has_pdf") else "",
        "json_url": f"/api/page/file?doc_name={qdoc}&page={qpage}&file_type=json" if info.get("has_json") else "",
        "md_url": f"/api/page/file?doc_name={qdoc}&page={qpage}&file_type=md" if info.get("has_md") else "",
        "image_url": f"/api/page/file?doc_name={qdoc}&page={qpage}&file_type=image" if info.get("has_image") else "",
    }
    return {
        "doc_name": doc,
        "page": int(page),
        "source_type": info.get("source_type", ""),
        "source_name": info.get("source_name", ""),
        "raw_text": info.get("raw_text", ""),
        "raw_json": info.get("raw_json", ""),
        "tables": info.get("tables", []),
        "files": files,
    }


@app.get("/api/page/file")
def api_page_file(doc_name: str, page: int, file_type: str):
    doc = (doc_name or "").strip()
    ftype = (file_type or "").strip().lower()
    if ftype not in {"pdf", "json", "md", "image"}:
        raise HTTPException(status_code=400, detail="file_type must be pdf/json/md/image")
    if not R.docs:
        R.docs = scan_docs(R.data_root)
    files = R.docs.get(doc, {}).get(int(page), {})
    target_key = ftype
    if ftype == "image":
        target_key = ""
        for k in ("png", "jpg", "jpeg", "webp"):
            if k in files:
                target_key = k
                break
    if target_key not in files:
        raise HTTPException(status_code=404, detail="Requested file type not found for page")
    path = files[target_key]
    media = {
        "pdf": "application/pdf",
        "json": "application/json",
        "md": "text/markdown",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(target_key, "application/octet-stream")
    return FileResponse(str(path), media_type=media)


@app.post("/api/rag/build")
def api_rag_build(scope: str = Form("all"), doc_name: str = Form("")):
    if not R.docs:
        R.docs = scan_docs(R.data_root)

    sc = (scope or "all").strip().lower()
    target_doc = (doc_name or "").strip()
    R.progress["index"]["running"] = True

    try:
        if sc == "selected" and target_doc:
            set_progress("index", 10, "building", f"Building RAG index for {target_doc} from extracted JSON")
            ok = build_doc_index(target_doc)
            if not ok:
                set_progress("index", 100, "failed", f"No extracted JSON/MD chunks for {target_doc}")
                return {"ok": False, "message": f"No extracted chunks found for {target_doc}"}
            set_progress("index", 100, "completed", f"RAG index ready for {target_doc}")
            log("status", f"RAG index built for {target_doc}")
            return {"ok": True, "message": f"RAG index built for {target_doc} (JSON-first)", "scope": "selected"}

        names = sorted(R.docs.keys())
        if not names:
            set_progress("index", 100, "failed", "No extracted docs found for index build")
            return {"ok": False, "message": "No docs found"}
        set_progress("index", 10, "building", f"Building global RAG from extracted JSON for {len(names)} docs")
        n = build_global_index(names)
        if n <= 0:
            set_progress("index", 100, "failed", "No extracted chunks found for global build")
            return {"ok": False, "message": "No extracted chunks found"}
        set_progress("index", 100, "completed", f"Global RAG index ready ({n} chunks)")
        log("status", f"Global RAG index built with {n} chunks")
        return {"ok": True, "message": f"Global RAG index built from extracted JSON ({n} chunks)", "scope": "all"}
    finally:
        R.progress["index"]["running"] = False


@app.get("/api/rag/status")
def api_rag_status():
    all_obj = load_index(ns("__ALL__"))
    all_chunks = len(all_obj[2]) if all_obj else 0
    return {
        "global_index_ready": bool(all_obj),
        "global_chunks": all_chunks,
        "current_doc": R.current_doc,
    }


@app.post("/api/tables/export")
def api_tables_export(req: ExportTablesReq):
    fmt = str(req.format or "excel").strip().lower()
    if fmt in {"xls", "xlsx"}:
        fmt = "excel"
    if fmt in {"doc", "docx"}:
        fmt = "word"
    if fmt not in {"excel", "word"}:
        raise HTTPException(status_code=400, detail="format must be excel or word")

    tables = list(req.tables or [])
    path, media = _export_tables_document(tables, str(req.title or "SCAL Combined Tables"), fmt)
    fname = path.name
    headers = {"Content-Disposition": f"attachment; filename={fname}"}
    return FileResponse(str(path), media_type=media, filename=fname, headers=headers)


@app.get("/api/models/options")
def api_models_options():
    backend = str(R.settings.get("backend", "inference_api"))
    try:
        if backend == "ollama":
            resp = _ollama_request("GET", "/api/tags", timeout=8)
            models = []
            for m in resp.get("models", []):
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name") or "")
                if not name:
                    continue
                models.append({"name": name, "label": name})
            return {
                "models": models,
                "default": R.model.get("model_name", "") or (models[0]["name"] if models else ""),
                "active": R.model.get("model_name", ""),
                "backend": "ollama",
            }

        if backend == "localai":
            names = _localai_models_list(timeout=8)
            models = [{"name": n, "label": n} for n in names]
            active_name = str(R.model.get("model_name") or "")
            return {
                "models": models,
                "default": active_name or (models[0]["name"] if models else ""),
                "active": active_name,
                "backend": "localai",
            }

        resp = _inference_request("GET", "/v1/models", timeout=8)
        return {
            "models": resp.get("models", []),
            "default": resp.get("default", ""),
            "active": resp.get("active", ""),
            "backend": "inference_api",
        }
    except Exception as e:
        return {"models": [], "default": "", "active": "", "backend": backend, "error": str(e)}


def _ollama_pull_worker(model_name: str):
    try:
        R.model.update({"loading": True, "target_model": model_name, "last_error": "", "backend": "ollama"})
        set_progress("model", 5, "pulling", f"Pulling {model_name} from Ollama registry")
        _ollama_request("POST", "/api/pull", {"name": model_name, "stream": False}, timeout=3600)
        set_progress("model", 90, "finalizing", f"Finalizing {model_name}")
        R.model.update({"loading": False, "target_model": "", "model_name": model_name, "loaded": True})
        set_progress("model", 100, "completed", f"Model ready in Ollama: {model_name}")
        log("status", f"Ollama pull completed: {model_name}")
    except Exception as e:
        R.model.update({"loading": False, "target_model": "", "loaded": False, "last_error": str(e)})
        set_progress("model", 100, "failed", str(e))
        log("error", f"Ollama pull failed: {e}")


@app.post("/api/models/pull")
def api_models_pull(model_name: str = Form(...)):
    backend = str(R.settings.get("backend", "inference_api"))
    if backend not in {"ollama", "localai"}:
        return {"ok": False, "message": "Model pull uses Ollama and is available for backend=ollama or localai"}
    name = (model_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model_name required")
    if R.model.get("loading"):
        return {"ok": False, "message": "Another model operation is running"}
    threading.Thread(target=_ollama_pull_worker, args=(name,), daemon=True).start()
    return {"ok": True, "message": f"Started pulling {name} via Ollama model loader"}


@app.post("/api/models/switch")
def api_models_switch(model_name: str = Form(...)):
    backend = str(R.settings.get("backend", "inference_api"))
    name = (model_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model_name required")

    if backend == "ollama":
        try:
            tags = _ollama_request("GET", "/api/tags", timeout=8)
            names = {str(m.get("name") or "") for m in tags.get("models", []) if isinstance(m, dict)}
            if name not in names:
                if R.model.get("loading"):
                    return {"ok": False, "message": "Another model operation is running"}
                threading.Thread(target=_ollama_pull_worker, args=(name,), daemon=True).start()
                return {"ok": True, "message": f"Model {name} not local yet. Started Ollama pull.", "backend": "ollama", "pull_started": True}
            R.model.update({"model_name": name, "loaded": True, "loading": False, "target_model": "", "last_error": "", "backend": "ollama"})
            set_progress("model", 100, "completed", f"Active model: {name} (Ollama)")
            return {"ok": True, "message": f"Switched model to {name} (Ollama)", "backend": "ollama"}
        except Exception as e:
            R.model.update({"loaded": False, "last_error": str(e), "backend": "ollama"})
            return {"ok": False, "message": str(e), "backend": "ollama"}

    if backend == "localai":
        try:
            names = set(_localai_models_list(timeout=8))
            if name not in names:
                return {
                    "ok": False,
                    "message": f"Model {name} not found in LocalAI model list",
                    "backend": "localai",
                }
            R.model.update({"model_name": name, "loaded": True, "loading": False, "target_model": "", "last_error": "", "backend": "localai"})
            set_progress("model", 100, "completed", f"Active model: {name} (LocalAI)")
            return {"ok": True, "message": f"Switched model to {name} (LocalAI)", "backend": "localai"}
        except Exception as e:
            R.model.update({"loaded": False, "last_error": str(e), "backend": "localai"})
            return {"ok": False, "message": str(e), "backend": "localai"}

    try:
        resp = _inference_request("POST", "/v1/models/load", {"model_name": name}, timeout=15)
        sync_model_state()
        return {"ok": bool(resp.get("ok", False)), "message": resp.get("message", ""), "backend": "inference_api"}
    except Exception as e:
        return {"ok": False, "message": str(e), "backend": "inference_api"}


@app.post("/api/models/unload")
def api_models_unload():
    backend = str(R.settings.get("backend", "inference_api"))
    if backend == "ollama":
        old = R.model.get("model_name", "")
        R.model.update({"loaded": False, "model_name": "", "target_model": "", "loading": False, "last_error": "", "backend": "ollama"})
        set_progress("model", 0, "idle", "No active Ollama model")
        return {"ok": True, "message": f"Ollama model context cleared ({old})"}

    if backend == "localai":
        old = R.model.get("model_name", "")
        R.model.update({"loaded": False, "model_name": "", "target_model": "", "loading": False, "last_error": "", "backend": "localai"})
        set_progress("model", 0, "idle", "No active LocalAI model")
        return {"ok": True, "message": f"LocalAI model context cleared ({old})"}

    try:
        resp = _inference_request("POST", "/v1/models/unload", {}, timeout=15)
        sync_model_state()
        return {"ok": bool(resp.get("ok", False)), "message": resp.get("message", "")}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.get("/api/chat/sessions")
def api_chat_sessions():
    return {"sessions": list_sessions()}


@app.post("/api/chat/session/new")
def api_chat_session_new(title: str = Form("")):
    s = create_session(title)
    return {"session": s}


@app.get("/api/chat/session/{session_id}")
def api_chat_session_get(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": s}


@app.post("/api/chat/session/{session_id}/title")
def api_chat_session_title(session_id: str, req: SessionTitleReq):
    data = _load_sessions()
    for s in data.get("sessions", []):
        if s.get("id") == session_id:
            title = str(req.title or "").strip() or "SCAL Chat"
            s["title"] = title[:80]
            s["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _save_sessions(data)
            return {"ok": True, "session": s}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/api/chat/session/{session_id}")
def api_chat_session_delete(session_id: str):
    data = _load_sessions()
    before = len(data.get("sessions", []))
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("id") != session_id]
    if len(data["sessions"]) == before:
        raise HTTPException(status_code=404, detail="Session not found")
    _save_sessions(data)
    return {"ok": True}


@app.get("/api/chat/session/{session_id}/export")
def api_chat_session_export(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    text = session_to_text(s)
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s.get("title") or "SCAL_Chat")).strip("_") or "SCAL_Chat"
    filename = f"{base}_{session_id}.txt"
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatReq):
    t_total0 = datetime.now()
    backend = str(R.settings.get("backend", "inference_api"))
    log("debug", f"chat.stream backend={backend} scope={req.scope} mode={req.response_mode}")
    filters = {"extraction_type": req.filter_extraction_type}
    mode = (req.response_mode or "fast").lower()
    gen_cfg = llm_generation_settings(mode)
    top_k = max(1, int(req.top_k or 24))

    if not R.docs:
        R.docs = scan_docs(R.data_root)

    doc_name = (req.doc_name or "").strip()
    scope = (req.scope or "all").lower()
    target_doc = "__ALL__"
    if scope == "selected" and doc_name:
        target_doc = doc_name

    retrieval_t0 = datetime.now()
    hits: list[dict[str, Any]] = []
    if not is_casual_chat(req.question):
        hits = await asyncio.get_event_loop().run_in_executor(
            _SEARCH_EXECUTOR,
            lambda: search(req.question, target_doc, filters, top_k=top_k),
        )
    retrieval_ms = (datetime.now() - retrieval_t0).total_seconds() * 1000.0
    log("debug", f"retrieval hits={len(hits)} retrieval_ms={retrieval_ms:.2f}")

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
                "snippet": str(h.get("text", ""))[:320],
            }
        )

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

    vision_images: list[str] = []
    if bool(req.use_pdf_vision):
        seen = set()
        for h in hits:
            m = h["meta"]
            doc = str(m.get("report_name") or "")
            page = int(m.get("page_number") or 0)
            key = (doc, page)
            if not doc or page <= 0 or key in seen:
                continue
            seen.add(key)
            files = R.docs.get(doc, {}).get(page, {})
            img_path = _first_image_for_page(files)
            if img_path is None:
                continue
            try:
                vision_images.append(_as_data_url(img_path))
            except Exception:
                continue
            if len(vision_images) >= 4:
                break

    if is_casual_chat(req.question):
        system = (
            "You are a friendly SCAL assistant. For casual chat, respond naturally and briefly. "
            "Do not cite files unless asked document questions."
        )
        user_prompt = req.question
        prefix = ""
    elif not hits:
        system = (
            "You are a helpful assistant in a SCAL app. If no document evidence is available, "
            "answer generally and clearly state it is not grounded in retrieved files."
        )
        user_prompt = req.question
        prefix = "(No retrieved document context found; general model response)\n\n"
    else:
        ctx = []
        for i, h in enumerate(hits, start=1):
            m = h["meta"]
            txt = h["text"]
            if len(txt) > 1400:
                txt = txt[:1400] + "..."
            block = f"[{i}] file={m.get('file_name')} page={m.get('page_number')} table={m.get('table_id')}\n{txt}"
            raw_html = str(m.get("raw_html") or "")
            if req.include_table_html and raw_html:
                if len(raw_html) > 2200:
                    raw_html = raw_html[:2200] + "..."
                block += f"\n\nHTML_TABLE:\n{raw_html}"
            ctx.append(block)
        context = "\n\n".join(ctx)
        system = (
            "You are a SCAL assistant focused on PDF extraction results. "
            "Treat HTML tables inside <table>...</table> as primary source structure, "
            "use retrieved evidence only, and cite [1],[2] references in answers."
        )
        task_hint = (req.prompt_template or "").strip()
        if task_hint:
            user_prompt = f"Task:\n{task_hint}\n\nQuestion:\n{req.question}\n\nEvidence:\n{context}"
        else:
            user_prompt = f"Question:\n{req.question}\n\nEvidence:\n{context}"
        prefix = ""

    sid = req.session_id
    if not sid:
        sid = create_session("SCAL Chat").get("id")

    payload = {
        "system_prompt": system,
        "user_prompt": user_prompt,
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 420)),
        "temperature": float(gen_cfg.get("temperature", 0.2)),
        "top_p": float(gen_cfg.get("top_p", 0.9)),
        "do_sample": bool(gen_cfg.get("do_sample", True)),
    }

    def _sse(evt: dict[str, Any]) -> str:
        return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    def streamer():
        answer_parts: list[str] = []
        infer_metrics: dict[str, Any] = {}
        try:
            if prefix:
                answer_parts.append(prefix)
                yield _sse({"type": "token", "text": prefix})

            if backend == "ollama":
                model_name = str(R.model.get("model_name") or "").strip()
                if not model_name:
                    tags = _ollama_request("GET", "/api/tags", timeout=8)
                    models = [str(m.get("name") or "") for m in tags.get("models", []) if isinstance(m, dict)]
                    if not models:
                        raise RuntimeError("No local Ollama models found. Pull a model first.")
                    model_name = models[0]
                    R.model.update({"model_name": model_name, "loaded": True, "backend": "ollama"})

                prompt = f"System:\n{system}\n\nUser:\n{user_prompt}"
                opayload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": int(gen_cfg.get("max_new_tokens", 420)),
                        "temperature": float(gen_cfg.get("temperature", 0.2)),
                        "top_p": float(gen_cfg.get("top_p", 0.9)),
                    },
                }
                if req.use_pdf_vision and vision_images:
                    opayload["images"] = [x.split(",", 1)[1] for x in vision_images]

                first_token_ms = None
                t_gen0 = datetime.now()
                for obj in _ollama_stream_generate(opayload, timeout=1200):
                    piece = str(obj.get("response") or "")
                    if piece:
                        if first_token_ms is None:
                            first_token_ms = (datetime.now() - t_gen0).total_seconds() * 1000.0
                        answer_parts.append(piece)
                        yield _sse({"type": "token", "text": piece})
                    if obj.get("done"):
                        eval_count = int(obj.get("eval_count") or 0)
                        eval_duration = float(obj.get("eval_duration") or 0)
                        gen_ms = eval_duration / 1_000_000.0 if eval_duration > 0 else (datetime.now() - t_gen0).total_seconds() * 1000.0
                        tok_s = (eval_count / max(eval_duration / 1_000_000_000.0, 1e-6)) if eval_duration > 0 else 0.0
                        infer_metrics = {
                            "generation_ms": round(gen_ms, 2),
                            "first_token_ms": round(float(first_token_ms or gen_ms), 2),
                            "answer_tokens": int(eval_count),
                            "tokens_per_sec": round(tok_s, 2),
                        }
                        break
            elif backend == "localai":
                model_name = str(R.model.get("model_name") or "").strip()
                if not model_name:
                    names = _localai_models_list(timeout=8)
                    if not names:
                        raise RuntimeError("No LocalAI models found. Start LocalAI and load at least one model.")
                    model_name = names[0]
                    R.model.update({"model_name": model_name, "loaded": True, "backend": "localai"})

                cpayload = {
                    "model": model_name,
                    "messages": [],
                    "stream": True,
                    "temperature": float(gen_cfg.get("temperature", 0.2)),
                    "top_p": float(gen_cfg.get("top_p", 0.9)),
                    "max_tokens": int(gen_cfg.get("max_new_tokens", 420)),
                }
                if req.use_pdf_vision and vision_images:
                    cpayload["messages"] = [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}]
                            + [{"type": "image_url", "image_url": {"url": u}} for u in vision_images],
                        },
                    ]
                else:
                    cpayload["messages"] = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ]

                t_gen0 = datetime.now()
                first_token_ms = None
                seen_tokens = 0
                for obj in _compat_stream_chat(LOCALAI_BASE_URL, cpayload, timeout=1200):
                    choices = obj.get("choices", []) if isinstance(obj, dict) else []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        piece = str(delta.get("content") or "")
                        if piece:
                            if first_token_ms is None:
                                first_token_ms = (datetime.now() - t_gen0).total_seconds() * 1000.0
                            seen_tokens += approx_token_count(piece)
                            answer_parts.append(piece)
                            yield _sse({"type": "token", "text": piece})

                    usage = obj.get("usage") if isinstance(obj, dict) else None
                    if isinstance(usage, dict):
                        answer_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or seen_tokens)
                        gen_ms = (datetime.now() - t_gen0).total_seconds() * 1000.0
                        infer_metrics = {
                            "generation_ms": round(gen_ms, 2),
                            "first_token_ms": round(float(first_token_ms or gen_ms), 2),
                            "answer_tokens": int(answer_tokens),
                            "tokens_per_sec": round(answer_tokens / max(gen_ms / 1000.0, 1e-6), 2),
                        }
                if not infer_metrics:
                    gen_ms = (datetime.now() - t_gen0).total_seconds() * 1000.0
                    answer_tokens = approx_token_count("".join(answer_parts))
                    infer_metrics = {
                        "generation_ms": round(gen_ms, 2),
                        "first_token_ms": round(float(first_token_ms or gen_ms), 2),
                        "answer_tokens": int(answer_tokens),
                        "tokens_per_sec": round(answer_tokens / max(gen_ms / 1000.0, 1e-6), 2),
                    }
            else:
                for ev in _inference_stream_events("/v1/chat/stream", payload, timeout=600):
                    if ev.get("type") == "token":
                        piece = str(ev.get("text") or "")
                        if piece:
                            answer_parts.append(piece)
                            yield _sse({"type": "token", "text": piece})
                    elif ev.get("type") == "metrics":
                        infer_metrics = ev.get("metrics") or {}

            answer = "".join(answer_parts).strip()
            append_session_messages(
                sid,
                [
                    {"role": "user", "content": req.question, "time": now()},
                    {"role": "assistant", "content": answer, "time": now(), "sources": reasoning},
                ],
            )
            log("debug", f"chat.stream complete tokens={infer_metrics.get('answer_tokens', 0)}")

            generation_ms = float(infer_metrics.get("generation_ms", 0) or 0)
            answer_tokens = int(infer_metrics.get("answer_tokens", approx_token_count(answer)) or 0)
            tok_s = float(infer_metrics.get("tokens_per_sec", 0) or 0)
            total_ms = (datetime.now() - t_total0).total_seconds() * 1000.0

            yield _sse(
                {
                    "type": "done",
                    "session_id": sid,
                    "answer": answer,
                    "reasoning": reasoning,
                    "sources": reasoning,
                    "tables": tables,
                    "raw_hits": hits,
                    "vision_images_used": len(vision_images),
                    "metrics": {
                        "backend": backend,
                        "response_mode": mode,
                        "model_name": R.model.get("model_name", ""),
                        "retrieval_ms": round(retrieval_ms, 2),
                        "generation_ms": round(generation_ms, 2),
                        "first_token_ms": round(float(infer_metrics.get("first_token_ms", generation_ms) or generation_ms), 2),
                        "total_ms": round(total_ms, 2),
                        "answer_tokens": answer_tokens,
                        "tokens_per_sec": round(tok_s, 2) if tok_s else round((answer_tokens / max(generation_ms / 1000.0, 1e-6)), 2),
                        "hits": len(hits),
                        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 0)),
                    },
                }
            )
        except Exception as e:
            log("error", f"stream failed: {e}")
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(streamer(), media_type="text/event-stream")
