from __future__ import annotations

import os
import re
import threading
import time
import traceback
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


APP_TITLE = "SCAL Local Inference API"
DEFAULT_MODEL = os.environ.get("SCAL_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

MODEL_CACHE_DIRS: dict[str, Path] = {
    "moonshotai/Kimi-K2.5": Path(r"D:\hf_cache\moonshotai\Kimi-K2.5"),
}

LLM_MODEL_OPTIONS = [
    {"name": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5-7B-Instruct"},
    {"name": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5-14B-Instruct"},
    {"name": "Qwen/Qwen3-30B-A3B-Instruct-2507", "label": "Qwen3-30B-A3B-Instruct-2507"},
    {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "label": "DeepSeek-R1-Distill-Qwen-32B"},
    {"name": "zai-org/GLM-4-32B-0414", "label": "GLM-4-32B-0414"},
    {"name": "moonshotai/Kimi-K2.5", "label": "Kimi-K2.5"},
]


class Runtime:
    def __init__(self):
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._tok = None
        self._model = None
        self.loaded = False
        self.model_name = DEFAULT_MODEL
        self.target_model = ""
        self.state = "idle"  # idle/loading/loaded/unloading/failed
        self.progress = {"percent": 0, "stage": "idle", "detail": ""}
        self.last_error = ""
        self.last_metrics: dict[str, Any] = {}


R = Runtime()
app = FastAPI(title=APP_TITLE)


class LoadReq(BaseModel):
    model_name: str


class ChatReq(BaseModel):
    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 420
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


def _set_progress(percent: int, stage: str, detail: str = ""):
    R.progress = {
        "percent": max(0, min(100, int(percent))),
        "stage": stage,
        "detail": detail,
    }


def _ensure_kimi_requirements(model_name: str):
    if not model_name.startswith("moonshotai/Kimi-K2"):
        return
    import transformers

    raw_ver = getattr(transformers, "__version__", "0.0.0")
    nums = [int(x) for x in re.findall(r"\d+", raw_ver)[:3]]
    while len(nums) < 3:
        nums.append(0)
    if tuple(nums) < (4, 57, 1):
        raise RuntimeError(f"Kimi-K2.5 requires transformers>=4.57.1, found {raw_ver}")
    try:
        import tiktoken  # noqa: F401
    except Exception:
        raise RuntimeError("Kimi-K2.5 requires tiktoken (install: pip install tiktoken)")


def _load_model_impl(model_name: str):
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with R._lock:
        R.state = "loading"
        R.target_model = model_name
        R.last_error = ""
        _set_progress(5, "starting", f"Preparing load: {model_name}")

        if R._model is not None or R._tok is not None:
            R.state = "unloading"
            _set_progress(10, "cleanup", "Releasing previous model")
            R._model = None
            R._tok = None
            R.loaded = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        cache_dir = MODEL_CACHE_DIRS.get(model_name)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir.parent.parent))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir.parent.parent / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir.parent.parent / "hub"))

        _ensure_kimi_requirements(model_name)

        _set_progress(20, "downloading", f"Tokenizer/config: {model_name}")
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

        _set_progress(45, "downloading", f"Weights: {model_name}")
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            ).eval()
        except Exception as e:
            msg = str(e)
            if "torchvision::nms" in msg or "Could not import module 'Qwen2ForCausalLM'" in msg:
                raise RuntimeError(
                    "Model load failed due to torchvision/transformers mismatch. "
                    "Uninstall torchvision and torchaudio in the env, then retry."
                )
            if "gated repo" in msg.lower() or "401" in msg:
                raise RuntimeError(
                    f"Model {model_name} is gated/private. Choose an open model or login to Hugging Face."
                )
            raise

        _set_progress(90, "finalizing", "Finishing load")
        R._tok = tok
        R._model = mdl
        R.loaded = True
        R.model_name = model_name
        R.target_model = ""
        R.state = "loaded"
        _set_progress(100, "completed", f"Model ready: {model_name}")


def _load_model_bg(model_name: str):
    try:
        _load_model_impl(model_name)
    except Exception as e:
        R.loaded = False
        R.state = "failed"
        R.target_model = ""
        R.last_error = str(e)
        _set_progress(100, "failed", str(e))
        traceback.print_exc()


def _unload_model_impl():
    import torch

    with R._lock:
        R.state = "unloading"
        _set_progress(20, "unloading", "Releasing model")
        R._model = None
        R._tok = None
        R.loaded = False
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        R.state = "idle"
        _set_progress(100, "completed", "Model unloaded")


@app.get("/v1/health")
def api_health():
    return {
        "ok": True,
        "state": R.state,
        "loaded": R.loaded,
        "model_name": R.model_name,
        "target_model": R.target_model,
        "last_error": R.last_error,
        "progress": R.progress,
        "busy": R._lock.locked() or R._gen_lock.locked(),
    }


@app.get("/v1/models")
def api_models():
    return {"models": LLM_MODEL_OPTIONS, "default": DEFAULT_MODEL, "active": R.model_name if R.loaded else ""}


@app.post("/v1/models/load")
def api_model_load(req: LoadReq):
    name = (req.model_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if R._lock.locked():
        return {"ok": False, "message": "Operation in progress", "state": R.state}
    if R.loaded and R.model_name == name:
        return {"ok": True, "message": "Model already loaded", "state": R.state}
    threading.Thread(target=_load_model_bg, args=(name,), daemon=True).start()
    return {"ok": True, "message": f"Loading started: {name}", "state": "loading"}


@app.post("/v1/models/unload")
def api_model_unload():
    if R._lock.locked():
        return {"ok": False, "message": "Operation in progress", "state": R.state}
    if not R.loaded and R._model is None and R._tok is None:
        return {"ok": True, "message": "Model already unloaded", "state": R.state}
    _unload_model_impl()
    return {"ok": True, "message": "Model unloaded", "state": R.state}


@app.post("/v1/chat/completions")
def api_chat(req: ChatReq):
    import torch

    if not R.loaded or R._tok is None or R._model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    msgs = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.user_prompt},
    ]
    t0 = time.perf_counter()
    with R._gen_lock:
        inp = R._tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda", dtype=torch.long)
        with torch.no_grad():
            out = R._model.generate(
                inp,
                max_new_tokens=int(req.max_new_tokens),
                temperature=float(req.temperature),
                do_sample=bool(req.do_sample),
                top_p=float(req.top_p),
            )
        answer = R._tok.decode(out[0][inp.shape[1] :], skip_special_tokens=True).strip()
    dt = (time.perf_counter() - t0) * 1000.0
    out_tokens = len(R._tok.encode(answer)) if answer else 0
    R.last_metrics = {
        "generation_ms": round(dt, 2),
        "answer_tokens": int(out_tokens),
        "tokens_per_sec": round((out_tokens / max(dt / 1000.0, 1e-6)), 2),
    }
    return {
        "ok": True,
        "model_name": R.model_name,
        "answer": answer,
        "metrics": R.last_metrics,
    }


@app.post("/v1/chat/stream")
def api_chat_stream(req: ChatReq):
    import torch
    from transformers import TextIteratorStreamer

    if not R.loaded or R._tok is None or R._model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    def event_stream():
        msgs = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt},
        ]
        with R._gen_lock:
            inp = R._tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda", dtype=torch.long)
            streamer = TextIteratorStreamer(R._tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = {
                "input_ids": inp,
                "max_new_tokens": int(req.max_new_tokens),
                "temperature": float(req.temperature),
                "do_sample": bool(req.do_sample),
                "top_p": float(req.top_p),
                "streamer": streamer,
            }

            answer_parts: list[str] = []

            def _producer():
                try:
                    with torch.no_grad():
                        R._model.generate(**kwargs)
                except Exception:
                    pass

            t0 = time.perf_counter()
            first_token_ms = None
            threading.Thread(target=_producer, daemon=True).start()

            for piece in streamer:
                if piece:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000.0
                    answer_parts.append(piece)
                    payload = {"type": "token", "text": piece}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            answer = "".join(answer_parts).strip()
            total_ms = (time.perf_counter() - t0) * 1000.0
            out_tokens = len(R._tok.encode(answer)) if answer else 0
            metrics = {
                "generation_ms": round(total_ms, 2),
                "first_token_ms": round(float(first_token_ms or total_ms), 2),
                "answer_tokens": int(out_tokens),
                "tokens_per_sec": round((out_tokens / max(total_ms / 1000.0, 1e-6)), 2),
            }
            R.last_metrics = metrics
            yield f"data: {json.dumps({'type': 'metrics', 'metrics': metrics}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/metrics")
def api_metrics():
    return {"ok": True, "model_name": R.model_name if R.loaded else "", "metrics": R.last_metrics}
