from __future__ import annotations

from fastapi import APIRouter

from ..services.web_olmocr_runtime import default_olmocr_prompt, get_llm, get_vlm


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/status")
def model_status():
    vlm = get_vlm()
    llm = get_llm()
    return {
        "vlm_loaded": vlm.loaded,
        "vlm_model": vlm.model_id,
        "llm_loaded": llm.loaded,
        "llm_model": llm.model_id,
        "default_prompt": default_olmocr_prompt(),
    }


@router.post("/load-vlm")
def load_vlm():
    vlm = get_vlm()
    try:
        vlm.load()
        return {"ok": True, "vlm_loaded": True, "vlm_model": vlm.model_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/load-llm")
def load_llm():
    llm = get_llm()
    try:
        llm.load()
        return {"ok": True, "llm_loaded": True, "llm_model": llm.model_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}
