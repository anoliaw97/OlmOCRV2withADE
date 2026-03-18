from __future__ import annotations

import gc
import os
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalChatLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.lock = Lock()

    def load(self):
        if self.loaded:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for local LLM in web app")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to("cuda").eval()
        self.loaded = True

    def ask(self, prompt: str, system: str | None = None, max_new_tokens: int = 700) -> str:
        with self.lock:
            if not self.loaded:
                self.load()

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to("cuda", dtype=torch.long)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                )

            text = self.tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
            del input_ids, out
            torch.cuda.empty_cache()
            gc.collect()
            return text.strip()


PRIMARY_RAG_MODEL = os.getenv("SCAL_RAG_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USECASE_MODEL = os.getenv("SCAL_USECASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

_rag_llm = LocalChatLLM(PRIMARY_RAG_MODEL)
_usecase_llm = LocalChatLLM(USECASE_MODEL)


def get_rag_llm() -> LocalChatLLM:
    return _rag_llm


def get_usecase_llm() -> LocalChatLLM:
    return _usecase_llm
