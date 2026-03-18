from __future__ import annotations

import base64
import gc
from io import BytesIO
from threading import Lock

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt


MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class VLMRuntime:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False
        self.model_id = MODEL_ID
        self.lock = Lock()

    def load(self):
        with self.lock:
            if self.loaded:
                return
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required for VLM")

            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()
            self.loaded = True

    def extract_page(self, pdf_path: str, page: int, prompt: str | None = None) -> dict:
        with self.lock:
            if not self.loaded:
                raise RuntimeError("VLM not loaded. Click Load VLM first.")

            use_prompt = prompt or build_no_anchoring_v4_yaml_prompt()
            image_base64 = render_pdf_to_base64png(pdf_path, page, target_longest_image_dim=1288)
            pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": use_prompt},
                        {"type": "image", "image": pil_image},
                    ],
                }
            ]

            chat_text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[chat_text], images=[pil_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            input_tokens = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=900, do_sample=True, temperature=0.1)

            output_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
            decoded = self.processor.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

            del inputs, out, pil_image, msgs
            gc.collect()
            torch.cuda.empty_cache()

            return {
                "raw_response": decoded,
                "prompt_used": use_prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }


class LLMRuntime:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.model_id = LLM_MODEL_ID
        self.lock = Lock()

    def load(self):
        with self.lock:
            if self.loaded:
                return
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required for LLM")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16).to("cuda").eval()
            self.loaded = True


_vlm = VLMRuntime()
_llm = LLMRuntime()


def get_vlm() -> VLMRuntime:
    return _vlm


def get_llm() -> LLMRuntime:
    return _llm


def default_olmocr_prompt() -> str:
    return build_no_anchoring_v4_yaml_prompt()
