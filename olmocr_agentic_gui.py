"""
olmOCR Agentic Document Extraction GUI - Integrated Version
=========================================================
Uses olmOCR package for PDF rendering and optimized prompts

Run from: C:/Users/Mining/Downloads/olmocr-main
Usage: python olmocr_agentic_gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import json
import threading
from datetime import datetime
from PIL import Image, ImageTk
import torch
import gc
import time
import tempfile
import base64
import io
import pandas as pd

# Add current directory to path for olmocr package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    OLMCOCR_AVAILABLE = True
except ImportError as e:
    OLMCOCR_AVAILABLE = False
    print(f"olmOCR package not found: {e}")

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try importing openai for Groq
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_OLMOCR_PROMPT = build_no_anchoring_v4_yaml_prompt() if OLMCOCR_AVAILABLE else """Attached is one page of a document. Return markdown with front matter: primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram."""

TABLE_PROMPT = """Extract table data from the image as JSON array. If no table: {"no_table": true}. Include all rows and columns."""

# Structure prompt for LLM to convert extracted text to structured format
STRUCTURE_PROMPT = """You are a data extraction expert. Convert the extracted document text into a well-structured JSON format.

Extract:
- Tables as JSON arrays
- Key-value pairs as objects
- Lists as arrays
- Headers and sections

Return ONLY valid JSON, no explanation."""

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resize_image(img, size=1288):
    """Resize image to target longest dimension (as per olmOCR spec)"""
    w, h = img.size
    if max(w, h) == size:
        return img
    if w > h:
        return img.resize((size, int(h * size / w)), Image.LANCZOS)
    return img.resize((int(w * size / h), size), Image.LANCZOS)

class VLMExtractor:
    def __init__(self, model_name=MODEL_ID):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.loaded = False

    def load_model(self):
        if self.loaded:
            return "already loaded"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"Loading model on {gpu_mem:.0f}GB GPU...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name, torch_dtype=torch.float16)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to("cuda").eval()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.loaded = True
        return f"loaded {self.model_name} ({gpu_mem:.0f}GB)"

    def extract(self, image_base64=None, image_path=None, prompt=None):
        if not self.loaded:
            self.load_model()
        
        clear_gpu()
        
        if image_base64:
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif image_path:
            img = Image.open(image_path).convert('RGB')
        else:
            raise ValueError("Must provide image_base64 or image_path")
        
        img = resize_image(img, 1288)  # Use olmOCR standard 1288px
        
        if prompt is None:
            prompt = DEFAULT_OLMOCR_PROMPT
        
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]}]
        
        txt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[txt], images=[img], padding=True, return_tensors="pt")
        
        device = "cuda"
        # Keep pixel_values as int64 (long), only convert other tensors to float16
        inputs = {k: v.to(device, dtype=torch.int64) if k == "pixel_values" else v.to(device, dtype=torch.float16) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, 
                temperature=0.0, 
                max_new_tokens=8000,  # olmOCR spec: 8000
                do_sample=False
            )
        
        output_tokens = out.shape[1] - input_ids.shape[1]
        
        decoded = self.processor.batch_decode(out[:, input_ids.shape[1]:], 
                                             skip_special_tokens=True)[0]
        
        del inputs, out, img, msgs
        clear_gpu()
        
        return {
            "raw_response": decoded,
            "prompt_used": prompt,
            "input_tokens": input_ids.shape[1],
            "output_tokens": output_tokens,
            "total_tokens": input_ids.shape[1] + output_tokens
        }

class IntelligentAssistant:
    def __init__(self, model_name=LLM_MODEL_ID):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.history = []

    def load_model(self):
        if self.loaded:
            return "already loaded"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to("cuda").eval()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.loaded = True
        return f"loaded {self.model_name} ({gpu_mem:.0f}GB)"

    def chat(self, message, system_context=None):
        if not self.loaded:
            self.load_model()
        
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        
        for msg in self.history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to("cuda", dtype=torch.long)  # Fix dtype
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=256, temperature=0.7, do_sample=True, top_p=0.9)
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        del inputs, outputs
        torch.cuda.empty_cache()
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        return response

    def optimize_prompt(self, doc_type, goal, example_fmt=None):
        if not self.loaded:
            self.load_model()
        
        system_msg = "You are an expert at writing prompts for document extraction with VLM."
        
        user_msg = f"""Create an optimized extraction prompt for:
- Document Type: {doc_type}
- Extraction Goal: {goal}
"""
        if example_fmt:
            user_msg += f"- Expected Format: {example_fmt}"
        
        user_msg += "\n\nRespond ONLY with the prompt text."
        
        return self.chat(user_msg, system_context=system_msg)

    def clear_history(self):
        self.history = []


class APILLM:
    """API-based LLM (Groq/OpenAI) for structuring output"""
    
    def __init__(self, provider="groq", api_key=None, model="llama-3.3-70b-versatile"):
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.loaded = False
        self.history = []
    
    def load_model(self):
        if not self.api_key:
            raise RuntimeError(f"No API key set for {self.provider}")
        
        if self.provider == "groq":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
            self.model = "llama-3.3-70b-versatile"
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4o-mini"
        
        self.loaded = True
        return f"loaded {self.provider} ({self.model})"
    
    def chat(self, message, system_context=None):
        if not self.loaded:
            self.load_model()
        
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        
        for msg in self.history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": result})
        
        return result
    
    def structure_data(self, extracted_text, format_type="json"):
        """Convert extracted text to structured format"""
        if format_type == "json":
            prompt = f"""Convert this extracted document text to clean JSON. Extract tables as arrays, key-value pairs as objects.

Text:
{extracted_text}

Return ONLY valid JSON, no explanation."""
        elif format_type == "excel":
            prompt = f"""Convert this extracted document text to a table format (CSV-like with columns).

Text:
{extracted_text}

Return as CSV format with headers."""
        else:
            prompt = extracted_text
        
        return self.chat(prompt, system_context=STRUCTURE_PROMPT)
    
    def clear_history(self):
        self.history = []

class OlmoCRAgenticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("olmOCR Agentic Document Extraction")
        self.root.geometry("1600x950")
        self.root.minsize(1400, 800)

        self.mode = tk.StringVar(value="single")
        self.selected_files = []
        self.pdf_pages = []
        self.selected_pages = []
        self.extracted_data = []
        self.output_dir = None
        self.current_page_idx = 0
        
        self.vlm = None
        self.llm = None
        self.stop_flag = False
        self.extracting = False
        self.current_prompt = DEFAULT_OLMOCR_PROMPT
        
        # LLM settings
        self.llm_provider = tk.StringVar(value="local")  # local, groq, openai
        self.api_key = ""
        self.api_llm = None  # API-based LLM
        self.structured_data = None  # Processed output
        
        self.setup_ui()
        
        self.log("🤖 olmOCR Agentic Extraction Ready!")
        self.log(f"olmOCR package: {'Loaded' if OLMCOCR_AVAILABLE else 'Not found'}")
        self.log("Workflow: Load VLM → Extract → Process Output → Export")

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top bar
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="📄 olmOCR Agentic Document Extraction", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(top_frame, text="Models: Not loaded", 
                                      foreground="gray", font=('Arial', 10))
        self.model_status.pack(side=tk.RIGHT, padx=10)
        
        # Control bar
        control_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="🔄 Load VLM", command=self.cmd_load_vlm, width=15).pack(side=tk.LEFT, padx=3)
        ttk.Button(control_frame, text="💬 Load LLM", command=self.cmd_load_llm, width=15).pack(side=tk.LEFT, padx=3)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Single", variable=self.mode, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Batch", variable=self.mode, value="batch").pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="📁 Select Files", command=self.select_files, width=18).pack(side=tk.LEFT, padx=10)
        self.file_label = ttk.Label(control_frame, text="No files", font=('Arial', 9))
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Select All", command=self.select_all_pages, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Deselect", command=self.deselect_all_pages, width=10).pack(side=tk.LEFT, padx=5)
        self.page_label = ttk.Label(control_frame, text="0 pages", font=('Arial', 9))
        self.page_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="💾 Output", command=self.select_output, width=12).pack(side=tk.LEFT, padx=10)
        self.output_label = ttk.Label(control_frame, text="Not set", font=('Arial', 9))
        self.output_label.pack(side=tk.LEFT, padx=5)
        
        # LLM Provider selector
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(control_frame, text="LLM:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Local", variable=self.llm_provider, value="local").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Groq", variable=self.llm_provider, value="groq").pack(side=tk.LEFT)
        
        # API Key input
        self.api_key_var = tk.StringVar(value=os.getenv("GROQ_API_KEY", ""))
        ttk.Entry(control_frame, textvariable=self.api_key_var, width=15, show="*").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Set API", command=self.set_api_key, width=8).pack(side=tk.LEFT, padx=2)
        
        # Main content
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        left_frame = ttk.Frame(main_paned, width=500)
        main_paned.add(left_frame, weight=1)
        self._create_left_panel(left_frame)
        
        right_frame = ttk.Frame(main_paned, width=900)
        main_paned.add(right_frame, weight=2)
        self._create_right_panel(right_frame)
        
        # Bottom
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _create_left_panel(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="📄 Document Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_page, width=10).pack(side=tk.LEFT)
        self.page_nav_label = ttk.Label(nav_frame, text="Page: 0 / 0", width=15)
        self.page_nav_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_page, width=10).pack(side=tk.LEFT)
        
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_canvas = tk.Canvas(canvas_frame, bg='#2d2d2d')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill=tk.Y)
        
        thumb_frame = ttk.LabelFrame(parent, text="Page Thumbnails", padding=5)
        thumb_frame.pack(fill=tk.X, pady=5)
        
        self.thumb_canvas = tk.Canvas(thumb_frame, bg='#1e1e1e', height=120)
        thumb_scroll = ttk.Scrollbar(thumb_frame, orient="horizontal", command=self.thumb_canvas.xview)
        self.thumb_canvas.configure(xscrollcommand=thumb_scroll.set)
        
        self.thumb_canvas.pack(side="left", fill=tk.X, expand=True)
        thumb_scroll.pack(fill=tk.X)
        
        self.thumb_frame_inner = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_frame_inner, anchor="nw")

    def _create_right_panel(self, parent):
        prompt_frame = ttk.LabelFrame(parent, text="📝 Extraction Prompt", padding="10")
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        prompt_top = ttk.Frame(prompt_frame)
        prompt_top.pack(fill=tk.X)
        
        ttk.Label(prompt_top, text="Type:").pack(side=tk.LEFT, padx=5)
        self.prompt_choice = ttk.Combobox(prompt_top, width=20, state="readonly")
        self.prompt_choice['values'] = ("Default (olmOCR v4)", "Table Extraction", "Custom")
        self.prompt_choice.current(0)
        self.prompt_choice.pack(side=tk.LEFT, padx=5)
        self.prompt_choice.bind('<<ComboboxSelected>>', lambda e: self.on_prompt_change())
        
        ttk.Button(prompt_top, text="✨ Optimize", command=self.optimize_prompt, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(prompt_top, text="💾 Save", command=self.save_prompt, width=8).pack(side=tk.LEFT, padx=2)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4, font=('Consolas', 9))
        self.prompt_text.pack(fill=tk.X, pady=5)
        self.prompt_text.insert("1.0", DEFAULT_OLMOCR_PROMPT)
        
        btn_frame = ttk.Frame(prompt_frame)
        btn_frame.pack(fill=tk.X)
        
        self.extract_btn = ttk.Button(btn_frame, text="▶ START EXTRACTION", command=self.start_extraction)
        self.extract_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self.stop_extraction, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Process Output buttons
        process_btn_frame = ttk.Frame(prompt_frame)
        process_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(process_btn_frame, text="🔧 Process Output (LLM)", command=self.process_output, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(process_btn_frame, text="📊 Export Excel", command=self.export_to_excel, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(process_btn_frame, text="📋 Export JSON", command=self.export_to_json, width=15).pack(side=tk.LEFT, padx=2)
        
        # Output
        output_frame = ttk.LabelFrame(parent, text="📊 Extraction Results", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        output_paned = ttk.PanedWindow(output_frame, orient=tk.HORIZONTAL)
        output_paned.pack(fill=tk.BOTH, expand=True)
        
        page_frame = ttk.Frame(output_paned)
        output_paned.add(page_frame, weight=1)
        
        ttk.Label(page_frame, text="Current Page", font=('Arial', 10, 'bold')).pack()
        self.page_preview_canvas = tk.Canvas(page_frame, bg='#1e1e1e')
        self.page_preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.page_token_var = tk.StringVar(value="Tokens: -")
        ttk.Label(page_frame, textvariable=self.page_token_var, font=('Arial', 9)).pack(pady=5)
        
        response_frame = ttk.Frame(output_paned)
        output_paned.add(response_frame, weight=2)
        
        ttk.Label(response_frame, text="Raw Response", font=('Arial', 10, 'bold')).pack()
        
        self.response_notebook = ttk.Notebook(response_frame)
        self.response_notebook.pack(fill=tk.BOTH, expand=True)
        
        raw_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(raw_frame, text="Raw Response")
        
        self.raw_text = scrolledtext.ScrolledText(raw_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.raw_text.pack(fill=tk.BOTH, expand=True)
        
        prompt_tab_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(prompt_tab_frame, text="Prompt Used")
        
        self.used_prompt_text = scrolledtext.ScrolledText(prompt_tab_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.used_prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # Chat
        chat_frame = ttk.LabelFrame(parent, text="💬 Chat & Logs", padding="5")
        chat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        chat_notebook = ttk.Notebook(chat_frame)
        chat_notebook.pack(fill=tk.X)
        
        chat_tab = ttk.Frame(chat_notebook)
        chat_notebook.add(chat_tab, text="Chat")
        
        self.chat_text = scrolledtext.ScrolledText(chat_tab, height=5, font=('Consolas', 8), bg='#1e1e2e', fg='#cdd6f4')
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.tag_config("user", foreground="#89b4fa")
        self.chat_text.tag_config("assistant", foreground="#a6e3a1")
        self.chat_text.tag_config("system", foreground="#9399b2")
        
        error_tab = ttk.Frame(chat_notebook)
        chat_notebook.add(error_tab, text="Error Log (Copyable)")
        
        self.error_text = scrolledtext.ScrolledText(error_tab, height=5, font=('Consolas', 8), bg='#2d1f1f', fg='#f0a0a0')
        self.error_text.pack(fill=tk.BOTH, expand=True)
        self.error_text.tag_config("error", foreground="#ff6666")
        
        chat_input_frame = ttk.Frame(chat_frame)
        chat_input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(chat_input_frame, text="You:").pack(side=tk.LEFT, padx=5)
        self.chat_input = ttk.Entry(chat_input_frame)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.chat_input.bind('<Return>', lambda e: self.send_chat())
        ttk.Button(chat_input_frame, text="Send", command=self.send_chat).pack(side=tk.LEFT, padx=5)

    def log(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def log_error(self, msg, exc_info=None):
        import traceback
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_msg = f"[{timestamp}] {msg}\n"
        if exc_info:
            error_msg += f"{traceback.format_exc()}\n"
        self.error_text.insert(tk.END, error_msg, "error")
        self.error_text.see(tk.END)
        self.status_label.config(text=f"Error: {msg}", foreground="red")

    def log_chat(self, msg, tag="system"):
        self.chat_text.insert(tk.END, msg + "\n", tag)
        self.chat_text.see(tk.END)

    # ===== FILE OPERATIONS =====
    
    def select_files(self):
        if self.mode.get() == "single":
            filename = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf"), ("Images", "*.png *.jpg *.jpeg")])
            if filename:
                self.selected_files = [filename]
                self.file_label.config(text=Path(filename).name)
                self.load_pdf_preview(filename)
        else:
            folder = filedialog.askdirectory()
            if folder:
                pdf_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.lower().endswith('.pdf')]
                if pdf_files:
                    self.selected_files = pdf_files
                    self.file_label.config(text=f"{len(pdf_files)} PDFs")
                    self.log(f"Found {len(pdf_files)} PDFs")
                    self.clear_preview()

    def load_pdf_preview(self, path):
        self.log(f"Loading {Path(path).name}...")
        
        if OLMCOCR_AVAILABLE:
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                self.pdf_pages = []
                
                for page_num in range(len(reader.pages)):
                    image_base64 = render_pdf_to_base64png(path, page_num + 1, target_longest_image_dim=1288)
                    img_data = base64.b64decode(image_base64)
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    self.pdf_pages.append(img)
                
                self.display_preview()
                self.display_thumbnails()
                self.log(f"✓ Loaded {len(self.pdf_pages)} pages (olmOCR rendering)")
            except Exception as e:
                self.log_error(f"olmOCR rendering failed: {e}")
                try:
                    from pdf2image import convert_from_path
                    self.pdf_pages = convert_from_path(path, dpi=150)
                    self.display_preview()
                    self.display_thumbnails()
                    self.log(f"✓ Loaded {len(self.pdf_pages)} pages (fallback)")
                except Exception as e2:
                    self.log_error(f"PDF loading failed: {e2}")
        else:
            try:
                from pdf2image import convert_from_path
                self.pdf_pages = convert_from_path(path, dpi=150)
                self.display_preview()
                self.display_thumbnails()
                self.log(f"✓ Loaded {len(self.pdf_pages)} pages")
            except Exception as e:
                self.log_error(f"PDF loading failed: {e}")

    def display_preview(self):
        self.preview_canvas.delete("all")
        
        if not self.pdf_pages or self.current_page_idx >= len(self.pdf_pages):
            return
        
        canvas_w = self.preview_canvas.winfo_width() or 480
        canvas_h = self.preview_canvas.winfo_height() or 600
        
        img = self.pdf_pages[self.current_page_idx]
        img_w, img_h = img.size
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.9
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        img_display = img.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_display)
        
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        self.page_nav_label.config(text=f"Page: {self.current_page_idx + 1} / {len(self.pdf_pages)}")
        
        if self.current_page_idx < len(self.extracted_data):
            self.display_result(self.extracted_data[self.current_page_idx])

    def display_result(self, result):
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert("1.0", result.get("raw_response", ""))
        
        self.used_prompt_text.delete("1.0", tk.END)
        self.used_prompt_text.insert("1.0", result.get("prompt_used", ""))
        
        tokens = result.get("total_tokens", 0)
        in_tok = result.get("input_tokens", 0)
        out_tok = result.get("output_tokens", 0)
        self.page_token_var.set(f"Tokens: In={in_tok} | Out={out_tok} | Total={tokens}")

    def display_thumbnails(self):
        for w in self.thumb_frame_inner.winfo_children():
            w.destroy()
        
        self.thumb_vars = []
        
        for i, page_img in enumerate(self.pdf_pages):
            thumb = page_img.copy()
            thumb.thumbnail((80, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            
            var = tk.BooleanVar(value=False)
            frame = ttk.Frame(self.thumb_frame_inner)
            
            cb = ttk.Checkbutton(frame, text=f"P{i+1}", variable=var, command=self.update_page_selection)
            cb.pack()
            
            lbl = ttk.Label(frame, image=photo)
            lbl.image = photo
            lbl.pack()
            
            def make_view(idx):
                return lambda e: self.view_page(idx)
            lbl.bind("<Button-1>", make_view(i))
            
            frame.pack(side=tk.LEFT, padx=2, pady=2)
            self.thumb_vars.append((i, var, photo))
        
        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))

    def view_page(self, idx):
        self.current_page_idx = idx
        self.display_preview()

    def prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.display_preview()

    def next_page(self):
        if self.current_page_idx < len(self.pdf_pages) - 1:
            self.current_page_idx += 1
            self.display_preview()

    def update_page_selection(self):
        self.selected_pages = [i for i, var, _ in self.thumb_vars if var.get()]
        self.page_label.config(text=f"{len(self.selected_pages)} pages")

    def select_all_pages(self):
        if hasattr(self, 'thumb_vars'):
            for _, var, _ in self.thumb_vars:
                var.set(True)
            self.update_page_selection()

    def deselect_all_pages(self):
        if hasattr(self, 'thumb_vars'):
            for _, var, _ in self.thumb_vars:
                var.set(False)
            self.update_page_selection()

    def clear_preview(self):
        self.preview_canvas.delete("all")
        self.pdf_pages = []
        self.selected_pages = []
        self.current_page_idx = 0
        self.page_nav_label.config(text="Page: 0 / 0")

    def select_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = folder
            self.output_label.config(text=Path(folder).name)
    
    # ===== API & OUTPUT PROCESSING =====
    
    def set_api_key(self):
        key = self.api_key_var.get().strip()
        if key:
            self.api_key = key
            self.log(f"API key set for {self.llm_provider.get()}")
        else:
            self.log("Please enter an API key")
    
    def process_output(self):
        """Process extracted data using LLM to structure it"""
        if not self.extracted_data:
            self.log("No extracted data to process. Run extraction first.")
            return
        
        provider = self.llm_provider.get()
        
        if provider == "local":
            # Use local LLM
            if not self.llm or not self.llm.loaded:
                self.log("Loading local LLM...")
                self.cmd_load_llm()
                time.sleep(2)
            
            if not self.llm or not self.llm.loaded:
                self.log("Local LLM not available")
                return
            llm = self.llm
        else:
            # Use API LLM
            if not self.api_key:
                self.log("Please set API key first")
                return
            
            try:
                if not hasattr(self, 'api_llm') or not self.api_llm:
                    self.api_llm = APILLM(provider=provider, api_key=self.api_key)
                    self.api_llm.load_model()
                llm = self.api_llm
            except Exception as e:
                self.log_error(f"API LLM error: {e}")
                return
        
        self.log("Processing output to structured format...")
        
        # Combine all extracted text
        all_text = ""
        for result in self.extracted_data:
            all_text += result.get("raw_response", "") + "\n\n"
        
        def do_structure():
            try:
                structured = llm.structure_data(all_text, format_type="json")
                self.structured_data = structured
                self.root.after(0, lambda: self.log("✓ Output structured! Click 'Export' to save."))
                self.root.after(0, lambda: self.display_structured_output(structured))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"Structure error: {err}"))
        
        threading.Thread(target=do_structure, daemon=True).start()
    
    def display_structured_output(self, data):
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert("1.0", data)
    
    def export_to_excel(self):
        """Export structured data to Excel"""
        if not self.structured_data:
            self.log("Process output first using 'Process Output' button")
            return
        
        try:
            # Try to parse as JSON
            data = json.loads(self.structured_data)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if contains tables
                tables = []
                for key, val in data.items():
                    if isinstance(val, list):
                        tables.append((key, pd.DataFrame(val)))
                
                if tables:
                    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
                    if filename:
                        with pd.ExcelWriter(filename) as writer:
                            for sheet_name, df in tables:
                                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                        self.log(f"✓ Saved to {filename}")
                        return
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame([{"data": str(data)}])
            
            filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
            if filename:
                df.to_excel(filename, index=False)
                self.log(f"✓ Saved to {filename}")
        except:
            # Save as text if not JSON
            filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.structured_data)
                self.log(f"✓ Saved to {filename}")
    
    def export_to_json(self):
        """Export structured data to JSON"""
        if not self.structured_data:
            self.log("Process output first")
            return
        
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            try:
                data = json.loads(self.structured_data)
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                self.log(f"✓ Saved to {filename}")
            except:
                with open(filename, 'w') as f:
                    f.write(self.structured_data)
                self.log(f"✓ Saved to {filename}")

    # ===== PROMPT =====
    
    def on_prompt_change(self):
        idx = self.prompt_choice.current()
        if idx == 0:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", DEFAULT_OLMOCR_PROMPT)
        elif idx == 1:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", TABLE_PROMPT)

    def save_prompt(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            with open(filename, 'w') as f:
                f.write(self.prompt_text.get("1.0", tk.END))
            self.log("Prompt saved")

    def optimize_prompt(self):
        if not self.llm or not self.llm.loaded:
            self.log("Loading LLM...")
            self.cmd_load_llm()
            time.sleep(2)
        
        if not self.llm or not self.llm.loaded:
            self.log("LLM not available")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Prompt Optimization")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Document Type:").pack(pady=5)
        doc_type = ttk.Entry(dialog, width=40)
        doc_type.pack(pady=5)
        doc_type.insert(0, "e.g., Invoice, Lab Report")
        
        ttk.Label(dialog, text="Extraction Goal:").pack(pady=5)
        goal = ttk.Entry(dialog, width=40)
        goal.pack(pady=5)
        goal.insert(0, "e.g., Extract line items")
        
        def run():
            try:
                optimized = self.llm.optimize_prompt(doc_type.get(), goal.get())
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", optimized)
                self.prompt_choice.current(2)
                self.log("✓ Prompt optimized!")
                dialog.destroy()
            except Exception as e:
                self.log_error(f"Optimization failed: {e}")
        
        ttk.Button(dialog, text="Optimize", command=run).pack(pady=20)

    # ===== EXTRACTION =====
    
    def start_extraction(self):
        if not self.vlm or not self.vlm.loaded:
            self.log("Loading VLM...")
            self.cmd_load_vlm()
            time.sleep(2)
        
        self.stop_flag = False
        self.extracting = True
        self.extract_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._run_extraction)
        thread.daemon = True
        thread.start()

    def stop_extraction(self):
        self.stop_flag = True
        self.log("Stopping...")

    def _run_extraction(self):
        try:
            if not self.output_dir:
                if self.selected_files:
                    self.output_dir = str(Path(self.selected_files[0]).parent / "extraction_output")
                    Path(self.output_dir).mkdir(exist_ok=True)
                    self.root.after(0, lambda: self.output_label.config(text=Path(self.output_dir).name))
            
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            self.extracted_data = []
            start_time = time.time()
            
            self.root.after(0, lambda: self.log("Extracting..."))
            
            if self.mode.get() == "single":
                self.current_page_idx = 0
                pages_to_process = self.selected_pages if self.selected_pages else list(range(len(self.pdf_pages)))
                
                for i, page_idx in enumerate(pages_to_process):
                    if self.stop_flag:
                        break
                    
                    self.root.after(0, lambda p=i+1, t=len(pages_to_process): self.log(f"Processing page {p}/{t}..."))
                    
                    if OLMCOCR_AVAILABLE and self.selected_files:
                        try:
                            image_base64 = render_pdf_to_base64png(self.selected_files[0], page_idx + 1, target_longest_image_dim=1288)
                            result = self.vlm.extract(image_base64=image_base64, prompt=prompt)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            self.root.after(0, lambda err=str(e), t=tb: self.log_error(f"Render error: {err}\nTraceback: {t[-500:]}"))
                            try:
                                temp_path = Path(self.output_dir or ".") / f"temp_p{page_idx}.png"
                                self.pdf_pages[page_idx].save(temp_path)
                                result = self.vlm.extract(image_path=str(temp_path), prompt=prompt)
                                temp_path.unlink()
                            except Exception as e2:
                                self.root.after(0, lambda err=str(e2): self.log_error(f"Fallback also failed: {err}"))
                                continue
                    else:
                        temp_path = Path(self.output_dir or ".") / f"temp_p{page_idx}.png"
                        self.pdf_pages[page_idx].save(temp_path)
                        result = self.vlm.extract(image_path=str(temp_path), prompt=prompt)
                        temp_path.unlink()
                    
                    self.extracted_data.append(result)
                    self.root.after(0, lambda idx=page_idx: self.view_page(idx))
            
            duration = time.time() - start_time
            total_tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.output_dir or ".") / f"extraction_{timestamp}.json"
            
            output_data = {
                "prompt": prompt,
                "pages": len(self.extracted_data),
                "total_tokens": total_tokens,
                "duration": round(duration, 2),
                "results": [{k: v for k, v in r.items()} for r in self.extracted_data]
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            self.root.after(0, lambda: self.log(
                f"✓ Done! {len(self.extracted_data)} pages, {total_tokens} tokens, {duration:.1f}s → {output_file.name}"))
        
        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_error(f"Extraction error: {err}"))
        finally:
            self.root.after(0, self._extraction_done)

    def _extraction_done(self):
        self.extracting = False
        self.progress_bar.stop()
        self.extract_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # ===== CHAT =====
    
    def send_chat(self):
        msg = self.chat_input.get().strip()
        if not msg:
            return
        
        self.chat_input.delete(0, tk.END)
        self.log_chat(f"You: {msg}", "user")
        
        thread = threading.Thread(target=self._process_chat, args=(msg,))
        thread.daemon = True
        thread.start()

    def _process_chat(self, msg):
        msg_lower = msg.lower()
        
        if "load vlm" in msg_lower:
            self.root.after(0, self.cmd_load_vlm)
            return
        if "load llm" in msg_lower:
            self.root.after(0, self.cmd_load_llm)
            return
        if msg_lower in ["extract", "start", "run"]:
            self.root.after(0, self.start_extraction)
            return
        if "stop" in msg_lower:
            self.root.after(0, self.stop_extraction)
            return
        if "help" in msg_lower:
            self.log_chat("Commands: 'load vlm', 'load llm', 'extract', 'stop', 'optimize'", "assistant")
            return
        
        if not self.llm or not self.llm.loaded:
            self.log_chat("Load LLM first: 'load llm'", "assistant")
            return
        
        try:
            response = self.llm.chat(msg)
            self.root.after(0, lambda r=response: self.log_chat(f"Assistant: {r}", "assistant"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_error(f"Chat error: {err}"))

    # ===== MODEL LOADING =====
    
    def cmd_load_vlm(self):
        if self.vlm and self.vlm.loaded:
            self.log("VLM already loaded")
            return
        
        self.log("Loading VLM (olmOCR-2-7B-1025-FP8)...")
        
        def load():
            try:
                self.vlm = VLMExtractor()
                result = self.vlm.load_model()
                self.root.after(0, lambda r=result: self._on_vlm_loaded(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"VLM Error: {err}"))
        
        threading.Thread(target=load, daemon=True).start()

    def _on_vlm_loaded(self, result):
        self.log(f"✓ {result}")
        self.update_model_status()

    def cmd_load_llm(self):
        if self.llm and self.llm.loaded:
            self.log("LLM already loaded")
            return
        
        self.log("Loading LLM (Qwen2.5-3B)...")
        
        def load():
            try:
                self.llm = IntelligentAssistant()
                result = self.llm.load_model()
                self.root.after(0, lambda r=result: self._on_llm_loaded(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"LLM Error: {err}"))
        
        threading.Thread(target=load, daemon=True).start()

    def _on_llm_loaded(self, result):
        self.log(f"✓ {result}")
        self.log_chat("Hi! I can help optimize prompts. Type 'help' for commands.", "assistant")
        self.update_model_status()

    def update_model_status(self):
        status = []
        if self.vlm and self.vlm.loaded:
            status.append("VLM")
        if self.llm and self.llm.loaded:
            status.append("LLM")
        
        if status:
            self.model_status.config(text=f"Models: {', '.join(status)}", foreground="green")
        else:
            self.model_status.config(text="Models: Not loaded", foreground="gray")

def main():
    if not TRANSFORMERS_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependencies", 
            "pip install torch transformers pillow pdf2image pandas pypdf")
        return
    
    if not torch.cuda.is_available():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("CUDA Required", "olmOCR needs CUDA GPU")
        return
    
    root = tk.Tk()
    app = OlmoCRAgenticGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
