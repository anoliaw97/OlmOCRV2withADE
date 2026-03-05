"""
olmOCR Post-Processor GUI
=========================
A standalone GUI for processing unstructured text/JSON into structured datasets.
Uses LLM (local or API) to extract and structure data into Excel/CSV.

Run from: C:/Users/Mining/Downloads/olmocr-main
Usage: python olmocr_postprocessor_gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import json
import threading
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Try importing transformers for local LLM
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try importing openai for API LLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LocalLLM:
    """Local LLM using Qwen2.5-3B-Instruct"""
    
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
            raise RuntimeError("CUDA GPU required for local LLM")
        
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to("cuda").eval()
        
        torch.cuda.empty_cache()
        
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
        inputs = inputs.to("cuda", dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=4096, temperature=0.1, do_sample=True, top_p=0.9)
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        del inputs, outputs
        torch.cuda.empty_cache()
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        self.history = []


class APILLM:
    """API-based LLM (Groq/OpenAI/Custom)"""
    
    def __init__(self, provider="groq", api_key=None, api_url=None, model=None):
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.client = None
        self.loaded = False
        self.history = []
    
    def load_model(self):
        if not self.api_key:
            raise RuntimeError("No API key set")
        
        if self.provider == "custom" and self.api_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        elif self.provider == "groq":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
            self.model = self.model or "llama-3.3-70b-versatile"
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.model = self.model or "gpt-4o-mini"
        
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
            temperature=0.1,
            max_tokens=4096
        )
        
        result = response.choices[0].message.content
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": result})
        
        return result
    
    def clear_history(self):
        self.history = []


class PostProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("olmOCR Post-Processor - Unstructured to Structured Dataset")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        self.input_file = None
        self.raw_text = ""
        self.structured_data = None
        self.records = []
        self.llm = None
        self.template_columns = []
        
        self.setup_ui()
        self.log("Post-Processor Ready!")
    
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top bar
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="olmOCR Post-Processor", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(top_frame, text="LLM: Not loaded", 
                                      foreground="gray", font=('Arial', 9))
        self.model_status.pack(side=tk.RIGHT, padx=10)
        
        # Control bar
        control_frame = ttk.Frame(self.root, padding=(5, 2))
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="Load LLM", command=self.cmd_load_llm, width=12).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(control_frame, text="Load File", command=self.load_input_file, width=12).pack(side=tk.LEFT, padx=2)
        self.file_label = ttk.Label(control_frame, text="No file loaded", font=('Arial', 8))
        self.file_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(control_frame, text="Load Template", command=self.load_template, width=12).pack(side=tk.LEFT, padx=2)
        self.template_label = ttk.Label(control_frame, text="No template", font=('Arial', 8))
        self.template_label.pack(side=tk.LEFT, padx=2)
        
        # Main content
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Input text
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        input_frame = ttk.LabelFrame(left_frame, text="Input Text / JSON Content", padding=5)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Prompt + Results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        right_notebook = ttk.Notebook(right_frame)
        right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Extraction Prompt
        prompt_tab = ttk.Frame(right_notebook)
        right_notebook.add(prompt_tab, text="Extraction Prompt")
        self._create_prompt_tab(prompt_tab)
        
        # Tab 2: Structured Output
        output_tab = ttk.Frame(right_notebook)
        right_notebook.add(output_tab, text="Structured Output")
        self._create_output_tab(output_tab)
        
        # Tab 3: Data Preview
        preview_tab = ttk.Frame(right_notebook)
        right_notebook.add(preview_tab, text="Data Preview")
        self._create_preview_tab(preview_tab)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=2)
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))
    
    def _create_prompt_tab(self, parent):
        # LLM Settings
        llm_frame = ttk.LabelFrame(parent, text="LLM Settings", padding="5")
        llm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        row1 = ttk.Frame(llm_frame)
        row1.pack(fill=tk.X)
        
        ttk.Label(row1, text="Provider:").pack(side=tk.LEFT, padx=(0, 4))
        self.llm_provider = ttk.Combobox(row1, width=12, state="readonly",
                                          values=("Local", "Groq", "OpenAI", "Custom"))
        self.llm_provider.current(0)
        self.llm_provider.pack(side=tk.LEFT, padx=(0, 8))
        self.llm_provider.bind('<<ComboboxSelected>>', lambda e: self._on_provider_change())
        
        ttk.Label(row1, text="API URL:").pack(side=tk.LEFT, padx=(0, 4))
        self.api_url_entry = ttk.Entry(row1, width=30)
        self.api_url_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.api_url_entry.insert(0, "https://api.groq.com/openai/v1")
        
        row2 = ttk.Frame(llm_frame)
        row2.pack(fill=tk.X, pady=(4, 0))
        
        ttk.Label(row2, text="API Key:").pack(side=tk.LEFT, padx=(0, 4))
        self.api_key_entry = ttk.Entry(row2, width=40, show="*")
        self.api_key_entry.pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Label(row2, text="Model:").pack(side=tk.LEFT, padx=(0, 4))
        self.model_entry = ttk.Entry(row2, width=20)
        self.model_entry.pack(side=tk.LEFT)
        self.model_entry.insert(0, "llama-3.3-70b-versatile")
        
        # Extraction Prompt
        prompt_frame = ttk.LabelFrame(parent, text="Extraction Prompt", padding="5")
        prompt_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, font=('Consolas', 9), height=8)
        self.prompt_text.pack(fill=tk.X)
        
        default_prompt = """You are a data extraction expert. Extract structured data from the following text.

Extract ALL records/rows of data as a JSON array. For each record:
- Use descriptive snake_case column names
- Extract only values that explicitly appear in the text
- Numbers should be numeric type, not strings
- If a value is not found, use null

Return ONLY a valid JSON array, no explanation."""
        
        self.prompt_text.insert("1.0", default_prompt)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.extract_btn = ttk.Button(btn_frame, text="Extract Data", command=self.start_extraction)
        self.extract_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ttk.Button(btn_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT)
    
    def _create_output_tab(self, parent):
        output_frame = ttk.LabelFrame(parent, text="Structured JSON Output", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, font=('Consolas', 9), bg='#1e1e2e', fg='#cdd6f4')
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons
        exp_frame = ttk.Frame(parent)
        exp_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Button(exp_frame, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(exp_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(exp_frame, text="Export JSON", command=self.export_json).pack(side=tk.LEFT)
    
    def _create_preview_tab(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="Data Table Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for data
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Record count
        self.record_count_label = ttk.Label(preview_frame, text="0 records")
        self.record_count_label.pack(anchor=tk.W, pady=(4, 0))
    
    def _on_provider_change(self):
        provider = self.llm_provider.get()
        if provider == "Local":
            self.api_url_entry.config(state='disabled')
            self.api_key_entry.config(state='disabled')
            self.model_entry.config(state='disabled')
        elif provider == "Groq":
            self.api_url_entry.config(state='normal')
            self.api_url_entry.delete(0, tk.END)
            self.api_url_entry.insert(0, "https://api.groq.com/openai/v1")
            self.api_key_entry.config(state='normal')
            self.model_entry.config(state='normal')
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, "llama-3.3-70b-versatile")
        elif provider == "OpenAI":
            self.api_url_entry.config(state='disabled')
            self.api_key_entry.config(state='normal')
            self.model_entry.config(state='normal')
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, "gpt-4o-mini")
        elif provider == "Custom":
            self.api_url_entry.config(state='normal')
            self.api_key_entry.config(state='normal')
            self.model_entry.config(state='normal')
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=msg, foreground="black")
        self.root.update_idletasks()
    
    def cmd_load_llm(self):
        provider = self.llm_provider.get()
        
        self.log(f"Loading {provider} LLM...")
        
        def load():
            try:
                if provider == "Local":
                    self.llm = LocalLLM()
                else:
                    api_key = self.api_key_entry.get().strip()
                    api_url = self.api_url_entry.get().strip() if provider == "Custom" else None
                    model = self.model_entry.get().strip()
                    self.llm = APILLM(provider=provider.lower(), api_key=api_key, api_url=api_url, model=model)
                
                result = self.llm.load_model()
                self.root.after(0, lambda r=result: self._on_llm_loaded(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log(f"Error: {err}"))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_llm_loaded(self, result):
        self.log(f"✓ {result}")
        self.model_status.config(text=f"LLM: {result}", foreground="green")
    
    def load_input_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Text/JSON", "*.txt;*.json;*.md"), ("All files", "*.*")]
        )
        if filename:
            self.input_file = filename
            self.file_label.config(text=Path(filename).name)
            
            try:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    # Extract text from common JSON structures
                    if isinstance(data, list):
                        parts = []
                        for item in data:
                            if isinstance(item, dict):
                                for key in ['raw_response', 'text', 'content', 'extracted_text']:
                                    if key in item:
                                        parts.append(str(item[key]))
                        content = "\n\n".join(parts) if parts else content
                    elif isinstance(data, dict):
                        parts = []
                        for key in ['raw_extraction', 'raw_extractions', 'extracted_text', 'results', 'text', 'content']:
                            if key in data:
                                v = data[key]
                                if isinstance(v, str):
                                    parts.append(v)
                                elif isinstance(v, list):
                                    parts.extend([str(x) for x in v if isinstance(x, (str, int, float))])
                        content = "\n\n".join(parts) if parts else json.dumps(data, indent=2)
                except json.JSONDecodeError:
                    pass  # Keep as plain text
                
                self.raw_text = content
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert("1.0", content[:50000])  # Limit display
                self.log(f"Loaded: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def load_template(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Excel/CSV", "*.xlsx;*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                else:
                    df = pd.read_excel(filename)
                
                self.template_columns = list(df.columns)
                self.template_label.config(text=f"{len(self.template_columns)} columns")
                self.log(f"Template loaded: {Path(filename).name} ({len(self.template_columns)} columns)")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load template: {e}")
    
    def start_extraction(self):
        if not self.llm or not self.llm.loaded:
            messagebox.showwarning("Warning", "Please load LLM first")
            return
        
        if not self.raw_text:
            messagebox.showwarning("Warning", "Please load an input file first")
            return
        
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        
        # Add template columns if available
        if self.template_columns:
            cols_str = ", ".join(self.template_columns)
            prompt += f"\n\nUse EXACTLY these column names: [{cols_str}]"
        
        # Combine with input text
        full_prompt = f"""{prompt}

INPUT TEXT TO PROCESS:
{self.raw_text}

Return ONLY valid JSON array, no explanation or markdown fences."""
        
        self.extract_btn.config(state=tk.DISABLED)
        self.log("Extracting data...")
        
        def extract():
            try:
                system_ctx = "You are a precise data extraction assistant. Return only valid JSON array."
                response = self.llm.chat(full_prompt, system_context=system_ctx)
                
                # Parse JSON
                records = self._parse_json_response(response)
                
                self.root.after(0, lambda: self._on_extraction_complete(records, response))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._on_extraction_error(err))
        
        threading.Thread(target=extract, daemon=True).start()
    
    def _parse_json_response(self, response):
        # Try to extract JSON from response
        text = response.strip()
        
        # Remove markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        
        # Try to find JSON array
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            # Try to find JSON array in text
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data if isinstance(data, list) else [data]
                except:
                    pass
        
        return []
    
    def _on_extraction_complete(self, records, raw_response):
        self.extract_btn.config(state=tk.NORMAL)
        
        if not records:
            self.log("No records extracted")
            messagebox.showwarning("Warning", "No records could be extracted from the input")
            return
        
        self.records = records
        self.structured_data = json.dumps(records, indent=2)
        
        # Update output
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", self.structured_data)
        
        # Update preview
        self._update_preview()
        
        self.log(f"✓ Extracted {len(records)} records")
    
    def _on_extraction_error(self, error):
        self.extract_btn.config(state=tk.NORMAL)
        self.log(f"Error: {error}")
        messagebox.showerror("Error", f"Extraction failed: {error}")
    
    def _update_preview(self):
        # Clear tree
        self.tree.delete(*self.tree.get_children())
        
        if not self.records:
            return
        
        # Get all columns
        all_cols = set()
        for record in self.records:
            if isinstance(record, dict):
                all_cols.update(record.keys())
        
        cols = list(all_cols)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add rows
        for record in self.records:
            if isinstance(record, dict):
                values = [str(record.get(col, "")) for col in cols]
                self.tree.insert("", tk.END, values=values)
        
        self.record_count_label.config(text=f"{len(self.records)} records")
    
    def clear_results(self):
        self.output_text.delete("1.0", tk.END)
        self.tree.delete(*self.tree.get_children())
        self.records = []
        self.structured_data = None
        self.record_count_label.config(text="0 records")
        self.log("Results cleared")
    
    def export_excel(self):
        if not self.records:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if not filename:
            return
        
        try:
            df = pd.DataFrame(self.records)
            
            # Reorder columns if template exists
            if self.template_columns:
                ordered = [c for c in self.template_columns if c in df.columns]
                extras = [c for c in df.columns if c not in self.template_columns]
                df = df[ordered + extras]
            
            # Convert to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)
                meta_df = pd.DataFrame([{
                    "source_file": self.input_file or "unknown",
                    "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "records_extracted": len(self.records),
                    "template_columns": ", ".join(self.template_columns),
                }])
                meta_df.to_excel(writer, sheet_name="_metadata", index=False)
            
            self.log(f"Exported to {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def export_csv(self):
        if not self.records:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not filename:
            return
        
        try:
            df = pd.DataFrame(self.records)
            
            if self.template_columns:
                ordered = [c for c in self.template_columns if c in df.columns]
                extras = [c for c in df.columns if c not in self.template_columns]
                df = df[ordered + extras]
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            df.to_csv(filename, index=False)
            self.log(f"Exported to {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def export_json(self):
        if not self.records:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if not filename:
            return
        
        try:
            output = {
                "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": self.input_file or "unknown",
                "template_columns": self.template_columns,
                "records": self.records,
            }
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            self.log(f"Exported to {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")


def main():
    if not TRANSFORMERS_AVAILABLE and not OPENAI_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependencies", 
            "Install at least one of: torch+transformers, or openai")
        return
    
    root = tk.Tk()
    app = PostProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
