"""
Simple Extraction Viewer
=======================
A basic viewer to render and display extracted text/JSON files.

Run: python olmocr_extraction_viewer.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
import json

def main():
    root = tk.Tk()
    root.title("Extraction Viewer")
    root.geometry("1000x700")
    
    # Top toolbar
    toolbar = ttk.Frame(root, padding=5)
    toolbar.pack(fill=tk.X)
    
    ttk.Button(toolbar, text="Open File", command=lambda: load_file()).pack(side=tk.LEFT, padx=5)
    
    file_label = ttk.Label(toolbar, text="No file loaded")
    file_label.pack(side=tk.LEFT, padx=10)
    
    # Main text area
    text_area = scrolledtext.ScrolledText(root, font=('Consolas', 10), wrap=tk.NONE)
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add horizontal scrollbar
    hscroll = ttk.Scrollbar(root, orient="horizontal", command=text_area.xview)
    hscroll.pack(fill=tk.X, padx=10, pady=(0, 10))
    text_area.configure(xscrollcommand=hscroll.set)
    
    current_file = [None]
    
    def load_file():
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Text/JSON", "*.txt;*.json;*.md"),
                ("All files", "*.*")
            ]
        )
        if not filename:
            return
        
        current_file[0] = filename
        file_label.config(text=Path(filename).name)
        
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to format JSON nicely
            try:
                data = json.loads(content)
                content = json.dumps(data, indent=2)
            except:
                pass
            
            text_area.delete("1.0", tk.END)
            text_area.insert("1.0", content)
            
        except Exception as e:
            text_area.delete("1.0", tk.END)
            text_area.insert("1.0", f"Error loading file: {e}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
