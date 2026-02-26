"""
Flask web app mirroring olmocr_agentic_gui.py
Single-page app with all GUI sections: Upload, Extract, Prompt Optimizer, Post-Process, Export, Screenshots
"""

import os
import io
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'RESULTS')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory state (mimics GUI state)
state = {
    'selected_file': None,
    'extracted_data': [],  # list of dicts: page_number, duration_s, total_tokens, raw_response
    'template_columns': [],
    'pp_records': [],
    'pp_model_used': '',
    'current_page_idx': 0,
}

def seconds_to_minutes(seconds):
    return round(seconds / 60.0, 2)

def load_angsi_results():
    """Load Angsi 1 Core.txt results and populate extracted_data"""
    results_path = os.path.join(app.config['RESULTS_FOLDER'], 'Angsi 1 Core.txt')
    if not os.path.exists(results_path):
        return False
    
    with open(results_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Parse the timing header section
    # Lines like: "PAGE 1  |  56.3s  |  1725 tok"
    page_pattern = re.compile(r'PAGE\s+(\d+)\s+\|\s+([\d.]+)s\s+\|\s+(\d+)\s+tok', re.IGNORECASE)
    
    # Parse the content sections
    # Each page section starts with "---" and contains YAML front matter
    sections = content.split('---')
    
    extracted = []
    page_timings = page_pattern.findall(content)
    
    for i, (page_num, duration_s, tokens) in enumerate(page_timings):
        raw_text = ""
        # Find the content for this page (between --- markers)
        for section in sections:
            if section.strip().startswith('primary_language:'):
                # This is a page section, extract raw text
                lines = section.strip().split('\n')
                # Skip YAML front matter
                text_lines = []
                in_yaml = True
                for line in lines:
                    if in_yaml and line.strip() == '':
                        in_yaml = False
                    if not in_yaml and line.strip():
                        text_lines.append(line.strip())
                raw_text = '\n'.join(text_lines[:10])  # First 10 non-empty lines
        
        extracted.append({
            'page_number': int(page_num),
            'duration_s': float(duration_s),
            'total_tokens': int(tokens),
            'raw_response': raw_text[:500] if raw_text else f"Page {page_num} content"
        })
    
    state['extracted_data'] = sorted(extracted, key=lambda x: x['page_number'])
    return True

@app.route('/')
def index():
    """Main single-page UI mirroring the Tkinter GUI"""
    return render_template('index.html', 
                         state=state,
                         results_folder=app.config['RESULTS_FOLDER'])

@app.route('/load_angsi')
def load_angsi():
    """Load Angsi results into the UI"""
    success = load_angsi_results()
    if success:
        return jsonify({'status': 'ok', 'pages': len(state['extracted_data'])})
    return jsonify({'status': 'error', 'message': 'Angsi results not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload PDF file"""
    file = request.files.get('pdf')
    if not file:
        return jsonify({'status': 'error', 'message': 'No file'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    state['selected_file'] = filepath
    return jsonify({'status': 'ok', 'filename': file.filename})

@app.route('/extract', methods=['POST'])
def extract():
    """Run extraction (simulated with Angsi data for now)"""
    if not state['selected_file'] and not state['extracted_data']:
        # Auto-load Angsi results as demo
        load_angsi_results()
    
    return jsonify({
        'status': 'ok',
        'pages': len(state['extracted_data']),
        'total_tokens': sum(p['total_tokens'] for p in state['extracted_data']),
        'total_duration_s': sum(p['duration_s'] for p in state['extracted_data'])
    })

@app.route('/page/<int:page_num>')
def get_page(page_num):
    """Get raw response for a specific page"""
    for page in state['extracted_data']:
        if page['page_number'] == page_num:
            return jsonify(page)
    return jsonify({'error': 'Page not found'}), 404

@app.route('/upload_template', methods=['POST'])
def upload_template():
    """Upload CSV/Excel template for post-processing"""
    file = request.files.get('template')
    if not file:
        return jsonify({'status': 'error', 'message': 'No file'})
    
    filename = file.filename.lower()
    content = file.read().decode('utf-8', errors='ignore')
    
    if filename.endswith('.csv'):
        lines = content.strip().split('\n')
        if lines:
            columns = [c.strip() for c in lines[0].split(',')]
    else:
        # Excel - just use generic columns for now
        columns = ['Column_A', 'Column_B', 'Column_C']
    
    state['template_columns'] = columns
    return jsonify({'status': 'ok', 'columns': columns})

@app.route('/clear_template', methods=['POST'])
def clear_template():
    """Clear template"""
    state['template_columns'] = []
    return jsonify({'status': 'ok'})

@app.route('/run_postprocess', methods=['POST'])
def run_postprocess():
    """Run post-process on full document (simulated)"""
    if not state['extracted_data']:
        return jsonify({'status': 'error', 'message': 'No extracted data'}), 400
    
    # Compile all raw text
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(
        p.get('raw_response', '') for p in state['extracted_data']
    )
    
    # Simulate post-process results (in real app, would call LLM)
    # Create sample records from the data
    records = []
    for page in state['extracted_data'][:5]:  # First 5 pages as sample
        records.append({
            'page': page['page_number'],
            'depth_ft': 8000 + page['page_number'] * 10,
            'porosity_pct': 15.0 + page['page_number'] * 0.5,
            'permeability_md': 10.0 + page['page_number'] * 2,
            'formation_factor': 40.0 + page['page_number'],
            'saturation_exponent': 2.0 + page['page_number'] * 0.02
        })
    
    state['pp_records'] = records
    state['pp_model_used'] = 'groq:llama-3.3-70b'
    
    return jsonify({
        'status': 'ok',
        'records': len(records),
        'model': state['pp_model_used']
    })

@app.route('/export')
def export_data():
    """Export data in various formats"""
    fmt = request.args.get('format', 'json')
    data = state.get('pp_records', [])
    
    if not data:
        return jsonify({'error': 'No data to export'}), 400
    
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    if fmt == 'json':
        return send_file(
            io.BytesIO(json.dumps(data, indent=2).encode()),
            mimetype='application/json',
            as_attachment=True,
            download_name='postprocess.json'
        )
    elif fmt == 'csv':
        return send_file(
            io.BytesIO(df.to_csv(index=False).encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='postprocess.csv'
        )
    elif fmt == 'xlsx':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            # Metadata sheet
            meta = pd.DataFrame([{
                'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source_file': os.path.basename(state.get('selected_file', 'unknown')),
                'model': state.get('pp_model_used', ''),
                'template_columns': ', '.join(state.get('template_columns', []))
            }])
            meta.to_excel(writer, sheet_name='_metadata', index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='postprocess.xlsx'
        )
    
    return jsonify({'error': 'Invalid format'}), 400

@app.route('/screenshots')
def screenshots():
    """List available screenshots"""
    results_dir = app.config['RESULTS_FOLDER']
    pngs = sorted([f for f in os.listdir(results_dir) if f.endswith('.png')])
    return jsonify(pngs)

@app.route('/img/<path:filename>')
def serve_image(filename):
    """Serve images from RESULTS folder"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), mimetype='image/png')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting OLM OCR Web GUI...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
