"""
Flask web app for Manga Generation Pipeline
Interactive interface for generating manga from text input
"""

import os
import sys
import json
import threading
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# ── Configure paths ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
STEP1_DIR = PROJECT_ROOT / "Step-1" / "code"
STEP2_DIR = PROJECT_ROOT / "step-2-layout"
STEP3_DIR = PROJECT_ROOT / "step-3"
WEB_OUTPUT = PROJECT_ROOT / "web_output"
UPLOADS_DIR = WEB_OUTPUT / "uploads"
RESULTS_DIR = WEB_OUTPUT / "results"

# Create directories
WEB_OUTPUT.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(STEP1_DIR))
sys.path.insert(0, str(STEP2_DIR))
sys.path.insert(0, str(STEP3_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

# ── Setup Flask ────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("manga_web_app")

# Store job status for each session
jobs = {}

# ── Import pipeline modules ───────────────────────────────────────────
try:
    from model_client import generate, get_current_model
    from layout_generator import convert_beats_to_layout
    from compositor import compose_page, CompositorConfig
    from text_extractor import TextExtractor
    from dialog_mapper import DialogMapper
    PIPELINE_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import pipeline modules: {e}")
    PIPELINE_AVAILABLE = False

# Import step1 function after paths are set up
try:
    from run_e2e_test import step1_extract_beats
    EXTRACT_BEATS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not import step1_extract_beats separately: {e}")
    EXTRACT_BEATS_AVAILABLE = False

# Check for Gemini API key
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if PIPELINE_AVAILABLE and not GEMINI_KEY:
    logger.warning("GEMINI_API_KEY not set - pipeline will fail at beat extraction")


class JobManager:
    """Manages generation job status and progress"""
    
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.message = "Waiting to start..."
        self.error = None
        self.result_path = None
        self.start_time = datetime.now()
        
    def update(self, status, progress, message):
        self.status = status
        self.progress = min(progress, 100)
        self.message = message
        logger.info(f"[{self.job_id}] {status}: {message} ({progress}%)")
        
    def set_error(self, error):
        self.error = str(error)
        self.status = "error"
        logger.error(f"[{self.job_id}] Error: {error}")
        
    def complete(self, result_path):
        self.result_path = result_path
        self.status = "completed"
        self.progress = 100
        self.message = "Generation complete!"
        logger.info(f"[{self.job_id}] Completed with result: {result_path}")
        
    def to_dict(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result_path": self.result_path,
            "elapsed_seconds": round(elapsed, 1)
        }


def run_pipeline_worker(job_id, text_input, options):
    """Worker thread that runs the manga pipeline"""
    try:
        job = jobs[job_id]
        
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("Pipeline not available. Check imports and requirements.")
        
        if not GEMINI_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        # Save input to temp file
        input_file = UPLOADS_DIR / f"{job_id}_input.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text_input)
        
        # Output directory
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Track counts for results
        beat_count = 0
        panels_count = 0
        
        # ===== STEP 1: Extract Beats =====
        job.update("extraction", 15, "Extracting narrative beats from text...")
        try:
            if not EXTRACT_BEATS_AVAILABLE:
                raise RuntimeError("Beat extraction module not available")
            
            # Limit text length for processing
            max_chars = 3000
            limited_text = text_input[:max_chars]
            
            beats_result = step1_extract_beats(text=limited_text, chunk_id=f"web_{job_id}")
            
            if not beats_result:
                raise ValueError("No beats extracted from text")
            
            # Convert beats to proper format if needed
            # If beats_result is a list, wrap it in a dict with 'beats' key
            if isinstance(beats_result, list):
                beats_list = beats_result
            elif isinstance(beats_result, dict) and "beats" in beats_result:
                beats_list = beats_result["beats"]
            else:
                beats_list = [beats_result] if isinstance(beats_result, dict) else []
            
            # Ensure each beat has an id
            for idx, beat in enumerate(beats_list):
                if isinstance(beat, dict) and "id" not in beat:
                    beat["id"] = f"beat_{idx + 1}"
            
            beats = {
                "page_id": job_id,
                "beats": beats_list
            }
            
            beat_count = len(beats_list)
            logger.info(f"[{job_id}] Extracted {beat_count} beats")
            
        except Exception as e:
            logger.error(f"[{job_id}] Beat extraction error: {e}")
            raise RuntimeError(f"Failed to extract beats: {str(e)}")
        
        # ===== STEP 2: Generate Layout =====
        job.update("layout", 35, "Generating manga page layout...")
        try:
            layout = convert_beats_to_layout(beats)
            
            if not layout:
                raise ValueError("Failed to generate layout")
            
            # Save layout for reference
            layout_file = output_dir / "layout.json"
            with open(layout_file, 'w') as f:
                json.dump(layout, f, indent=2)
            
            panels_count = len(layout.get('panels', []))
            logger.info(f"[{job_id}] Generated layout with {panels_count} panels")
            
        except Exception as e:
            logger.error(f"[{job_id}] Layout generation error: {e}")
            raise RuntimeError(f"Failed to generate layout: {str(e)}")
        
        # ===== STEP 3: Generate Panels =====
        job.update("generation", 65, "Generating images with Stable Diffusion (this may take a minute)...")
        try:
            from run_e2e_test import (
                load_sd_pipeline, step3_generate_panels, PipelineMetrics, DRY_RUN
            )
            
            if DRY_RUN:
                logger.warning(f"[{job_id}] Dry run mode - skipping image generation")
                panels_dir = output_dir / "panels"
                panels_dir.mkdir(exist_ok=True)
            else:
                pipe = load_sd_pipeline()
                metrics = PipelineMetrics()
                panels_dir = step3_generate_panels(layout, pipe, metrics)
            
            logger.info(f"[{job_id}] Generated panels")
            
        except Exception as e:
            logger.error(f"[{job_id}] Panel generation error: {e}")
            raise RuntimeError(f"Failed to generate panels: {str(e)}")
        
        # ===== STEP 4: Composite Page =====
        job.update("composition", 90, "Compositing final manga page...")
        try:
            from run_e2e_test import step3_composite
            
            final_page = step3_composite(layout, panels_dir)
            
            if not final_page or not final_page.exists():
                raise ValueError("Composite failed - no output file")
            
            # Move to web output
            web_final_page = output_dir / "final_page.png"
            if final_page != web_final_page:
                import shutil
                shutil.copy2(final_page, web_final_page)
                final_page = web_final_page
            
            logger.info(f"[{job_id}] Composite complete: {final_page}")
            
        except Exception as e:
            logger.error(f"[{job_id}] Composite error: {e}")
            raise RuntimeError(f"Failed to composite page: {str(e)}")
        
        # ===== Save Metadata =====
        results_meta = {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "final_page": str(final_page.relative_to(RESULTS_DIR).as_posix()),
            "beats_count": beat_count,
            "panels_count": panels_count,
            "options": options,
            "generated_at": datetime.now().isoformat()
        }
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results_meta, f, indent=2)
        
        job.complete(str(final_page.relative_to(RESULTS_DIR).as_posix()))
        
    except Exception as e:
        logger.error(f"[{job_id}] Pipeline fatal error: {traceback.format_exc()}")
        job.set_error(e)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Start a new generation job"""
    try:
        data = request.json
        text_input = data.get('text', '').strip()
        options = data.get('options', {})
        
        if not text_input:
            return jsonify({"error": "Text input is required"}), 400
        
        if len(text_input) < 50:
            return jsonify({"error": "Text input must be at least 50 characters"}), 400
        
        # Create job
        job_id = f"job_{int(datetime.now().timestamp() * 1000)}"
        jobs[job_id] = JobManager(job_id)
        
        # Start worker thread
        thread = threading.Thread(
            target=run_pipeline_worker,
            args=(job_id, text_input, options),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": jobs[job_id].to_dict()
        })
        
    except Exception as e:
        logger.error(f"API error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def api_status(job_id):
    """Get job status"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(jobs[job_id].to_dict())


@app.route('/api/results/<job_id>', methods=['GET'])
def api_results(job_id):
    """Get job results"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job.status != "completed":
        return jsonify({"error": f"Job status is {job.status}"}), 400
    
    result_dir = RESULTS_DIR / job_id
    results_file = result_dir / "results.json"
    
    if not results_file.exists():
        return jsonify({"error": "Results file not found"}), 404
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get list of generated files
    generated_files = []
    if result_dir.exists():
        for item in result_dir.rglob('*'):
            if item.is_file() and item.suffix in ['.png', '.jpg', '.json']:
                rel_path = str(item.relative_to(RESULTS_DIR))
                generated_files.append({
                    "path": rel_path,
                    "type": "image" if item.suffix in ['.png', '.jpg'] else "data"
                })
    
    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "results": results,
        "files": generated_files
    })


@app.route('/results/<path:filepath>', methods=['GET'])
def serve_result(filepath):
    """Serve generated files"""
    try:
        result_path = RESULTS_DIR / filepath
        
        # Security check
        if not str(result_path.resolve()).startswith(str(RESULTS_DIR.resolve())):
            return {"error": "Invalid path"}, 403
        
        if not result_path.exists():
            return {"error": "File not found"}, 404
        
        if result_path.is_file():
            return send_file(result_path)
        
        return {"error": "Path is not a file"}, 400
        
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        return {"error": str(e)}, 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "pipeline_available": PIPELINE_AVAILABLE,
        "active_jobs": len([j for j in jobs.values() if j.status in ["pending", "initializing", "extraction", "layout", "generation", "composition"]])
    })


# ── Error handlers ───────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🎨 Manga Generation Web Interface")
    logger.info("=" * 60)
    logger.info(f"Pipeline available: {PIPELINE_AVAILABLE}")
    logger.info(f"Output directory: {WEB_OUTPUT}")
    logger.info("Starting server on http://localhost:5000")
    logger.info("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
