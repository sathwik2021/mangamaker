# Flask web app for Manga Generation Pipeline
# Interactive interface for generating manga from text input
#

import os
import sys
import json
import threading
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import traceback

# Load .env if present so GEMINI_API_KEY and other runtime env vars are available.
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
except ImportError:
    pass

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
# sys.path.insert(0, str(PROJECT_ROOT / "tests"))  # FIX 0: Removed redundant tests search path

# ── Setup Flask ────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
# FIX 7: Tighten CORS origins
CORS(app, origins=["http://localhost:5000"])

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

# FIX 6: Module-level shared pipeline cache and serializing lock
_pipeline_lock = threading.Lock()
_cached_pipe = None

def get_shared_pipeline():
    global _cached_pipe
    with _pipeline_lock:
        if _cached_pipe is None:
            from run_e2e_test import load_sd_pipeline
            _cached_pipe = load_sd_pipeline()
            logger.info("Shared SD pipeline loaded: pipe_id=%s", id(_cached_pipe))
        else:
            logger.info("Reusing shared SD pipeline: pipe_id=%s", id(_cached_pipe))
        return _cached_pipe

# FIX 7: API Auth token config
API_TOKEN = os.getenv("MANGA_APP_TOKEN", "")

@app.before_request
def check_auth():
    if request.path.startswith('/api/generate') and API_TOKEN:
        if request.headers.get('X-App-Token') != API_TOKEN:
            return jsonify({"error": "unauthorized"}), 401


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


def chunk_beats_into_pages(beats: list, max_chars_per_page: int = 400) -> list:
    """
    Groups beats into page-sized chunks based on cumulative dialogue length.
    A page with heavy dialogue gets fewer beats; a page with mostly visual/action beats gets more.
    Returns a list of list of beat dicts.
    """
    pages = []
    current_page = []
    current_chars = 0
    
    for beat in beats:
        dialogue = beat.get("dialogue", "")
        if isinstance(dialogue, list):
            dialogue_str = " ".join(dialogue)
        else:
            dialogue_str = str(dialogue)
            
        # +40 buffer for visual/action beats with no dialogue
        beat_chars = len(dialogue_str) + 40
        if current_chars + beat_chars > max_chars_per_page and current_page:
            pages.append(current_page)
            current_page = []
            current_chars = 0
        current_page.append(beat)
        current_chars += beat_chars
        
    if current_page:
        pages.append(current_page)
    return pages


def run_pipeline_worker(job_id, text_input, options):
    """Worker thread that runs the manga pipeline.
    
    Known Limitation: Characters are not visually consistent across pages.
    Each page's SD generation is independent.
    """
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
        total_panels_count = 0
        
        # Override pipeline config from request options before running the SD stages.
        try:
            from run_e2e_test import configure_pipeline, CONFIG as PIPELINE_CONFIG
            configure_pipeline(options)
            options["applied_guidance_scale"] = str(PIPELINE_CONFIG.sd_guidance_scale)
        except Exception as exc:
            logger.warning("[%s] Could not configure pipeline from options: %s", job_id, exc)

        # ===== STEP 1: Extract Beats =====
        job.update("extraction", 15, "Extracting narrative beats from text...")
        try:
            if not EXTRACT_BEATS_AVAILABLE:
                raise RuntimeError("Beat extraction module not available")
            
            # FIX 8: Log input truncation warning and output to metadata
            max_chars = 3000
            if len(text_input) > max_chars:
                logger.warning(f"[{job_id}] Input truncated from {len(text_input)} to {max_chars} chars")
            limited_text = text_input[:max_chars]
            
            beats_result = step1_extract_beats(text=limited_text, chunk_id=f"web_{job_id}")
            
            if not beats_result:
                raise ValueError("No beats extracted from text")
            
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
            
            # Apply max_beats option if set
            max_beats_opt = options.get('max_beats')
            if max_beats_opt:
                try:
                    limit = int(max_beats_opt)
                    if limit > 0:
                        beats_list = beats_list[:limit]
                        logger.info(f"[{job_id}] Sliced beats to max_beats: {limit}")
                except Exception as ex:
                    logger.warning(f"[{job_id}] Could not parse max_beats option: {ex}")
            
            beat_count = len(beats_list)
            logger.info(f"[{job_id}] Extracted {beat_count} beats")
            
        except Exception as e:
            logger.error(f"[{job_id}] Beat extraction error: {e}")
            raise RuntimeError(f"Failed to extract beats: {str(e)}")
            
        # ===== Group Beats into Pages (FIX C) =====
        pages_list = chunk_beats_into_pages(beats_list, max_chars_per_page=400)
        num_pages = len(pages_list)
        logger.info(f"[{job_id}] Grouped beats into {num_pages} pages. Note: Character visual consistency between pages is not guaranteed.")
        
        final_pages = []
        
        from run_e2e_test import (
            step3_generate_panels, PipelineMetrics, DRY_RUN, step3_composite
        )
        
        for p_idx, page_chunk in enumerate(pages_list):
            page_num = p_idx + 1
            page_job_id = f"{job_id}_p{page_num}"
            page_beats = {
                "page_id": page_job_id,
                "beats": page_chunk
            }
            
            # ===== STEP 2: Generate Layout =====
            job.update("layout", int(35 + (p_idx / num_pages) * 20), f"Generating layout for page {page_num}/{num_pages}...")
            try:
                layout = convert_beats_to_layout(page_beats)
                if not layout:
                    raise ValueError(f"Failed to generate layout for page {page_num}")
                
                # Save layout for reference
                layout_file = output_dir / f"layout_p{page_num}.json"
                with open(layout_file, 'w') as f:
                    json.dump(layout, f, indent=2)
                
                page_panels_count = len(layout.get('panels', []))
                total_panels_count += page_panels_count
                logger.info(f"[{job_id}] Generated layout for page {page_num} with {page_panels_count} panels")
                
            except Exception as e:
                logger.error(f"[{job_id}] Layout generation error on page {page_num}: {e}")
                raise RuntimeError(f"Failed to generate layout for page {page_num}: {str(e)}")
                
            # ===== STEP 3: Generate Panels =====
            job.update("generation", int(55 + (p_idx / num_pages) * 25), f"Generating panels for page {page_num}/{num_pages}...")
            try:
                if DRY_RUN:
                    logger.warning(f"[{job_id}] Dry run mode - skipping image generation for page {page_num}")
                    panels_dir = output_dir / "panels" / page_job_id
                    panels_dir.mkdir(exist_ok=True, parents=True)
                else:
                    pipe = get_shared_pipeline()
                    logger.info(
                        "[%s] Using shared SD pipeline for page=%s page_num=%d/%d pipe_id=%s",
                        job_id,
                        page_job_id,
                        page_num,
                        num_pages,
                        id(pipe),
                    )
                    metrics = PipelineMetrics()
                    with _pipeline_lock:
                        panels_dir = step3_generate_panels(layout, pipe, metrics)
                        
                logger.info(f"[{job_id}] Generated panels for page {page_num}")
                
            except Exception as e:
                logger.error(f"[{job_id}] Panel generation error on page {page_num}: {e}")
                raise RuntimeError(f"Failed to generate panels for page {page_num}: {str(e)}")
                
            # ===== STEP 4: Composite Page =====
            job.update("composition", int(85 + (p_idx / num_pages) * 10), f"Compositing page {page_num}/{num_pages}...")
            try:
                final_page = step3_composite(layout, panels_dir)
                
                if not final_page or not final_page.exists():
                    raise ValueError(f"Composite failed for page {page_num} - no output file")
                
                # Move to web output
                web_final_page = output_dir / f"page_{page_num}_full.png"
                if final_page != web_final_page:
                    shutil.copy2(final_page, web_final_page)
                    final_page = web_final_page
                    
                final_pages.append(final_page)
                logger.info(f"[{job_id}] Composite complete for page {page_num}: {final_page}")
                
            except Exception as e:
                logger.error(f"[{job_id}] Composite error on page {page_num}: {e}")
                raise RuntimeError(f"Failed to composite page {page_num}: {str(e)}")
                
        # ===== Save Metadata =====
        page_paths = [str(p.relative_to(RESULTS_DIR).as_posix()) for p in final_pages]
        results_meta = {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "final_page": page_paths[0] if page_paths else "",
            "pages": page_paths,
            "beats_count": beat_count,
            "panels_count": total_panels_count,
            "options": options,
            "generated_at": datetime.now().isoformat(),
            "input_truncated": len(text_input) > max_chars,
            "original_length": len(text_input)
        }
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results_meta, f, indent=2)
            
        job.complete(page_paths[0] if page_paths else "")
        
    except Exception as e:
        logger.error(f"[{job_id}] Pipeline fatal error: {traceback.format_exc()}")
        job.set_error(e)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


# FIX 9: Evict stale jobs logic
JOB_TTL_SECONDS = 3600

def _evict_stale_jobs():
    now = datetime.now()
    stale = []
    for jid, j in jobs.items():
        if (now - j.start_time).total_seconds() > JOB_TTL_SECONDS:
            stale.append(jid)
    for jid in stale:
        logger.info(f"Evicting stale job {jid}")
        del jobs[jid]
        # Clean results directory
        job_dir = RESULTS_DIR / jid
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Start a new generation job"""
    # FIX 9: Clean stale jobs on request
    _evict_stale_jobs()
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
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
