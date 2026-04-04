import os
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models in priority order (best first, fallback down the list)
GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-3-flash-preview",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-3.1-flash-lite-preview",
]

CANVAS_WIDTH  = 1800
CANVAS_HEIGHT = 2400

# Step 1 saves here → Step 2 reads from here
INPUT_DIR  = r"C:\Users\peech\Documents\model-1\step1_outputs"

# Step 2 saves here → Step 3 will read from here
OUTPUT_DIR = r"C:\Users\peech\Documents\model-1\step-2-layout\output"