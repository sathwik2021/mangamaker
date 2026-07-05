# config.py
import os

# ── Local paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DRIVE_ROOT = BASE_DIR
RAW_NOVELS_DIR = os.path.join(BASE_DIR, "raw_novels")
CLEANED_NOVELS_DIR = os.path.join(BASE_DIR, "cleaned_novels")
STEP1_OUTPUTS_DIR = os.path.join(BASE_DIR, "step1_outputs")
CHECKPOINT_FILE = os.path.join(STEP1_OUTPUTS_DIR, "checkpoint.json")
SUMMARY_FILE = os.path.join(STEP1_OUTPUTS_DIR, "summary.json")
LOG_FILE = os.path.join(STEP1_OUTPUTS_DIR, "pipeline.log")

# ── Multi-novel paths ───────────────────────────────────────────────────────
MULTI_NOVEL_CHECKPOINT_FILE = os.path.join(STEP1_OUTPUTS_DIR, "multi_novel_checkpoint.json")
MULTI_NOVEL_SUMMARY_FILE = os.path.join(STEP1_OUTPUTS_DIR, "multi_novel_summary.json")
MULTI_NOVEL_LOG_FILE = os.path.join(STEP1_OUTPUTS_DIR, "multi_novel_pipeline.log")

# ── Fine-tuning paths ───────────────────────────────────────────────────────
FINETUNE_DIR = os.path.join(BASE_DIR, "finetune")
FINETUNE_DATASET_FILE = os.path.join(FINETUNE_DIR, "dataset.jsonl")
FINETUNE_DATASET_TRAIN_FILE = os.path.join(FINETUNE_DIR, "train.jsonl")
FINETUNE_DATASET_EVAL_FILE = os.path.join(FINETUNE_DIR, "eval.jsonl")
FINETUNE_OUTPUT_DIR = os.path.join(FINETUNE_DIR, "checkpoints")
FINETUNE_FINAL_MODEL_DIR = os.path.join(FINETUNE_DIR, "final_model")
FINETUNE_LOG_FILE = os.path.join(FINETUNE_DIR, "finetune.log")

# ── Fine-tuning hyperparameters ─────────────────────────────────────────────
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 2
FINETUNE_GRAD_ACCUMULATION_STEPS = 8
FINETUNE_LEARNING_RATE = 2e-4
FINETUNE_MAX_SEQ_LENGTH = 2048
FINETUNE_WARMUP_RATIO = 0.05
FINETUNE_WEIGHT_DECAY = 0.01
FINETUNE_EVAL_SPLIT = 0.1          # 10% of pages held out for eval
FINETUNE_SAVE_STEPS = 50
FINETUNE_EVAL_STEPS = 50
FINETUNE_LOGGING_STEPS = 10

# ── LoRA / QLoRA settings ───────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Model settings ──────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_TEMPERATURE = 0.2
MODEL_MAX_NEW_TOKENS = 250
USE_4BIT_QUANTIZATION = False

# ── Chunker settings ────────────────────────────────────────────────────────
TARGET_BEATS_MIN = 8
TARGET_BEATS_MAX = 12
TARGET_BEATS_MID = 10
WORDS_PER_BEAT_ESTIMATE = 60
CHUNK_OVERLAP_SENTENCES = 2

# Sentence patterns that indicate scene boundaries
SCENE_BOUNDARY_PATTERNS = [
    r"\*\s*\*\s*\*",           # asterisk dividers
    r"^\s*[-—]{3,}\s*$",       # dashes
    r"^\s*#{1,3}\s",           # markdown headings
    r"CHAPTER\s+[IVXLCDM\d]+", # chapter markers
]

# ── Validator settings ──────────────────────────────────────────────────────
MAX_VALIDATION_RETRIES = 3
EMOTION_INTENSITY_MIN = 1
EMOTION_INTENSITY_MAX = 10

VALID_BEAT_TYPES = [
    "action",
    "dialogue",
    "reaction",
    "description",
    "transition",
]

# ── Error codes ─────────────────────────────────────────────────────────────
ERROR_CODES = {
    "E001": "Invalid or missing beat type",
    "E002": "Beat order not sequential or id/order mismatch",
    "E003": "Character referenced but not in characters list",
    "E004": "Dialogue beat missing or empty text field",
    "E005": "Action beat missing or empty verb field",
    "E006": "Reaction beat missing or empty emotion field",
    "E007": "emotional_flow is empty or not a list of strings",
    "E008": "Empty required array (beats, characters, or emotional_flow)",
}
