# pipeline.py
import gc
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch

import config
from chunker import chunk_text
from drive_manager import DriveManager
from extractor import Extractor
from validator import Validator

# ── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    """Configure root logger to write to Drive log file and stdout."""
    os.makedirs(config.STEP1_OUTPUTS_DIR, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Orchestrates Step 1 of the Novel-to-Manga system:
        1. Mount Drive and create directory structure
        2. Chunk cleaned novel text
        3. For each chunk: extract → validate → retry loop → save
        4. Checkpoint after each successful save
        5. Generate final summary
    """

    def __init__(self):
        self.drive = DriveManager()
        self.extractor = Extractor()
        self.validator = Validator()

    def _clear_gpu_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def _process_chunk(
        self,
        chunk: Dict[str, Any],
        processed_ids: List[str],
    ) -> bool:
        """
        Run the extract → validate → retry loop for a single chunk.

        Returns True on success, False if all retries exhausted.
        """
        chunk_id: str = chunk["chunk_id"]
        chunk_text_content: str = chunk["text"]
        feedback: Optional[str] = None

        for attempt in range(1, config.MAX_VALIDATION_RETRIES + 1):
            logger.info(
                "Chunk %s — attempt %d/%d", chunk_id, attempt, config.MAX_VALIDATION_RETRIES
            )

            # ── Extract ────────────────────────────────────────────────────
            try:
                page = self.extractor.extract(chunk_text_content, chunk_id, feedback=feedback)
            except RuntimeError as oom_exc:
                if "out of memory" in str(oom_exc).lower():
                    logger.warning("OOM on chunk %s (attempt %d) — clearing cache and retrying", chunk_id, attempt)
                    self._clear_gpu_cache()
                    continue
                raise

            if page is None:
                feedback = "The model returned output that could not be parsed as JSON. Return ONLY a valid JSON object."
                logger.warning("Chunk %s attempt %d: JSON parse failed", chunk_id, attempt)
                continue

            # ── Validate ───────────────────────────────────────────────────
            result = self.validator.validate(page)

            if result.valid:
                logger.info("Chunk %s validated successfully on attempt %d", chunk_id, attempt)
                try:
                    self.drive.save_page(page, chunk_id)
                except OSError:
                    # save_page already logged the error
                    return False
                processed_ids.append(chunk_id)
                self.drive.save_checkpoint(processed_ids)
                return True
            else:
                feedback = result.feedback_string()
                logger.warning(
                    "Chunk %s attempt %d validation failed:\n%s", chunk_id, attempt, feedback
                )

        # All retries exhausted
        logger.error(
            "Chunk %s FAILED after %d attempts — logging and skipping",
            chunk_id, config.MAX_VALIDATION_RETRIES,
        )
        self.drive.log_failure(
            chunk_id=chunk_id,
            reason=f"Exceeded {config.MAX_VALIDATION_RETRIES} retries",
            errors=feedback.split("\n") if feedback else [],
        )
        return False

    def run(self, novel_filename: str, resume: bool = True) -> None:
        """
        Run the full Step 1 pipeline.

        Parameters
        ----------
        novel_filename : str  – filename within cleaned_novels_dir
        resume         : bool – if True, skip already-processed chunks
        """
        _setup_logging()

        # ── Drive setup ────────────────────────────────────────────────────
        self.drive.mount_drive()
        self.drive.create_directory_structure()

        # ── Load novel ─────────────────────────────────────────────────────
        logger.info("Reading novel: %s", novel_filename)
        novel_text = self.drive.read_cleaned_novel(novel_filename)
        logger.info("Novel loaded: %d characters", len(novel_text))

        # ── Chunk ──────────────────────────────────────────────────────────
        logger.info("Chunking text…")
        chunks = chunk_text(novel_text)
        logger.info("Total chunks: %d", len(chunks))

        # ── Checkpoint resume ─────────────────────────────────────────────
        processed_ids: List[str] = []
        if resume:
            checkpoint = self.drive.load_checkpoint()
            processed_ids = checkpoint.get("processed_chunk_ids", [])
            logger.info("Resume mode: %d chunks already processed", len(processed_ids))

        # ── Load model ─────────────────────────────────────────────────────
        logger.info("Loading Phi-3 model…")
        self.extractor.load_model()

        # ── Process each chunk ────────────────────────────────────────────
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            if resume and chunk_id in processed_ids:
                logger.info("Skipping already-processed chunk %s", chunk_id)
                continue

            logger.info(
                "Processing chunk %s | words=%d | estimated_beats=%d",
                chunk_id, chunk["word_count"], chunk["estimated_beats"],
            )

            self._process_chunk(chunk, processed_ids)

        # ── Final summary ─────────────────────────────────────────────────
        self.drive.generate_summary(total_chunks=len(chunks))
        logger.info(
            "Pipeline complete. Success: %d | Failed: %d",
            self.drive.success_count,
            self.drive.failed_count,
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Novel → Beat JSON")
    parser.add_argument("novel_filename", help="Filename inside cleaned_novels_dir on Drive")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume/checkpointing")
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.run(novel_filename=args.novel_filename, resume=not args.no_resume)
