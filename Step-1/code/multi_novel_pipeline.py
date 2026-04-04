# multi_novel_pipeline.py
#
# Discovers every .txt file in CLEANED_NOVELS_DIR, runs the Step 1 pipeline
# on each one, and records per-novel outcomes.
#
# Usage (Colab cell):
#   from multi_novel_pipeline import MultiNovelPipeline
#   mnp = MultiNovelPipeline()
#   mnp.run(resume=True)
#
# Or from the command line:
#   python multi_novel_pipeline.py [--no-resume]

import gc
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

import config
from chunker import chunk_text
from drive_manager import DriveManager
from extractor import Extractor
from validator import Validator

# ── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    os.makedirs(config.STEP1_OUTPUTS_DIR, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.MULTI_NOVEL_LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


# ── Multi-novel checkpoint helpers ───────────────────────────────────────────

def _load_multi_checkpoint() -> Dict[str, Any]:
    """
    Returns:
        {
          "completed_novels": ["novel_a.txt", ...],
          "failed_novels":    ["novel_b.txt", ...],
          "novel_chunk_map":  { "novel_a.txt": ["chunk_id_1", ...], ... }
        }
    """
    if os.path.exists(config.MULTI_NOVEL_CHECKPOINT_FILE):
        try:
            with open(config.MULTI_NOVEL_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Multi-novel checkpoint loaded: %d completed, %d failed",
                len(data.get("completed_novels", [])),
                len(data.get("failed_novels", [])),
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read multi-novel checkpoint (%s); starting fresh", exc)
    return {"completed_novels": [], "failed_novels": [], "novel_chunk_map": {}}


def _save_multi_checkpoint(checkpoint: Dict[str, Any]) -> None:
    checkpoint["updated_at"] = datetime.utcnow().isoformat()
    try:
        with open(config.MULTI_NOVEL_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
    except OSError as exc:
        logger.error("Failed to save multi-novel checkpoint: %s", exc)


def _save_multi_summary(results: List[Dict[str, Any]], total_novels: int) -> None:
    summary = {
        "total_novels": total_novels,
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "novels": results,
        "generated_at": datetime.utcnow().isoformat(),
    }
    try:
        with open(config.MULTI_NOVEL_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Multi-novel summary written to %s", config.MULTI_NOVEL_SUMMARY_FILE)
    except OSError as exc:
        logger.error("Could not write multi-novel summary: %s", exc)


# ── Single-novel processing (reuses Step 1 components) ───────────────────────

class _NovelProcessor:
    """
    Processes one novel: chunk → extract → validate → save.
    Shares the already-loaded Extractor and Validator across novels.
    """

    def __init__(self, drive: DriveManager, extractor: Extractor, validator: Validator):
        self.drive = drive
        self.extractor = extractor
        self.validator = validator

    def _clear_gpu_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _process_chunk(
        self,
        chunk: Dict[str, Any],
        processed_ids: List[str],
    ) -> bool:
        chunk_id: str = chunk["chunk_id"]
        chunk_text_content: str = chunk["text"]
        feedback: Optional[str] = None

        for attempt in range(1, config.MAX_VALIDATION_RETRIES + 1):
            logger.info("  Chunk %s — attempt %d/%d", chunk_id, attempt, config.MAX_VALIDATION_RETRIES)

            try:
                page = self.extractor.extract(chunk_text_content, chunk_id, feedback=feedback)
            except RuntimeError as oom_exc:
                if "out of memory" in str(oom_exc).lower():
                    logger.warning("  OOM on chunk %s — clearing cache and retrying", chunk_id)
                    self._clear_gpu_cache()
                    continue
                raise

            if page is None:
                feedback = "The model returned output that could not be parsed as JSON. Return ONLY a valid JSON object."
                logger.warning("  Chunk %s attempt %d: JSON parse failed", chunk_id, attempt)
                continue

            result = self.validator.validate(page)

            if result.valid:
                try:
                    self.drive.save_page(page, chunk_id)
                except OSError:
                    return False
                processed_ids.append(chunk_id)
                return True
            else:
                feedback = result.feedback_string()
                logger.warning("  Chunk %s validation failed:\n%s", chunk_id, feedback)

        logger.error("  Chunk %s FAILED after %d attempts — skipping", chunk_id, config.MAX_VALIDATION_RETRIES)
        self.drive.log_failure(
            chunk_id=chunk_id,
            reason=f"Exceeded {config.MAX_VALIDATION_RETRIES} retries",
            errors=feedback.split("\n") if feedback else [],
        )
        return False

    def process_novel(
        self,
        filename: str,
        already_processed_chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Process a single novel file.

        Returns a result dict:
            { "filename": str, "status": "completed"|"failed",
              "chunks_total": int, "chunks_success": int, "chunks_failed": int,
              "processed_chunk_ids": [...] }
        """
        logger.info("══ Processing novel: %s ══", filename)

        try:
            novel_text = self.drive.read_cleaned_novel(filename)
        except OSError as exc:
            logger.error("Cannot read '%s': %s — skipping novel", filename, exc)
            return {
                "filename": filename,
                "status": "failed",
                "reason": f"File read error: {exc}",
                "chunks_total": 0,
                "chunks_success": 0,
                "chunks_failed": 0,
                "processed_chunk_ids": [],
            }

        logger.info("  Loaded %d characters", len(novel_text))

        try:
            chunks = chunk_text(novel_text)
        except Exception as exc:
            logger.error("Chunking failed for '%s': %s — skipping novel", filename, exc)
            return {
                "filename": filename,
                "status": "failed",
                "reason": f"Chunking error: {exc}",
                "chunks_total": 0,
                "chunks_success": 0,
                "chunks_failed": 0,
                "processed_chunk_ids": [],
            }

        logger.info("  %d chunks generated", len(chunks))

        processed_ids: List[str] = list(already_processed_chunk_ids)
        success_count = 0
        fail_count = 0

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in already_processed_chunk_ids:
                logger.info("  Skipping already-processed chunk %s", chunk_id)
                success_count += 1
                continue

            ok = self._process_chunk(chunk, processed_ids)
            if ok:
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            "  Novel '%s' done: %d success, %d failed",
            filename, success_count, fail_count,
        )

        return {
            "filename": filename,
            "status": "completed",
            "chunks_total": len(chunks),
            "chunks_success": success_count,
            "chunks_failed": fail_count,
            "processed_chunk_ids": processed_ids,
        }


# ── MultiNovelPipeline ────────────────────────────────────────────────────────

class MultiNovelPipeline:
    """
    Discovers all .txt files in CLEANED_NOVELS_DIR and runs Step 1 on each.

    Behaviour:
        - Skips novels already marked completed in the multi-novel checkpoint.
        - On per-novel failure: logs the error, records the novel as failed,
          and continues to the next novel.
        - Saves the multi-novel checkpoint after every novel.
        - Generates a final multi-novel summary JSON.
    """

    def __init__(self):
        self.drive = DriveManager()
        self.extractor = Extractor()
        self.validator = Validator()

    def _discover_novels(self) -> List[str]:
        """Return sorted list of .txt filenames in CLEANED_NOVELS_DIR."""
        try:
            all_files = os.listdir(config.CLEANED_NOVELS_DIR)
        except OSError as exc:
            logger.error("Cannot list cleaned_novels directory: %s", exc)
            return []
        novels = sorted(f for f in all_files if f.lower().endswith(".txt"))
        logger.info("Discovered %d .txt novel(s) in %s", len(novels), config.CLEANED_NOVELS_DIR)
        return novels

    def run(self, resume: bool = True) -> None:
        """
        Run Step 1 on every .txt novel in CLEANED_NOVELS_DIR.

        Parameters
        ----------
        resume : bool
            If True, skip novels already completed and skip chunk IDs already
            processed within a partially-complete novel.
        """
        _setup_logging()

        # ── Drive setup ────────────────────────────────────────────────────
        self.drive.mount_drive()
        self.drive.create_directory_structure()
        os.makedirs(config.FINETUNE_DIR, exist_ok=True)

        # ── Discover novels ────────────────────────────────────────────────
        novels = self._discover_novels()
        if not novels:
            logger.error("No .txt files found in %s — aborting", config.CLEANED_NOVELS_DIR)
            return

        # ── Load multi-novel checkpoint ────────────────────────────────────
        checkpoint = _load_multi_checkpoint() if resume else {
            "completed_novels": [],
            "failed_novels": [],
            "novel_chunk_map": {},
        }
        completed_novels: List[str] = checkpoint.get("completed_novels", [])
        novel_chunk_map: Dict[str, List[str]] = checkpoint.get("novel_chunk_map", {})

        # ── Load model once for all novels ────────────────────────────────
        logger.info("Loading Phi-3 model (shared across all novels)…")
        self.extractor.load_model()

        processor = _NovelProcessor(self.drive, self.extractor, self.validator)
        results: List[Dict[str, Any]] = []

        # ── Iterate over novels ────────────────────────────────────────────
        for filename in novels:
            if resume and filename in completed_novels:
                logger.info("Skipping already-completed novel: %s", filename)
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "chunks_total": 0,
                    "chunks_success": 0,
                    "chunks_failed": 0,
                })
                continue

            already_processed = novel_chunk_map.get(filename, [])

            try:
                result = processor.process_novel(filename, already_processed)
            except Exception as exc:
                logger.error(
                    "Unexpected error processing '%s': %s — skipping novel", filename, exc,
                    exc_info=True,
                )
                result = {
                    "filename": filename,
                    "status": "failed",
                    "reason": f"Unexpected exception: {exc}",
                    "chunks_total": 0,
                    "chunks_success": 0,
                    "chunks_failed": 0,
                    "processed_chunk_ids": [],
                }

            results.append(result)

            # ── Update multi-novel checkpoint ──────────────────────────────
            if result["status"] == "completed":
                if filename not in completed_novels:
                    completed_novels.append(filename)
            novel_chunk_map[filename] = result.get("processed_chunk_ids", [])

            checkpoint["completed_novels"] = completed_novels
            checkpoint["novel_chunk_map"] = novel_chunk_map
            if result["status"] == "failed":
                failed = checkpoint.get("failed_novels", [])
                if filename not in failed:
                    failed.append(filename)
                checkpoint["failed_novels"] = failed

            _save_multi_checkpoint(checkpoint)

            # ── Clear GPU cache between novels ─────────────────────────────
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Final summary ──────────────────────────────────────────────────
        _save_multi_summary(results, total_novels=len(novels))
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        logger.info(
            "All novels processed. Completed: %d | Failed: %d | Skipped: %d",
            completed, failed, skipped,
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1 — all novels in cleaned_novels/")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and reprocess everything from scratch",
    )
    args = parser.parse_args()

    mnp = MultiNovelPipeline()
    mnp.run(resume=not args.no_resume)
