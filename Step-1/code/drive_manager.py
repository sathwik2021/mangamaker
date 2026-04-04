# drive_manager.py
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class DriveManager:
    """
    Manages Google Drive I/O for the manga pipeline.

    Directory structure (all under DRIVE_ROOT):
        manga_pipeline/
            raw_novels/
            cleaned_novels/
            step1_outputs/
                pages/        ← validated beat JSON files
                failed/       ← failed chunk logs
                checkpoint.json
                summary.json
                pipeline.log
    """

    PAGES_DIR = os.path.join(config.STEP1_OUTPUTS_DIR, "pages")
    FAILED_DIR = os.path.join(config.STEP1_OUTPUTS_DIR, "failed")

    def __init__(self):
        self._success_count: int = 0
        self._failed_count: int = 0
        self._error_log: List[Dict[str, Any]] = []

    # ── Initialization ───────────────────────────────────────────────────────

    def mount_drive(self) -> None:
        """Mount Google Drive (Colab only)."""
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive", force_remount=False)
            logger.info("Google Drive mounted")
        except ImportError:
            logger.warning("google.colab not available — assuming Drive is already mounted")

    def create_directory_structure(self) -> None:
        """Create all required directories on Drive."""
        dirs = [
            config.DRIVE_ROOT,
            config.RAW_NOVELS_DIR,
            config.CLEANED_NOVELS_DIR,
            config.STEP1_OUTPUTS_DIR,
            self.PAGES_DIR,
            self.FAILED_DIR,
        ]
        for d in dirs:
            _ensure_dir(d)
            logger.debug("Ensured directory: %s", d)
        logger.info("Directory structure ready")

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load existing checkpoint.

        Returns dict: { "processed_chunk_ids": [...] }
        """
        if os.path.exists(config.CHECKPOINT_FILE):
            try:
                with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Checkpoint loaded: %d chunks already processed", len(data.get("processed_chunk_ids", [])))
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read checkpoint file (%s); starting fresh", exc)
        return {"processed_chunk_ids": []}

    def save_checkpoint(self, processed_chunk_ids: List[str]) -> None:
        """Persist the list of processed chunk IDs."""
        data = {
            "processed_chunk_ids": processed_chunk_ids,
            "updated_at": datetime.utcnow().isoformat(),
        }
        try:
            with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Checkpoint saved (%d ids)", len(processed_chunk_ids))
        except OSError as exc:
            logger.error("Failed to save checkpoint: %s", exc)

    # ── Page I/O ─────────────────────────────────────────────────────────────

    def save_page(self, page: Dict[str, Any], chunk_id: str) -> str:
        """
        Save a validated page JSON to Drive.

        Returns the file path written.
        """
        filename = f"page_{chunk_id}.json"
        filepath = os.path.join(self.PAGES_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(page, f, indent=2, ensure_ascii=False)
            self._success_count += 1
            logger.info("[SUCCESS] Saved page for chunk %s → %s", chunk_id, filepath)
            return filepath
        except OSError as exc:
            self._failed_count += 1
            logger.error("[ERROR] Could not save page for chunk %s: %s", chunk_id, exc)
            raise

    def log_failure(self, chunk_id: str, reason: str, errors: Optional[List[str]] = None) -> None:
        """Log a failed chunk to the failed directory and internal error log."""
        entry = {
            "chunk_id": chunk_id,
            "reason": reason,
            "errors": errors or [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._error_log.append(entry)
        self._failed_count += 1

        filename = f"failed_{chunk_id}.json"
        filepath = os.path.join(self.FAILED_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            logger.warning("[FAILED] chunk %s logged to %s | reason: %s", chunk_id, filepath, reason)
        except OSError as exc:
            logger.error("Could not write failure log for chunk %s: %s", chunk_id, exc)

    # ── Summary ──────────────────────────────────────────────────────────────

    def generate_summary(self, total_chunks: int) -> None:
        """Write a summary JSON file with pipeline run statistics."""
        summary = {
            "total_chunks": total_chunks,
            "success_count": self._success_count,
            "failed_count": self._failed_count,
            "success_rate": (
                round(self._success_count / total_chunks, 4) if total_chunks else 0.0
            ),
            "errors": self._error_log,
            "generated_at": datetime.utcnow().isoformat(),
        }
        try:
            with open(config.SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Summary written to %s", config.SUMMARY_FILE)
        except OSError as exc:
            logger.error("Could not write summary file: %s", exc)

    # ── Read helpers ─────────────────────────────────────────────────────────

    def read_cleaned_novel(self, filename: str) -> str:
        """Read a cleaned novel text file from Drive."""
        filepath = os.path.join(config.CLEANED_NOVELS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def failed_count(self) -> int:
        return self._failed_count
