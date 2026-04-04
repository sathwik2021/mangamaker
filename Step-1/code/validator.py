# validator.py
import logging
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


class ValidationResult:
    """Holds the outcome of a single validation pass."""

    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []          # human-readable messages
        self.error_codes: List[str] = []     # E001–E008

    def add_error(self, code: str, message: str) -> None:
        self.valid = False
        self.errors.append(f"[{code}] {message}")
        self.error_codes.append(code)

    def feedback_string(self) -> str:
        return "\n".join(self.errors)


class Validator:
    """
    Strictly validates beat-based JSON pages against the architecture schema.

    Error codes:
        E001 – Invalid or missing beat type
        E002 – Beat order not sequential or id/order mismatch
        E003 – Character referenced but not in characters list
        E004 – Dialogue beat missing or empty text field
        E005 – Action beat missing or empty verb field
        E006 – Reaction beat missing or empty emotion field
        E007 – emotional_flow is empty or not a list of strings
        E008 – Empty required array (beats, characters, or emotional_flow)
    """

    # ── Required top-level keys ─────────────────────────────────────────────
    _REQUIRED_TOP_LEVEL = {"page_id", "source_chunk_id", "characters", "beats", "emotional_flow"}

    # ── Required keys per beat ──────────────────────────────────────────────
    _REQUIRED_BEAT_KEYS = {"id", "order", "type", "characters", "causes", "description",
                           "text", "verb", "emotion", "intensity"}

    def validate(self, page: Dict[str, Any]) -> ValidationResult:
        """
        Validate a page dict.  Mutates ``page`` in place to clamp emotion
        intensity values that are out of range.

        Returns a ValidationResult (valid=True means the page may be saved).
        """
        result = ValidationResult()

        # ── 1. Top-level keys present ───────────────────────────────────────
        for key in self._REQUIRED_TOP_LEVEL:
            if key not in page:
                result.add_error("E008", f"Missing required top-level key: '{key}'")

        if not result.valid:
            return result

        characters_list: List[str] = page.get("characters", [])
        beats: List[Any] = page.get("beats", [])
        emotional_flow: Any = page.get("emotional_flow", [])

        # ── 2. Non-empty arrays (E008) ──────────────────────────────────────
        if not isinstance(characters_list, list) or len(characters_list) == 0:
            result.add_error("E008", "Top-level 'characters' must be a non-empty list")

        if not isinstance(beats, list) or len(beats) == 0:
            result.add_error("E008", "Top-level 'beats' must be a non-empty list")

        if not isinstance(emotional_flow, list) or len(emotional_flow) == 0:
            result.add_error("E007", "'emotional_flow' must be a non-empty list of strings")
        else:
            # ── 3. emotional_flow strings (E007) ────────────────────────────
            for idx, item in enumerate(emotional_flow):
                if not isinstance(item, str) or not item.strip():
                    result.add_error(
                        "E007",
                        f"'emotional_flow[{idx}]' must be a non-empty string, got: {repr(item)}",
                    )

        if not result.valid:
            return result

        # ── 4. Validate each beat ───────────────────────────────────────────
        for beat_idx, beat in enumerate(beats):
            if not isinstance(beat, dict):
                result.add_error("E001", f"Beat at index {beat_idx} is not a dict")
                continue

            self._validate_beat(beat, beat_idx, characters_list, result)

        # ── 5. Order must be sequential 1..N (E002) ─────────────────────────
        if isinstance(beats, list) and all(isinstance(b, dict) for b in beats):
            orders = [b.get("order") for b in beats]
            expected = list(range(1, len(beats) + 1))
            if orders != expected:
                result.add_error(
                    "E002",
                    f"Beat 'order' values must be sequential 1..{len(beats)}, got: {orders}",
                )

        return result

    def _validate_beat(
        self,
        beat: Dict[str, Any],
        beat_idx: int,
        characters_list: List[str],
        result: ValidationResult,
    ) -> None:
        """Validate a single beat dict, mutating emotion intensity if needed."""

        # ── E001: beat type ─────────────────────────────────────────────────
        beat_type = beat.get("type")
        if beat_type not in config.VALID_BEAT_TYPES:
            result.add_error(
                "E001",
                f"Beat[{beat_idx}] has invalid type '{beat_type}'. "
                f"Must be one of {config.VALID_BEAT_TYPES}",
            )
            return  # can't do type-specific checks without a valid type

        # ── E002: id / order consistency ────────────────────────────────────
        order = beat.get("order")
        beat_id = beat.get("id")
        if not isinstance(order, int):
            result.add_error("E002", f"Beat[{beat_idx}] 'order' must be an integer, got: {repr(order)}")
        else:
            expected_id = f"beat_{order}"
            if beat_id != expected_id:
                result.add_error(
                    "E002",
                    f"Beat[{beat_idx}] id '{beat_id}' does not match expected '{expected_id}'",
                )

        # ── E003: characters referenced must exist ───────────────────────────
        beat_chars = beat.get("characters", [])
        if not isinstance(beat_chars, list):
            result.add_error("E003", f"Beat[{beat_idx}] 'characters' must be a list")
        else:
            for char in beat_chars:
                if char not in characters_list:
                    result.add_error(
                        "E003",
                        f"Beat[{beat_idx}] references character '{char}' not in top-level characters",
                    )

        # ── Type-specific checks ─────────────────────────────────────────────
        if beat_type == "dialogue":
            # E004: dialogue must have non-empty text
            text_val = beat.get("text", "")
            if not isinstance(text_val, str) or not text_val.strip():
                result.add_error("E004", f"Beat[{beat_idx}] (dialogue) must have non-empty 'text' field")

        elif beat_type == "action":
            # E005: action must have non-empty verb
            verb_val = beat.get("verb", "")
            if not isinstance(verb_val, str) or not verb_val.strip():
                result.add_error("E005", f"Beat[{beat_idx}] (action) must have non-empty 'verb' field")

        elif beat_type == "reaction":
            # E006: reaction must have non-empty emotion
            emotion_val = beat.get("emotion", "")
            if not isinstance(emotion_val, str) or not emotion_val.strip():
                result.add_error("E006", f"Beat[{beat_idx}] (reaction) must have non-empty 'emotion' field")

        # ── Clamp intensity (in-place mutation) ──────────────────────────────
        intensity = beat.get("intensity")
        if isinstance(intensity, (int, float)):
            clamped = int(max(config.EMOTION_INTENSITY_MIN, min(config.EMOTION_INTENSITY_MAX, intensity)))
            beat["intensity"] = clamped

        # ── causes must be a list ─────────────────────────────────────────────
        causes = beat.get("causes", [])
        if not isinstance(causes, list):
            result.add_error("E008", f"Beat[{beat_idx}] 'causes' must be a list (can be empty)")
