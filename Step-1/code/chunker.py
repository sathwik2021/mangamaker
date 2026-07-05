# chunker.py
import re
import uuid
from typing import List, Dict, Any

import config


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
    raw = sentence_endings.split(text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def _is_scene_boundary(sentence: str) -> bool:
    """Return True if the sentence looks like a scene boundary marker."""
    for pattern in config.SCENE_BOUNDARY_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _estimate_beats(sentences: List[str]) -> int:
    """Estimate the number of beats represented by a list of sentences."""
    word_count = sum(len(s.split()) for s in sentences)
    return max(1, round(word_count / config.WORDS_PER_BEAT_ESTIMATE))


def chunk_text(text: str, overlap: int = config.CHUNK_OVERLAP_SENTENCES) -> List[Dict[str, Any]]:
    """
    Chunk novel text into segments targeting TARGET_BEATS_MIN–TARGET_BEATS_MAX beats each.

    Parameters
    ----------
    text    : cleaned novel text
    overlap : number of sentences to overlap between consecutive chunks

    Returns
    -------
    List of chunk dicts, each containing:
        chunk_id        : str  – unique identifier
        text            : str  – chunk text content
        estimated_beats : int  – estimated beat count
        word_count      : int  – total words in chunk
        start_sentence  : int  – inclusive index into global sentence list
        end_sentence    : int  – exclusive index into global sentence list
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current_start = 0

    while current_start < len(sentences):
        accumulated: List[str] = []
        i = current_start

        while i < len(sentences):
            sentence = sentences[i]

            # Always add the sentence first
            accumulated.append(sentence)

            # Check if we've hit a natural scene boundary (after adding)
            at_boundary = _is_scene_boundary(sentence)

            # Estimate beats for the accumulated sentences
            estimated = _estimate_beats(accumulated)

            # If we're in the target range and at a boundary, or exceeded max, stop
            if at_boundary and config.TARGET_BEATS_MIN <= estimated <= config.TARGET_BEATS_MAX:
                i += 1
                break
            elif estimated >= config.TARGET_BEATS_MAX:
                i += 1
                break

            i += 1

        if not accumulated:
            break

        word_count = sum(len(s.split()) for s in accumulated)
        estimated_beats = _estimate_beats(accumulated)

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": " ".join(accumulated),
            "estimated_beats": estimated_beats,
            "word_count": word_count,
            "start_sentence": current_start,
            "end_sentence": i,
        }
        chunks.append(chunk)

        # Advance with overlap
        next_start = i - overlap if (i - overlap) > current_start else i
        if next_start <= current_start:
            next_start = current_start + 1
        current_start = next_start

    return chunks
