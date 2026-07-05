"""
dialog_mapper.py — Map extracted text to narrative beats semantically

Links dialog, monologue, and narration to specific beats using position-based
windows and keyword overlap heuristics.
"""

import logging
from typing import List, Dict, Any, Optional
from text_extractor import TextSegment

logger = logging.getLogger("dialog_mapper")


class DialogMapper:
    """Map extracted text segments to corresponding story beats."""
    
    def __init__(self, use_semantic_matching: bool = False):
        """
        Initialize mapper.
        
        Args:
            use_semantic_matching: If True, use embeddings for similarity.
                                   If False, use position + keyword heuristics.
        """
        self.use_semantic_matching = use_semantic_matching
        self.logger = logger
    
    def _position_window(
        self,
        beat_position: int,
        beat_span: int,
        window_size: int = 500
    ) -> tuple:
        """
        Calculate position window around a beat.
        
        Beat position is approximate (based on text byte offset).
        Return (start, end) byte positions where text for this beat should be found.
        """
        start = max(0, beat_position - window_size)
        end = beat_position + beat_span + window_size
        return (start, end)
    
    def _keyword_overlap(self, text: str, beat_description: str) -> float:
        """
        Score text against beat description using keyword overlap.
        
        Lower scoring: generic words (the, a, is, etc.)
        Higher scoring: action verbs, emotional words, specific nouns.
        """
        # Extract keywords from beat description
        beat_words = set(beat_description.lower().split())
        text_words = set(text.lower().split())
        
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'be', 'been', 'of', 'to', 'in',
            'for', 'and', 'or', 'but', 'that', 'this', 'it', 'with', 'from', 'as'
        }
        
        # Remove stop words
        beat_words -= stop_words
        text_words -= stop_words
        
        if not beat_words or not text_words:
            return 0.0
        
        # Jaccard similarity
        overlap = beat_words & text_words
        union = beat_words | text_words
        
        return len(overlap) / len(union) if union else 0.0
    
    def _extract_emotions(self, text: str) -> List[str]:
        """Extract emotion keywords from text."""
        emotions = {
            'pain': ['pain', 'hurt', 'ache', 'throb', 'agony', 'suffer'],
            'fear': ['fear', 'scared', 'terrified', 'afraid', 'panic', 'dread'],
            'confusion': ['confused', 'confused', 'why', 'what', 'bewildered', 'disoriented'],
            'anger': ['angry', 'furious', 'rage', 'livid', 'mad'],
            'sadness': ['sad', 'sorrow', 'grief', 'melancholy', 'lonely', 'depressed'],
            'joy': ['happy', 'joy', 'glad', 'delighted', 'cheerful'],
            'surprise': ['surprised', 'shocked', 'astonished', 'sudden'],
        }
        
        text_lower = text.lower()
        found_emotions = []
        
        for emotion, words in emotions.items():
            if any(word in text_lower for word in words):
                found_emotions.append(emotion)
        
        return found_emotions
    
    def map_text_to_beats(
        self,
        extracted_text: Dict[str, List[TextSegment]],
        beats: List[Dict[str, Any]],
        source_text: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich beats with text_content field by mapping extracted text segments.
        
        Strategy:
        1. For each beat, use position window to find nearby text
        2. Score text relevance using keyword overlap
        3. Assign top-scoring dialog, monologue, narration to beat
        4. Add `text_content` field to beat
        5. Estimate speech_bubble_count and narrative_priority
        
        Args:
            extracted_text: Output from TextExtractor.extract_all()
            beats: List of beats from Gemini extraction
            source_text: Original full text (for position context)
        
        Returns:
            beats with added "text_content" field
        """
        self.logger.info(f"Mapping {len(beats)} beats to extracted text")
        
        enriched_beats = []
        
        for beat_idx, beat in enumerate(beats):
            # Estimate beat position in source (rough heuristic)
            # Beats are extracted sequentially, so assign proportional positions
            beat_position = int((beat_idx / max(len(beats), 1)) * len(source_text))
            beat_span = len(source_text) // len(beats) if beats else 100
            
            # Get position window for this beat
            window_start, window_end = self._position_window(beat_position, beat_span)
            
            # Collect text within window, scored by relevance
            beat_description = beat.get("description", "")
            
            # Score and filter text segments
            candidate_dialog = []
            candidate_monologue = []
            candidate_narration = []
            
            # Dialog: position-based + keyword match
            for seg in extracted_text.get("dialog", []):
                if window_start <= seg.position <= window_end:
                    score = self._keyword_overlap(seg.text, beat_description)
                    candidate_dialog.append((score, seg))
            
            # Monologue: position-based + keyword match + emotion match
            for seg in extracted_text.get("monologue", []):
                if window_start <= seg.position <= window_end:
                    score = self._keyword_overlap(seg.text, beat_description)
                    
                    # Boost score if emotions match
                    beat_emotions = self._extract_emotions(beat_description)
                    seg_emotions = self._extract_emotions(seg.text)
                    if beat_emotions and seg_emotions:
                        overlap = len(set(beat_emotions) & set(seg_emotions)) / len(set(beat_emotions) | set(seg_emotions))
                        score *= (1.0 + overlap)
                    
                    candidate_monologue.append((score, seg))
            
            # Narration: position-based + relevance
            for seg in extracted_text.get("narration", []):
                if window_start <= seg.position <= window_end:
                    score = self._keyword_overlap(seg.text, beat_description)
                    candidate_narration.append((score, seg))
            
            # Sort by score and pick top N
            candidate_dialog.sort(reverse=True, key=lambda x: x[0])
            candidate_monologue.sort(reverse=True, key=lambda x: x[0])
            candidate_narration.sort(reverse=True, key=lambda x: x[0])
            
            # Assign to beat (convert TextSegment objects to dicts for JSON serialization)
            text_content = {
                "dialog": [
                    {
                        "type": seg.type,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "bubble_style": seg.bubble_style,
                        "position": seg.position
                    }
                    for _, seg in candidate_dialog[:2]
                ],  # Max 2 dialogs per beat
                "monologue": [
                    {
                        "type": seg.type,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "bubble_style": seg.bubble_style,
                        "position": seg.position
                    }
                    for _, seg in candidate_monologue[:1]
                ],  # Max 1 monologue
                "narration": [
                    {
                        "type": seg.type,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "bubble_style": seg.bubble_style,
                        "position": seg.position
                    }
                    for _, seg in candidate_narration[:1]
                ]  # Max 1 narration
            }
            
            # Count bubbles and estimate narrative priority
            bubble_count = len(text_content["dialog"]) + len(text_content["monologue"])
            has_narration = len(text_content["narration"]) > 0
            
            # Narrative priority: high if substantial text, low if visual-only
            narrative_priority = min(1.0, (bubble_count * 0.3) + (0.4 if has_narration else 0.0))
            
            # Add to enriched beat
            enriched_beat = dict(beat)
            enriched_beat["text_content"] = text_content
            enriched_beat["speech_bubble_count"] = bubble_count
            enriched_beat["has_narration_box"] = has_narration
            enriched_beat["narrative_priority"] = narrative_priority
            
            enriched_beats.append(enriched_beat)
            
            if text_content["dialog"] or text_content["monologue"]:
                self.logger.debug(
                    f"Beat {beat_idx}: {bubble_count} bubbles, "
                    f"priority={narrative_priority:.2f}"
                )
        
        self.logger.info(
            f"✅ Mapped beats: avg {sum(b.get('speech_bubble_count', 0) for b in enriched_beats) / len(enriched_beats):.1f} "
            f"bubbles per beat"
        )
        
        return enriched_beats
