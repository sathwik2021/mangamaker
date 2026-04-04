"""
text_extractor.py — Extract and classify text from novel excerpts

Separates dialog, internal monologue, and narration for semantic mapping to beats.
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("text_extractor")


@dataclass
class TextSegment:
    """Single extracted text segment."""
    type: str  # "dialog", "monologue", "narration"
    text: str
    speaker: str = None  # who is speaking/thinking
    bubble_style: str = "speech"  # speech, thought, scream, whisper, narration_box
    position: int = 0  # byte offset in source


class TextExtractor:
    """Extract and classify all text content from raw novel excerpt."""
    
    def __init__(self):
        self.logger = logger
    
    def extract_dialog(self, text: str) -> List[TextSegment]:
        """
        Extract quoted speech/dialog.
        
        Patterns:
        - "Direct quote"
        - "Quote?" 
        - Quote with em-dash: "Speaking"—Zhou said
        """
        segments = []
        
        # Pattern 1: Simple quotes "text"
        # Match: "any text inside quotes"
        pattern_quote = r'"([^"]+)"'
        
        for match in re.finditer(pattern_quote, text):
            quote_text = match.group(1).strip()
            
            if len(quote_text) > 2:  # Filter out tiny fragments
                # Detect exclamations vs questions
                if quote_text.endswith('!'):
                    style = "scream" if len(quote_text.split()) <= 2 else "speech"  # Short exclamations are screams
                elif quote_text.endswith('?'):
                    style = "speech"
                else:
                    style = "speech"
                
                segments.append(TextSegment(
                    type="dialog",
                    text=quote_text,
                    bubble_style=style,
                    position=match.start()
                ))
        
        return segments
    
    def extract_internal_monologue(self, text: str) -> List[TextSegment]:
        """
        Extract internal thoughts/monologue.
        
        Patterns:
        - *italicized text* (thinking)
        - _underscored text_ (thinking)
        - First-person questions: "Why would I..." (no quotes but first-person)
        - Parenthetical asides: (I was confused)
        """
        segments = []
        
        # Pattern 1: *italicized* or _underscored_
        for pattern in [r'\*([^*]+)\*', r'_([^_]+)_']:
            for match in re.finditer(pattern, text):
                thought_text = match.group(1).strip()
                if len(thought_text) > 3:
                    segments.append(TextSegment(
                        type="monologue",
                        text=thought_text,
                        bubble_style="thought",
                        position=match.start()
                    ))
        
        # Pattern 2: Standalone first-person internal questions/statements
        # Look for sentences starting with "I" or "Why" that appear alone
        # This is heuristic-based and position-dependent
        first_person_patterns = [
            r'\n(I[^.\n]{20,200}[.?!])',  # I... sentence
            r'\n(Why[^.\n]{15,200}[?])',    # Why...?
            r'\n(How[^.\n]{15,200}[?])',    # How...?
            r'\n(What[^.\n]{15,200}[?])',   # What...?
        ]
        
        for pattern in first_person_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                thought_text = match.group(1).strip()
                if len(thought_text) > 5 and '"' not in thought_text:  # Avoid quoted text
                    segments.append(TextSegment(
                        type="monologue",
                        text=thought_text,
                        bubble_style="thought",
                        position=match.start()
                    ))
        
        # Pattern 3: Parenthetical thoughts (I was confused)
        for match in re.finditer(r'\(([^)]{10,150})\)', text):
            paren_text = match.group(1).strip()
            # Filter for actual internal monologue (not just scene description)
            if any(word in paren_text.lower() for word in ['i', 'me', 'my', 'think', 'feel', 'know']):
                segments.append(TextSegment(
                    type="monologue",
                    text=paren_text,
                    bubble_style="thought",
                    position=match.start()
                ))
        
        return segments
    
    def extract_narration(self, text: str) -> List[TextSegment]:
        """
        Extract narration/scene description.
        
        Strategy: Find paragraph-length chunks that are NOT dialog/monologue.
        Typically 3rd-person descriptive sentences.
        """
        segments = []
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        offset = 0
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip if paragraph is dialog-like or too short
            if len(para) < 20:
                offset += len(para) + 2
                continue
            
            # Skip if it's mostly quoted (more than 50% quotes)
            quote_count = len(re.findall(r'"[^"]*"', para))
            quote_length = sum(len(m.group(0)) for m in re.finditer(r'"[^"]*"', para))
            if quote_length / len(para) > 0.5 and quote_count > 2:
                offset += len(para) + 2
                continue
            
            # Skip if it's mostly enclosed in parens/italics (internal monologue)
            special_chars = para.count('(') + para.count('*') + para.count('_')
            if special_chars > len(para) / 10:
                offset += len(para) + 2
                continue
            
            # This is narration
            # Split into sentences for better readability
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:  # Only significant sentences
                    segments.append(TextSegment(
                        type="narration",
                        text=sentence,
                        bubble_style="narration_box",
                        position=offset
                    ))
                offset += len(sentence) + 1
            
            offset += 2  # paragraph break
        
        return segments
    
    def extract_all(self, text: str) -> Dict[str, List[TextSegment]]:
        """
        Extract all text types from source.
        
        Returns:
        {
            "dialog": [...],
            "monologue": [...],
            "narration": [...]
        }
        """
        self.logger.info(f"Extracting text from {len(text)} characters")
        
        dialog = self.extract_dialog(text)
        monologue = self.extract_internal_monologue(text)
        narration = self.extract_narration(text)
        
        result = {
            "dialog": dialog,
            "monologue": monologue,
            "narration": narration
        }
        
        self.logger.info(
            f"✅ Extracted {len(dialog)} dialog, "
            f"{len(monologue)} monologue, {len(narration)} narration segments"
        )
        
        return result
    
    def to_json(self, extracted: Dict[str, List[TextSegment]]) -> Dict:
        """Convert extracted segments to JSON-serializable format."""
        return {
            "dialog": [
                {
                    "type": seg.type,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "bubble_style": seg.bubble_style,
                    "position": seg.position
                }
                for seg in extracted["dialog"]
            ],
            "monologue": [
                {
                    "type": seg.type,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "bubble_style": seg.bubble_style,
                    "position": seg.position
                }
                for seg in extracted["monologue"]
            ],
            "narration": [
                {
                    "type": seg.type,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "bubble_style": seg.bubble_style,
                    "position": seg.position
                }
                for seg in extracted["narration"]
            ]
        }
