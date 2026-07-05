# ============================================================================
# NOVEL TEXT CLEANING PIPELINE - FINAL VERSION
# For Gutenberg .txt files
# ============================================================================

import re
import os
from pathlib import Path

class GutenbergCleaner:
    """Cleans raw Gutenberg text files for beat extraction"""
    
    def __init__(self, raw_dir: str, cleaned_dir: str):
        self.raw_dir = Path(raw_dir)
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_all(self):
        """Clean all .txt files in raw directory"""
        txt_files = list(self.raw_dir.glob("*.txt")) + list(self.raw_dir.glob("*.TXT"))
        
        for txt_file in txt_files:
            print(f"Cleaning: {txt_file.name}")
            cleaned = self.clean_file(txt_file)
            
            # Save cleaned version
            output_path = self.cleaned_dir / f"{txt_file.stem}_clean.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            print(f"  → Saved to: {output_path.name}")
    
    def clean_file(self, file_path: Path) -> str:
        """Clean a single Gutenberg file"""
        # Use errors='replace' to handle encoding issues safely
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Step 1: Remove Gutenberg header
        text = self._remove_header(text)
        
        # Step 2: Remove Gutenberg footer
        text = self._remove_footer(text)
        
        # Step 3: Remove front matter (dedications, prefaces, TOC, etc.)
        text = self._remove_front_matter(text)
        
        # Step 4: Normalize quotes (curly quotes ONLY - NO single quote conversion)
        text = self._normalize_quotes(text)
        
        # Step 5: Remove page numbers and artifacts
        text = self._remove_artifacts(text)
        
        # Step 6: Normalize whitespace (preserve paragraph structure)
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_header(self, text: str) -> str:
        """Remove everything before the actual story starts"""
        # Common Gutenberg start markers
        start_patterns = [
            r'\*\*\* ?START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'\*\*\* ?START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'Project Gutenberg eBook.*?\n\n',
            r'The Project Gutenberg eBook.*?\n\n',
        ]
        
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[match.end():]
                break
        
        # Also look for "*** START OF" variations
        match = re.search(r'\*\*\* ?START OF .*?\*\*\*', text, re.IGNORECASE)
        if match:
            text = text[match.end():]
        
        return text
    
    def _remove_footer(self, text: str) -> str:
        """Remove everything after the story ends"""
        # Common Gutenberg end markers
        end_patterns = [
            r'\*\*\* ?END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'\*\*\* ?END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'End of the Project Gutenberg EBook.*?',
            r'End of Project Gutenberg EBook.*?',
        ]
        
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[:match.start()]
                break
        
        return text
    
    def _remove_front_matter(self, text: str) -> str:
        """Remove dedications, prefaces, TOC, etc. before first chapter.
        Handles various chapter header formats including ALL CAPS."""
        
        # ------------------------------------------------------------------
        # Remove structured Table of Contents blocks FIRST
        # (Handles: CONTENTS + Roman numerals list)
        # ------------------------------------------------------------------
        
        toc_patterns = [
            # CONTENTS ... I. Introduction
            r'CONTENTS\s+.*?(?=\n\s*[IVXLC]+\.\s*\n)',
            
            # CONTENTS ... CHAPTER I
            r'CONTENTS\s+.*?(?=\n\s*CHAPTER\s+[IVXLC\d]+)',
            
            # CONTENTS ... Chapter 1
            r'CONTENTS\s+.*?(?=\n\s*Chapter\s+\d+)',
            
            # Also handle "TABLE OF CONTENTS"
            r'TABLE\s+OF\s+CONTENTS\s+.*?(?=\n\s*[IVXLC]+\.\s*\n)',
            r'TABLE\s+OF\s+CONTENTS\s+.*?(?=\n\s*CHAPTER\s+[IVXLC\d]+)',
            
            # Handle "CONTENTS." with period
            r'CONTENTS\.\s+.*?(?=\n\s*[IVXLC]+\.\s*\n)',
        ]
        
        for pattern in toc_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Also remove any standalone "CONTENTS" line
        text = re.sub(r'^\s*CONTENTS\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove "LIST OF ILLUSTRATIONS" blocks
        illustration_patterns = [
            r'LIST\s+OF\s+ILLUSTRATIONS\s+.*?(?=\n\s*[IVXLC]+\.\s*\n)',
            r'ILLUSTRATIONS\s+.*?(?=\n\s*CHAPTER\s+)',
        ]
        
        for pattern in illustration_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # ------------------------------------------------------------------
        # Now find first chapter marker
        # ------------------------------------------------------------------
        
        # Comprehensive pattern for chapter headers
        chapter_patterns = [
            r'\bCHAPTER\s+THE\s+[A-Z]+\b',        # CHAPTER THE FIRST
            r'\bCHAPTER\s+[IVXLC]+\b',            # CHAPTER I, CHAPTER V
            r'\bCHAPTER\s+\d+\b',                  # CHAPTER 1, CHAPTER 23
            r'\bCHAPTER\s+[A-Z]+\b',               # CHAPTER ONE, CHAPTER FIRST (ALL CAPS)
            r'\bChapter\s+\d+\b',                   # Chapter 1
            r'\bChapter\s+[IVXLC]+\b',              # Chapter I
            r'\bCHAP\.\s+[IVXLC]+\b',               # CHAP. I
            r'\bCHAP\.\s+\d+\b',                    # CHAP. 1
            r'\bCHAP\.[IVXLC]+\b',                  # CHAP.I (no space)
            r'\bCHAPTER[IVXLC]+\b',                 # CHAPTERI (no space)
            r'\bCHAPTER\s+[A-Z]+\s+[A-Z]+\b',       # CHAPTER THE FIRST (word based)
            r'\bTHE\s+FIRST\s+CHAPTER\b',           # THE FIRST CHAPTER (alternate order)
            r'\bCHAPTER\s+[A-Z]+\s*$',              # CHAPTER at line end
        ]
        
        # Try each pattern with IGNORECASE flag
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Found a chapter marker - cut everything before it
                # But keep the chapter header itself
                print(f"  Found chapter marker: '{match.group(0)}'")  # Debug output
                return text[match.start():]
        
        # Fallback: look for any line that contains chapter-like text in first 100 lines
        lines = text.split('\n')
        for i, line in enumerate(lines[:100]):  # Check first 100 lines
            if re.search(r'\b(?:CHAPTER|Chapter|CHAP\.)\b', line, re.IGNORECASE):
                print(f"  Found chapter line {i+1}: '{line[:50]}...'")  # Debug output
                return '\n'.join(lines[i:])
        
        # If still no chapter found, try a simpler approach - look for any CHAPTER text
        simple_match = re.search(r'\bCHAPTER\b', text, re.IGNORECASE)
        if simple_match:
            print(f"  Found 'CHAPTER' at position {simple_match.start()}")  # Debug output
            return text[simple_match.start():]
        
        print("  No chapter markers found - returning full text")  # Debug output
        return text  # If no chapter found, return whole text
    
    def _normalize_quotes(self, text: str) -> str:
        """Convert ONLY curly quotes to straight quotes.
        NEVER convert single quotes used for possession or contractions."""
        
        # Curly double quotes to straight double
        text = text.replace('“', '"').replace('”', '"')
        
        # Curly single quotes to straight single
        text = text.replace('‘', "'").replace('’', "'")
        
        # DO NOT attempt to convert straight single quotes to double
        # This preserves possessives like "hero's" and contractions like "don't"
        
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove page numbers, illustration markers, etc."""
        # Remove page numbers (common formats)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'\[Pg \d+\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Illustration:?.*?\]', '', text, flags=re.IGNORECASE)
        
        # Remove multiple underscores or asterisks
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'\*{3,}', '', text)
        
        # Remove table of contents lines (common patterns)
        text = re.sub(r'^.*\.\.+.*\d+$', '', text, flags=re.MULTILINE)
        
        # Remove "Transcriber's Notes" sections
        tn_match = re.search(r'Transcriber\'?s Note.*?\n\n', text, re.IGNORECASE | re.DOTALL)
        if tn_match:
            text = text[:tn_match.start()] + text[tn_match.end():]
        
        # Remove "***" separators
        text = re.sub(r'\n\s*\*{3,}\s*\n', '\n\n', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Fix excessive blank lines while preserving paragraph structure."""
        
        # Replace multiple blank lines with max 2 (preserves paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at line ends
        text = re.sub(r' +\n', '\n', text)
        
        # Ensure chapter headers have space around them
        text = re.sub(r'(?<!\n\n)(CHAPTER [A-Z0-9]+)', r'\n\n\1\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\n\n)(Chapter \d+)', r'\n\n\1\n\n', text, flags=re.IGNORECASE)
        
        # Fix common Gutenberg line breaks (remove single hard wraps but keep paragraph breaks)
        # This is a heuristic - join lines that don't end with sentence punctuation
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            if i < len(lines) - 1:
                # If line doesn't end with sentence punctuation and next line isn't empty
                if (lines[i] and not lines[i].rstrip().endswith(('.', '!', '?')) and 
                    lines[i+1] and not lines[i+1].startswith(' ') and
                    not lines[i+1].startswith('CHAPTER') and
                    not lines[i+1].startswith('Chapter')):
                    # Join with previous line
                    fixed_lines.append(lines[i] + ' ' + lines[i+1])
                    i += 2
                    continue
            fixed_lines.append(lines[i])
            i += 1
        
        text = '\n'.join(fixed_lines)
        
        return text
    
    def split_into_chapters(self, cleaned_file: Path) -> list:
        """Split cleaned text into chapters (optional)"""
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Common chapter markers
        chapter_patterns = [
            r'CHAPTER [IVXLC]+\.?',
            r'CHAPTER \d+\.?',
            r'Chapter \d+\.?',
            r'CHAP\. [IVXLC]+\.?',
            r'^[IVXLC]+\.\s+',  # Roman numerals at start of line
            r'CHAPTER [A-Z]+',   # CHAPTER ONE, CHAPTER FIRST
        ]
        
        # Combine patterns
        pattern = '|'.join(f'({p})' for p in chapter_patterns)
        
        # Find all chapter starts
        chapters = []
        last_pos = 0
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            if last_pos > 0:
                chapter_text = text[last_pos:match.start()].strip()
                if chapter_text:
                    chapters.append(chapter_text)
            last_pos = match.start()
        
        # Add final chapter
        if last_pos > 0:
            chapters.append(text[last_pos:].strip())
        
        # If no chapters found, return whole text as single "chapter"
        return chapters if chapters else [text]
    
    def preview_cleaning(self, file_path: Path, num_lines: int = 20):
        """Preview before/after cleaning for a file"""
        print(f"\nPreview for: {file_path.name}")
        print("=" * 60)
        
        # Original
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            original = f.read()
        
        print("ORIGINAL (first 1000 chars):")
        print("-" * 40)
        print(original[:1000])
        
        # Cleaned
        cleaned = self.clean_file(file_path)
        print("\n\nCLEANED (first 1000 chars):")
        print("-" * 40)
        print(cleaned[:1000])
        print("=" * 60)


# ============================================================================
# VERIFICATION SCRIPT - IMPROVED WITH SAFER TOC CHECK
# ============================================================================

def verify_cleaning_quality(cleaned_file: Path):
    """Check if cleaning preserved important narrative elements"""
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    checks = {
        "Has dialogue quotes": bool(re.search(r'"[^"]*"', text)),
        "Has possessives preserved": bool(re.search(r"\w+'\w+", text)),  # hero's, don't
        "Has chapter markers": bool(re.search(r'CHAPTER|Chapter', text, re.IGNORECASE)),
        "Has paragraph breaks": bool(re.search(r'\n\n', text)),
        "No Gutenberg header": not bool(re.search(r'START OF .*PROJECT GUTENBERG', text, re.IGNORECASE)),
        "No Gutenberg footer": not bool(re.search(r'END OF .*PROJECT GUTENBERG', text, re.IGNORECASE)),
        "No page numbers": not bool(re.search(r'\n\s*\d+\s*\n', text)),
        "No standalone TOC": not bool(re.search(r'^\s*CONTENTS\s*$', text, re.IGNORECASE | re.MULTILINE)),
        "No illustration list": not bool(re.search(r'^\s*LIST OF ILLUSTRATIONS\s*$', text, re.IGNORECASE | re.MULTILINE)),
        "No transcriber notes": not bool(re.search(r'Transcriber', text, re.IGNORECASE)),
    }
    
    print(f"\nVerification for: {cleaned_file.name}")
    print("-" * 40)
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    # Count dialogue lines as a quality metric
    dialogue_lines = len(re.findall(r'"[^"]*"', text))
    print(f"\n📊 Dialogue lines detected: {dialogue_lines}")
    
    # Show first few lines to verify chapter start
    print("\n📖 First 5 lines after cleaning:")
    lines = text.split('\n')[:5]
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}: {line[:80]}")
    
    return all(checks.values())


# ============================================================================
# BATCH PROCESSING FUNCTION
# ============================================================================

def batch_clean_novels(raw_dir: str = "raw_novels", cleaned_dir: str = "cleaned_novels"):
    """Clean all novels and generate a summary report"""
    
    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    
    print(f"📁 Raw directory: {raw_dir}")
    print(f"📁 Cleaned directory: {cleaned_dir}")
    print("=" * 60)
    
    # Check if raw directory has files
    txt_files = list(Path(raw_dir).glob("*.txt")) + list(Path(raw_dir).glob("*.TXT"))
    if not txt_files:
        print("❌ No .txt files found in raw directory!")
        print("Please add your downloaded novels to:")
        print(f"  {raw_dir}")
        return
    
    print(f"✅ Found {len(txt_files)} text files to clean")
    print("=" * 60)
    
    # Run cleaning
    cleaner = GutenbergCleaner(raw_dir, cleaned_dir)
    cleaner.clean_all()
    
    print("\n✅ Cleaning complete!")
    print("=" * 60)
    
    # Verify all cleaned files
    sample_files = list(Path(cleaned_dir).glob("*_clean.txt"))
    if sample_files:
        print("\n🔍 VERIFICATION SUMMARY")
        print("=" * 60)
        
        results = []
        for sample in sample_files:
            result = verify_cleaning_quality(sample)
            results.append((sample.name, result))
            print("-" * 40)
        
        # Summary
        passed = sum(1 for _, r in results if r)
        print(f"\n📊 Summary: {passed}/{len(results)} files passed all checks")
        
        # Show preview of first file
        print("\n📖 Sample preview (first file):")
        with open(sample_files[0], 'r', encoding='utf-8') as f:
            preview = f.read()[:500]
        print(preview)
        print("...")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Set your paths - modify these to match your Google Drive structure
    RAW_DIR = "raw_novels"
    CLEANED_DIR = "cleaned_novels"
    
    # For local testing, use:
    # RAW_DIR = "raw_novels"
    # CLEANED_DIR = "cleaned_novels"
    
    # Run batch cleaning
    batch_clean_novels(RAW_DIR, CLEANED_DIR)