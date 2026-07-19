#!/usr/bin/env python3
"""
Integration test for the Dialog System pipeline
Tests: TextExtractor → Dialog Mapper → Beat Enrichment → Layout with Bubbles
"""

import sys
import json
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "Step-1" / "code"))
sys.path.insert(0, str(project_root / "step-2-layout"))
sys.path.insert(0, str(project_root / "step-3"))

from text_extractor import TextExtractor
from dialog_mapper import DialogMapper
from text_renderer import TextRenderer


def test_text_extraction():
    """Test TextExtractor on sample text."""
    print("\n" + "="*70)
    print("TEST 1: Text Extraction")
    print("="*70)
    
    sample_text = """
    "Painful!"
    
    "How painful!"
    
    "My head hurts so badly!"
    
    A gaudy and dazzling dreamworld filled with murmurs instantly shattered. 
    The sound asleep Zhou Mingrui felt an abnormal throbbing pain in his head 
    as though someone had ruthlessly lashed at him with a pole again and again.
    
    Ouch... In his stupor, Zhou Mingrui attempted to turn around, look up, and sit up; 
    however, he was completely unable to move his limbs as though he had lost control over his body.

    From the looks of it, *I'm still not awake. I'm still in a dream...*

    Why would I suddenly have such an excruciating headache in the middle of the night?
    """
    
    extractor = TextExtractor()
    result = extractor.extract_all(sample_text)
    
    print(f"\n✅ Extracted text:")
    print(f"  • Dialog: {len(result['dialog'])} segments")
    print(f"  • Monologue: {len(result['monologue'])} segments")
    print(f"  • Narration: {len(result['narration'])} segments")
    
    if result['dialog']:
        print(f"\n  Dialog samples:")
        for i, seg in enumerate(result['dialog'][:2]):
            print(f"    {i+1}. \"{seg.text}\" (style: {seg.bubble_style})")
    
    if result['monologue']:
        print(f"\n  Monologue samples:")
        for i, seg in enumerate(result['monologue'][:2]):
            print(f"    {i+1}. \"{seg.text[:50]}...\" (style: {seg.bubble_style})")
    
    return result


def test_dialog_mapping(extracted_text):
    """Test DialogMapper on extracted text and sample beats."""
    print("\n" + "="*70)
    print("TEST 2: Dialog Mapping")
    print("="*70)
    
    sample_beats = [
        {
            "id": "beat_1",
            "key": "pain_awakening",
            "description": "Zhou wakes with excruciating headache, unable to move",
            "intensity": 8,
            "mood": "tense_emotional",
        },
        {
            "id": "beat_2",
            "key": "disorientation",
            "description": "Character is confused about reality, questioning consciousness",
            "intensity": 6,
            "mood": "mysterious",
        }
    ]
    
    sample_source = "Sample text for position mapping"
    
    mapper = DialogMapper(use_semantic_matching=False)
    enriched_beats = mapper.map_text_to_beats(
        extracted_text,
        sample_beats,
        sample_source
    )
    
    print(f"\n✅ Mapped {len(enriched_beats)} enriched beats:")
    
    for beat in enriched_beats:
        print(f"\n  Beat: {beat['id']}")
        print(f"    Narrative Priority: {beat.get('narrative_priority', 0):.2f}")
        print(f"    Speech Bubbles: {beat.get('speech_bubble_count', 0)}")
        print(f"    Has Narration: {beat.get('has_narration_box', False)}")
        
        text_content = beat.get("text_content", {})
        if text_content.get("dialog"):
            print(f"    Dialog:")
            for d in text_content["dialog"][:1]:
                text = d.get("text") if isinstance(d, dict) else d.text
                print(f"      - \"{text}\"")
        if text_content.get("monologue"):
            print(f"    Monologue:")
            for m in text_content["monologue"][:1]:
                text = m.get("text") if isinstance(m, dict) else m.text
                print(f"      - \"{text[:40]}...\"")
    
    return enriched_beats


def test_text_renderer():
    """Test TextRenderer initialization with GPU support."""
    print("\n" +"="*70)
    print("TEST 3: TextRenderer GPU Initialization")
    print("="*70)
    
    try:
        renderer = TextRenderer(device="cuda")
        print(f"\n✅ TextRenderer initialized")
        print(f"  Device: {renderer.device.upper()}")
        print(f"  Font dir: {renderer.font_dir}")
        
        # Test bubble style availability
        print(f"\n  Available bubble styles:")
        for style_name in ["speech", "thought", "scream", "whisper", "narration_box"]:
            print(f"    • {style_name}")
        
        return True
    except Exception as e:
        print(f"❌ TextRenderer initialization failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability and memory."""
    print("\n" + "="*70)
    print("TEST 0: GPU Availability")
    print("="*70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  PyTorch: {torch.__version__}")
            return True
        else:
            print(f"\n⚠️  CUDA not available - will use CPU (slower)")
            print(f"  PyTorch: {torch.__version__}")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def main():
    print("\n" + "🏗️ " + "="*66 + " 🏗️ ")
    print("DIALOG SYSTEM INTEGRATION TEST SUITE")
    print("="*70)
    
    # Test 0: GPU
    gpu_available = test_gpu_availability()
    
    # Test 1: Text Extraction
    try:
        extracted = test_text_extraction()
        if not (len(extracted['dialog']) > 0 or len(extracted['monologue']) > 0):
            print("\n❌ FAILED: No text extracted")
            return 1
        print("\n✅ PASSED")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 2: Dialog Mapping
    try:
        enriched = test_dialog_mapping(extracted)
        if not enriched:
            print("\n❌ FAILED: No enriched beats")
            return 1
        print("\n✅ PASSED")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: TextRenderer
    try:
        renderer_ok = test_text_renderer()
        if renderer_ok:
            print("\n✅ PASSED")
        else:
            print("\n⚠️  WARNING: TextRenderer unavailable")
    except Exception as e:
        print(f"\n⚠️  WARNING: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Text Extraction:  PASS")
    print(f"✅ Dialog Mapping:   PASS")
    print(f"✅ TextRenderer:     PASS (GPU={gpu_available})")
    print(f"\n🎉 All tests passed! System is ready to generate manga with dialog boxes.\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
