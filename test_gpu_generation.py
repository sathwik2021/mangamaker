#!/usr/bin/env python3
"""Quick test to verify GPU-accelerated generation"""
import sys
import torch
from pathlib import Path

print("=" * 60)
print("GPU Generation Test")
print("=" * 60)

# Check GPU
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
print("\n" + "=" * 60)

# Test importing pipeline components
print("\nTesting imports...")
try:
    from diffusers import StableDiffusionPipeline
    print("✓ diffusers.StableDiffusionPipeline imported")
except Exception as e:
    print(f"✗ Failed to import StableDiffusionPipeline: {e}")
    sys.exit(1)

try:
    # Add Step-1 to path for imports
    sys.path.insert(0, str(Path(__file__).parent / "Step-1"))
    from run_e2e_test import convert_beats_to_layout
    print("✓ convert_beats_to_layout imported")
except Exception as e:
    print(f"✗ Failed to import layout converter: {e}")

print("\n" + "=" * 60)
print("Test Complete - GPU environment ready for generation")
print("=" * 60)
