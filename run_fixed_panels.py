#!/usr/bin/env python
import sys
import os
from pathlib import Path

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['USE_CLIP'] = '1'
os.environ['SD_STEPS'] = '40'
os.environ['SD_GUIDANCE'] = '7.5'
os.environ['NUM_CANDIDATES'] = '1'
os.environ['MAX_RETRIES'] = '1'
os.environ['TORCH_DEVICE'] = 'cuda'  # Use GPU if available, falls back to CPU
os.environ['USE_LORA'] = '0'  # DISABLE LoRA - it's trained on different manga content

# Import from main run_e2e_test
sys.path.insert(0, str(Path.cwd()))
from run_e2e_test import (
    PipelineConfig, 
    PipelineMetrics,
    load_sd_pipeline,
    step3_generate_panels,
    step3_composite,
)

import json

# Load cached beats and layout
beats_json = json.load(open('test_output/beats.json'))
layout_json = json.load(open('test_output/layout/layout.json'))

print('\n' + '='*70)
print('  🎬 PANEL GENERATION WITH IMPROVED CLIP FILTERING')
print('='*70)
print(f'  Config: 1 candidates, 1 retries, CLIP=True, LoRA=True')
print()

# Initialize pipeline
config = PipelineConfig.from_env()
print(f'  [INFO] SD Steps: {config.sd_steps}, CFG: {config.sd_guidance_scale}, Candidates: {config.num_candidates}')

# Generate panels (Step 3b)
output_dir = Path('test_output')
panels_dir = output_dir / 'panels' / 'page_chunk_001'
panels_dir.mkdir(parents=True, exist_ok=True)

try:
    print('\n' + '-'*70)
    print('  STEP 3b: Layout → Panels (with CLIP character filtering)')
    print('-'*70)
    
    # Load SD pipeline
    pipe = load_sd_pipeline()
    metrics = PipelineMetrics()
    
    # Generate panels
    panels_dir = step3_generate_panels(layout_json, pipe, metrics)
    
    print('\n' + '-'*70)
    print('  STEP 3c: Panels → Composition')
    print('-'*70)
    
    # Composite final page
    out_page = step3_composite(layout_json, panels_dir)
    
    print(f'  ✓ Final page composed successfully!')
    print(f'     Output: {out_page}')
    print('\n' + '='*70)
    print('  🎬 PIPELINE COMPLETE')
    print('='*70)
    
except Exception as e:
    print(f'  [ERROR] {e}')
    import traceback
    traceback.print_exc()

