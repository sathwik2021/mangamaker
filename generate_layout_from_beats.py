#!/usr/bin/env python3
"""Generate layout from beats"""
import json
from pathlib import Path

# Load beats
beats_data = json.load(open('test_output/beats.json'))
beats_list = beats_data.get('beats', [])

# Create layout
layout = {
    'version': '1.0',
    'canvasWidth': 1800,
    'canvasHeight': 2400,
    'panels': []
}

num_panels = len(beats_list)
panel_height = 2400 / num_panels if num_panels > 0 else 2400
y_offset = 0

for i, beat in enumerate(beats_list):
    panel = {
        'id': f'panel_{i}',
        'x': 0,
        'y': int(y_offset),
        'width': 1800,
        'height': int(panel_height),
        'beat_index': i,
        'beat_key': beat.get('key', f'beat_{i}'),
        'description': beat.get('description', ''),
        'scene_type': beat.get('scene_type', ''),
        'visual_notes': beat.get('visual_notes', ''),
        'prompt_tags': beat.get('prompt_tags', '')
    }
    layout['panels'].append(panel)
    y_offset += panel_height

# Save layout
Path('test_output/layout').mkdir(exist_ok=True, parents=True)
with open('test_output/layout/layout.json', 'w', encoding='utf-8') as f:
    json.dump(layout, f, indent=2, ensure_ascii=False)

print(f'✅ Layout created with {num_panels} panels')
print('   Saved to: test_output/layout/layout.json')
