import json
from validator import Validator

mock_json = {
  "page_id": "page_chunk_sample_001",
  "source_chunk_id": "chunk_sample_001",
  "characters": ["Elara", "Elias", "Archmage"],
  "beats": [
    {
      "id": "beat_1",
      "order": 1,
      "type": "action",
      "characters": ["Elara"],
      "verb": "pushed",
      "causes": [],
      "description": "Elara pushes a heavy wooden door open, dust dancing in moonlight.",
      "text": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_2",
      "order": 2,
      "type": "dialogue",
      "characters": ["Elara", "Elias"],
      "text": "We shouldn't be here, Elias,",
      "causes": ["beat_1"],
      "description": "Elara whispers, shivering from a cold draft.",
      "verb": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_3",
      "order": 3,
      "type": "action",
      "characters": ["Elias"],
      "verb": "stepped",
      "causes": ["beat_2"],
      "description": "Elias steps forward and points at a glowing manuscript.",
      "text": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_4",
      "order": 4,
      "type": "dialogue",
      "characters": ["Elias", "Elara"],
      "text": "This is it, Elara. The Chronicle of Shadows. If we don't destroy it tonight, the Archmage will find it.",
      "causes": ["beat_3"],
      "description": "Elias points trembling finger at the glowing blue manuscript.",
      "verb": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_5",
      "order": 5,
      "type": "transition",
      "characters": ["Archmage"],
      "causes": ["beat_4"],
      "description": "Floorboards creak as a towering figure in a cloak emerges.",
      "text": "",
      "verb": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_6",
      "order": 6,
      "type": "dialogue",
      "characters": ["Archmage"],
      "text": "You are too late, children,",
      "causes": ["beat_5"],
      "description": "The Archmage laughs softly, malice in his voice.",
      "verb": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_7",
      "order": 7,
      "type": "reaction",
      "characters": ["Elara"],
      "emotion": "terror",
      "intensity": 8,
      "causes": ["beat_6"],
      "description": "Elara reacts to the Archmage's threat.",
      "text": "",
      "verb": ""
    },
    {
      "id": "beat_8",
      "order": 8,
      "type": "action",
      "characters": ["Elara"],
      "verb": "drew",
      "causes": ["beat_7"],
      "description": "Elara draws her dagger, protecting her brother.",
      "text": "",
      "emotion": "",
      "intensity": 0
    },
    {
      "id": "beat_9",
      "order": 9,
      "type": "dialogue",
      "characters": ["Elara"],
      "text": "We won't let you take it!",
      "causes": ["beat_8"],
      "description": "Elara shouts bravely.",
      "verb": "",
      "emotion": "",
      "intensity": 0
    }
  ],
  "emotional_flow": ["apprehension", "determination", "dread", "terror", "defiance"]
}

print("Running Validator Custom JSON...")
validator = Validator()
res = validator.validate(mock_json)

if res.valid:
    print("VALIDATION PASSED PERFECTLY!\n")
    print("=== MOCK PIPELINE OUTPUT ===")
    print(json.dumps(mock_json, indent=2))
else:
    print("VALIDATION FAILED:", res.errors)
