import os
import json
import logging
import sys

# Append the directory to make imports work from command line
sys.path.append(os.path.dirname(__file__))

from extractor import Extractor
from validator import Validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sample_text = """The heavy wooden door groaned as Elara pushed it open. Dust danced in the shafts of moonlight piercing the gloom of the old library. 'We shouldn't be here, Elias,' she whispered, shivering as a cold draft swept past them.
Elias stepped forward, his eyes scanning the endless rows of ancient tomes. He pointed a trembling finger at a glowing blue manuscript resting on a pedestal. 'This is it, Elara. The Chronicle of Shadows. If we don't destroy it tonight, the Archmage will find it.'
Suddenly, the floorboards creaked. From the shadows emerged a towering figure wrapped in a dark cloak. The Archmage laughed softly. 'You are too late, children,' his voice echoed, dripping with malice.
Elara drew her dagger, stepping protectively in front of her brother. 'We won't let you take it!' she shouted, terror gripping her."""

def run():
    print("=== INITIALIZING EXTRACTOR (Downloading model weights if necessary) ===")
    extractor = Extractor()
    extractor.load_model()
    
    print("\n=== EXTRACTING BEATS (Running Phi-3 Inference) ===")
    chunk_id = "chunk_sample_001"
    raw_json = extractor.extract(chunk_text=sample_text, chunk_id=chunk_id)
    
    print("\n=== RAW OUTPUT ===")
    if raw_json:
        print(json.dumps(raw_json, indent=2))
    else:
        print("Extraction failed to return valid JSON.")
        return
        
    print("\n=== RUNNING VALIDATOR ===")
    validator = Validator()
    res = validator.validate(raw_json)
    if res.valid:
        print("Validation: PASSED! The output strictly adheres to the schema.")
    else:
        print(f"Validation: FAILED - {res.errors}")

if __name__ == "__main__":
    run()
