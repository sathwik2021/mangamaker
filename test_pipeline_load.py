import torch
import os
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model_id = "runwayml/stable-diffusion-v1-5"
lora_path = "./Step-4-lora-training/output/final_lora"

print(f"Loading base model: {model_id}...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    print("Base model loaded OK.")
except Exception as e:
    print(f"Failed to load base model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if os.path.exists(lora_path):
    print(f"Loading LoRA from {lora_path}...")
    try:
        pipe.load_lora_weights(lora_path)
        print("LoRA loaded OK.")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"LoRA path {lora_path} not found, skipping.")
