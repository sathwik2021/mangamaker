import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

working = [
    "models/gemini-2.5-flash",
    "models/gemma-3-1b-it",
    "models/gemma-3-4b-it",
    "models/gemma-3-12b-it",
    "models/gemma-3-27b-it",
    "models/gemma-3n-e4b-it",
    "models/gemma-3n-e2b-it",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-3-flash-preview",
    "models/gemini-3.1-flash-lite-preview",
    "models/gemini-robotics-er-1.5-preview",
]

print(f"{'Model':<45} {'Input Tokens':>15} {'Output Tokens':>15}")
print("-" * 80)

for model_name in working:
    try:
        info = client.models.get(model=model_name)
        input_tokens  = getattr(info, "input_token_limit",  "N/A")
        output_tokens = getattr(info, "output_token_limit", "N/A")
        print(f"{model_name:<45} {str(input_tokens):>15} {str(output_tokens):>15}")
    except Exception as e:
        print(f"{model_name:<45} ERROR: {str(e)[:40]}")