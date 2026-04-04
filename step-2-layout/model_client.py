import os
import google.genai as genai
from config import GEMINI_API_KEY, GEMINI_MODELS

client = genai.Client(api_key=GEMINI_API_KEY)

_current_model_index = 0

def get_current_model():
    return GEMINI_MODELS[_current_model_index]

def generate(prompt: str) -> str:
    global _current_model_index

    while _current_model_index < len(GEMINI_MODELS):
        model_name = GEMINI_MODELS[_current_model_index]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            # Check for safety filter block
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                if response.prompt_feedback.block_reason is not None:
                    print(f"⚠️  {model_name} blocked due to: {response.prompt_feedback.block_reason}")
                    raise ValueError(f"Prompt blocked by safety filter: {response.prompt_feedback.block_reason}")
            
            # Check if response.text is None (can happen with safety filters)
            if response.text is None:
                raise ValueError(f"{model_name} returned None response (possible safety filter or content block)")
            
            return response.text

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"⚠️  {model_name} quota exhausted — switching to next model...")
                _current_model_index += 1
                if _current_model_index < len(GEMINI_MODELS):
                    print(f"🔄 Now using: {GEMINI_MODELS[_current_model_index]}")
                else:
                    raise RuntimeError("❌ All models exhausted. Try again tomorrow.")
            else:
                raise