import google.genai as genai
from config import GEMINI_API_KEYS, GEMINI_MODELS

_current_model_index = 0
_current_api_key_index = 0


def _current_model() -> str:
    return GEMINI_MODELS[_current_model_index]


def _current_api_key() -> str:
    if GEMINI_API_KEYS:
        return GEMINI_API_KEYS[_current_api_key_index % len(GEMINI_API_KEYS)]
    raise RuntimeError("No Gemini API keys configured")


def _make_client(api_key: str):
    return genai.Client(api_key=api_key)


def _rotate_api_key() -> bool:
    global _current_api_key_index
    if len(GEMINI_API_KEYS) <= 1:
        return False
    _current_api_key_index = (_current_api_key_index + 1) % len(GEMINI_API_KEYS)
    return True


def _rotate_model() -> bool:
    global _current_model_index, _current_api_key_index
    _current_model_index += 1
    _current_api_key_index = 0
    return _current_model_index < len(GEMINI_MODELS)


def get_current_model() -> str:
    return _current_model()


def generate(prompt: str) -> str:
    while True:
        model_name = _current_model()
        api_key = _current_api_key()
        client = _make_client(api_key)
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
            if any(code in err for code in ["429", "503", "RESOURCE_EXHAUSTED"]):
                if _rotate_api_key():
                    print(f"⚠️  {model_name} rate limited — switching to next API key (key {_current_api_key_index + 1}/{len(GEMINI_API_KEYS)}) and retrying same model...")
                    continue
                if _rotate_model():
                    new_model = _current_model()
                    print(f"⚠️  {model_name} quota exhausted — switching to next model {new_model}")
                    continue
                raise RuntimeError("❌ All models and API keys exhausted. Try again later.")
            raise
