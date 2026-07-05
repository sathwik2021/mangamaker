from model_client import generate, get_current_model

print(f"Starting with model: {get_current_model()}")

response = generate("Say hello in one sentence.")
print(f"✅ Response: {response}")
print(f"Active model: {get_current_model()}")