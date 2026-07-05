import json
from extractor import _extract_json_from_text, _build_user_prompt
from transformers import AutoTokenizer

print("=== TEST 1: extractor.py Prompt ===")
prompt = _build_user_prompt("Text excerpt.", "test_123")
print("Prompt contains page_id:", "page_id: page_test_123" in prompt)

print("\n=== TEST 2: extractor.py JSON Regex Recovery ===")
# Example of truncated JSON with an unclosed string
truncated_output = '''```json
{
  "page_id": "page_test_123",
  "beats": [
    {
      "id": "beat_1",
      "text": "This is unclosed'''

parsed = _extract_json_from_text(truncated_output)
print("Parsed JSON:", parsed is not None)
if parsed:
    print("Keys found:", parsed.keys())
    print("Text value:", parsed['beats'][0]['text'])

print("\n=== TEST 3: finetune.py Token Masking Logic ===")
try:
    from finetune import FineTuner
    import config
    import datasets
    
    # We will just test the tokenizer logic by instantiating the class
    # To avoid downloading Phi-3 if not cached, we'll try just using a distilgpt2 tokenizer.
    ft = FineTuner()
    print("Loading test tokenizer (distilgpt2 for speed)...")
    ft.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    ft.tokenizer.pad_token = ft.tokenizer.eos_token
    ft.tokenizer.padding_side = "right"
    
    # Mock text following the <|assistant|>\n format
    mock_text = "System: Hello User: World <|assistant|>\n{\"test\": 123}"
    ds = datasets.Dataset.from_dict({"text": [mock_text]})
    
    # Run tokenize logic
    tokenized = ft._tokenize_dataset(ds)
    input_ids = tokenized["input_ids"][0]
    labels = tokenized["labels"][0]
    
    print(f"Total Sequence Length: {len(input_ids)}")
    print(f"Labels length: {len(labels)}")
    
    # Find how many -100 are there
    masked_count = labels.count(-100)
    print(f"Masked tokens (-100): {masked_count}")
    print(f"Unmasked tokens (actual target): {len(input_ids) - masked_count}")
    if len(input_ids) == len(labels) and masked_count > 0:
        print("Masking logic test passed!")
    else:
        print("Masking logic failed.")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to test finetune.py logic.")
