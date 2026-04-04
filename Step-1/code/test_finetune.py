from finetune import FineTuner
from transformers import AutoTokenizer
import datasets
import traceback

print("=== STARTING FINETUNE MASKING TEST ===")
try:
    ft = FineTuner()
    ft.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    ft.tokenizer.pad_token = ft.tokenizer.eos_token
    ft.tokenizer.padding_side = "right"
    
    mock_text = "System: Hello User: World <|assistant|>\n{\"test\": 123}"
    ds = datasets.Dataset.from_dict({"text": [mock_text]})
    print("Dataset created, running _tokenize_dataset...")
    tokenized = ft._tokenize_dataset(ds)
    input_ids = tokenized["input_ids"][0]
    labels = tokenized["labels"][0]
    
    print(f"Total Sequence Length: {len(input_ids)}")
    print(f"Labels length: {len(labels)}")
    
    masked_count = labels.count(-100)
    print(f"Masked tokens (-100): {masked_count}")
    print(f"Unmasked tokens: {len(input_ids) - masked_count}")
except Exception as e:
    traceback.print_exc()
