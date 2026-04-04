import json
import re

raw = '''```json
{
  "page_id": "page_test_123",
  "beats": [
    {
      "id": "beat_1",
      "text": "This is unclosed'''

brace_match = re.search(r"(\{.*)", raw, re.DOTALL)
candidate = brace_match.group(1)

for end in range(len(candidate), len(candidate) // 2, -1):
    trimmed = candidate[:end]
    if trimmed.count('"') % 2 != 0:
        trimmed += '"'
    open_braces = trimmed.count("{") - trimmed.count("}")
    open_brackets = trimmed.count("[") - trimmed.count("]")
    closed = trimmed + "]" * open_brackets + "}" * open_braces
    try:
        json.loads(closed)
        print(f"SUCCESS at end={end}")
        print("Closed string:")
        print(closed)
        break
    except json.JSONDecodeError as e:
        if end == len(candidate):
            print(f"Failed at end={end}: {e}")
            print(f"FAILED closed string:\n{closed}")
