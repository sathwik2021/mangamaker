import json

with open('model1-step1.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open('extracted_notebook.txt', 'w', encoding='utf-8') as f:
    for cell in notebook.get('cells', []):
        f.write(f"\n--- {cell.get('cell_type', 'unknown').upper()} ---\n")
        source = cell.get('source', [])
        if isinstance(source, list):
            f.write(''.join(source))
        else:
            f.write(source)
        f.write("\n")
