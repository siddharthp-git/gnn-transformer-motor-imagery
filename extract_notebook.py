import json
import sys

def extract_code(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            print(f"\n# Cell {i}")
            print(source)

if __name__ == "__main__":
    extract_code(sys.argv[1])
