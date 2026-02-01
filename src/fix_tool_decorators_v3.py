"""Fix @tool decorators by directly modifying line array"""
import json
import re

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 8 (tool implementations)
cell_idx = 8
cell_lines = nb['cells'][cell_idx]['source']

print(f"Fixing tool decorators in cell {cell_idx}...")
print(f"Total lines: {len(cell_lines)}")

# Process each line
fixed_lines = []
for i, line in enumerate(cell_lines):
    # Fix 1: Remove 'name=' from tool decorator
    if re.search(r'^\s+name="[^"]+",', line):
        # Replace 'name="xyz",' with '"xyz",'
        line = re.sub(r'name=', '', line)
        print(f"Line {i}: Removed 'name=' keyword")

    # Fix 2: Remove 'description=' from tool decorator
    if re.search(r'^\s+description=\(', line):
        # Replace 'description=(' with just '('
        line = re.sub(r'description=', '', line)
        print(f"Line {i}: Removed 'description=' keyword")

    # Fix 3: Remove 'input_schema=' from tool decorator
    if re.search(r'^\s+input_schema=\{', line):
        # Replace 'input_schema={' with just '{'
        line = re.sub(r'input_schema=', '', line)
        print(f"Line {i}: Removed 'input_schema=' keyword")

    fixed_lines.append(line)

# Update notebook
nb['cells'][cell_idx]['source'] = fixed_lines

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Fixed cell {cell_idx}")
print("Changes: Removed 'name=', 'description=', 'input_schema=' keywords")
print("\nNEXT: In Jupyter, close and reopen the notebook, then re-run Cell 8")
