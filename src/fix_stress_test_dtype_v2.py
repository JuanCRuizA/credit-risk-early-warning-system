"""Fix run_stress_test function - add dtype conversion before predictions"""
import json
import re

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with run_stress_test function
stress_test_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def run_stress_test(' in source and '3.9 WHAT-IF STRESS TESTING' in source:
            stress_test_cell_idx = idx
            print(f"Found stress test cell at index {idx}")
            break

if stress_test_cell_idx is None:
    print("ERROR: Could not find run_stress_test cell")
    exit(1)

# Get cell source as list of lines
cell_lines = nb['cells'][stress_test_cell_idx]['source']

# Find the line indices we need to modify
x_stressed_line_idx = None
x_baseline_line_idx = None

for i, line in enumerate(cell_lines):
    if 'X_stressed = X_stressed[feature_names]' in line:
        x_stressed_line_idx = i
    if 'X_baseline = X_baseline[feature_names]' in line:
        x_baseline_line_idx = i

if x_stressed_line_idx is None or x_baseline_line_idx is None:
    print("ERROR: Could not find target lines")
    exit(1)

print(f"Found X_stressed line at index {x_stressed_line_idx}")
print(f"Found X_baseline line at index {x_baseline_line_idx}")

# Insert dtype conversion code after X_stressed line
# We need to insert after x_stressed_line_idx and before the next non-empty line
dtype_conversion_stressed = [
    "    \n",
    "    # Convert object columns to numeric for XGBoost\n",
    "    for col in X_stressed.columns:\n",
    "        if X_stressed[col].dtype == \"object\":\n",
    "            X_stressed[col] = pd.Categorical(X_stressed[col]).codes\n"
]

# Insert dtype conversion code after X_baseline line
dtype_conversion_baseline = [
    "    \n",
    "    # Convert object columns to numeric for XGBoost\n",
    "    for col in X_baseline.columns:\n",
    "        if X_baseline[col].dtype == \"object\":\n",
    "            X_baseline[col] = pd.Categorical(X_baseline[col]).codes\n"
]

# Check if conversion already exists
stressed_section = ''.join(cell_lines[x_stressed_line_idx:x_stressed_line_idx+10])
baseline_section = ''.join(cell_lines[x_baseline_line_idx:x_baseline_line_idx+10])

if 'Convert object columns to numeric for XGBoost' in stressed_section:
    print("Dtype conversion for X_stressed already exists, skipping...")
else:
    # Insert after X_stressed line
    cell_lines = cell_lines[:x_stressed_line_idx+1] + dtype_conversion_stressed + cell_lines[x_stressed_line_idx+1:]
    print("Added dtype conversion for X_stressed")
    # Update x_baseline_line_idx since we inserted lines
    x_baseline_line_idx += len(dtype_conversion_stressed)

if 'Convert object columns to numeric for XGBoost' in baseline_section:
    print("Dtype conversion for X_baseline already exists, skipping...")
else:
    # Insert after X_baseline line
    cell_lines = cell_lines[:x_baseline_line_idx+1] + dtype_conversion_baseline + cell_lines[x_baseline_line_idx+1:]
    print("Added dtype conversion for X_baseline")

# Update the cell
nb['cells'][stress_test_cell_idx]['source'] = cell_lines

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Fixed run_stress_test function in cell {stress_test_cell_idx}")
print("\nNow:")
print("  1. Close and re-open the notebook")
print("  2. Restart kernel")
print("  3. Re-run all cells from Cell 1")
