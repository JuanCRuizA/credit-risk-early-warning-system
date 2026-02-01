"""Fix run_stress_test function to convert object dtypes before XGBoost predictions"""
import json

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with run_stress_test function (Cell with "3.9 WHAT-IF STRESS TESTING")
stress_test_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def run_stress_test(' in source and 'baseline_pd = model.predict_proba(X_baseline)' in source:
            stress_test_cell_idx = idx
            print(f"Found stress test cell at index {idx}")
            break

if stress_test_cell_idx is None:
    print("ERROR: Could not find run_stress_test cell")
    exit(1)

# Get cell source
cell_source = ''.join(nb['cells'][stress_test_cell_idx]['source'])

# Fix 1: Add dtype conversion for X_stressed (before line 2142)
# Insert after: X_stressed = X_stressed[feature_names]
# Insert before: # Get baseline and stressed PD scores

fix1_old = """    X_stressed = X_stressed[feature_names]

    # Get baseline and stressed PD scores"""

fix1_new = """    X_stressed = X_stressed[feature_names]

    # Convert object columns to numeric for XGBoost
    for col in X_stressed.columns:
        if X_stressed[col].dtype == "object":
            X_stressed[col] = pd.Categorical(X_stressed[col]).codes

    # Get baseline and stressed PD scores"""

cell_source = cell_source.replace(fix1_old, fix1_new)

# Fix 2: Add dtype conversion for X_baseline (after X_baseline column order)
# Insert after: X_baseline = X_baseline[feature_names]
# Insert before: baseline_pd = model.predict_proba(X_baseline)

fix2_old = """    X_baseline = X_baseline[feature_names]

    baseline_pd = model.predict_proba(X_baseline)[:, 1]"""

fix2_new = """    X_baseline = X_baseline[feature_names]

    # Convert object columns to numeric for XGBoost
    for col in X_baseline.columns:
        if X_baseline[col].dtype == "object":
            X_baseline[col] = pd.Categorical(X_baseline[col]).codes

    baseline_pd = model.predict_proba(X_baseline)[:, 1]"""

cell_source = cell_source.replace(fix2_old, fix2_new)

# Update cell source
nb['cells'][stress_test_cell_idx]['source'] = cell_source.split('\n')

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Fixed run_stress_test function in cell {stress_test_cell_idx}")
print("\nChanges made:")
print("  - Added dtype conversion for X_stressed before predictions")
print("  - Added dtype conversion for X_baseline before predictions")
print("\nPlease restart kernel and re-run Cell 19 (3.9 WHAT-IF STRESS TESTING)")
