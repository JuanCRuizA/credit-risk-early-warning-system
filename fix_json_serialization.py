"""Fix json.dumps() calls to handle NumPy types in log_audit_event function"""
import json

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with log_audit_event function (Cell 4)
log_audit_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def log_audit_event(' in source and 'audit_logger.info(' in source:
            log_audit_cell_idx = idx
            print(f"Found log_audit_event cell at index {idx}")
            break

if log_audit_cell_idx is None:
    print("ERROR: Could not find log_audit_event cell")
    exit(1)

# Get cell source
cell_source = ''.join(nb['cells'][log_audit_cell_idx]['source'])

# Fix 1: Add default=str to json.dumps in audit_logger.info call
fix1_old = 'f"Data: {json.dumps(data_summary) if data_summary else \'N/A\'} | "'
fix1_new = 'f"Data: {json.dumps(data_summary, default=str) if data_summary else \'N/A\'} | "'

cell_source = cell_source.replace(fix1_old, fix1_new)

# Fix 2: Add default=str to json.dumps in database insert
fix2_old = 'json.dumps(data_summary) if data_summary else None,'
fix2_new = 'json.dumps(data_summary, default=str) if data_summary else None,'

cell_source = cell_source.replace(fix2_old, fix2_new)

# Update cell source
nb['cells'][log_audit_cell_idx]['source'] = cell_source.split('\n')

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Fixed json.dumps() calls in cell {log_audit_cell_idx}")
print("\nChanges made:")
print("  - Added default=str to json.dumps() in audit_logger.info()")
print("  - Added default=str to json.dumps() in database insert")
print("\nThis will convert NumPy float32/int64 types to strings for JSON serialization")
print("\nPlease restart kernel and re-run from Cell 1")
