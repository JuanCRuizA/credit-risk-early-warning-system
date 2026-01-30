"""Comprehensive diagnostic and fix for notebook tool decorator issues"""
import json

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("DIAGNOSTIC REPORT")
print("=" * 70)

# Check Cell 1 imports
cell_1 = ''.join(nb['cells'][1]['source'])
print("\n1. Cell 1 Import Block:")
print("-" * 70)
import_start = cell_1.find('try:')
import_end = cell_1.find('except ImportError:') + 100
print(cell_1[import_start:import_end])

# Check Cell 7 - Old tool schemas (should be removed/commented)
cell_7 = ''.join(nb['cells'][7]['source'])
print("\n2. Cell 7 - OLD TOOL SCHEMAS (Problem!):")
print("-" * 70)
print(f"Cell 7 defines JSON schemas for the OLD manual SDK approach")
print(f"These are no longer needed with @tool decorators")
print(f"Length: {len(cell_7)} characters")

# Check if Cell 8 has the correct decorator syntax
cell_8 = nb['cells'][8]['source']
line_384 = cell_8[384] if len(cell_8) > 384 else ""
print("\n3. Cell 8 Line 384:")
print("-" * 70)
print(f"Content: {line_384.strip()}")
if 'name=' in line_384:
    print("[ERROR] Still has keyword args!")
else:
    print("[OK] Has positional args")

print("\n" + "=" * 70)
print("ROOT CAUSE IDENTIFIED")
print("=" * 70)
print("""
The notebook has TWO competing patterns:
1. Cell 7: Old JSON schemas (SQL_QUERY_TOOL, etc.) for manual SDK
2. Cell 8: New @tool decorators for claude-agent-sdk

Solution: Cell 7 must be COMMENTED OUT or REMOVED entirely.
The @tool decorators in Cell 8 are self-contained and don't need Cell 7.
""")

# Fix: Comment out Cell 7
print("\nAPPLYING FIX: Commenting out Cell 7...")
cell_7_lines = nb['cells'][7]['source']
commented_lines = []
for line in cell_7_lines:
    if line.strip() and not line.startswith('#'):
        commented_lines.append('# ' + line)
    else:
        commented_lines.append(line)

# Update notebook
nb['cells'][7]['source'] = commented_lines

# Add a note at the top of Cell 7
note = [
    "# " + "=" * 68 + "\n",
    "# NOTE: This cell has been COMMENTED OUT\n",
    "# The JSON schemas below were for the OLD manual Anthropic SDK approach.\n",
    "# Cell 8 now uses @tool decorators which are self-contained.\n",
    "# " + "=" * 68 + "\n",
    "\n"
]
nb['cells'][7]['source'] = note + nb['cells'][7]['source']

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Cell 7 has been commented out")
print("\nNEXT STEPS:")
print("1. In Jupyter: Restart Kernel (Kernel → Restart)")
print("2. Run cells 1-8 sequentially")
print("3. Cell 8 should now work without errors")
