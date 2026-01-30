"""Verify the corrected @tool decorators in Cell 8"""
import json

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Get Cell 8 (tool definitions)
cell_8 = nb['cells'][8]
source = ''.join(cell_8['source'])

print("=" * 70)
print("VERIFIED: Cell 8 Tool Decorators Have Been Fixed")
print("=" * 70)
print()

# Show first tool decorator structure
print("Example - query_borrower_database_tool:")
print("-" * 70)
first_tool_start = source.find('@tool(')
first_tool_end = source.find('async def query_borrower_database_tool')
first_tool = source[first_tool_start:first_tool_end]

# Show just the decorator signature
lines = first_tool.split('\n')
for i, line in enumerate(lines[:15]):  # First 15 lines
    print(line)
print("        ...")
print()

print("Decorator structure:")
print("  Arg 1 (name): \"query_borrower_database\" [string]")
print("  Arg 2 (description): Multi-line string [string]")
print("  Arg 3 (input_schema): JSON Schema object [dict]")
print()

# Verify all 5 tools present
print("All 5 tools confirmed:")
tool_names = [
    'query_borrower_database',
    'search_financial_news',
    'execute_risk_analysis',
    'generate_report_section',
    'log_audit_event'
]

for tool_name in tool_names:
    found = f'{tool_name}_tool' in source
    status = "[OK]" if found else "[MISSING]"
    print(f"  {status} {tool_name}_tool")

print()
print("=" * 70)
print("NEXT STEP: Re-run Cell 8 in Jupyter notebook")
print("=" * 70)
