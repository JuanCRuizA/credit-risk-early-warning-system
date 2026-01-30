"""Fix @tool decorator signatures in notebook Cell 8 (TOOL IMPLEMENTATION FUNCTIONS)"""
import json
import re

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with "@tool(" (should be around index 7)
tool_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '@tool(' in source and 'query_borrower_database' in source:
            tool_cell_idx = idx
            print(f"Found tool definitions cell at index {idx}")
            break

if tool_cell_idx is None:
    print("ERROR: Could not find tool definitions cell")
    exit(1)

# Get the cell source
cell_source = ''.join(nb['cells'][tool_cell_idx]['source'])

# Fix 1: query_borrower_database_tool
cell_source = re.sub(
    r'@tool\(\s*name="query_borrower_database",\s*description=\(',
    '@tool(\n        "query_borrower_database",\n        (',
    cell_source,
    flags=re.DOTALL
)

# Replace input_schema={...} with just the schema dict as 3rd positional arg
# Pattern: Find the closing paren of description, then match input_schema={...}
cell_source = re.sub(
    r'\),\s*input_schema=(\{[^}]+\{[^}]+\}[^}]+\})\s*\)',
    r'),\n        \1\n    )',
    cell_source,
    flags=re.DOTALL
)

# Fix 2-5: Do the same for other tools
# search_financial_news_tool
cell_source = re.sub(
    r'@tool\(\s*name="search_financial_news",\s*description=',
    '@tool(\n        "search_financial_news",\n        ',
    cell_source
)

# execute_risk_analysis_tool
cell_source = re.sub(
    r'@tool\(\s*name="execute_risk_analysis",\s*description=',
    '@tool(\n        "execute_risk_analysis",\n        ',
    cell_source
)

# generate_report_section_tool
cell_source = re.sub(
    r'@tool\(\s*name="generate_report_section",\s*description=',
    '@tool(\n        "generate_report_section",\n        ',
    cell_source
)

# log_audit_event_tool
cell_source = re.sub(
    r'@tool\(\s*name="log_audit_event",\s*description=',
    '@tool(\n        "log_audit_event",\n        ',
    cell_source
)

# Save the fixed cell
nb['cells'][tool_cell_idx]['source'] = cell_source.split('\n')

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Fixed @tool decorators in cell {tool_cell_idx}")
print("Please re-run the cell in Jupyter")
