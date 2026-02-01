"""Fix @tool decorator signatures - convert from kwargs to positional args"""
import json

# Load notebook
with open('notebooks/05_portfolio_surveillance.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find tool definitions cell
tool_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '@tool(' in source and 'query_borrower_database_tool' in source:
            tool_cell_idx = idx
            print(f"Found tool definitions cell at index {idx}")
            break

if tool_cell_idx is None:
    print("ERROR: Could not find tool definitions cell")
    exit(1)

# The fix: Replace the @tool(...) decorator patterns
# Old pattern: @tool(name="...", description=..., input_schema=...)
# New pattern: @tool("name", "description", {...})

cell_lines = nb['cells'][tool_cell_idx]['source']
cell_source = ''.join(cell_lines)

# Manual replacements for each tool
replacements = [
    # Tool 1: query_borrower_database
    (
        '''@tool(
        name="query_borrower_database",
        description=(''',
        '''@tool(
        "query_borrower_database",
        ('''
    ),
    # Tool 2: search_financial_news
    (
        '''@tool(
        name="search_financial_news",
        description=(''',
        '''@tool(
        "search_financial_news",
        ('''
    ),
    # Tool 3: execute_risk_analysis
    (
        '''@tool(
        name="execute_risk_analysis",
        description=(''',
        '''@tool(
        "execute_risk_analysis",
        ('''
    ),
    # Tool 4: generate_report_section
    (
        '''@tool(
        name="generate_report_section",
        description=(''',
        '''@tool(
        "generate_report_section",
        ('''
    ),
    # Tool 5: log_audit_event
    (
        '''@tool(
        name="log_audit_event",
        description=(''',
        '''@tool(
        "log_audit_event",
        ('''
    ),
]

for old, new in replacements:
    cell_source = cell_source.replace(old, new)

# Now fix the input_schema kwarg -> positional arg pattern
# Replace: ),\n        input_schema=
# With: ),\n
cell_source = cell_source.replace(
    '),\n        input_schema=',
    '),\n        '
)

# Split back into lines
nb['cells'][tool_cell_idx]['source'] = cell_source.split('\n')

# Write back
with open('notebooks/05_portfolio_surveillance.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Fixed all 5 @tool decorators in cell {tool_cell_idx}")
print("\nChanges made:")
print("  - Converted name='...' to positional arg 1")
print("  - Converted description=... to positional arg 2")
print("  - Converted input_schema={...} to positional arg 3")
print("\nPlease re-run the cell in Jupyter notebook.")
