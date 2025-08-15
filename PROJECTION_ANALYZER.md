# IFS Cloud Projection Analyzer

The IFS Cloud Projection Analyzer provides comprehensive AST (Abstract Syntax Tree) analysis of IFS Cloud projection files, making it incredibly helpful for Copilot and other tools to understand projection structure.

## Features

- ✅ **Real File Compatibility** - Tested with actual IFS Cloud projection files
- ✅ **Complete AST** - Parses all projection elements (header, entitysets, entities, actions, functions, etc.)
- ✅ **JSON Serializable** - Easy integration with APIs and tools
- ✅ **Both Full & Partial** - Handles both complete and partial projection files
- ✅ **Rich Metadata** - Extracts component, layer, description, category, includes, etc.

## Usage

### Basic Analysis

```python
from ifs_cloud_mcp_server.projection_analyzer import ProjectionAnalyzer

# Initialize analyzer
analyzer = ProjectionAnalyzer()

# Read projection file
with open('AccountsHandling.projection', 'r') as f:
    content = f.read()

# Analyze and get AST
ast = analyzer.analyze(content)

# Access parsed data
print(f"Projection: {ast.name}")
print(f"Component: {ast.component}")
print(f"Layer: {ast.layer}")
print(f"Description: {ast.description}")
print(f"Entity Sets: {len(ast.entitysets)}")
print(f"Entities: {len(ast.entities)}")
print(f"Actions: {len(ast.actions)}")
print(f"Functions: {len(ast.functions)}")
```

### Using with File Path

```python
from ifs_cloud_mcp_server.projection_analyzer import analyze_projection_file

# Direct file analysis
ast = analyze_projection_file('path/to/projection.projection')
```

### JSON Export

```python
# Convert to dictionary for JSON serialization
ast_dict = ast.to_dict()

import json
json_data = json.dumps(ast_dict, indent=2)
```

## AST Structure

The `ProjectionAST` contains:

### Header Information

- `name`: Projection name
- `component`: IFS component (e.g., "ORDER", "ACCRUL")
- `layer`: Layer designation (e.g., "Core", "Custom")
- `description`: Human-readable description
- `category`: Category (e.g., "Users")
- `projection_type`: FULL or PARTIAL

### Content Elements

- `includes`: List of included fragments
- `entitysets`: Entity set definitions with context and where clauses
- `entities`: Entity overrides with attributes and references
- `actions`: Action definitions with parameters
- `functions`: Function definitions with parameters
- `structures`: Structure definitions
- `enumerations`: Enumeration definitions
- `queries`: Query definitions
- `virtuals`: Virtual definitions
- `summaries`: Summary definitions
- `singletons`: Singleton definitions

### Example AST Output

```json
{
  "projection_type": "full",
  "name": "AccountsHandling",
  "component": "ACCRUL",
  "layer": "Core",
  "description": "Accounts Overview",
  "category": "Users",
  "includes": [
    { "name": "AccountsConsolidationSelector", "type": "fragment" },
    { "name": "AccountCommonHandling", "type": "fragment" }
  ],
  "entitysets": [
    {
      "name": "AccountSet",
      "entity": "Account",
      "context": "Company(Company)"
    }
  ],
  "entities": [
    {
      "name": "Account",
      "type": "entity",
      "attributes": {
        "entity_attributes": [
          { "name": "Account", "type": "Text", "properties": {} }
        ],
        "references": [{ "name": "CompanyRef", "target": "CompanyFinance" }]
      }
    }
  ],
  "actions": [
    {
      "name": "ValidateGetSelectedCompany",
      "return_type": "Text",
      "parameters": [{ "name": "VarListText", "type": "List<Text>" }]
    }
  ],
  "functions": [
    {
      "name": "GetSelectedCompany",
      "return_type": "Text",
      "parameters": [{ "name": "FullSelection", "type": "Text" }]
    }
  ]
}
```

## Copilot Integration Examples

### 1. Code Completion Helper

```python
def get_projection_context(file_path: str) -> dict:
    """Get projection context for Copilot suggestions"""
    ast = analyze_projection_file(file_path)

    return {
        'available_entities': [e.name for e in ast.entities],
        'available_actions': [a['name'] for a in ast.actions],
        'available_functions': [f['name'] for f in ast.functions],
        'component': ast.component,
        'layer': ast.layer,
        'entity_attributes': {
            e.name: e.attributes.get('entity_attributes', [])
            for e in ast.entities
        }
    }
```

### 2. Validation Helper

```python
def validate_projection_references(ast: ProjectionAST) -> list:
    """Validate entity references in projection"""
    issues = []
    entity_names = {e.name for e in ast.entities}

    for entityset in ast.entitysets:
        if entityset['entity'] not in entity_names:
            issues.append(f"EntitySet '{entityset['name']}' references unknown entity '{entityset['entity']}'")

    return issues
```

### 3. Documentation Generator

```python
def generate_projection_docs(ast: ProjectionAST) -> str:
    """Generate documentation from projection AST"""
    docs = f"""
# {ast.name}

**Component:** {ast.component}
**Layer:** {ast.layer}
**Description:** {ast.description}

## Entity Sets
{chr(10).join(f"- **{es['name']}**: {es['entity']}" for es in ast.entitysets)}

## Actions
{chr(10).join(f"- **{a['name']}**: {a['return_type']}" for a in ast.actions)}

## Functions
{chr(10).join(f"- **{f['name']}**: {f['return_type']}" for f in ast.functions)}
"""
    return docs
```

## Testing

The analyzer has been tested with real IFS Cloud projection files from the ACCRUL component:

- ✅ AccountsHandling.projection (2 entity sets, 1 entity, 1 action, 2 functions)
- ✅ AccountGroupsHandling.projection (2 entity sets, 1 entity)
- ✅ AccountingPeriodsHandling.projection (2 entity sets, 4 entities)
- ✅ And many more...

This ensures the analyzer works with real-world IFS Cloud projections, not just synthetic examples.

## Error Handling & Syntax Recovery

The analyzer provides comprehensive error recovery with detailed diagnostics and fix suggestions:

### Basic Error Recovery

```python
# Analyze projection with syntax errors
from ifs_cloud_mcp_server.projection_analyzer import ProjectionAnalyzer, DiagnosticSeverity

analyzer = ProjectionAnalyzer(strict_mode=False)  # Enable error recovery

broken_content = '''
projection;  // Missing name
component order;  // Wrong case
layer Core;

entityset TestSet for {  // Missing entity name
    context Company(Company);
}
'''

ast = analyzer.analyze(broken_content)
print(f"Valid: {ast.is_valid}")           # False
print(f"Errors: {len(ast.get_errors())}")  # 2
print(f"Warnings: {len(ast.get_warnings())}")  # 1

# Get detailed diagnostics
for diagnostic in ast.diagnostics:
    severity = diagnostic.severity.name
    print(f"{severity}: Line {diagnostic.line}: {diagnostic.message}")
    if diagnostic.fix_suggestion:
        print(f"  Fix: {diagnostic.fix_suggestion}")
```

### Diagnostic Types

The analyzer provides four severity levels:

- **ERROR**: Critical syntax issues that prevent proper parsing
- **WARNING**: Potential issues or style violations
- **INFO**: General information about the projection
- **HINT**: Suggestions for improvements

### Error Recovery Features

1. **Missing Components**: Detects missing projection name, component, layer
2. **Syntax Errors**: Identifies malformed entitysets, entities, actions
3. **Naming Conventions**: Warns about uppercase/lowercase issues
4. **Incomplete Blocks**: Detects unclosed braces, missing keywords
5. **Reference Validation**: Checks for undefined entity references
6. **Quote Handling**: Identifies partial or missing quotes

### Strict vs Non-Strict Mode

```python
# Strict mode: Fails on any error
strict_analyzer = ProjectionAnalyzer(strict_mode=True)

# Non-strict mode: Recovers from errors and continues parsing
lenient_analyzer = ProjectionAnalyzer(strict_mode=False)
```

### JSON Output with Diagnostics

The AST includes diagnostic information in JSON export:

```python
ast_dict = ast.to_dict()
print(f"Diagnostics: {len(ast_dict['diagnostics'])}")

# Each diagnostic includes:
# - severity: ERROR, WARNING, INFO, HINT
# - line: Line number where issue occurs
# - message: Human-readable description
# - code: Machine-readable error code
# - fix_suggestion: Suggested fix (optional)
```

### Error Recovery Testing

The analyzer has been tested with various syntax error scenarios:

- ✅ Missing projection names and components
- ✅ Malformed entityset declarations
- ✅ Unclosed blocks and missing keywords
- ✅ Invalid naming conventions
- ✅ Partial quotes and incomplete descriptions
- ✅ Mixed error scenarios with detailed diagnostics
