"""
IFS Cloud Projection File AST Analyzer with Syntax Error Recovery

This module provides comprehensive analysis of IFS Cloud projection files,
including both full and partial projections, returning structured AST data.
It supports syntax error recovery and provides detailed feedback on issues.
"""

import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class ProjectionType(Enum):
    FULL = "full"
    PARTIAL = "partial"


class DiagnosticSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    """Represents a syntax error, warning, or other diagnostic message"""

    severity: DiagnosticSeverity
    message: str
    line: int
    column: int = 0
    length: int = 0
    source: str = "projection-analyzer"
    code: Optional[str] = None
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "length": self.length,
            "source": self.source,
            "code": self.code,
            "fix_suggestion": self.fix_suggestion,
        }


class NodeType(Enum):
    PROJECTION = "projection"
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    REFERENCE = "reference"
    ARRAY = "array"
    ACTION = "action"
    FUNCTION = "function"
    COMMAND = "command"
    LUDEPENDENCY = "ludependency"
    ENTITYSET = "entityset"
    SELECTOR = "selector"
    NAVIGATOR = "navigator"
    LIST = "list"
    CARD = "card"
    GROUP = "group"
    FIELD = "field"
    FIELDSET = "fieldset"
    TAB = "tab"
    PAGE = "page"
    ASSISTANT = "assistant"
    CHART = "chart"
    SEARCHCONTEXT = "searchcontext"


@dataclass
class ASTAttribute:
    """Represents an attribute with optional properties"""

    name: str
    data_type: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    annotations: List[str] = field(default_factory=list)


@dataclass
class ASTNode:
    """Base AST node for all projection elements"""

    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["ASTNode"] = field(default_factory=list)
    parent: Optional["ASTNode"] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    raw_content: Optional[str] = None


@dataclass
class ProjectionAST:
    """Root AST for a projection file with diagnostic support"""

    projection_type: ProjectionType
    name: str
    component: Optional[str] = None
    layer: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    base_projection: Optional[str] = None
    includes: List[Dict[str, str]] = field(default_factory=list)
    ludependencies: List[str] = field(default_factory=list)
    entitysets: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[ASTNode] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    structures: List[Dict[str, Any]] = field(default_factory=list)
    enumerations: List[Dict[str, Any]] = field(default_factory=list)
    queries: List[Dict[str, Any]] = field(default_factory=list)
    virtuals: List[Dict[str, Any]] = field(default_factory=list)
    summaries: List[Dict[str, Any]] = field(default_factory=list)
    singletons: List[Dict[str, Any]] = field(default_factory=list)
    client_metadata: List[ASTNode] = field(default_factory=list)
    all_nodes: List[ASTNode] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(
        default_factory=list
    )  # New: syntax errors and warnings
    is_valid: bool = True  # New: overall validity status

    def add_error(
        self,
        message: str,
        line: int,
        column: int = 0,
        code: str = None,
        fix_suggestion: str = None,
    ):
        """Add an error diagnostic"""
        self.diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=message,
                line=line,
                column=column,
                code=code,
                fix_suggestion=fix_suggestion,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        message: str,
        line: int,
        column: int = 0,
        code: str = None,
        fix_suggestion: str = None,
    ):
        """Add a warning diagnostic"""
        self.diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                message=message,
                line=line,
                column=column,
                code=code,
                fix_suggestion=fix_suggestion,
            )
        )

    def add_info(self, message: str, line: int, column: int = 0, code: str = None):
        """Add an info diagnostic"""
        self.diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.INFO,
                message=message,
                line=line,
                column=column,
                code=code,
            )
        )

    def add_hint(
        self,
        message: str,
        line: int,
        column: int = 0,
        code: str = None,
        fix_suggestion: str = None,
    ):
        """Add a hint diagnostic"""
        self.diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.HINT,
                message=message,
                line=line,
                column=column,
                code=code,
                fix_suggestion=fix_suggestion,
            )
        )

    def get_errors(self) -> List[Diagnostic]:
        """Get only error diagnostics"""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR]

    def get_warnings(self) -> List[Diagnostic]:
        """Get only warning diagnostics"""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert AST to dictionary for JSON serialization"""
        return {
            "projection_type": (
                self.projection_type.value if self.projection_type else None
            ),
            "name": self.name,
            "component": self.component,
            "layer": self.layer,
            "description": self.description,
            "category": self.category,
            "base_projection": self.base_projection,
            "includes": self.includes,
            "ludependencies": self.ludependencies,
            "entitysets": self.entitysets,
            "entities": [
                {"name": e.name, "type": e.node_type.value, "attributes": e.attributes}
                for e in self.entities
            ],
            "actions": self.actions,
            "functions": self.functions,
            "structures": self.structures,
            "enumerations": self.enumerations,
            "queries": self.queries,
            "virtuals": self.virtuals,
            "summaries": self.summaries,
            "singletons": self.singletons,
            "imports": self.imports,
            "annotations": self.annotations,
            "node_count": len(self.all_nodes),
            "is_valid": self.is_valid,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings()),
        }


class ProjectionAnalyzer:
    """Analyzer for IFS Cloud projection files with syntax error recovery"""

    def __init__(self, strict_mode: bool = False):
        self.current_line = 0
        self.lines = []
        self.strict_mode = strict_mode  # If True, stops on first error

        # Regex patterns for parsing
        self.patterns = {
            "projection_header": re.compile(
                r"^projection\s+(\w+)\s*(?::\s*(\w+))?\s*;?"
            ),
            "partial_projection": re.compile(r"^fragment\s+projection\s+(\w+)"),
            "entity": re.compile(r"^entity\s+(\w+)"),
            "attribute": re.compile(r"^attribute\s+(\w+)\s+(\w+)"),
            "reference": re.compile(r"^reference\s+(\w+)"),
            "array": re.compile(r"^array\s+(\w+)"),
            "action": re.compile(r"^action\s+(\w+)"),
            "function": re.compile(r"^function\s+(\w+)"),
            "entityset": re.compile(r"^entityset\s+(\w+)\s+for\s+(\w+)"),
            "component": re.compile(r"^component\s+(\w+)"),
            "layer": re.compile(r"^layer\s+(\w+)"),
            "description": re.compile(r'^description\s+"([^"]*)"'),
            "category": re.compile(r"^category\s+(\w+)"),
            "include": re.compile(r"^include\s+fragment\s+(\w+)"),
        }

        # Common syntax errors and their fixes
        self.common_fixes = {
            "missing_semicolon": "Add semicolon (;) at the end of the line",
            "missing_bracket": "Add closing bracket (})",
            "invalid_identifier": "Use valid identifier (alphanumeric and underscore only)",
            "missing_entity": "Entity name is required after entityset declaration",
            "invalid_action_syntax": "Action syntax: action <name> <return_type>? { ... }",
            "invalid_function_syntax": "Function syntax: function <name> <return_type>? { ... }",
        }

    def _safe_parse(self, ast: ProjectionAST, parse_func, *args, **kwargs):
        """Safely execute a parsing function with error recovery"""
        try:
            return parse_func(ast, *args, **kwargs)
        except Exception as e:
            line_num = kwargs.get("line_num", self.current_line + 1)
            ast.add_error(
                f"Parse error: {str(e)}",
                line_num,
                fix_suggestion="Check syntax and formatting",
            )
            if self.strict_mode:
                raise
            return None

    def analyze(self, content: str, file_path: str = "") -> ProjectionAST:
        """Analyze projection file content and return AST with error recovery"""
        self.lines = content.split("\n")
        self.current_line = 0

        # Clean content - remove comments but preserve line structure
        cleaned_lines = []
        for line in self.lines:
            # Remove SQL-style comments
            if line.strip().startswith("--"):
                cleaned_lines.append("")  # Keep line numbering
                continue
            # Remove inline comments but keep the line
            cleaned_line = re.sub(r"//.*$", "", line)
            cleaned_line = re.sub(r"/\*.*?\*/", "", cleaned_line)
            cleaned_lines.append(cleaned_line)

        self.lines = cleaned_lines

        # Determine projection type and parse header
        projection_info = self._parse_projection_header()

        ast = ProjectionAST(
            projection_type=projection_info["type"],
            name=projection_info["name"],
            base_projection=projection_info.get("base"),
        )

        # Only flag projection name as error if it's truly missing from a substantial file
        if not projection_info["name"] or projection_info["name"] == "Unknown":
            # Check if this looks like a real projection file with content
            content_lines = [
                line
                for line in self.lines
                if line.strip() and not line.strip().startswith("--")
            ]
            if len(content_lines) > 3:  # Only flag if there's substantial content
                ast.add_error(
                    "Missing or invalid projection name",
                    1,
                    code="missing_projection_name",
                    fix_suggestion="Add 'projection <ProjectionName>;' at the beginning of the file",
                )
            elif len(content_lines) > 0:  # Just a hint for minimal files
                ast.add_hint(
                    "Projection name might be missing",
                    1,
                    code="missing_projection_name",
                    fix_suggestion="Consider adding 'projection <ProjectionName>;'",
                )

        # Parse IFS Cloud specific header information with error recovery
        self._safe_parse(ast, self._parse_ifs_header)

        # Parse includes with validation
        self._safe_parse(ast, self._parse_includes)

        # Parse main content sections with error recovery
        self._safe_parse(ast, self._parse_entitysets)
        self._safe_parse(ast, self._parse_entities)
        self._safe_parse(ast, self._parse_actions)
        self._safe_parse(ast, self._parse_functions)

        # Post-processing validations
        self._validate_structure(ast)

        return ast

    def _validate_structure(self, ast: ProjectionAST):
        """Validate the overall structure and relationships - conservatively"""
        # Only flag missing components if there's substantial content
        if not ast.component and (ast.entitysets or ast.entities or ast.actions):
            ast.add_hint(
                "Component declaration might be missing",
                1,
                code="missing_component",
                fix_suggestion="Consider adding 'component <COMPONENT_NAME>;' after projection declaration",
            )

        if not ast.layer and (ast.entitysets or ast.entities or ast.actions):
            ast.add_hint(
                "Layer declaration might be missing",
                1,
                code="missing_layer",
                fix_suggestion="Consider adding 'layer <LayerName>;' after component declaration",
            )

        # Only validate entity references conservatively - avoid false positives
        entity_names = {e.name for e in ast.entities}

        # Define common IFS Cloud base entities and patterns to avoid false warnings
        common_entities = {
            "Account",
            "Company",
            "CustomerOrder",
            "Part",
            "Site",
            "Project",
            "Person",
            "InventoryLocation",
            "InventoryPart",
            "PurchaseOrder",
            "WorkOrder",
            "User",
            "Currency",
            "Country",
            "Language",
            "Document",
            "Note",
            "Address",
            "Contact",
        }

        for entityset in ast.entitysets:
            entity_ref = entityset.get("entity", "")
            # Only warn if entity name looks user-defined and is clearly not a base entity
            if (
                entity_ref
                and entity_ref not in entity_names
                and entity_ref not in common_entities
                and not entity_ref.endswith("Entity")  # Likely base entity
                and not entity_ref.endswith("View")  # Likely view entity
                and len(entity_ref) > 15
            ):  # Only warn on obviously custom long names

                ast.add_hint(
                    f"EntitySet '{entityset['name']}' references '{entity_ref}' which might be defined elsewhere",
                    0,
                    code="external_entity_reference",
                    fix_suggestion=f"Verify that entity '{entity_ref}' is available in scope",
                )

        # Only note empty projections if they're completely empty (not just partial)
        if (
            not ast.entitysets
            and not ast.entities
            and not ast.actions
            and not ast.functions
            and not ast.queries
            and len(self.lines) < 10
        ):
            ast.add_info(
                "Projection appears to be empty or minimal", 1, code="empty_projection"
            )

    def _parse_ifs_header(self, ast: ProjectionAST):
        """Parse IFS Cloud specific header (component, layer, description, category)"""
        for i, line in enumerate(self.lines[:20]):  # Check first 20 lines
            line = line.strip().rstrip(";")
            line_num = i + 1

            try:
                if line.startswith("component "):
                    component = line.replace("component ", "").strip()
                    if not component:
                        ast.add_error(
                            "Empty component declaration",
                            line_num,
                            code="empty_component",
                            fix_suggestion="Provide component name: component <COMPONENT_NAME>;",
                        )
                    # Only suggest uppercase for clearly component-like names (avoid false positives)
                    elif (
                        component.isalpha()
                        and len(component) > 2
                        and component != component.upper()
                    ):
                        ast.add_hint(
                            f"Component '{component}' is typically uppercase in IFS Cloud",
                            line_num,
                            code="component_naming",
                            fix_suggestion="Consider using uppercase: "
                            + component.upper(),
                        )
                    ast.component = component
                elif line.startswith("layer "):
                    layer = line.replace("layer ", "").strip()
                    if not layer:
                        ast.add_error(
                            "Empty layer declaration",
                            line_num,
                            code="empty_layer",
                            fix_suggestion="Provide layer name: layer <LayerName>;",
                        )
                    ast.layer = layer
                elif line.startswith("description "):
                    desc = line.replace("description ", "").strip()
                    if desc.startswith('"') and desc.endswith('"'):
                        desc = desc[1:-1]
                    elif '"' in desc and not (
                        desc.startswith('"') or desc.endswith('"')
                    ):
                        # Only warn if quotes are clearly misplaced, not just different style
                        if desc.count('"') == 1:
                            ast.add_hint(
                                "Description might need complete quoting",
                                line_num,
                                code="partial_quotes",
                                fix_suggestion='Consider: description "Your description here";',
                            )
                    ast.description = desc
                elif line.startswith("category "):
                    category = line.replace("category ", "").strip()
                    if not category:
                        ast.add_error(
                            "Empty category declaration",
                            line_num,
                            code="empty_category",
                            fix_suggestion="Provide category name: category <CategoryName>;",
                        )
                    ast.category = category
            except Exception as e:
                ast.add_error(
                    f"Error parsing header at line {line_num}: {str(e)}", line_num
                )

    def _parse_includes(self, ast: ProjectionAST):
        """Parse include fragment statements"""
        for i, line in enumerate(self.lines):
            line = line.strip().rstrip(";")
            if line.startswith("include fragment "):
                fragment_name = line.replace("include fragment ", "").strip()
                ast.includes.append({"name": fragment_name, "type": "fragment"})

    def _parse_entitysets(self, ast: ProjectionAST):
        """Parse entityset definitions with error recovery"""
        for i, line in enumerate(self.lines):
            line = line.strip()
            line_num = i + 1

            if line.startswith("entityset "):
                try:
                    # Parse entityset definition
                    match = re.match(r"entityset\s+(\w+)\s+for\s+(\w+)\s*{?", line)
                    if match:
                        entityset = {"name": match.group(1), "entity": match.group(2)}

                        # Only suggest naming convention if it's clearly wrong (starts with lowercase)
                        if entityset["name"][0].islower():
                            ast.add_hint(
                                f"EntitySet name '{entityset['name']}' typically starts with uppercase",
                                line_num,
                                code="naming_convention",
                                fix_suggestion=f"Consider PascalCase: {entityset['name'][0].upper() + entityset['name'][1:]}",
                            )

                        # Look for additional properties in following lines
                        j = i + 1
                        brace_count = 1 if "{" in line else 0
                        while j < len(self.lines) and brace_count > 0:
                            prop_line = self.lines[j].strip().rstrip(";")

                            # Count braces to know when section ends
                            brace_count += prop_line.count("{") - prop_line.count("}")

                            if prop_line.startswith("context "):
                                context = prop_line.replace("context ", "").strip()
                                if not context:
                                    ast.add_error(
                                        "Empty context declaration",
                                        j + 1,
                                        code="empty_context",
                                        fix_suggestion="Provide context: context Company(Company);",
                                    )
                                entityset["context"] = context
                            elif prop_line.startswith("where = "):
                                where_clause = prop_line.replace("where = ", "").strip()
                                if where_clause.startswith(
                                    '"'
                                ) and where_clause.endswith('"'):
                                    where_clause = where_clause[1:-1]
                                elif where_clause == "":
                                    # Only flag completely empty where clauses as errors
                                    ast.add_error(
                                        "Empty where clause",
                                        j + 1,
                                        code="empty_where",
                                        fix_suggestion='Provide condition: where = "your_condition";',
                                    )
                                # Accept any non-empty where clause without validation
                                if where_clause:
                                    entityset["where"] = where_clause

                            j += 1
                            if j >= len(self.lines):
                                ast.add_error(
                                    f"Unclosed entityset block for '{entityset['name']}'",
                                    line_num,
                                    code="unclosed_block",
                                    fix_suggestion="Add closing brace }",
                                )
                                break

                        ast.entitysets.append(entityset)
                    else:
                        # Failed to parse - try to give helpful error
                        if "for" not in line:
                            ast.add_error(
                                "EntitySet syntax error: missing 'for' keyword",
                                line_num,
                                code="missing_for_keyword",
                                fix_suggestion="Use: entityset <SetName> for <EntityName> { ... }",
                            )
                        else:
                            ast.add_error(
                                f"EntitySet syntax error in line: {line}",
                                line_num,
                                code="entityset_syntax_error",
                                fix_suggestion="Check entityset syntax: entityset <SetName> for <EntityName> { ... }",
                            )
                except Exception as e:
                    ast.add_error(
                        f"Error parsing entityset at line {line_num}: {str(e)}",
                        line_num,
                    )

    def _parse_entities(self, ast: ProjectionAST):
        """Parse entity overrides and definitions"""
        i = 0
        while i < len(self.lines):
            line = self.lines[i].strip()

            if line.startswith("@Override") or line.startswith("entity "):
                # Found an entity definition
                if line.startswith("@Override"):
                    i += 1
                    if i < len(self.lines):
                        line = self.lines[i].strip()

                match = re.match(r"entity\s+(\w+)\s*{?", line)
                if match:
                    entity_name = match.group(1)
                    entity_node = ASTNode(
                        node_type=NodeType.ENTITY,
                        name=entity_name,
                        attributes={"entity_attributes": [], "references": []},
                    )

                    # Parse entity content
                    i += 1
                    while i < len(self.lines):
                        entity_line = self.lines[i].strip()

                        if entity_line.startswith("}"):
                            break
                        elif entity_line.startswith("attribute "):
                            # Parse attribute
                            attr_match = re.match(
                                r"attribute\s+(\w+)\s+(\w+)", entity_line
                            )
                            if attr_match:
                                attr = {
                                    "name": attr_match.group(1),
                                    "type": attr_match.group(2),
                                    "properties": {},
                                }
                                entity_node.attributes["entity_attributes"].append(attr)
                        elif entity_line.startswith("reference "):
                            # Parse reference
                            ref_match = re.match(
                                r"reference\s+(\w+)\s*\([^)]+\)\s+to\s+(\w+)",
                                entity_line,
                            )
                            if ref_match:
                                ref = {
                                    "name": ref_match.group(1),
                                    "target": ref_match.group(2),
                                }
                                entity_node.attributes["references"].append(ref)

                        i += 1

                    ast.entities.append(entity_node)

            i += 1

    def _parse_actions(self, ast: ProjectionAST):
        """Parse action definitions"""
        for i, line in enumerate(self.lines):
            line = line.strip()
            if line.startswith("action "):
                # Parse action
                match = re.match(r"action\s+(\w+)\s+(\w+)?\s*{?", line)
                if match:
                    action = {
                        "name": match.group(1),
                        "return_type": match.group(2) if match.group(2) else "void",
                        "parameters": [],
                    }

                    # Look for parameters in following lines
                    j = i + 1
                    while j < len(self.lines) and not self.lines[j].strip().startswith(
                        "}"
                    ):
                        param_line = self.lines[j].strip().rstrip(";")

                        if param_line.startswith("parameter "):
                            param_match = re.match(
                                r"parameter\s+(\w+)\s+(\w+)", param_line
                            )
                            if param_match:
                                action["parameters"].append(
                                    {
                                        "name": param_match.group(1),
                                        "type": param_match.group(2),
                                    }
                                )

                        j += 1
                        if j < len(self.lines) and "}" in self.lines[j]:
                            break

                    ast.actions.append(action)

    def _parse_functions(self, ast: ProjectionAST):
        """Parse function definitions"""
        for i, line in enumerate(self.lines):
            line = line.strip()
            if line.startswith("function "):
                # Parse function
                match = re.match(r"function\s+(\w+)\s+(\w+)?\s*{?", line)
                if match:
                    function = {
                        "name": match.group(1),
                        "return_type": match.group(2) if match.group(2) else "void",
                        "parameters": [],
                    }

                    # Look for parameters in following lines
                    j = i + 1
                    while j < len(self.lines) and not self.lines[j].strip().startswith(
                        "}"
                    ):
                        param_line = self.lines[j].strip().rstrip(";")

                        if param_line.startswith("parameter "):
                            param_match = re.match(
                                r"parameter\s+(\w+)\s+(\w+)", param_line
                            )
                            if param_match:
                                function["parameters"].append(
                                    {
                                        "name": param_match.group(1),
                                        "type": param_match.group(2),
                                    }
                                )

                        j += 1
                        if j < len(self.lines) and "}" in self.lines[j]:
                            break

                    ast.functions.append(function)

    def _parse_projection_header(self) -> Dict[str, Any]:
        """Parse projection header to determine type and name"""
        for i, line in enumerate(self.lines[:10]):  # Check first 10 lines
            line = line.strip()

            # Check for partial projection
            partial_match = self.patterns["partial_projection"].match(line)
            if partial_match:
                return {"type": ProjectionType.PARTIAL, "name": partial_match.group(1)}

            # Check for full projection
            proj_match = self.patterns["projection_header"].match(line)
            if proj_match:
                return {
                    "type": ProjectionType.FULL,
                    "name": proj_match.group(1),
                    "base": proj_match.group(2) if proj_match.group(2) else None,
                }

        # Default if no header found
        return {"type": ProjectionType.FULL, "name": "Unknown"}
