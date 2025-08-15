#!/usr/bin/env python3
"""
IFS Cloud PLSQL File Analyzer

This module provides conservative AST analysis for IFS Cloud PLSQL (.plsql) files,
focusing on understanding business logic structure including:

- Package headers and declarations
- Public and private procedures/functions
- Constants and type declarations
- Override annotations
- Layer declarations
- Business validation patterns
- Error handling structures
- Complete AST generation for PLSQL constructs

Designed with a conservative approach to minimize false positives while providing
comprehensive structural analysis for AI-assisted customization development.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosticSeverity(Enum):
    """Diagnostic severity levels"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    """Represents a diagnostic message with location information"""

    message: str
    line: int
    character: int = 0
    end_line: Optional[int] = None
    end_character: Optional[int] = None
    severity: DiagnosticSeverity = DiagnosticSeverity.INFO
    code: Optional[str] = None
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic to dictionary format"""
        return {
            "message": self.message,
            "line": self.line,
            "character": self.character,
            "end_line": self.end_line or self.line,
            "end_character": self.end_character or 0,
            "severity": self.severity.value,
            "code": self.code,
            "fix_suggestion": self.fix_suggestion,
        }


class NodeType(Enum):
    """PLSQL AST node types"""

    PACKAGE = "package"
    PROCEDURE = "procedure"
    FUNCTION = "function"
    TYPE_DECLARATION = "type_declaration"
    CONSTANT_DECLARATION = "constant_declaration"
    VARIABLE_DECLARATION = "variable_declaration"
    CURSOR_DECLARATION = "cursor_declaration"
    EXCEPTION_DECLARATION = "exception_declaration"
    OVERRIDE_ANNOTATION = "override_annotation"
    IGNORE_UNIT_TEST_ANNOTATION = "ignore_unit_test_annotation"
    LAYER_DECLARATION = "layer_declaration"
    COMMENT_BLOCK = "comment_block"
    ERROR_HANDLING = "error_handling"
    VALIDATION_BLOCK = "validation_block"
    PRAGMA_DECLARATION = "pragma_declaration"


@dataclass
class PLSQLASTNode:
    """AST node for PLSQL files"""

    node_type: NodeType
    name: str
    start_line: int
    end_line: int
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List["PLSQLASTNode"] = field(default_factory=list)

    def add_child(self, child: "PLSQLASTNode"):
        """Add a child node"""
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization"""
        return {
            "node_type": self.node_type.value,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "properties": self.properties,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class PLSQLAnalysisResult:
    """Result of PLSQL file analysis"""

    is_valid: bool
    diagnostics: List[Diagnostic]
    ast: Optional[PLSQLASTNode] = None
    logical_unit: Optional[str] = None
    component: Optional[str] = None
    layer: Optional[str] = None
    public_methods: List[Dict[str, Any]] = field(default_factory=list)
    private_methods: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    types: List[Dict[str, Any]] = field(default_factory=list)
    business_validations: List[Dict[str, Any]] = field(default_factory=list)
    error_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def get_errors(self) -> List[Diagnostic]:
        """Get only error diagnostics"""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR]

    def get_warnings(self) -> List[Diagnostic]:
        """Get only warning diagnostics"""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING]


class ConservativePLSQLAnalyzer:
    """
    Conservative PLSQL analyzer for IFS Cloud business logic files

    Focuses on structural analysis with minimal false positives while providing
    comprehensive understanding of PLSQL package organization and business patterns.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the PLSQL analyzer

        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        self.diagnostics: List[Diagnostic] = []

        # Regex patterns for PLSQL constructs
        self.patterns = {
            # Header patterns
            "logical_unit": re.compile(r"--\s*Logical unit:\s*(\w+)", re.IGNORECASE),
            "component": re.compile(r"--\s*Component:\s*(\w+)", re.IGNORECASE),
            "layer": re.compile(r"^layer\s+(\w+)\s*;", re.IGNORECASE),
            # Method declarations
            "procedure": re.compile(r"^\s*PROCEDURE\s+(\w+)", re.IGNORECASE),
            "function": re.compile(r"^\s*FUNCTION\s+(\w+)", re.IGNORECASE),
            # Annotations
            "override": re.compile(r"^\s*@Override\s*$", re.IGNORECASE),
            "ignore_unit_test": re.compile(
                r"^\s*@IgnoreUnitTest\s+(\w+)", re.IGNORECASE
            ),
            # Declarations
            "constant": re.compile(r"^\s*(\w+)\s+CONSTANT\s+(\w+)", re.IGNORECASE),
            "type_declaration": re.compile(r"^\s*TYPE\s+(\w+)\s+IS", re.IGNORECASE),
            "cursor": re.compile(r"^\s*CURSOR\s+(\w+)", re.IGNORECASE),
            "exception": re.compile(r"^\s*(\w+)\s+EXCEPTION", re.IGNORECASE),
            "pragma": re.compile(r"^\s*PRAGMA\s+(\w+)", re.IGNORECASE),
            # Business patterns
            "error_sys_call": re.compile(r"Error_SYS\.(\w+)\s*\(", re.IGNORECASE),
            "validation_check": re.compile(
                r"IF.*THEN\s*Error_SYS", re.IGNORECASE | re.DOTALL
            ),
            "api_call": re.compile(r"(\w+_API)\.(\w+)", re.IGNORECASE),
            # Sections
            "public_declarations": re.compile(
                r".*PUBLIC DECLARATIONS.*", re.IGNORECASE
            ),
            "private_declarations": re.compile(
                r".*PRIVATE DECLARATIONS.*", re.IGNORECASE
            ),
            "implementation_methods": re.compile(
                r".*LU SPECIFIC IMPLEMENTATION METHODS.*", re.IGNORECASE
            ),
            # Comments and documentation
            "comment_block": re.compile(r"^\s*--.*", re.IGNORECASE),
            "history_entry": re.compile(r"--\s*(\d{6})\s+(\w+)\s+(.+)", re.IGNORECASE),
        }

        # Business validation patterns
        self.business_patterns = {
            "record_validation": re.compile(
                r"Record_General|Record_Not_Exist", re.IGNORECASE
            ),
            "field_validation": re.compile(
                r"Field_General|Field_Not_Exist", re.IGNORECASE
            ),
            "currency_validation": re.compile(
                r"Currency.*Valid|Curr.*Balance", re.IGNORECASE
            ),
            "date_validation": re.compile(
                r"valid_from.*valid_until|FROM.*TO", re.IGNORECASE
            ),
            "code_part_validation": re.compile(
                r"Code_Part.*Valid|Accounting_Code_Part", re.IGNORECASE
            ),
        }

    def analyze(self, content: str) -> PLSQLAnalysisResult:
        """
        Analyze PLSQL file content and return structured results

        Args:
            content: PLSQL file content as string

        Returns:
            PLSQLAnalysisResult with AST and diagnostics
        """
        self.diagnostics = []

        if not content.strip():
            return PLSQLAnalysisResult(
                is_valid=False,
                diagnostics=[
                    Diagnostic(
                        "Empty PLSQL file",
                        1,
                        0,
                        severity=DiagnosticSeverity.ERROR,
                        code="empty_file",
                    )
                ],
                ast=None,
            )

        lines = content.split("\n")

        # Parse header information
        logical_unit = self._extract_logical_unit(lines)
        component = self._extract_component(lines)
        layer = self._extract_layer(lines)

        # Build AST
        ast = self._build_ast(lines)

        # Extract structured information
        public_methods = self._extract_public_methods(lines)
        private_methods = self._extract_private_methods(lines)
        constants = self._extract_constants(lines)
        types = self._extract_types(lines)
        business_validations = self._extract_business_validations(lines)
        error_patterns = self._extract_error_patterns(lines)

        # Perform validations
        self._validate_structure(lines, logical_unit, component, layer)

        is_valid = (
            len([d for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR])
            == 0
        )

        return PLSQLAnalysisResult(
            is_valid=is_valid,
            diagnostics=self.diagnostics.copy(),
            ast=ast,
            logical_unit=logical_unit,
            component=component,
            layer=layer,
            public_methods=public_methods,
            private_methods=private_methods,
            constants=constants,
            types=types,
            business_validations=business_validations,
            error_patterns=error_patterns,
        )

    def _extract_logical_unit(self, lines: List[str]) -> Optional[str]:
        """Extract logical unit name from header comments"""
        for line in lines[:50]:  # Check first 50 lines
            match = self.patterns["logical_unit"].search(line)
            if match:
                return match.group(1)
        return None

    def _extract_component(self, lines: List[str]) -> Optional[str]:
        """Extract component name from header comments"""
        for line in lines[:50]:
            match = self.patterns["component"].search(line)
            if match:
                return match.group(1)
        return None

    def _extract_layer(self, lines: List[str]) -> Optional[str]:
        """Extract layer declaration"""
        for i, line in enumerate(lines):
            match = self.patterns["layer"].match(line.strip())
            if match:
                return match.group(1)
        return None

    def _build_ast(self, lines: List[str]) -> PLSQLASTNode:
        """Build comprehensive AST from PLSQL content"""
        root = PLSQLASTNode(NodeType.PACKAGE, "package_root", 0, len(lines) - 1)

        current_section = None
        current_method = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Check for section headers
            if self.patterns["public_declarations"].search(line):
                current_section = "public"
                i += 1
                continue
            elif self.patterns["private_declarations"].search(line):
                current_section = "private"
                i += 1
                continue
            elif self.patterns["implementation_methods"].search(line):
                current_section = "implementation"
                i += 1
                continue

            # Parse declarations and methods
            node = self._parse_line(line, i, current_section)
            if node:
                # For methods, try to find the end
                if node.node_type in [NodeType.PROCEDURE, NodeType.FUNCTION]:
                    end_line = self._find_method_end(lines, i)
                    node.end_line = end_line
                    current_method = node

                root.add_child(node)

            i += 1

        return root

    def _parse_line(
        self, line: str, line_number: int, section: Optional[str]
    ) -> Optional[PLSQLASTNode]:
        """Parse a single line and create appropriate AST node"""

        # Check for annotations
        if self.patterns["override"].match(line):
            return PLSQLASTNode(
                NodeType.OVERRIDE_ANNOTATION,
                "@Override",
                line_number,
                line_number,
                {"section": section},
            )

        ignore_test_match = self.patterns["ignore_unit_test"].match(line)
        if ignore_test_match:
            return PLSQLASTNode(
                NodeType.IGNORE_UNIT_TEST_ANNOTATION,
                f"@IgnoreUnitTest {ignore_test_match.group(1)}",
                line_number,
                line_number,
                {"test_type": ignore_test_match.group(1), "section": section},
            )

        # Check for layer declaration
        layer_match = self.patterns["layer"].match(line)
        if layer_match:
            return PLSQLASTNode(
                NodeType.LAYER_DECLARATION,
                layer_match.group(1),
                line_number,
                line_number,
                {"layer": layer_match.group(1)},
            )

        # Check for procedures
        proc_match = self.patterns["procedure"].match(line)
        if proc_match:
            return PLSQLASTNode(
                NodeType.PROCEDURE,
                proc_match.group(1),
                line_number,
                line_number,
                {
                    "name": proc_match.group(1),
                    "section": section,
                    "visibility": "public" if section == "public" else "private",
                },
            )

        # Check for functions
        func_match = self.patterns["function"].match(line)
        if func_match:
            return PLSQLASTNode(
                NodeType.FUNCTION,
                func_match.group(1),
                line_number,
                line_number,
                {
                    "name": func_match.group(1),
                    "section": section,
                    "visibility": "public" if section == "public" else "private",
                },
            )

        # Check for constants
        const_match = self.patterns["constant"].match(line)
        if const_match:
            return PLSQLASTNode(
                NodeType.CONSTANT_DECLARATION,
                const_match.group(1),
                line_number,
                line_number,
                {
                    "name": const_match.group(1),
                    "type": const_match.group(2),
                    "section": section,
                },
            )

        # Check for types
        type_match = self.patterns["type_declaration"].match(line)
        if type_match:
            return PLSQLASTNode(
                NodeType.TYPE_DECLARATION,
                type_match.group(1),
                line_number,
                line_number,
                {"name": type_match.group(1), "section": section},
            )

        return None

    def _find_method_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a method (procedure/function)"""
        # Look for END followed by method name or just END;
        method_line = lines[start_line]
        method_match = self.patterns["procedure"].match(
            method_line.strip()
        ) or self.patterns["function"].match(method_line.strip())

        if not method_match:
            return start_line

        method_name = method_match.group(1)

        # Track BEGIN/END blocks
        begin_count = 0
        in_method = False

        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip().upper()

            # Skip comments
            if line.startswith("--"):
                continue

            # Look for IS/AS to indicate start of method body
            if "IS" in line or "AS" in line:
                in_method = True
                continue

            if in_method:
                # Count BEGIN/END blocks
                if line.startswith("BEGIN"):
                    begin_count += 1
                elif line.startswith("END"):
                    if begin_count > 0:
                        begin_count -= 1
                    else:
                        # This should be the end of our method
                        # Check if it's followed by method name
                        if method_name.upper() in line or line.endswith(";"):
                            return i

        # If we can't find the end, estimate based on next method or end of file
        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()
            if (
                self.patterns["procedure"].match(line)
                or self.patterns["function"].match(line)
                or self.patterns["override"].match(line)
            ):
                return i - 1

        return len(lines) - 1

    def _extract_public_methods(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract public procedures and functions"""
        methods = []
        in_public_section = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if self.patterns["public_declarations"].search(stripped):
                in_public_section = True
                continue
            elif self.patterns["private_declarations"].search(stripped):
                in_public_section = False
                continue
            elif self.patterns["implementation_methods"].search(stripped):
                in_public_section = False
                continue

            if in_public_section:
                proc_match = self.patterns["procedure"].match(stripped)
                func_match = self.patterns["function"].match(stripped)

                if proc_match:
                    methods.append(
                        {
                            "type": "procedure",
                            "name": proc_match.group(1),
                            "line": i + 1,
                            "visibility": "public",
                        }
                    )
                elif func_match:
                    methods.append(
                        {
                            "type": "function",
                            "name": func_match.group(1),
                            "line": i + 1,
                            "visibility": "public",
                        }
                    )

        return methods

    def _extract_private_methods(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract private procedures and functions"""
        methods = []
        in_implementation = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if self.patterns["implementation_methods"].search(stripped):
                in_implementation = True
                continue

            if in_implementation:
                proc_match = self.patterns["procedure"].match(stripped)
                func_match = self.patterns["function"].match(stripped)

                if proc_match:
                    methods.append(
                        {
                            "type": "procedure",
                            "name": proc_match.group(1),
                            "line": i + 1,
                            "visibility": "private",
                        }
                    )
                elif func_match:
                    methods.append(
                        {
                            "type": "function",
                            "name": func_match.group(1),
                            "line": i + 1,
                            "visibility": "private",
                        }
                    )

        return methods

    def _extract_constants(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract constant declarations"""
        constants = []

        for i, line in enumerate(lines):
            match = self.patterns["constant"].match(line.strip())
            if match:
                constants.append(
                    {"name": match.group(1), "type": match.group(2), "line": i + 1}
                )

        return constants

    def _extract_types(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract type declarations"""
        types = []

        for i, line in enumerate(lines):
            match = self.patterns["type_declaration"].match(line.strip())
            if match:
                types.append({"name": match.group(1), "line": i + 1})

        return types

    def _extract_business_validations(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract business validation patterns"""
        validations = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Look for Error_SYS calls
            error_match = self.patterns["error_sys_call"].search(stripped)
            if error_match:
                validation_type = "unknown"
                for pattern_name, pattern in self.business_patterns.items():
                    if pattern.search(stripped):
                        validation_type = pattern_name
                        break

                validations.append(
                    {
                        "type": validation_type,
                        "error_method": error_match.group(1),
                        "line": i + 1,
                        "pattern": (
                            stripped[:100] + "..." if len(stripped) > 100 else stripped
                        ),
                    }
                )

        return validations

    def _extract_error_patterns(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract error handling patterns"""
        error_patterns = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # API calls that might indicate error handling
            api_match = self.patterns["api_call"].search(stripped)
            if api_match and "Error" in api_match.group(1):
                error_patterns.append(
                    {
                        "api": api_match.group(1),
                        "method": api_match.group(2),
                        "line": i + 1,
                        "pattern": stripped,
                    }
                )

        return error_patterns

    def _validate_structure(
        self,
        lines: List[str],
        logical_unit: Optional[str],
        component: Optional[str],
        layer: Optional[str],
    ):
        """Validate overall PLSQL file structure"""

        # Check for required elements (conservative approach)
        if not logical_unit and self.strict_mode:
            self._add_diagnostic(
                "Logical unit declaration missing in header comments",
                1,
                DiagnosticSeverity.HINT,
                "missing_logical_unit",
                "Add '-- Logical unit: <LogicalUnitName>' in header",
            )

        if not component and self.strict_mode:
            self._add_diagnostic(
                "Component declaration missing in header comments",
                1,
                DiagnosticSeverity.HINT,
                "missing_component",
                "Add '-- Component: <COMPONENT>' in header",
            )

        if not layer:
            self._add_diagnostic(
                "Layer declaration missing",
                1,
                DiagnosticSeverity.WARNING,
                "missing_layer",
                "Add 'layer <LayerName>;' declaration",
            )

        # Check for common sections
        has_public_declarations = any(
            self.patterns["public_declarations"].search(line) for line in lines
        )
        has_private_declarations = any(
            self.patterns["private_declarations"].search(line) for line in lines
        )

        if not has_public_declarations and len(lines) > 100:
            self._add_diagnostic(
                "PUBLIC DECLARATIONS section not found",
                1,
                DiagnosticSeverity.HINT,
                "missing_public_section",
                "Consider adding '-- PUBLIC DECLARATIONS --' section",
            )

        if not has_private_declarations and len(lines) > 100:
            self._add_diagnostic(
                "PRIVATE DECLARATIONS section not found",
                1,
                DiagnosticSeverity.HINT,
                "missing_private_section",
                "Consider adding '-- PRIVATE DECLARATIONS --' section",
            )

    def _add_diagnostic(
        self,
        message: str,
        line: int,
        severity: DiagnosticSeverity,
        code: Optional[str] = None,
        fix_suggestion: Optional[str] = None,
    ):
        """Add a diagnostic message"""
        self.diagnostics.append(
            Diagnostic(
                message=message,
                line=line,
                severity=severity,
                code=code,
                fix_suggestion=fix_suggestion,
            )
        )


def analyze_plsql_file(content: str, strict_mode: bool = True) -> PLSQLAnalysisResult:
    """
    Convenience function to analyze PLSQL file content

    Args:
        content: PLSQL file content
        strict_mode: Whether to apply strict validation rules

    Returns:
        PLSQLAnalysisResult with analysis results
    """
    analyzer = ConservativePLSQLAnalyzer(strict_mode=strict_mode)
    return analyzer.analyze(content)


if __name__ == "__main__":
    # Example usage
    sample_plsql = """
    -----------------------------------------------------------------------------
    --
    --  Logical unit: CustomerOrder
    --  Component:    ORDER
    --
    -----------------------------------------------------------------------------
    
    layer Core;
    
    -------------------- PUBLIC DECLARATIONS ------------------------------------
    
    TYPE Order_Rec IS RECORD
       (order_no    VARCHAR2(12),
        customer_no VARCHAR2(20));
    
    -------------------- PRIVATE DECLARATIONS -----------------------------------
    
    order_status_error_  CONSTANT VARCHAR2(30) := 'ORDER_STATUS_ERROR';
    
    -------------------- LU SPECIFIC IMPLEMENTATION METHODS ---------------------
    
    @Override
    PROCEDURE Check_Insert___ (
       newrec_ IN OUT customer_order_tab%ROWTYPE )
    IS
    BEGIN
       IF newrec_.order_date > SYSDATE THEN
          Error_SYS.Record_General(lu_name_, 'FUTUREORDER: Order date cannot be in the future');
       END IF;
       super(newrec_);
    END Check_Insert___;
    
    @IgnoreUnitTest TrivialFunction  
    FUNCTION Get_Order_Status (
       order_no_ IN VARCHAR2 ) RETURN VARCHAR2
    IS
    BEGIN
       RETURN Customer_Order_API.Get_Objstate(order_no_);
    END Get_Order_Status;
    """

    result = analyze_plsql_file(sample_plsql)
    print(f"Analysis complete - Valid: {result.is_valid}")
    print(f"Found {len(result.public_methods)} public methods")
    print(f"Found {len(result.private_methods)} private methods")
    print(f"Found {len(result.constants)} constants")
    print(f"Found {len(result.business_validations)} business validations")
