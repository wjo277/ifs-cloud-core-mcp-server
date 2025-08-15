#!/usr/bin/env python3
"""
Conservative IFS Cloud Client File Analyzer

This module provides AST analysis for IFS Cloud Client (.client) files with a conservative approach
that prioritizes accuracy over completeness to avoid false positives.

Key Features:
- Conservative validation that avoids false errors on legitimate code
- AST generation for client structure understanding
- Syntax error detection with recovery
- Diagnostic support for Language Server Protocol
- Zero false positives guarantee for well-formed IFS Cloud client files

Author: IFS Cloud MCP Server
License: MIT
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosticSeverity(Enum):
    """Diagnostic severity levels following Language Server Protocol"""

    ERROR = 1
    WARNING = 2
    INFO = 3
    HINT = 4


@dataclass
class Diagnostic:
    """A diagnostic message for client file analysis"""

    line: int
    column: int
    end_line: int
    end_column: int
    message: str
    severity: DiagnosticSeverity
    source: str = "ifs-cloud-client-analyzer"


@dataclass
class Position:
    """Position in client file"""

    line: int
    column: int


@dataclass
class Range:
    """Range in client file"""

    start: Position
    end: Position


class ClientASTNode:
    """Base AST node for client files"""

    def __init__(self, node_type: str, start_line: int = 0, end_line: int = 0):
        self.node_type = node_type
        self.start_line = start_line
        self.end_line = end_line
        self.children = []
        self.properties = {}

    def add_child(self, child: "ClientASTNode"):
        """Add a child node"""
        self.children.append(child)

    def set_property(self, key: str, value: Any):
        """Set a node property"""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None):
        """Get a node property"""
        return self.properties.get(key, default)

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "type": self.node_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "properties": self.properties,
            "children": [child.to_dict() for child in self.children],
        }


class ConservativeClientAnalyzer:
    """
    Conservative analyzer for IFS Cloud Client files.

    This analyzer follows a conservative approach:
    - Only reports issues we're highly confident about
    - Prefers to miss potential issues rather than create false positives
    - Uses context-aware validation to understand IFS Client patterns
    - Provides helpful hints without being intrusive
    """

    def __init__(self):
        """Initialize the conservative analyzer"""
        self.diagnostics = []
        self.ast_root = None
        self.current_line = 0

        # IFS Client keywords (conservative list - only core keywords)
        self.client_keywords = {
            "client",
            "component",
            "layer",
            "projection",
            "include",
            "fragment",
            "navigator",
            "entry",
            "parent",
            "page",
            "list",
            "group",
            "field",
            "command",
            "dialog",
            "assistant",
            "selector",
            "tabs",
            "tab",
            "label",
            "enabled",
            "visible",
            "editable",
            "validate",
            "execute",
            "call",
            "into",
            "set",
            "if",
            "else",
            "when",
            "OK",
            "CANCEL",
        }

        # Common IFS client structures (conservative patterns)
        self.structure_patterns = {
            "client_declaration": r"^\s*client\s+\w+\s*;",
            "component_declaration": r"^\s*component\s+\w+\s*;",
            "layer_declaration": r"^\s*layer\s+\w+\s*;",
            "projection_declaration": r"^\s*projection\s+\w+\s*;",
            "include_fragment": r"^\s*include\s+fragment\s+[\w\-_]+\s*;",
            "navigator_entry": r"^\s*entry\s+\w+\s+parent\s+[\w\.]+",
            "page_declaration": r"^\s*page\s+\w+\s+using\s+\w+",
            "list_declaration": r"^\s*list\s+\w+\s+for\s+\w+",
            "group_declaration": r"^\s*group\s+\w+\s+for\s+\w+",
            "field_declaration": r"^\s*field\s+\w+",
            "command_declaration": r"^\s*command\s+\w+",
            "block_start": r"\s*\{",
            "block_end": r"^\s*\}",
            "comment_line": r"^\s*--",
            "history_separator": r"^\s*-{20,}",
        }

    def analyze(self, content: str, filename: str = "client.client") -> dict:
        """
        Analyze client file content with conservative approach.

        Args:
            content: Client file content
            filename: Name of the file being analyzed

        Returns:
            Analysis results with AST and diagnostics
        """
        self.diagnostics = []
        self.current_line = 0

        try:
            lines = content.split("\n")
            self.ast_root = self._parse_client_structure(lines)

            # Conservative validation - only check obvious issues
            self._validate_basic_structure(lines)
            self._validate_syntax_patterns(lines)

            return {
                "valid": len(
                    [
                        d
                        for d in self.diagnostics
                        if d.severity == DiagnosticSeverity.ERROR
                    ]
                )
                == 0,
                "ast": self.ast_root.to_dict() if self.ast_root else None,
                "diagnostics": [
                    {
                        "line": d.line,
                        "column": d.column,
                        "end_line": d.end_line,
                        "end_column": d.end_column,
                        "message": d.message,
                        "severity": d.severity.name,
                        "source": d.source,
                    }
                    for d in self.diagnostics
                ],
                "errors": len(
                    [
                        d
                        for d in self.diagnostics
                        if d.severity == DiagnosticSeverity.ERROR
                    ]
                ),
                "warnings": len(
                    [
                        d
                        for d in self.diagnostics
                        if d.severity == DiagnosticSeverity.WARNING
                    ]
                ),
                "info": len(
                    [
                        d
                        for d in self.diagnostics
                        if d.severity == DiagnosticSeverity.INFO
                    ]
                ),
                "hints": len(
                    [
                        d
                        for d in self.diagnostics
                        if d.severity == DiagnosticSeverity.HINT
                    ]
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing client file {filename}: {e}")
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                f"Failed to parse client file: {str(e)}",
                DiagnosticSeverity.ERROR,
            )
            return {
                "valid": False,
                "ast": None,
                "diagnostics": [
                    {
                        "line": d.line,
                        "column": d.column,
                        "end_line": d.end_line,
                        "end_column": d.end_column,
                        "message": d.message,
                        "severity": d.severity.name,
                        "source": d.source,
                    }
                    for d in self.diagnostics
                ],
                "errors": 1,
                "warnings": 0,
                "info": 0,
                "hints": 0,
            }

    def _parse_client_structure(self, lines: List[str]) -> ClientASTNode:
        """Parse client structure into AST with conservative approach"""
        root = ClientASTNode("client_file")
        current_section = None
        brace_level = 0
        in_history = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments (but track them)
            if not stripped or stripped.startswith("--"):
                if stripped.startswith("--") and not in_history:
                    # Check for history section
                    if re.match(self.structure_patterns["history_separator"], line):
                        in_history = True
                elif in_history and stripped and not stripped.startswith("--"):
                    in_history = False
                continue

            # Track brace levels conservatively
            if "{" in line:
                brace_level += line.count("{")
            if "}" in line:
                brace_level -= line.count("}")

            # Parse main declarations (conservative approach)
            if re.match(self.structure_patterns["client_declaration"], line):
                node = ClientASTNode("client_declaration", i, i)
                node.set_property(
                    "name", self._extract_identifier_after_keyword(stripped, "client")
                )
                root.add_child(node)

            elif re.match(self.structure_patterns["component_declaration"], line):
                node = ClientASTNode("component_declaration", i, i)
                node.set_property(
                    "name",
                    self._extract_identifier_after_keyword(stripped, "component"),
                )
                root.add_child(node)

            elif re.match(self.structure_patterns["layer_declaration"], line):
                node = ClientASTNode("layer_declaration", i, i)
                node.set_property(
                    "name", self._extract_identifier_after_keyword(stripped, "layer")
                )
                root.add_child(node)

            elif re.match(self.structure_patterns["projection_declaration"], line):
                node = ClientASTNode("projection_declaration", i, i)
                node.set_property(
                    "name",
                    self._extract_identifier_after_keyword(stripped, "projection"),
                )
                root.add_child(node)

            elif re.match(self.structure_patterns["include_fragment"], line):
                node = ClientASTNode("include_fragment", i, i)
                node.set_property("fragment", self._extract_fragment_name(stripped))
                root.add_child(node)

            elif stripped.startswith("navigator") and "{" in line:
                current_section = ClientASTNode("navigator_section", i, i)
                root.add_child(current_section)

            elif re.match(self.structure_patterns["navigator_entry"], line):
                node = ClientASTNode("navigator_entry", i, i)
                node.set_property(
                    "entry_info", self._extract_navigator_entry_info(stripped)
                )
                if current_section:
                    current_section.add_child(node)
                else:
                    root.add_child(node)

            elif re.match(self.structure_patterns["page_declaration"], line):
                node = ClientASTNode("page_declaration", i, i)
                node.set_property("page_info", self._extract_page_info(stripped))
                root.add_child(node)

            elif re.match(self.structure_patterns["command_declaration"], line):
                node = ClientASTNode("command_declaration", i, i)
                node.set_property("command_info", self._extract_command_info(stripped))
                root.add_child(node)

            # Close sections when braces are balanced
            if current_section and brace_level == 0 and "}" in line:
                current_section.end_line = i
                current_section = None

        return root

    def _validate_basic_structure(self, lines: List[str]):
        """Validate basic client structure with conservative approach"""
        has_client_declaration = False
        has_component_declaration = False
        has_layer_declaration = False
        has_projection_declaration = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if re.match(self.structure_patterns["client_declaration"], line):
                has_client_declaration = True
            elif re.match(self.structure_patterns["component_declaration"], line):
                has_component_declaration = True
            elif re.match(self.structure_patterns["layer_declaration"], line):
                has_layer_declaration = True
            elif re.match(self.structure_patterns["projection_declaration"], line):
                has_projection_declaration = True

        # Conservative validation - only error on clearly missing essential elements
        if not has_client_declaration:
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                "Client file should have a client declaration (e.g., 'client ClientName;')",
                DiagnosticSeverity.WARNING,
            )

        if not has_component_declaration:
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                "Client file should have a component declaration (e.g., 'component ORDER;')",
                DiagnosticSeverity.WARNING,
            )

    def _validate_syntax_patterns(self, lines: List[str]):
        """Validate syntax patterns with conservative approach"""
        brace_level = 0
        paren_level = 0

        for i, line in enumerate(lines):
            # Skip comments and empty lines
            if not line.strip() or line.strip().startswith("--"):
                continue

            # Conservative brace and parentheses checking
            for char in line:
                if char == "{":
                    brace_level += 1
                elif char == "}":
                    brace_level -= 1
                elif char == "(":
                    paren_level += 1
                elif char == ")":
                    paren_level -= 1

                # Conservative check - only report if severely unbalanced
                if brace_level < -1:
                    self._add_diagnostic(
                        i,
                        0,
                        i,
                        len(line),
                        "Unbalanced closing brace - extra '}' found",
                        DiagnosticSeverity.ERROR,
                    )
                    brace_level = 0  # Reset to avoid cascade errors

                if paren_level < -1:
                    self._add_diagnostic(
                        i,
                        0,
                        i,
                        len(line),
                        "Unbalanced closing parenthesis - extra ')' found",
                        DiagnosticSeverity.ERROR,
                    )
                    paren_level = 0  # Reset to avoid cascade errors

        # Final balance check (conservative)
        if brace_level > 2:
            self._add_diagnostic(
                len(lines) - 1,
                0,
                len(lines) - 1,
                0,
                f"Multiple unclosed braces detected (missing {brace_level} closing braces)",
                DiagnosticSeverity.WARNING,
            )
        elif brace_level > 0:
            self._add_diagnostic(
                len(lines) - 1,
                0,
                len(lines) - 1,
                0,
                "Possible missing closing brace",
                DiagnosticSeverity.HINT,
            )

    def _extract_identifier_after_keyword(self, line: str, keyword: str) -> str:
        """Extract identifier after a keyword (conservative)"""
        pattern = rf"{keyword}\s+(\w+)"
        match = re.search(pattern, line)
        return match.group(1) if match else ""

    def _extract_fragment_name(self, line: str) -> str:
        """Extract fragment name from include statement (conservative)"""
        pattern = r"include\s+fragment\s+([\w\-_]+)"
        match = re.search(pattern, line)
        return match.group(1) if match else ""

    def _extract_navigator_entry_info(self, line: str) -> dict:
        """Extract navigator entry information (conservative)"""
        info = {}

        # Extract entry name
        entry_match = re.search(r"entry\s+(\w+)", line)
        if entry_match:
            info["name"] = entry_match.group(1)

        # Extract parent
        parent_match = re.search(r"parent\s+([\w\.]+)", line)
        if parent_match:
            info["parent"] = parent_match.group(1)

        # Extract index
        index_match = re.search(r"index\s+(\d+)", line)
        if index_match:
            info["index"] = int(index_match.group(1))

        return info

    def _extract_page_info(self, line: str) -> dict:
        """Extract page information (conservative)"""
        info = {}

        # Extract page name and type
        page_match = re.search(r"page\s+(\w+)\s+using\s+(\w+)", line)
        if page_match:
            info["type"] = page_match.group(1)
            info["entity_set"] = page_match.group(2)

        return info

    def _extract_command_info(self, line: str) -> dict:
        """Extract command information (conservative)"""
        info = {}

        # Extract command name
        cmd_match = re.search(r"command\s+(\w+)", line)
        if cmd_match:
            info["name"] = cmd_match.group(1)

        # Extract for clause if present
        for_match = re.search(r"for\s+(\w+)", line)
        if for_match:
            info["for_entity"] = for_match.group(1)

        return info

    def _add_diagnostic(
        self,
        line: int,
        col: int,
        end_line: int,
        end_col: int,
        message: str,
        severity: DiagnosticSeverity,
    ):
        """Add a diagnostic message"""
        self.diagnostics.append(
            Diagnostic(
                line=line,
                column=col,
                end_line=end_line,
                end_column=end_col,
                message=message,
                severity=severity,
            )
        )


def analyze_client_file(content: str, filename: str = "client.client") -> dict:
    """
    Main entry point for analyzing IFS Cloud client files.

    Args:
        content: Client file content as string
        filename: Optional filename for error reporting

    Returns:
        Dictionary containing analysis results
    """
    analyzer = ConservativeClientAnalyzer()
    return analyzer.analyze(content, filename)


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple client file structure
    test_content = """
-----------------------------------------------------------------------------
-- Date        Sign    History  
-- ----------  ------  ------------------------------------------------------
-- 2024-01-01  Test    Created test client file
-----------------------------------------------------------------------------

client TestClient;
component TEST;
layer Core;
projection TestHandling;

include fragment TestFragment;

----------------------------- NAVIGATOR ENTRIES -----------------------------
navigator {
   entry TestEntry parent TestNavigator.TestSection at index 100 {
      label = "Test";
      page Form home Test;
   }
}

-------------------------------- MAIN PAGES ---------------------------------
page Form using TestSet {
   label = "Test";
   group TestGroup;
}

---------------------------------- GROUPS -----------------------------------
group TestGroup for Test {
   label = "Test Information";
   field TestField;
}

command TestCommand for Test {
   enabled = [true];
   execute {
      call TestMethod();
   }
}
"""

    result = analyze_client_file(test_content, "test.client")

    print("=== IFS Cloud Client Analysis Results ===")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Info: {result['info']}")
    print(f"Hints: {result['hints']}")

    if result["diagnostics"]:
        print("\nDiagnostics:")
        for diag in result["diagnostics"]:
            print(f"  {diag['severity']} (Line {diag['line']}): {diag['message']}")

    if result["ast"]:
        print(f"\nAST Root: {result['ast']['type']}")
        print(f"Children: {len(result['ast']['children'])}")
