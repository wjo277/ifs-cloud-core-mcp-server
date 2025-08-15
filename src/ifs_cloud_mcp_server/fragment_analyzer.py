#!/usr/bin/env python3
"""
Conservative IFS Cloud Fragment File Analyzer

This module provides AST analysis for IFS Cloud Fragment (.fragment) files by combining
both client and projection analyzers. Fragment files contain both client-side UI components
and server-side projection elements separated by clear section markers.

Key Features:
- Leverages both existing conservative analyzers (client & projection)
- Handles mixed client/projection content appropriately
- Conservative validation maintaining zero false positives
- Section-aware parsing and analysis
- Complete AST generation for both sections
- Diagnostic support with appropriate context

Author: IFS Cloud MCP Server
License: MIT
"""

import re
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Import our existing analyzers
from .client_analyzer import (
    ConservativeClientAnalyzer,
    DiagnosticSeverity,
    Diagnostic,
    ClientASTNode,
)
from .projection_analyzer import ProjectionAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FragmentSection(Enum):
    """Fragment file sections"""

    HEADER = "header"
    CLIENT_FRAGMENTS = "client_fragments"
    PROJECTION_FRAGMENTS = "projection_fragments"


@dataclass
class FragmentSectionInfo:
    """Information about a fragment section"""

    section_type: FragmentSection
    start_line: int
    end_line: int
    content_lines: List[str]


class FragmentASTNode:
    """AST node for fragment files combining client and projection elements"""

    def __init__(self, node_type: str, start_line: int = 0, end_line: int = 0):
        self.node_type = node_type
        self.start_line = start_line
        self.end_line = end_line
        self.children = []
        self.properties = {}
        self.section = None

    def add_child(self, child: "FragmentASTNode"):
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
            "section": self.section.value if self.section else None,
            "properties": self.properties,
            "children": [child.to_dict() for child in self.children],
        }


class ConservativeFragmentAnalyzer:
    """
    Conservative analyzer for IFS Cloud Fragment files.

    This analyzer combines the client and projection analyzers to handle fragment files
    that contain both client-side UI components and server-side projection elements.

    Maintains the same conservative philosophy:
    - Zero false positives on legitimate IFS Cloud fragments
    - Context-aware analysis for different sections
    - Appropriate diagnostic severity levels
    - Graceful error recovery
    """

    def __init__(self):
        """Initialize the conservative fragment analyzer"""
        self.diagnostics = []
        self.ast_root = None
        self.client_analyzer = ConservativeClientAnalyzer()
        self.projection_analyzer = ProjectionAnalyzer(strict_mode=False)

        # Fragment-specific patterns
        self.section_patterns = {
            "fragment_declaration": r"^\s*fragment\s+\w+\s*;",
            "component_declaration": r"^\s*component\s+\w+\s*;",
            "layer_declaration": r"^\s*layer\s+\w+\s*;",
            "description_declaration": r"^\s*description\s+",
            "client_section": r"^\s*-+\s*CLIENT\s+FRAGMENTS\s*-+",
            "projection_section": r"^\s*-+\s*PROJECTION\s+FRAGMENTS\s*-+",
            "include_fragment": r"^\s*include\s+fragment\s+[\w\-_]+\s*;",
        }

    def analyze(self, content: str, filename: str = "fragment.fragment") -> dict:
        """
        Analyze fragment file content with conservative approach.

        Args:
            content: Fragment file content
            filename: Name of the file being analyzed

        Returns:
            Analysis results with combined AST and diagnostics
        """
        self.diagnostics = []

        try:
            lines = content.split("\n")

            # Parse fragment sections
            sections = self._parse_fragment_sections(lines)

            # Generate combined AST
            self.ast_root = self._build_fragment_ast(lines, sections)

            # Validate fragment structure
            self._validate_fragment_structure(lines, sections)

            # Analyze each section with appropriate analyzer
            for section in sections:
                if section.section_type == FragmentSection.CLIENT_FRAGMENTS:
                    self._analyze_client_section(section)
                elif section.section_type == FragmentSection.PROJECTION_FRAGMENTS:
                    self._analyze_projection_section(section)

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
                "sections": {
                    "header": any(
                        s.section_type == FragmentSection.HEADER for s in sections
                    ),
                    "client_fragments": any(
                        s.section_type == FragmentSection.CLIENT_FRAGMENTS
                        for s in sections
                    ),
                    "projection_fragments": any(
                        s.section_type == FragmentSection.PROJECTION_FRAGMENTS
                        for s in sections
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing fragment file {filename}: {e}")
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                f"Failed to parse fragment file: {str(e)}",
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
                "sections": {
                    "header": False,
                    "client_fragments": False,
                    "projection_fragments": False,
                },
            }

    def _parse_fragment_sections(self, lines: List[str]) -> List[FragmentSectionInfo]:
        """Parse fragment file into sections"""
        sections = []
        current_section_type = FragmentSection.HEADER
        current_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip comments and empty lines for section detection
            if not stripped or stripped.startswith("-- "):
                continue

            # Check for section markers
            if re.match(self.section_patterns["client_section"], line):
                # Close previous section
                if current_start < i:
                    sections.append(
                        FragmentSectionInfo(
                            current_section_type,
                            current_start,
                            i - 1,
                            lines[current_start:i],
                        )
                    )
                current_section_type = FragmentSection.CLIENT_FRAGMENTS
                current_start = i + 1

            elif re.match(self.section_patterns["projection_section"], line):
                # Close previous section
                if current_start < i:
                    sections.append(
                        FragmentSectionInfo(
                            current_section_type,
                            current_start,
                            i - 1,
                            lines[current_start:i],
                        )
                    )
                current_section_type = FragmentSection.PROJECTION_FRAGMENTS
                current_start = i + 1

        # Close final section
        if current_start < len(lines):
            sections.append(
                FragmentSectionInfo(
                    current_section_type,
                    current_start,
                    len(lines) - 1,
                    lines[current_start:],
                )
            )

        return sections

    def _build_fragment_ast(
        self, lines: List[str], sections: List[FragmentSectionInfo]
    ) -> FragmentASTNode:
        """Build combined AST from fragment sections"""
        root = FragmentASTNode("fragment_file")

        for section in sections:
            section_node = FragmentASTNode(
                f"{section.section_type.value}_section",
                section.start_line,
                section.end_line,
            )
            section_node.section = section.section_type

            # Parse section-specific content
            if section.section_type == FragmentSection.HEADER:
                self._parse_header_section(section_node, section.content_lines)
            elif section.section_type == FragmentSection.CLIENT_FRAGMENTS:
                self._parse_client_section_ast(
                    section_node, section.content_lines, section.start_line
                )
            elif section.section_type == FragmentSection.PROJECTION_FRAGMENTS:
                self._parse_projection_section_ast(
                    section_node, section.content_lines, section.start_line
                )

            root.add_child(section_node)

        return root

    def _parse_header_section(self, section_node: FragmentASTNode, lines: List[str]):
        """Parse fragment header section"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue

            if re.match(self.section_patterns["fragment_declaration"], line):
                node = FragmentASTNode("fragment_declaration", i, i)
                node.set_property(
                    "name", self._extract_identifier_after_keyword(stripped, "fragment")
                )
                section_node.add_child(node)

            elif re.match(self.section_patterns["component_declaration"], line):
                node = FragmentASTNode("component_declaration", i, i)
                node.set_property(
                    "name",
                    self._extract_identifier_after_keyword(stripped, "component"),
                )
                section_node.add_child(node)

            elif re.match(self.section_patterns["layer_declaration"], line):
                node = FragmentASTNode("layer_declaration", i, i)
                node.set_property(
                    "name", self._extract_identifier_after_keyword(stripped, "layer")
                )
                section_node.add_child(node)

            elif re.match(self.section_patterns["description_declaration"], line):
                node = FragmentASTNode("description_declaration", i, i)
                node.set_property("description", self._extract_description(stripped))
                section_node.add_child(node)

            elif re.match(self.section_patterns["include_fragment"], line):
                node = FragmentASTNode("include_fragment", i, i)
                node.set_property("fragment", self._extract_fragment_name(stripped))
                section_node.add_child(node)

    def _parse_client_section_ast(
        self, section_node: FragmentASTNode, lines: List[str], start_line_offset: int
    ):
        """Parse client section using client analyzer"""
        if not lines:
            return

        # Create a pseudo-client file content for the client analyzer
        content = "\n".join(lines)

        try:
            # Use client analyzer but handle it gracefully
            client_ast = self.client_analyzer._parse_client_structure(lines)

            # Convert client AST to fragment AST nodes
            for child in client_ast.children:
                fragment_node = self._convert_client_ast_node(child, start_line_offset)
                fragment_node.section = FragmentSection.CLIENT_FRAGMENTS
                section_node.add_child(fragment_node)

        except Exception as e:
            logger.debug(f"Client section parsing issue (non-critical): {e}")
            # Conservative approach - don't fail on client section issues

    def _parse_projection_section_ast(
        self, section_node: FragmentASTNode, lines: List[str], start_line_offset: int
    ):
        """Parse projection section using projection analyzer"""
        if not lines:
            return

        # Create a pseudo-projection file content
        content = "\n".join(lines)

        try:
            # Use projection analyzer but handle it gracefully
            projection_ast = self.projection_analyzer.analyze(content)

            # Convert projection AST to fragment AST nodes
            # The projection analyzer returns a ProjectionAST object
            if hasattr(projection_ast, "entities"):
                for entity in projection_ast.entities:
                    fragment_node = FragmentASTNode(
                        "entity", start_line_offset, start_line_offset
                    )
                    fragment_node.set_property(
                        "name", entity.name if hasattr(entity, "name") else "Unknown"
                    )
                    fragment_node.section = FragmentSection.PROJECTION_FRAGMENTS
                    section_node.add_child(fragment_node)

            if hasattr(projection_ast, "queries"):
                for query in projection_ast.queries:
                    fragment_node = FragmentASTNode(
                        "query", start_line_offset, start_line_offset
                    )
                    fragment_node.set_property(
                        "name",
                        (
                            query.get("name", "Unknown")
                            if isinstance(query, dict)
                            else "Unknown"
                        ),
                    )
                    fragment_node.section = FragmentSection.PROJECTION_FRAGMENTS
                    section_node.add_child(fragment_node)

            if hasattr(projection_ast, "actions"):
                for action in projection_ast.actions:
                    fragment_node = FragmentASTNode(
                        "action", start_line_offset, start_line_offset
                    )
                    fragment_node.set_property(
                        "name",
                        (
                            action.get("name", "Unknown")
                            if isinstance(action, dict)
                            else "Unknown"
                        ),
                    )
                    fragment_node.section = FragmentSection.PROJECTION_FRAGMENTS
                    section_node.add_child(fragment_node)

            if hasattr(projection_ast, "functions"):
                for function in projection_ast.functions:
                    fragment_node = FragmentASTNode(
                        "function", start_line_offset, start_line_offset
                    )
                    fragment_node.set_property(
                        "name",
                        (
                            function.get("name", "Unknown")
                            if isinstance(function, dict)
                            else "Unknown"
                        ),
                    )
                    fragment_node.section = FragmentSection.PROJECTION_FRAGMENTS
                    section_node.add_child(fragment_node)

        except Exception as e:
            logger.debug(f"Projection section parsing issue (non-critical): {e}")
            # Conservative approach - don't fail on projection section issues

    def _convert_client_ast_node(
        self, client_node, line_offset: int
    ) -> FragmentASTNode:
        """Convert client AST node to fragment AST node"""
        fragment_node = FragmentASTNode(
            client_node.node_type,
            client_node.start_line + line_offset,
            client_node.end_line + line_offset,
        )

        # Copy properties
        for key, value in client_node.properties.items():
            fragment_node.set_property(key, value)

        # Convert children recursively
        for child in client_node.children:
            fragment_child = self._convert_client_ast_node(child, line_offset)
            fragment_node.add_child(fragment_child)

        return fragment_node

    def _validate_fragment_structure(
        self, lines: List[str], sections: List[FragmentSectionInfo]
    ):
        """Validate fragment structure with conservative approach"""
        has_fragment_declaration = False
        has_component_declaration = False

        # Check header section for essential declarations
        header_section = next(
            (s for s in sections if s.section_type == FragmentSection.HEADER), None
        )
        if header_section:
            for line in header_section.content_lines:
                if re.match(self.section_patterns["fragment_declaration"], line):
                    has_fragment_declaration = True
                elif re.match(self.section_patterns["component_declaration"], line):
                    has_component_declaration = True

        # Conservative validation - only warn about clearly missing elements
        if not has_fragment_declaration:
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                "Fragment file should have a fragment declaration (e.g., 'fragment FragmentName;')",
                DiagnosticSeverity.WARNING,
            )

        if not has_component_declaration:
            self._add_diagnostic(
                0,
                0,
                0,
                0,
                "Fragment file should have a component declaration (e.g., 'component ORDER;')",
                DiagnosticSeverity.WARNING,
            )

    def _analyze_client_section(self, section: FragmentSectionInfo):
        """Analyze client section using client analyzer"""
        if not section.content_lines:
            return

        content = "\n".join(section.content_lines)

        try:
            # Analyze with client analyzer
            result = self.client_analyzer.analyze(
                content, "fragment_client_section", skip_header_validation=True
            )

            # Add diagnostics with line offset
            for diag_dict in result["diagnostics"]:
                self._add_diagnostic(
                    diag_dict["line"] + section.start_line,
                    diag_dict["column"],
                    diag_dict["end_line"] + section.start_line,
                    diag_dict["end_column"],
                    f"Client section: {diag_dict['message']}",
                    DiagnosticSeverity[diag_dict["severity"]],
                )

        except Exception as e:
            logger.debug(f"Client section analysis issue (non-critical): {e}")

    def _analyze_projection_section(self, section: FragmentSectionInfo):
        """Analyze projection section using projection analyzer"""
        if not section.content_lines:
            return

        content = "\n".join(section.content_lines)

        try:
            # Analyze with projection analyzer
            projection_ast = self.projection_analyzer.analyze(content)

            # The projection analyzer doesn't return diagnostics in the same format
            # For now, we'll just do basic validation
            if not projection_ast or projection_ast.name == "Unknown":
                self._add_diagnostic(
                    section.start_line,
                    0,
                    section.start_line,
                    0,
                    "Projection section: Could not parse projection structure",
                    DiagnosticSeverity.HINT,
                )

        except Exception as e:
            logger.debug(f"Projection section analysis issue (non-critical): {e}")
            # Conservative approach - only add a hint, not an error
            self._add_diagnostic(
                section.start_line,
                0,
                section.start_line,
                0,
                f"Projection section: Analysis issue (non-critical)",
                DiagnosticSeverity.HINT,
            )

    def _extract_identifier_after_keyword(self, line: str, keyword: str) -> str:
        """Extract identifier after a keyword (conservative)"""
        pattern = rf"{keyword}\s+(\w+)"
        match = re.search(pattern, line)
        return match.group(1) if match else ""

    def _extract_description(self, line: str) -> str:
        """Extract description text"""
        pattern = r'description\s+"([^"]*)"'
        match = re.search(pattern, line)
        return match.group(1) if match else ""

    def _extract_fragment_name(self, line: str) -> str:
        """Extract fragment name from include statement"""
        pattern = r"include\s+fragment\s+([\w\-_]+)"
        match = re.search(pattern, line)
        return match.group(1) if match else ""

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
                source="ifs-cloud-fragment-analyzer",
            )
        )


def analyze_fragment_file(content: str, filename: str = "fragment.fragment") -> dict:
    """
    Main entry point for analyzing IFS Cloud fragment files.

    Args:
        content: Fragment file content as string
        filename: Optional filename for error reporting

    Returns:
        Dictionary containing analysis results
    """
    analyzer = ConservativeFragmentAnalyzer()
    return analyzer.analyze(content, filename)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample fragment file
    test_content = """
--------------------------------------------------------------------------------------
-- Date        Sign    History
-- ----------  ------  ---------------------------------------------------------------
-- 2024-01-01  Test    Created test fragment
--------------------------------------------------------------------------------------

fragment TestFragment;
component TEST;
layer Core;
description "Test fragment for validation";

include fragment TestIncludeFragment;

----------------------------- CLIENT FRAGMENTS ------------------------------

command TestCommand for TestEntity {
   enabled = [true];
   execute {
      call TestMethod();
   }
}

group TestGroup for TestEntity {
   label = "Test Group";
   field TestField;
}

--------------------------- PROJECTION FRAGMENTS ----------------------------

entity TestEntity {
   crud = Read, Update;
   
   attribute TestField Text;
}

action TestMethod {
   ludependencies = TestEntity;
}
"""

    result = analyze_fragment_file(test_content, "test.fragment")

    print("=== IFS Cloud Fragment Analysis Results ===")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Info: {result['info']}")
    print(f"Hints: {result['hints']}")

    print(f"\nSections detected:")
    print(f"  Header: {result['sections']['header']}")
    print(f"  Client Fragments: {result['sections']['client_fragments']}")
    print(f"  Projection Fragments: {result['sections']['projection_fragments']}")

    if result["diagnostics"]:
        print("\nDiagnostics:")
        for diag in result["diagnostics"]:
            print(f"  {diag['severity']} (Line {diag['line']}): {diag['message']}")

    if result["ast"]:
        print(f"\nAST Root: {result['ast']['type']}")
        print(f"Sections: {len(result['ast']['children'])}")
        for section in result["ast"]["children"]:
            print(f"  {section['type']}: {len(section['children'])} elements")
