#!/usr/bin/env python3
"""
Example showing how to integrate the IFS Cloud Projection Analyzer with VS Code and Copilot.

This demonstrates the analyzer's capabilities for:
- Real-time syntax checking
- AST-based code completion
- Error diagnostics with fix suggestions
- Intelligent refactoring support
"""

import sys
import os
import json
from typing import Dict, List, Any

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ifs_cloud_mcp_server.projection_analyzer import (
    ProjectionAnalyzer,
    DiagnosticSeverity,
)


class CopilotIntegration:
    """
    Integration layer for VS Code Copilot to analyze IFS Cloud projection files
    """

    def __init__(self):
        self.analyzer = ProjectionAnalyzer(strict_mode=False)

    def get_diagnostics(self, content: str) -> List[Dict[str, Any]]:
        """
        Get VS Code compatible diagnostics for a projection file

        Returns:
            List of diagnostic objects compatible with VS Code Language Server Protocol
        """
        ast = self.analyzer.analyze(content)

        diagnostics = []
        for diag in ast.diagnostics:
            vscode_severity = {
                DiagnosticSeverity.ERROR: 1,  # Error
                DiagnosticSeverity.WARNING: 2,  # Warning
                DiagnosticSeverity.INFO: 3,  # Information
                DiagnosticSeverity.HINT: 4,  # Hint
            }.get(diag.severity, 3)

            diagnostic = {
                "range": {
                    "start": {"line": diag.line - 1, "character": 0},
                    "end": {"line": diag.line - 1, "character": 999},
                },
                "severity": vscode_severity,
                "message": diag.message,
                "code": diag.code,
                "source": "ifs-cloud-projection-analyzer",
            }

            if diag.fix_suggestion:
                diagnostic["data"] = {"fix_suggestion": diag.fix_suggestion}

            diagnostics.append(diagnostic)

        return diagnostics

    def get_completion_items(
        self, content: str, line: int, character: int
    ) -> List[Dict[str, Any]]:
        """
        Get code completion suggestions based on current AST context

        Returns:
            List of completion items compatible with VS Code Language Server Protocol
        """
        ast = self.analyzer.analyze(content)
        completions = []

        # Get the current line content to understand context
        lines = content.split("\n")
        if line >= len(lines):
            return []

        current_line = lines[line][:character].strip().lower()

        # Basic completion suggestions based on context
        if not current_line or current_line.endswith(";"):
            # At top level - suggest main projection keywords
            completions.extend(
                [
                    {
                        "label": "projection",
                        "kind": 14,  # Keyword
                        "detail": "Projection declaration",
                        "insertText": "projection ${1:ProjectionName};",
                        "insertTextFormat": 2,  # Snippet
                    },
                    {
                        "label": "component",
                        "kind": 14,
                        "detail": "Component declaration",
                        "insertText": "component ${1:COMPONENT_NAME};",
                        "insertTextFormat": 2,
                    },
                    {
                        "label": "entityset",
                        "kind": 14,
                        "detail": "Entity set definition",
                        "insertText": "entityset ${1:SetName} for ${2:EntityName} {\n\t$0\n}",
                        "insertTextFormat": 2,
                    },
                ]
            )

        elif "entityset" in current_line and "{" not in current_line:
            # Inside entityset definition
            completions.extend(
                [
                    {
                        "label": "context",
                        "kind": 14,
                        "detail": "Context definition",
                        "insertText": "context ${1:Context}(${2:Parameter});",
                        "insertTextFormat": 2,
                    },
                    {
                        "label": "where",
                        "kind": 14,
                        "detail": "Where clause",
                        "insertText": 'where = "${1:condition}";',
                        "insertTextFormat": 2,
                    },
                ]
            )

        # Add entity references from the AST
        for entity in ast.entities:
            entity_name = (
                entity.name
                if hasattr(entity, "name")
                else entity.get("name", "Unknown")
            )
            completions.append(
                {
                    "label": entity_name,
                    "kind": 7,  # Class
                    "detail": f"Entity from {ast.name}",
                    "insertText": entity_name,
                }
            )

        return completions

    def get_hover_info(self, content: str, line: int, character: int) -> Dict[str, Any]:
        """
        Get hover information for symbols in projection files

        Returns:
            Hover information compatible with VS Code Language Server Protocol
        """
        ast = self.analyzer.analyze(content)

        # Find what symbol is under the cursor
        lines = content.split("\n")
        if line >= len(lines):
            return {}

        current_line = lines[line]
        # Simple word extraction at character position
        start = character
        end = character

        # Find word boundaries
        while start > 0 and current_line[start - 1].isalnum():
            start -= 1
        while end < len(current_line) and current_line[end].isalnum():
            end += 1

        word = current_line[start:end]

        # Look for the word in AST elements
        for entity in ast.entities:
            entity_name = (
                entity.name
                if hasattr(entity, "name")
                else entity.get("name", "Unknown")
            )
            if entity_name == word:
                return {
                    "contents": {
                        "kind": "markdown",
                        "value": f"**Entity**: `{entity_name}`\n\nDefined in projection `{ast.name}`",
                    }
                }

        for entityset in ast.entitysets:
            entityset_name = (
                entityset.get("name", "Unknown")
                if isinstance(entityset, dict)
                else getattr(entityset, "name", "Unknown")
            )
            if entityset_name == word:
                entity_ref = (
                    entityset.get("entity", "Unknown")
                    if isinstance(entityset, dict)
                    else getattr(entityset, "entity", "Unknown")
                )
                context_ref = (
                    entityset.get("context", "Unknown")
                    if isinstance(entityset, dict)
                    else getattr(entityset, "context", "Unknown")
                )
                return {
                    "contents": {
                        "kind": "markdown",
                        "value": f"**EntitySet**: `{entityset_name}`\n\nEntity: `{entity_ref}`\nContext: `{context_ref}`",
                    }
                }

        return {}

    def get_document_symbols(self, content: str) -> List[Dict[str, Any]]:
        """
        Get document outline/symbols for navigation

        Returns:
            Document symbols compatible with VS Code Language Server Protocol
        """
        ast = self.analyzer.analyze(content)
        symbols = []

        # Add projection as root symbol
        symbols.append(
            {
                "name": ast.name or "Unknown Projection",
                "kind": 5,  # Class
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 999},
                },
                "selectionRange": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 999},
                },
            }
        )

        # Add entitysets
        for entityset in ast.entitysets:
            entityset_name = (
                entityset.get("name", "Unknown")
                if isinstance(entityset, dict)
                else getattr(entityset, "name", "Unknown")
            )
            symbols.append(
                {
                    "name": f"EntitySet: {entityset_name}",
                    "kind": 13,  # Module
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 999},
                    },
                    "selectionRange": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 999},
                    },
                }
            )

        # Add entities
        for entity in ast.entities:
            entity_name = (
                entity.name
                if hasattr(entity, "name")
                else entity.get("name", "Unknown")
            )
            symbols.append(
                {
                    "name": f"Entity: {entity_name}",
                    "kind": 7,  # Class
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 999},
                    },
                    "selectionRange": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 999},
                    },
                }
            )

        return symbols


def demo_copilot_features():
    """
    Demonstrate the Copilot integration features
    """
    print("🤖 IFS Cloud Projection Analyzer - Copilot Integration Demo")
    print("=" * 60)

    integration = CopilotIntegration()

    # Sample projection with some issues for demonstration
    sample_content = """
    projection AccountsHandling;
    component accrul;  // Should be uppercase
    layer Core;
    description "Accounts Overview";
    
    entityset AccountSet for Account {
        context Company(Company);
        where = "status = 'ACTIVE'";
    }
    
    entityset MissingEntitySet for {  // Missing entity name
        context Company(Company);
    }
    
    @Override
    entity Account {
        attribute AccountCode Text;
        reference CompanyRef(Company) to Company(Company);
    """

    # 1. Get diagnostics (error checking)
    print("1. 🔍 Diagnostics (Error Detection)")
    print("-" * 30)
    diagnostics = integration.get_diagnostics(sample_content)

    for diag in diagnostics:
        severity_name = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}[
            diag["severity"]
        ]
        line_num = diag["range"]["start"]["line"] + 1
        print(f"   {severity_name}: Line {line_num}: {diag['message']}")
        if "data" in diag and "fix_suggestion" in diag["data"]:
            print(f"   💡 Fix: {diag['data']['fix_suggestion']}")

    # 2. Get completion suggestions
    print(f"\n2. 📝 Code Completion (at line 15, position 0)")
    print("-" * 30)
    completions = integration.get_completion_items(sample_content, 15, 0)

    for completion in completions[:5]:  # Show first 5
        print(f"   {completion['label']}: {completion['detail']}")

    # 3. Get hover information
    print(f"\n3. 📖 Hover Info (for 'Account' on line 7)")
    print("-" * 30)
    hover = integration.get_hover_info(sample_content, 6, 25)
    if hover:
        print(f"   {hover['contents']['value']}")
    else:
        print("   No hover info available")

    # 4. Get document symbols
    print(f"\n4. 🗂️  Document Symbols (Outline)")
    print("-" * 30)
    symbols = integration.get_document_symbols(sample_content)

    for symbol in symbols:
        kind_name = {5: "Projection", 13: "EntitySet", 7: "Entity"}.get(
            symbol["kind"], "Other"
        )
        print(f"   {kind_name}: {symbol['name']}")

    print("\n" + "=" * 60)
    print("✅ Copilot integration demo completed!")
    print("🚀 This analyzer is ready for VS Code extension integration!")


if __name__ == "__main__":
    demo_copilot_features()
