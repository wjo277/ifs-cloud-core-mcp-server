#!/usr/bin/env python3
"""
IFS Cloud Client Analyzer - Copilot Integration Example

This example demonstrates how the client analyzer can be integrated with GitHub Copilot
and VS Code to provide intelligent assistance for IFS Cloud client file development.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ifs_cloud_mcp_server.client_analyzer import analyze_client_file


class ClientCopilotIntegration:
    """Integration class for GitHub Copilot and VS Code"""

    def __init__(self):
        self.analyzer_cache = {}

    def get_client_context(self, file_content: str, filename: str) -> dict:
        """
        Extract context from client file for Copilot suggestions

        Returns structured information about the client file that can be used
        to provide more relevant code suggestions.
        """
        # Use cached analysis if available
        cache_key = hash(file_content)
        if cache_key in self.analyzer_cache:
            result = self.analyzer_cache[cache_key]
        else:
            result = analyze_client_file(file_content, filename)
            self.analyzer_cache[cache_key] = result

        if not result["ast"]:
            return {}

        ast = result["ast"]
        context = {
            "file_type": "ifs_cloud_client",
            "valid": result["valid"],
            "client_info": {},
            "structural_elements": [],
            "included_fragments": [],
            "navigator_entries": [],
            "page_definitions": [],
            "commands": [],
            "issues": result["diagnostics"],
        }

        # Extract detailed information from AST
        for child in ast["children"]:
            child_type = child["type"]

            if child_type == "client_declaration":
                context["client_info"]["name"] = child["properties"].get("name", "")

            elif child_type == "component_declaration":
                context["client_info"]["component"] = child["properties"].get(
                    "name", ""
                )

            elif child_type == "layer_declaration":
                context["client_info"]["layer"] = child["properties"].get("name", "")

            elif child_type == "projection_declaration":
                context["client_info"]["projection"] = child["properties"].get(
                    "name", ""
                )

            elif child_type == "include_fragment":
                context["included_fragments"].append(
                    child["properties"].get("fragment", "")
                )

            elif child_type == "navigator_section":
                # Extract navigator entries from this section
                for nav_child in child.get("children", []):
                    if nav_child["type"] == "navigator_entry":
                        context["navigator_entries"].append(nav_child["properties"])

            elif child_type == "page_declaration":
                context["page_definitions"].append(child["properties"])

            elif child_type == "command_declaration":
                context["commands"].append(child["properties"])

            context["structural_elements"].append(child_type)

        # Remove duplicates from structural elements
        context["structural_elements"] = list(set(context["structural_elements"]))

        return context

    def suggest_completions(
        self, file_content: str, cursor_position: tuple, filename: str
    ) -> list:
        """
        Suggest code completions based on current context

        Args:
            file_content: Current file content
            cursor_position: (line, column) tuple
            filename: File name

        Returns:
            List of completion suggestions
        """
        context = self.get_client_context(file_content, filename)
        line, column = cursor_position

        # Get current line
        lines = file_content.split("\n")
        if line >= len(lines):
            return []

        current_line = lines[line][:column]
        suggestions = []

        # Context-aware suggestions based on current line
        stripped_line = current_line.strip()

        if not stripped_line:
            # Beginning of line - suggest common declarations
            suggestions.extend(
                [
                    "client ",
                    "component ",
                    "layer Core;",
                    "projection ",
                    "include fragment ",
                    "navigator {",
                    "page ",
                    "list ",
                    "group ",
                    "field ",
                    "command ",
                    "dialog ",
                ]
            )

        elif stripped_line.startswith("client ") and not stripped_line.endswith(";"):
            # Client declaration - suggest completion
            client_name = context["client_info"].get("name", "ClientName")
            suggestions.append(f"{client_name};")

        elif stripped_line.startswith("component ") and not stripped_line.endswith(";"):
            # Component declaration - suggest common components
            suggestions.extend(["ORDER;", "PURCH;", "INVENT;", "PROJ;", "MFG;"])

        elif stripped_line.startswith("projection ") and not stripped_line.endswith(
            ";"
        ):
            # Projection declaration - suggest based on client name
            client_name = context["client_info"].get("name", "")
            if client_name:
                suggestions.append(f"{client_name}Handling;")
                suggestions.append(f"{client_name}Service;")

        elif "include fragment" in stripped_line and not stripped_line.endswith(";"):
            # Fragment inclusion - suggest common fragments
            suggestions.extend(
                [
                    "UserAllowedSiteLovSelector;",
                    "OutputTypeLovSelector;",
                    "MpccomPhraseTextLovSelector;",
                    "TaxCodeRestrictedSelector;",
                    "DocumentText;",
                ]
            )

        elif stripped_line.startswith("entry ") and "parent" not in stripped_line:
            # Navigator entry - suggest parent structure
            suggestions.extend(
                [
                    "parent OrderNavigator.SalesPart at index 100",
                    "parent PurchNavigator.ProcurementOrder at index 100",
                    "parent InventNavigator.InventoryPart at index 100",
                ]
            )

        elif stripped_line.startswith("page ") and "using" not in stripped_line:
            # Page declaration - suggest common page types
            suggestions.extend(
                ["Form using EntitySet", "List using EntitySet", "Dialog for Entity"]
            )

        elif stripped_line.startswith("field ") and "{" not in stripped_line:
            # Field declaration - suggest common field properties
            suggestions.extend([" {", "Ref {", " {\n   size = Small;\n}"])

        return suggestions

    def provide_hover_info(
        self, file_content: str, cursor_position: tuple, filename: str
    ) -> str:
        """
        Provide hover information at cursor position

        Args:
            file_content: Current file content
            cursor_position: (line, column) tuple
            filename: File name

        Returns:
            Hover information string
        """
        context = self.get_client_context(file_content, filename)
        line, column = cursor_position

        lines = file_content.split("\n")
        if line >= len(lines):
            return ""

        current_line = lines[line]
        word_at_cursor = self._get_word_at_position(current_line, column)

        if not word_at_cursor:
            return ""

        # Provide context-specific information
        hover_info = []

        if word_at_cursor == "client":
            client_name = context["client_info"].get("name", "Unknown")
            hover_info.append(f"**IFS Cloud Client**: {client_name}")
            hover_info.append(
                "Defines the client-side presentation layer for IFS Cloud applications"
            )

        elif word_at_cursor == "component":
            component = context["client_info"].get("component", "Unknown")
            hover_info.append(f"**IFS Cloud Component**: {component}")
            hover_info.append("Specifies the logical component this client belongs to")

        elif word_at_cursor == "projection":
            projection = context["client_info"].get("projection", "Unknown")
            hover_info.append(f"**IFS Cloud Projection**: {projection}")
            hover_info.append(
                "References the server-side projection that provides data services"
            )

        elif word_at_cursor in context["included_fragments"]:
            hover_info.append(f"**Fragment**: {word_at_cursor}")
            hover_info.append("Reusable UI component included in this client")

        elif word_at_cursor == "navigator":
            nav_count = len(context["navigator_entries"])
            hover_info.append(f"**Navigator**: {nav_count} entries")
            hover_info.append("Defines navigation structure and menu entries")

        elif word_at_cursor == "page":
            page_count = len(context["page_definitions"])
            hover_info.append(f"**Pages**: {page_count} definitions")
            hover_info.append("Defines user interface pages and layouts")

        elif word_at_cursor == "command":
            cmd_count = len(context["commands"])
            hover_info.append(f"**Commands**: {cmd_count} definitions")
            hover_info.append("Defines user actions and business logic")

        # Add diagnostic information if there are issues
        if context["issues"]:
            error_count = len(
                [d for d in context["issues"] if d["severity"] == "ERROR"]
            )
            warning_count = len(
                [d for d in context["issues"] if d["severity"] == "WARNING"]
            )

            if error_count > 0:
                hover_info.append(f"⚠️ {error_count} error(s) detected")
            if warning_count > 0:
                hover_info.append(f"💡 {warning_count} warning(s)")

        return "\n\n".join(hover_info) if hover_info else ""

    def _get_word_at_position(self, line: str, column: int) -> str:
        """Extract word at given column position"""
        if column >= len(line):
            return ""

        # Find word boundaries
        start = column
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1

        end = column
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1

        return line[start:end]


def demo_copilot_integration():
    """Demonstrate Copilot integration features"""

    print("IFS Cloud Client Analyzer - Copilot Integration Demo")
    print("=" * 60)

    # Load a real client file for demonstration
    test_file = "_work/order/model/order/SalesChargeType.client"

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        # Use sample content if real file not available
        content = """
client SalesChargeType;
component ORDER;
layer Core;
projection SalesChargeTypeHandling;

include fragment UserAllowedSiteLovSelector;
include fragment TaxCodeRestrictedSelector;

navigator {
   entry SalesChargeTypeNavEntry parent OrderNavigator.SalesPart at index 600 {
      label = "Sales Charge Type";
      page Form home SalesChargeType;
   }
}

page Form using SalesChargeTypeSet {
   label = "Sales Charge Type";
   group SalesChargeTypeGroup;
}

command DocumentTextCommand for SalesChargeTypeDesc {
   label = "Document Text";
   enabled = [true];
   execute {
      assistant DocumentText(NoteId, LabelTextVar);
   }
}
"""

    # Initialize integration
    integration = ClientCopilotIntegration()

    # Demo 1: Context extraction
    print("\n1. Context Extraction")
    print("-" * 20)
    context = integration.get_client_context(content, test_file)

    print(f"Client Name: {context['client_info'].get('name', 'N/A')}")
    print(f"Component: {context['client_info'].get('component', 'N/A')}")
    print(f"Projection: {context['client_info'].get('projection', 'N/A')}")
    print(f"Fragments: {len(context['included_fragments'])}")
    print(f"Navigator Entries: {len(context['navigator_entries'])}")
    print(f"Commands: {len(context['commands'])}")
    print(f"Valid: {context['valid']}")

    # Demo 2: Code completion
    print("\n2. Code Completion Suggestions")
    print("-" * 30)

    # Test completion at different positions
    test_positions = [
        ("After 'client '", "client ", (0, 7)),
        ("After 'include fragment '", "include fragment ", (0, 17)),
        ("After 'field '", "field ", (0, 6)),
    ]

    for desc, test_line, pos in test_positions:
        suggestions = integration.suggest_completions(test_line, pos, "test.client")
        print(f"{desc}: {suggestions[:3]}...")  # Show first 3 suggestions

    # Demo 3: Hover information
    print("\n3. Hover Information")
    print("-" * 20)

    hover_tests = [("client", 5), ("component", 5), ("projection", 5), ("navigator", 5)]

    for word, col in hover_tests:
        test_content = f"some text {word} more text"
        hover_info = integration.provide_hover_info(
            test_content, (0, col + 10), "test.client"
        )
        if hover_info:
            first_line = hover_info.split("\n")[0]
            print(f"'{word}': {first_line}")

    print(f"\n{'='*60}")
    print("Integration demo completed!")


if __name__ == "__main__":
    demo_copilot_integration()
