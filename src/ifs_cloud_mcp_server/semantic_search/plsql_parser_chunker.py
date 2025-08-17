#!/usr/bin/env python3
"""
New parser-based PL/SQL chunking implementation to replace the manual parser
"""

from pathlib import Path
from typing import List, Tuple, Optional
from ..plsql_analyzer import ConservativePLSQLAnalyzer, NodeType


def chunk_plsql_with_parser(
    content: str,
) -> List[Tuple[str, int, int, str, Optional[str]]]:
    """
    Parse PL/SQL content using the robust ConservativePLSQLAnalyzer.

    Returns chunks in the format expected by the data loader:
    List of (content, start_line, end_line, chunk_type, function_name) tuples
    """
    chunks = []
    lines = content.split("\n")

    if not content.strip():
        return chunks

    # Initialize the robust PL/SQL analyzer
    analyzer = ConservativePLSQLAnalyzer(strict_mode=False)

    # Analyze the content to get AST and method information
    result = analyzer.analyze(content)

    # Extract chunks from the AST
    def extract_from_ast(node, parent_depth=0):
        if node.node_type in [NodeType.FUNCTION, NodeType.PROCEDURE]:
            # Get the function/procedure content with proper boundaries
            start_line = node.start_line
            end_line = node.end_line

            # Extract the content with bounds checking
            if end_line < len(lines):
                raw_content = "\n".join(lines[start_line : end_line + 1])

                # Add to chunks in the expected format
                chunks.append(
                    (
                        raw_content.strip(),
                        start_line + 1,  # 1-based line numbers
                        end_line + 1,
                        f"plsql_{node.node_type.value}",
                        node.name,
                    )
                )

        # Recursively process children (handles nested functions properly)
        for child in node.children:
            extract_from_ast(child, parent_depth + 1)

    # Process the AST if available
    if result.ast:
        extract_from_ast(result.ast)

    # Fallback: extract from method lists if AST parsing missed anything
    all_methods = result.public_methods + result.private_methods
    existing_names = {chunk[4] for chunk in chunks}

    for method in all_methods:
        if method["name"] not in existing_names:
            # Find the method boundaries using the analyzer's robust method
            method_line = method["line"] - 1  # Convert to 0-based
            if method_line < len(lines):
                end_line = analyzer._find_method_end(lines, method_line)

                if end_line > method_line:
                    raw_content = "\n".join(lines[method_line : end_line + 1])

                    chunks.append(
                        (
                            raw_content.strip(),
                            method_line + 1,
                            end_line + 1,
                            f"plsql_{method['type']}",
                            method["name"],
                        )
                    )

    return chunks
