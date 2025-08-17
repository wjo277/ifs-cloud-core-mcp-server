#!/usr/bin/env python3
"""
Analyze and show examples of high-complexity functions (score 15+)
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ifs_cloud_mcp_server.semantic_search.ai_summarizer import AISummarizer
from ifs_cloud_mcp_server.semantic_search.data_structures import CodeChunk


def analyze_high_complexity_examples():
    """Find and analyze examples of functions with complexity score 15+"""

    # Initialize AI summarizer to access the complexity calculation
    summarizer = AISummarizer()

    # Look in _work directory for some sample files
    work_dir = Path("_work")
    if not work_dir.exists():
        print("❌ _work directory not found")
        return

    high_complexity_examples = []
    analyzed_count = 0

    # Analyze files from a few different modules
    sample_modules = ["accrul", "adcom", "appsrv", "fndwf", "mwo"]

    for module in sample_modules:
        module_dir = work_dir / module / "source" / module / "database"
        if not module_dir.exists():
            print(f"⚠️ Module {module} not found, skipping")
            continue

        print(f"\n🔍 Analyzing module: {module}")

        # Look at first few .plsql files in each module
        plsql_files = list(module_dir.glob("*.plsql"))[
            :10
        ]  # Just first 10 files per module

        for plsql_file in plsql_files:
            try:
                with open(plsql_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Split content into function/procedure chunks (simplified)
                chunks = split_into_chunks(content, str(plsql_file))

                for chunk in chunks:
                    analyzed_count += 1

                    # Calculate complexity score
                    content_lower = chunk.processed_content.lower()
                    lines = chunk.processed_content.split("\n")
                    complexity_score = summarizer._calculate_complexity(
                        content_lower, lines
                    )

                    # Keep examples with score 15+
                    if complexity_score >= 15:
                        should_skip = summarizer._should_skip_chunk(chunk)
                        high_complexity_examples.append(
                            {
                                "file": plsql_file.name,
                                "module": module,
                                "function_name": getattr(
                                    chunk, "function_name", "unknown"
                                ),
                                "complexity_score": complexity_score,
                                "passes_filter": not should_skip,
                                "content_preview": chunk.processed_content[:500],
                                "full_content": chunk.processed_content,
                            }
                        )

                        if len(high_complexity_examples) >= 20:  # Limit to 20 examples
                            break

                if len(high_complexity_examples) >= 20:
                    break

            except Exception as e:
                print(f"❌ Error processing {plsql_file}: {e}")
                continue

        if len(high_complexity_examples) >= 20:
            break

    # Sort by complexity score (highest first)
    high_complexity_examples.sort(key=lambda x: x["complexity_score"], reverse=True)

    print(f"\n🎯 Analysis Results:")
    print(f"📊 Analyzed {analyzed_count} code chunks")
    print(
        f"🔥 Found {len(high_complexity_examples)} examples with complexity score 15+"
    )
    print("=" * 80)

    # Show top examples
    for i, example in enumerate(high_complexity_examples[:10], 1):
        print(f"\n#{i} COMPLEXITY SCORE: {example['complexity_score']}")
        print(f"📁 File: {example['module']}/{example['file']}")
        print(f"🔧 Function: {example['function_name']}")
        print(f"✅ Passes Filter: {example['passes_filter']}")
        print(f"📝 Content Preview:")
        print("-" * 60)
        print(example["content_preview"])
        if len(example["content_preview"]) >= 500:
            print("... [truncated]")
        print("-" * 60)

        # Show what makes it complex
        analyze_complexity_factors(example["full_content"], example["complexity_score"])
        print("=" * 80)


def split_into_chunks(content: str, file_path: str) -> list:
    """Simplified chunk extraction - look for PROCEDURE and FUNCTION declarations"""
    chunks = []
    lines = content.split("\n")
    current_chunk = []
    current_function = None
    indent_level = 0
    in_function = False

    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip().upper()

        # Look for function/procedure starts
        if (
            line_stripped.startswith("PROCEDURE ")
            or line_stripped.startswith("FUNCTION ")
        ) and not line_stripped.endswith(";"):
            # Save previous chunk if exists
            if current_chunk and in_function:
                chunk_content = "\n".join(current_chunk)
                if len(chunk_content.strip()) > 50:  # Only keep substantial chunks
                    chunk = CodeChunk(
                        chunk_id=f"{file_path}:{current_function}",
                        file_path=file_path,
                        start_line=line_num - len(current_chunk),
                        end_line=line_num - 1,
                        raw_content=chunk_content,
                        processed_content=chunk_content,
                        chunk_type="function",
                        language="plsql",
                    )
                    chunk.function_name = current_function or "unknown"
                    chunks.append(chunk)

            # Start new chunk
            current_chunk = [line]
            current_function = extract_function_name(line)
            in_function = True
            indent_level = 0

        elif in_function:
            current_chunk.append(line)

            # Track BEGIN/END to know when function ends
            if "BEGIN" in line_stripped:
                indent_level += 1
            elif "END" in line_stripped and ";" in line_stripped:
                indent_level -= 1
                if indent_level <= 0:
                    # End of function
                    chunk_content = "\n".join(current_chunk)
                    if len(chunk_content.strip()) > 50:
                        chunk = CodeChunk(
                            chunk_id=f"{file_path}:{current_function}",
                            file_path=file_path,
                            start_line=line_num - len(current_chunk) + 1,
                            end_line=line_num,
                            raw_content=chunk_content,
                            processed_content=chunk_content,
                            chunk_type="function",
                            language="plsql",
                        )
                        chunk.function_name = current_function or "unknown"
                        chunks.append(chunk)

                    current_chunk = []
                    current_function = None
                    in_function = False

    # Handle last chunk
    if current_chunk and in_function and len("\n".join(current_chunk).strip()) > 50:
        chunk_content = "\n".join(current_chunk)
        chunk = CodeChunk(
            chunk_id=f"{file_path}:{current_function}",
            file_path=file_path,
            start_line=len(lines) - len(current_chunk) + 1,
            end_line=len(lines),
            raw_content=chunk_content,
            processed_content=chunk_content,
            chunk_type="function",
            language="plsql",
        )
        chunk.function_name = current_function or "unknown"
        chunks.append(chunk)

    return chunks


def extract_function_name(line: str) -> str:
    """Extract function/procedure name from declaration line"""
    line = line.strip()
    if line.upper().startswith("PROCEDURE "):
        name_part = line[10:].split("(")[0].split()[0]
        return name_part.strip()
    elif line.upper().startswith("FUNCTION "):
        name_part = line[9:].split("(")[0].split()[0]
        return name_part.strip()
    return "unknown"


def analyze_complexity_factors(content: str, score: int):
    """Analyze what contributes to the complexity score"""
    content_lower = content.lower()

    factors = []

    # Control structures
    control_counts = {
        "LOOP": content_lower.count("loop") * 2,
        "WHILE": content_lower.count("while") * 2,
        "FOR": content_lower.count("for") * 2,
        "IF": content_lower.count("if") * 1,
        "CASE": content_lower.count("case") * 2,
        "WHEN": content_lower.count("when") * 1,
        "CURSOR": content_lower.count("cursor") * 2,
        "BULK COLLECT": content_lower.count("bulk collect") * 2,
        "FORALL": content_lower.count("forall") * 2,
    }

    # Exception handling
    exception_counts = {
        "EXCEPTION": content_lower.count("exception") * 2,
        "RAISE": content_lower.count("raise") * 2,
        "PRAGMA": content_lower.count("pragma exception_init") * 2,
    }

    # Database operations
    db_operations = ["select", "insert", "update", "delete", "merge"]
    db_count = sum(1 for op in db_operations if op in content_lower)
    if db_count > 2:
        factors.append(f"DB Operations: {db_count} (+2)")
    elif db_count > 0:
        factors.append(f"DB Operations: {db_count} (+1)")

    # Transaction control
    transaction_counts = {
        "COMMIT": content_lower.count("commit") * 2,
        "ROLLBACK": content_lower.count("rollback") * 2,
        "SAVEPOINT": content_lower.count("savepoint") * 2,
    }

    # Add significant factors
    for name, count in {
        **control_counts,
        **exception_counts,
        **transaction_counts,
    }.items():
        if count > 0:
            factors.append(
                f"{name}: {count // (2 if name in ['LOOP', 'WHILE', 'FOR', 'CASE', 'CURSOR', 'BULK COLLECT', 'FORALL', 'EXCEPTION', 'RAISE', 'PRAGMA', 'COMMIT', 'ROLLBACK', 'SAVEPOINT'] else 1)} occurrences (+{count})"
            )

    # Line count factor
    lines = content.split("\n")
    non_comment_lines = [
        line for line in lines if line.strip() and not line.strip().startswith("--")
    ]
    if len(non_comment_lines) > 100:
        factors.append(f"Lines: {len(non_comment_lines)} (+2)")
    elif len(non_comment_lines) > 50:
        factors.append(f"Lines: {len(non_comment_lines)} (+1)")

    # Function calls
    function_call_count = content_lower.count("(") - content_lower.count("--")
    if function_call_count > 10:
        factors.append(f"Function calls: {function_call_count} (+2)")
    elif function_call_count > 5:
        factors.append(f"Function calls: {function_call_count} (+1)")

    print(f"🧮 Complexity Factors (Total Score: {score}):")
    for factor in factors[:8]:  # Show top 8 factors
        print(f"   • {factor}")


if __name__ == "__main__":
    analyze_high_complexity_examples()
