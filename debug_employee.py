#!/usr/bin/env python3
"""Debug the employee name validation query specifically."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def debug_employee_validation():
    """Debug why employee name validation doesn't find Employee.plsql first."""

    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    query = "employee name validation"
    results = indexer.search_deduplicated(query, limit=10)

    print(f"🔍 Query: '{query}'")
    print(f"📊 Found {len(results)} results:")

    employee_plsql_found = False
    employee_position = None

    for i, result in enumerate(results, 1):
        marker = "🎯" if "Employee.plsql" in result.name else f"{i}."
        if "Employee.plsql" in result.name:
            employee_plsql_found = True
            employee_position = i

        line_info = f"({result.line_count} lines)" if result.line_count else ""
        module_info = f"[{result.module}]" if result.module else ""
        print(
            f"  {marker} {result.name} - Score: {result.score:.1f} {line_info} {module_info}"
        )

        # Show entity name if available
        if result.entity_name:
            print(f"      Entity: {result.entity_name}")

    if employee_plsql_found:
        print(f"\n✅ Employee.plsql found at position {employee_position}")
        if employee_position > 1:
            print(
                f"💡 Analysis: Employee.plsql needs higher ranking for validation queries"
            )
    else:
        print(f"\n❌ Employee.plsql not found in top 10 results")

    # Show what terms the query expands to
    indexer_terms = indexer._parse_query_terms(query)
    print(f"\n🔧 Query terms expanded to: {indexer_terms}")


if __name__ == "__main__":
    debug_employee_validation()
