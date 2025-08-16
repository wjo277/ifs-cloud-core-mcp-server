#!/usr/bin/env python3
"""Quick validation of the main search improvement: ExpenseHeader.plsql should be top result for authorization queries."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def quick_test():
    """Test that ExpenseHeader.plsql is now top result for expense authorization queries."""
    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print("❌ Index not found")
        return False

    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    # The key test case from the analysis
    results = indexer.search_deduplicated("expense sheet authorization", limit=5)

    if not results:
        print("❌ No results")
        return False

    top_result = results[0]
    print(f"🔍 Query: 'expense sheet authorization'")
    print(f"🏆 Top Result: {top_result.name} (Score: {top_result.score:.2f})")
    print(f"📄 Type: {top_result.type} | Lines: {top_result.line_count}")

    # Success criteria: ExpenseHeader.plsql should be top result with high score
    if "ExpenseHeader.plsql" in top_result.name and top_result.score > 100:
        print("✅ SUCCESS: ExpenseHeader.plsql is top result with high score!")
        print(
            "✅ IMPROVEMENT VERIFIED: Authorization queries now find business logic files first"
        )
        return True
    else:
        print(f"⚠️  Expected ExpenseHeader.plsql as top result, got {top_result.name}")
        return False


if __name__ == "__main__":
    success = quick_test()
    print(
        f"\n{'🎉 SEARCH IMPROVEMENTS WORKING' if success else '❌ IMPROVEMENTS NEED REVIEW'}"
    )
    sys.exit(0 if success else 1)
