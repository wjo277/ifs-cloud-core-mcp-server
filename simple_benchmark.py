#!/usr/bin/env python3
"""Simple test of one benchmark query."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def quick_benchmark_test():
    """Test one benchmark query to see current performance."""

    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found")
        return False

    print(
        f"🔍 Testing: 'project transaction approval' -> expect ProjectTransaction.plsql"
    )
    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    results = indexer.search_deduplicated("project transaction approval", limit=5)

    if not results:
        print("❌ No results")
        return False

    print(f"\n📊 Results:")
    expected_found = False
    for i, result in enumerate(results, 1):
        marker = "🎯" if "ProjectTransaction.plsql" in result.name else f"{i}."
        if "ProjectTransaction.plsql" in result.name:
            expected_found = True
            expected_position = i
        print(
            f"  {marker} {result.name} (Score: {result.score:.1f}) ({result.line_count} lines)"
        )

    if expected_found:
        print(f"✅ Found ProjectTransaction.plsql at position {expected_position}")
        return expected_position == 1
    else:
        print(f"❌ ProjectTransaction.plsql not found in top 5")
        return False


if __name__ == "__main__":
    success = quick_benchmark_test()
    print(f"\n{'✅ SUCCESS' if success else '🔧 NEEDS IMPROVEMENT'}")
    sys.exit(0 if success else 1)
