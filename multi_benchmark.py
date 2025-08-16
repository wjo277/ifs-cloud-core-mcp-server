#!/usr/bin/env python3
"""Test multiple benchmark queries to validate improvements."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def test_multiple_benchmarks():
    """Test several benchmark queries."""

    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found")
        return False

    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    test_cases = [
        ("project transaction approval", "ProjectTransaction.plsql"),
        ("project posting", "ProjectTransPosting.plsql"),
        (
            "employee name validation",
            "CompanyPerson.plsql",
        ),  # GUI: "Employee File" -> CompanyPerson.plsql
        ("activity creation", "Activity.plsql"),
        ("expense sheet lines", "ExpenseDetail.plsql"),
    ]

    results = []

    for query, expected in test_cases:
        print(f"\n🔍 Testing: '{query}' -> expect {expected}")

        search_results = indexer.search_deduplicated(query, limit=5)

        if not search_results:
            print("❌ No results")
            results.append(False)
            continue

        found_position = None
        for i, result in enumerate(search_results, 1):
            if expected.lower() in result.name.lower():
                found_position = i
                break

        print(
            f"📊 Top result: {search_results[0].name} (Score: {search_results[0].score:.1f})"
        )

        if found_position == 1:
            print(f"✅ SUCCESS - {expected} found at position 1!")
            results.append(True)
        elif found_position:
            print(f"⚠️  PARTIAL - {expected} found at position {found_position}")
            results.append(False)
        else:
            print(f"❌ MISSED - {expected} not found in top 5")
            results.append(False)

    success_rate = sum(results) / len(results)
    print(f"\n📊 Success Rate: {sum(results)}/{len(results)} ({success_rate*100:.1f}%)")

    return success_rate >= 0.8  # 80% success threshold


if __name__ == "__main__":
    success = test_multiple_benchmarks()
    print(f"\n{'🎉 BENCHMARKS PASSED' if success else '🔧 MORE IMPROVEMENTS NEEDED'}")
    sys.exit(0 if success else 1)
