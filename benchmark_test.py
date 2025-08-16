#!/usr/bin/env python3
"""Benchmark test for additional search query patterns."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def run_benchmark_tests():
    """Test the search algorithm against the new benchmark queries."""

    # Setup
    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found at {index_path}")
        return False

    print(f"🔍 Running benchmark tests with index: {index_path}")
    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    # Benchmark test cases from user
    benchmark_queries = [
        {
            "query": "project transaction approval",
            "expected": "ProjectTransaction.plsql",
            "pattern": "entity + action + business_logic",
        },
        {
            "query": "project posting",
            "expected": "ProjectTransPosting.plsql",
            "pattern": "entity + action",
        },
        {
            "query": "employee name validation",
            "expected": "Employee.plsql",
            "pattern": "entity + property + validation",
        },
        {
            "query": "activity creation",
            "expected": "Activity.plsql",
            "pattern": "entity + lifecycle_action",
        },
        {
            "query": "expense sheet lines",
            "expected": "ExpenseDetail.plsql",
            "pattern": "master_entity + detail_reference",
        },
        {
            "query": "expense sheet project connection",
            "expected": "ExpenseHeader.plsql",
            "pattern": "entity + integration_reference",
        },
        {
            "query": "payment modification authorization",
            "expected": "PaymentAddress.plsql",
            "pattern": "entity + modification + authorization",
        },
        {
            "query": "global item creation rules",
            "expected": "PartCatalog.plsql",
            "pattern": "scope + entity + lifecycle + rules",
        },
        {
            "query": "per-company item stocking rules",
            "expected": "InventoryPart.plsql",
            "pattern": "scope + entity + domain_action + rules",
        },
        {
            "query": "per-company item purchase rules",
            "expected": "PurchasePart.plsql",
            "pattern": "scope + entity + domain_action + rules",
        },
    ]

    print("\n" + "=" * 80)
    print("BENCHMARK TEST RESULTS")
    print("=" * 80)

    successes = 0
    total_tests = len(benchmark_queries)

    for i, test in enumerate(benchmark_queries, 1):
        print(f"\n🧪 Test {i}: {test['pattern']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        print("-" * 60)

        try:
            results = indexer.search_deduplicated(test["query"], limit=5)

            if not results:
                print("❌ No results returned")
                continue

            # Check if expected result is in top 5
            found_position = None
            for pos, result in enumerate(results, 1):
                if test["expected"].lower() in result.name.lower():
                    found_position = pos
                    break

            # Display results
            print(f"📊 Results ({len(results)} found):")
            for j, result in enumerate(results, 1):
                marker = "🎯" if j == found_position else f"{j}."
                score_info = f"(Score: {result.score:.1f})"
                line_info = f"({result.line_count} lines)" if result.line_count else ""
                print(f"  {marker} {result.name} {score_info} {line_info}")

            # Evaluate result
            if found_position == 1:
                print(f"✅ SUCCESS: {test['expected']} found at position 1!")
                successes += 1
            elif found_position and found_position <= 3:
                print(
                    f"⚠️  PARTIAL: {test['expected']} found at position {found_position} (top 3)"
                )
            elif found_position:
                print(
                    f"🔍 FOUND: {test['expected']} found at position {found_position}"
                )
            else:
                print(f"❌ MISSED: {test['expected']} not found in top 5 results")

        except Exception as e:
            print(f"❌ Search failed: {e}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"✅ Perfect matches (position 1): {successes}/{total_tests} ({successes/total_tests*100:.1f}%)"
    )

    # Identify patterns that need improvement
    print(f"\n📋 PATTERN ANALYSIS:")
    patterns_tested = {}
    for test in benchmark_queries:
        pattern = test["pattern"]
        if pattern not in patterns_tested:
            patterns_tested[pattern] = []
        patterns_tested[pattern].append(test["query"])

    for pattern, queries in patterns_tested.items():
        print(f"  • {pattern}: {len(queries)} queries")

    return successes >= total_tests * 0.7  # 70% success rate threshold


if __name__ == "__main__":
    success = run_benchmark_tests()
    print(f"\n{'🎉 BENCHMARK PASSED' if success else '🔧 IMPROVEMENTS NEEDED'}")
    sys.exit(0 if success else 1)
