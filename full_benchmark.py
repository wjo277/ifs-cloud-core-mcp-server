#!/usr/bin/env python3
"""Full benchmark test for all 10 original search queries with GUI mappings."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def run_full_benchmark():
    """Test all 10 original benchmark queries with proper GUI-to-backend mappings."""

    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found")
        return False

    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    # Complete benchmark queries with GUI-aware mappings
    benchmark_queries = [
        ("project transaction approval", "ProjectTransaction.plsql"),
        ("project posting", "ProjectTransPosting.plsql"),
        (
            "employee name validation",
            "CompanyPerson.plsql",
        ),  # GUI: "Employee File" -> CompanyPerson.plsql
        ("activity creation", "Activity.plsql"),
        ("expense sheet lines", "ExpenseDetail.plsql"),
        ("expense sheet project connection", "ExpenseHeader.plsql"),
        ("payment modification authorization", "PaymentAddress.plsql"),
        ("global item creation rules", "PartCatalog.plsql"),
        ("per-company item stocking rules", "InventoryPart.plsql"),
        ("per-company item purchase rules", "PurchasePart.plsql"),
    ]

    print("\n" + "=" * 80)
    print("COMPLETE BENCHMARK TEST - GUI-AWARE SEARCH RANKING")
    print("=" * 80)

    successes = 0
    partials = 0
    total_tests = len(benchmark_queries)

    for i, (query, expected) in enumerate(benchmark_queries, 1):
        print(f"\n🧪 Test {i}/10: {query}")
        print(f"Expected: {expected}")
        print("-" * 60)

        try:
            results = indexer.search_deduplicated(query, limit=5)

            if not results:
                print("❌ No results")
                continue

            # Check position of expected result
            found_position = None
            for pos, result in enumerate(results, 1):
                if expected.lower() in result.name.lower():
                    found_position = pos
                    break

            # Display results
            for j, result in enumerate(results, 1):
                marker = "🎯" if j == found_position else f"{j}."
                score_info = f"Score: {result.score:.1f}"
                line_info = f"({result.line_count} lines)" if result.line_count else ""
                print(f"  {marker} {result.name} - {score_info} {line_info}")

            # Evaluate result
            if found_position == 1:
                print(f"✅ PERFECT: {expected} at position 1!")
                successes += 1
            elif found_position and found_position <= 3:
                print(f"⚠️  GOOD: {expected} at position {found_position} (top 3)")
                partials += 1
            elif found_position:
                print(f"🔍 FOUND: {expected} at position {found_position}")
            else:
                print(f"❌ MISSED: {expected} not in top 5")

        except Exception as e:
            print(f"❌ Search failed: {e}")

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)

    perfect_rate = successes / total_tests * 100
    good_rate = (successes + partials) / total_tests * 100

    print(
        f"🎯 Perfect matches (position 1): {successes}/{total_tests} ({perfect_rate:.1f}%)"
    )
    print(
        f"📊 Good matches (top 3): {successes + partials}/{total_tests} ({good_rate:.1f}%)"
    )

    print(f"\n🚀 SEARCH ALGORITHM IMPROVEMENTS:")
    print(f"   ✅ GUI-to-Backend mapping (Employee File -> CompanyPerson.plsql)")
    print(f"   ✅ Compound entity patterns (project + posting -> ProjectTransPosting)")
    print(f"   ✅ Action-entity recognition (activity + creation -> Activity)")
    print(f"   ✅ Business logic prioritization (.plsql files boosted)")
    print(f"   ✅ Content analysis for authorization/validation patterns")
    print(f"   ✅ Module context awareness for domain alignment")

    # Success if 80%+ perfect matches or 90%+ good matches
    success = perfect_rate >= 80 or good_rate >= 90

    print(f"\n🏆 BENCHMARK STATUS: {'PASSED' if success else 'NEEDS IMPROVEMENT'}")
    return success


if __name__ == "__main__":
    success = run_full_benchmark()
    sys.exit(0 if success else 1)
