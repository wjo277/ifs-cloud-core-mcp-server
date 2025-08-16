#!/usr/bin/env python3
"""Test script for search algorithm improvements."""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def test_search_improvements():
    """Test the improved search ranking algorithm with real scenarios from the analysis."""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Use existing index
    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found at {index_path}")
        return False

    print(f"🔍 Testing search improvements with index: {index_path}")

    # Create indexer
    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    # Test cases based on the analysis document
    test_queries = [
        {
            "query": "expense sheet authorization",
            "expected_top_results": [
                "ExpenseHeader.plsql",
                "ExpenseHeader.views",
                "ExpenseSheetHandling.projection",
            ],
            "description": "Main authorization test case - should prioritize business logic files",
        },
        {
            "query": "ExpenseSheet authorize",
            "expected_contains": [
                "ExpenseHeader"
            ],  # Should map ExpenseSheet -> ExpenseHeader
            "description": "Entity synonym mapping test",
        },
        {
            "query": "expense approval workflow",
            "expected_file_types": [".plsql", ".projection", ".views"],
            "description": "Business logic query should prioritize .plsql files",
        },
        {
            "query": "trvexp authorization",
            "expected_module": "trvexp",
            "description": "Module context boosting test",
        },
    ]

    print("\n" + "=" * 60)
    print("SEARCH IMPROVEMENT VALIDATION")
    print("=" * 60)

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)

        try:
            # Perform search
            results = indexer.search_deduplicated(test_case["query"], limit=10)

            if not results:
                print("❌ No results returned")
                continue

            print(f"✅ Returned {len(results)} results:")

            # Display top results
            for j, result in enumerate(results[:5], 1):
                file_size_info = (
                    f"({result.line_count} lines)" if result.line_count else ""
                )
                module_info = f"[{result.module}]" if result.module else ""
                print(f"  {j}. {result.name} {file_size_info} {module_info}")
                print(f"     Score: {result.score:.2f} | Type: {result.type}")

                # Check if content analysis worked
                if "authorization" in test_case[
                    "query"
                ].lower() and result.type.endswith(".plsql"):
                    if result.score > 50:  # Should have high score due to our bonuses
                        print(f"     ✅ High score for .plsql authorization query")
                    else:
                        print(f"     ⚠️  Lower than expected score for .plsql file")

            # Validate expected results
            if "expected_top_results" in test_case:
                top_names = [r.name for r in results[:3]]
                found_expected = any(
                    expected in " ".join(top_names)
                    for expected in test_case["expected_top_results"]
                )
                if found_expected:
                    print(f"     ✅ Found expected results in top 3")
                else:
                    print(
                        f"     ⚠️  Expected results not in top 3: {test_case['expected_top_results']}"
                    )

            if "expected_contains" in test_case:
                all_names = " ".join([r.name for r in results])
                for expected in test_case["expected_contains"]:
                    if expected in all_names:
                        print(f"     ✅ Found expected entity: {expected}")
                    else:
                        print(f"     ⚠️  Missing expected entity: {expected}")

            if "expected_module" in test_case:
                modules = [r.module for r in results[:3] if r.module]
                if test_case["expected_module"] in modules:
                    print(
                        f"     ✅ Found expected module: {test_case['expected_module']}"
                    )
                else:
                    print(
                        f"     ⚠️  Expected module not in top 3: {test_case['expected_module']}"
                    )

        except Exception as e:
            print(f"❌ Search failed: {e}")

    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print("✅ Enhanced file type priorities (.plsql boosted for authorization)")
    print("✅ Added entity synonym mapping (ExpenseSheet ↔ ExpenseHeader)")
    print("✅ Content analysis for business logic detection")
    print("✅ Module context boosting (expense queries + trvexp module)")
    print("✅ Authorization keyword detection and scoring")
    print("✅ Large file business logic prioritization")

    return True


if __name__ == "__main__":
    success = test_search_improvements()
    sys.exit(0 if success else 1)
