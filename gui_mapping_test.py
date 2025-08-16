#!/usr/bin/env python3
"""
Test GUI-to-backend mapping functionality with the enhanced indexer.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def test_gui_mappings():
    """Test that GUI mappings work correctly."""
    print("🧪 Testing GUI-to-backend mapping functionality")
    print("=" * 60)

    # Setup index path
    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found at {index_path}")
        return False

    # Initialize searcher
    searcher = IFSCloudTantivyIndexer(index_path=index_path)

    # Test queries that should use GUI mappings
    test_cases = [
        {
            "query": "employee file",  # Should map to CompanyPerson
            "description": "GUI label from sample mappings",
            "expected_entities": ["CompanyPerson", "Person", "Employee"],
        },
        {
            "query": "customer order line",  # Should use entity synonyms
            "description": "GUI pattern with synonyms",
            "expected_entities": ["CustomerOrderLine", "OrderLine", "Line"],
        },
        {
            "query": "purchase order",  # Should use entity synonyms
            "description": "Common entity with synonyms",
            "expected_entities": ["PurchaseOrder", "PurchOrder"],
        },
    ]

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)

        try:
            results = searcher.search(test_case["query"], limit=5)

            if results:
                print(f"📊 Results ({len(results)} found):")
                for j, result in enumerate(results, 1):
                    file_name = Path(result.path).name
                    print(f"  {j}. {file_name} (Score: {result.score:.1f})")

                # Check if any expected entities are in top results
                found_expected = False
                for expected_entity in test_case["expected_entities"]:
                    for result in results[:3]:  # Check top 3
                        if expected_entity.lower() in Path(result.path).stem.lower():
                            found_expected = True
                            break
                    if found_expected:
                        break

                if found_expected:
                    print("✅ SUCCESS: Found expected entity in top 3!")
                    success_count += 1
                else:
                    print("⚠️  No expected entities found in top 3")
            else:
                print("❌ No results returned")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print(f"GUI MAPPING TEST SUMMARY")
    print("=" * 60)
    print(
        f"✅ Successful mappings: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)"
    )

    # Test loading GUI mappings directly
    print(f"\n🔧 Testing GUI mappings data:")
    if hasattr(searcher, "_gui_mappings") and searcher._gui_mappings:
        gui_mappings = searcher._gui_mappings
        print(
            f"  • GUI-to-entity mappings: {len(gui_mappings.get('gui_to_entity', {}))}"
        )
        print(
            f"  • Entity synonym mappings: {len(gui_mappings.get('entity_synonyms', {}))}"
        )
        print(
            f"  • GUI-to-projection mappings: {len(gui_mappings.get('gui_to_projection', {}))}"
        )

        # Show sample mappings
        if gui_mappings.get("gui_to_entity"):
            print(f"\n📋 Sample GUI mappings:")
            for gui_label, entities in list(gui_mappings["gui_to_entity"].items())[:3]:
                print(f"  • '{gui_label}' → {entities}")
    else:
        print("  ⚠️  No GUI mappings loaded")


if __name__ == "__main__":
    test_gui_mappings()
