#!/usr/bin/env python3
"""Test the specific query that was failing."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


def test_global_item_query():
    """Test the specific query that was causing the error."""

    # Setup
    data_dir = (
        Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        / "ifs_cloud_mcp_server"
    )
    index_path = data_dir / "indexes" / "25.1.0"

    if not index_path.exists():
        print(f"❌ Index not found at {index_path}")
        return False

    print(f"🔍 Testing query: 'global item creation rules'")
    indexer = IFSCloudTantivyIndexer(index_path=index_path)

    try:
        results = indexer.search("global item creation rules", limit=5)

        if results:
            print(f"✅ SUCCESS: Found {len(results)} results")
            print("📊 Results:")
            for i, result in enumerate(results, 1):
                file_name = Path(result.path).name
                print(f"  {i}. {file_name} (Score: {result.score:.1f})")

            # Check if PartCatalog.plsql is in results
            found_partcatalog = False
            for result in results:
                if "partcatalog" in Path(result.path).stem.lower():
                    found_partcatalog = True
                    break

            if found_partcatalog:
                print("🎯 PartCatalog.plsql found in results!")
            else:
                print("⚠️  PartCatalog.plsql not found, but query executed successfully")

        else:
            print("⚠️  No results returned (but no error)")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_global_item_query()
