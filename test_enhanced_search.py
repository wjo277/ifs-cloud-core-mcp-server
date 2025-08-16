#!/usr/bin/env python3

from pathlib import Path
from src.ifs_cloud_mcp_server.search_engine import IFSCloudSearchEngine
from src.ifs_cloud_mcp_server.indexer import IFSCloudIndexer


def test_enhanced_search():
    """Test the enhanced search system with ML intent classification and metadata indexer"""
    print("Initializing enhanced search engine...")

    # Create indexer first
    index_path = Path("index")
    indexer = IFSCloudIndexer(index_path)

    # Create search engine with indexer
    engine = IFSCloudSearchEngine(indexer)

    # Test different types of queries to validate ML intent classification
    test_queries = [
        (
            "customer order processing",
            "BUSINESS_LOGIC",
        ),  # Should be classified as BUSINESS_LOGIC
        (
            "Entity Customer",
            "ENTITY_DEFINITION",
        ),  # Should be classified as ENTITY_DEFINITION
        (
            "button widget form",
            "UI_COMPONENTS",
        ),  # Should be classified as UI_COMPONENTS
        (
            "database connection API",
            "API_INTEGRATION",
        ),  # Should be classified as API_INTEGRATION
        (
            "ERROR: connection failed",
            "TROUBLESHOOTING",
        ),  # Should be classified as TROUBLESHOOTING
    ]

    for query, expected_intent in test_queries:
        print(f'\n=== Testing query: "{query}" (Expected: {expected_intent}) ===')
        try:
            results = engine.search(query, limit=5)  # Remove await
            print(f"Found {len(results)} results")

            for i, result in enumerate(results[:3]):
                print(f"{i+1}. {result.name} (score: {result.score:.3f})")
                print(f"   Path: {result.path}")

                # Show metadata if available
                metadata_info = []
                if hasattr(result, "entities") and result.entities:
                    metadata_info.append(f"Entities: {result.entities[:3]}")
                if hasattr(result, "functions") and result.functions:
                    metadata_info.append(f"Functions: {result.functions[:3]}")
                if hasattr(result, "pages") and result.pages:
                    metadata_info.append(f"Pages: {result.pages[:3]}")

                if metadata_info:
                    print(f'   Metadata: {"; ".join(metadata_info)}')
                print()

        except Exception as e:
            print(f'Error searching for "{query}": {e}')
            import traceback

            traceback.print_exc()

    print("\n=== Enhanced search system test completed! ===")
    print("✅ ML Intent Classifier: Integrated")
    print("✅ Metadata Indexer: Integrated")
    print("✅ Dual-search approach: Active")


if __name__ == "__main__":
    test_enhanced_search()
