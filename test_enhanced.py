"""Test script to verify the enhanced IFS Cloud MCP Server functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer
from ifs_cloud_mcp_server.parsers import IFSFileParser
from ifs_cloud_mcp_server.config import ConfigManager


async def test_parser():
    """Test the file parser with sample data."""
    print("Testing IFS File Parser...")

    parser = IFSFileParser()

    # Test with sample entity content
    entity_sample = """<?xml version="1.0" encoding="UTF-8"?>
<ENTITY xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="urn:ifsworld-com:schemas:entity_entity">
   <NAME>CustomerOrder</NAME>
   <COMPONENT>ORDER</COMPONENT>
   <BASED_ON>BusinessObject</BASED_ON>
   <ATTRIBUTES>
      <ATTRIBUTE>
         <NAME>OrderNo</NAME>
         <DATATYPE>TEXT</DATATYPE>
      </ATTRIBUTE>
   </ATTRIBUTES>
</ENTITY>"""

    result = parser.parse(entity_sample, ".entity")
    print(f"Entity parse result: {result}")

    # Test with sample projection content
    projection_sample = """projection CustomerOrderHandling;
component ORDER;
layer Core;
description "Customer Order Handling";

entityset CustomerOrderSet for CustomerOrder {
   context Company(Company);
}

entity CustomerOrder {
   attribute OrderNo Text;
   reference CompanyRef(Company) to Company(Company);
}"""

    result = parser.parse(projection_sample, ".projection")
    print(f"Projection parse result: {result}")


async def test_indexer():
    """Test the indexer with sample files."""
    print("\nTesting Indexer...")

    # Create a temporary index
    indexer = IFSCloudTantivyIndexer("./test_index", create_new=True)

    # Test with a sample file from _work directory
    work_dir = Path("_work")
    if work_dir.exists():
        sample_files = list(work_dir.rglob("*.entity"))[:3]  # Get first 3 entity files

        for file_path in sample_files:
            print(f"Indexing: {file_path}")
            success = await indexer.index_file(file_path)
            print(f"Success: {success}")

        # Commit changes
        indexer.commit()

        # Test search
        print("\nTesting search...")
        results = indexer.search("CustomerOrder", limit=5)
        print(f"Search results: {len(results)} found")
        for result in results:
            print(f"  - {result.path} (score: {result.score:.2f})")

    indexer.close()


async def test_config():
    """Test the configuration manager."""
    print("\nTesting Configuration Manager...")

    config = ConfigManager()

    # Test setting and getting path
    test_path = Path("_work").absolute()
    if test_path.exists():
        success = config.set_core_codes_path(str(test_path))
        print(f"Set path success: {success}")

        retrieved_path = config.get_core_codes_path()
        print(f"Retrieved path: {retrieved_path}")
    else:
        print("_work directory not found, skipping path test")


async def main():
    """Run all tests."""
    print("IFS Cloud MCP Server - Enhanced Functionality Test")
    print("=" * 50)

    try:
        await test_parser()
        await test_indexer()
        await test_config()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
