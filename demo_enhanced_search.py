#!/usr/bin/env python3
"""
Demo script showing IFS Cloud metadata-enhanced search capabilities
"""

import asyncio
import logging
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ifs_cloud_mcp_server.indexer import IFSCloudIndexer
from ifs_cloud_mcp_server.metadata_extractor import (
    MetadataManager,
    DatabaseMetadataExtractor,
)
from ifs_cloud_mcp_server.extract_metadata import MCPMetadataExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_metadata():
    """Create sample metadata for demonstration"""
    sample_results = {
        "logical_units": [
            {
                "MODULE": "ORDER",
                "LU_NAME": "CustomerOrder",
                "LU_PROMPT": "Customer Order",
                "BASE_TABLE": "CUSTOMER_ORDER_TAB",
                "BASE_VIEW": "CUSTOMER_ORDER",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": "",
            },
            {
                "MODULE": "ORDER",
                "LU_NAME": "CustomerOrderLine",
                "LU_PROMPT": "Customer Order Line",
                "BASE_TABLE": "CUSTOMER_ORDER_LINE_TAB",
                "BASE_VIEW": "CUSTOMER_ORDER_LINE",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": "",
            },
            {
                "MODULE": "PERSON",
                "LU_NAME": "CompanyPerson",
                "LU_PROMPT": "Employee",
                "BASE_TABLE": "COMPANY_PERSON_TAB",
                "BASE_VIEW": "COMPANY_PERSON",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": "",
            },
            {
                "MODULE": "INVOIC",
                "LU_NAME": "CustomerInvoice",
                "LU_PROMPT": "Customer Invoice",
                "BASE_TABLE": "INVOICE_TAB",
                "BASE_VIEW": "CUSTOMER_INVOICE",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": "",
            },
            {
                "MODULE": "PURCH",
                "LU_NAME": "PurchaseOrder",
                "LU_PROMPT": "Purchase Order",
                "BASE_TABLE": "PURCHASE_ORDER_TAB",
                "BASE_VIEW": "PURCHASE_ORDER",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": "",
            },
        ],
        "modules": [
            {"MODULE": "ORDER", "LU_COUNT": 505, "TABLE_COUNT": 283, "LU_TYPES": "L"},
            {"MODULE": "PERSON", "LU_COUNT": 361, "TABLE_COUNT": 225, "LU_TYPES": "L"},
            {"MODULE": "INVOIC", "LU_COUNT": 311, "TABLE_COUNT": 153, "LU_TYPES": "L"},
            {"MODULE": "PURCH", "LU_COUNT": 366, "TABLE_COUNT": 175, "LU_TYPES": "L"},
        ],
        "domain_mappings": [
            {
                "LU_NAME": "CustomerOrder",
                "PACKAGE_NAME": "CUSTOMER_ORDER_API",
                "DB_VALUE": "PLANNED",
                "CLIENT_VALUE": "Planned",
            },
            {
                "LU_NAME": "CustomerOrder",
                "PACKAGE_NAME": "CUSTOMER_ORDER_API",
                "DB_VALUE": "RELEASED",
                "CLIENT_VALUE": "Released",
            },
            {
                "LU_NAME": "PurchaseOrder",
                "PACKAGE_NAME": "PURCHASE_ORDER_API",
                "DB_VALUE": "CONFIRMED",
                "CLIENT_VALUE": "Confirmed",
            },
        ],
        "views": [
            {
                "LU_NAME": "CustomerOrder",
                "VIEW_NAME": "CUSTOMER_ORDER",
                "VIEW_TYPE": "A",
                "VIEW_PROMPT": "Customer Orders",
                "VIEW_COMMENT": "Main view for customer orders",
            },
            {
                "LU_NAME": "CustomerOrderLine",
                "VIEW_NAME": "CUSTOMER_ORDER_LINE",
                "VIEW_TYPE": "A",
                "VIEW_PROMPT": "Customer Order Lines",
                "VIEW_COMMENT": "Customer order line details",
            },
            {
                "LU_NAME": "CompanyPerson",
                "VIEW_NAME": "COMPANY_PERSON",
                "VIEW_TYPE": "A",
                "VIEW_PROMPT": "Employees",
                "VIEW_COMMENT": "Company employee information",
            },
        ],
    }

    return sample_results


def demo_metadata_extraction():
    """Demonstrate metadata extraction process"""
    print("\n" + "=" * 80)
    print("DEMO: IFS Cloud Metadata Extraction")
    print("=" * 80)

    # Create sample data
    sample_results = create_sample_metadata()

    print(f"Sample data created with:")
    print(f"  - {len(sample_results['logical_units'])} logical units")
    print(f"  - {len(sample_results['modules'])} modules")
    print(f"  - {len(sample_results['domain_mappings'])} domain mappings")
    print(f"  - {len(sample_results['views'])} views")

    # Extract and save metadata
    extractor = MCPMetadataExtractor()
    success = extractor.extract_and_save_metadata("25.1.0-DEMO", sample_results)

    if success:
        print("✅ Sample metadata extracted and saved successfully")
    else:
        print("❌ Sample metadata extraction failed")
        return False

    return True


def demo_enhanced_search():
    """Demonstrate enhanced search capabilities"""
    print("\n" + "=" * 80)
    print("DEMO: Enhanced Search with Metadata")
    print("=" * 80)

    # Create indexer
    index_path = Path.cwd() / "demo_index"
    indexer = IFSCloudIndexer(index_path, create_new=True)

    # First, save the metadata to the correct location
    extractor = MCPMetadataExtractor(indexer)
    sample_results = create_sample_metadata()
    success = extractor.extract_and_save_metadata("25.1.0-DEMO", sample_results)

    if not success:
        print("❌ Failed to save demo metadata")
        return

    try:
        # Set IFS version (will load the demo metadata)
        success = indexer.set_ifs_version("25.1.0-DEMO")
        if not success:
            print("❌ Failed to load demo metadata")
            return

        print("✅ Enhanced search engine initialized with demo metadata")

        # Test various search scenarios
        test_queries = [
            "customer order",
            "employee information",
            "purchase order",
            "invoice processing",
            "order line details",
        ]

        for query in test_queries:
            print(f"\n🔍 Search: '{query}'")
            print("-" * 40)

            # Perform enhanced search
            try:
                results = indexer.enhanced_search(query)

                if results:
                    for i, result in enumerate(results[:3]):  # Show top 3
                        print(f"  {i+1}. {result.file_path}")
                        print(f"     Type: {result.content_type}")
                        print(f"     Module: {result.module or 'Unknown'}")
                        print(f"     LU: {result.logical_unit or 'Unknown'}")
                        print(
                            f"     Description: {result.business_description or 'N/A'}"
                        )
                        print(f"     Confidence: {result.confidence:.1f}%")
                        if result.related_entities:
                            print(
                                f"     Related: {', '.join(result.related_entities[:3])}"
                            )
                        print()
                else:
                    print("  No results found")

            except Exception as e:
                print(f"  Error during search: {e}")

        # Test related search suggestions
        print(f"\n💡 Related search suggestions for 'customer order':")
        suggestions = indexer.suggest_related_searches("customer order")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

        # Show module statistics
        print(f"\n📊 Module Statistics:")
        stats = indexer.get_module_statistics()
        for module, info in stats.items():
            print(f"  {module}: {info['lu_count']} logical units")

    finally:
        indexer.close()


def demo_instruction_generation():
    """Demonstrate instruction generation for real database extraction"""
    print("\n" + "=" * 80)
    print("DEMO: Real Database Extraction Instructions")
    print("=" * 80)

    extractor = MCPMetadataExtractor()
    extractor.print_extraction_instructions()


def main():
    """Run all demonstrations"""
    print("IFS CLOUD METADATA-ENHANCED SEARCH DEMONSTRATION")
    print("=" * 60)

    # Demo 1: Metadata extraction
    if not demo_metadata_extraction():
        print("❌ Metadata extraction demo failed - stopping")
        return

    # Demo 2: Enhanced search
    try:
        demo_enhanced_search()
    except Exception as e:
        print(f"❌ Enhanced search demo failed: {e}")
        logger.exception("Enhanced search demo error")

    # Demo 3: Instruction generation
    demo_instruction_generation()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Connect to your IFS Cloud database using MCP SQLcl")
    print("2. Run the SQL queries shown in the instructions")
    print("3. Process the results using the metadata extraction utility")
    print("4. Enjoy enhanced search with business context!")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
