#!/usr/bin/env python3
"""
IFS Cloud Metadata Extraction Utility

Extracts metadata from IFS Cloud database using MCP SQLcl server
and saves it for offline use in search enhancement.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPMetadataExtractor:
    """Extracts metadata using MCP SQLcl server"""

    def __init__(self, indexer_instance=None):
        """Initialize with optional indexer instance"""
        self.indexer = indexer_instance
        self.extraction_queries = self._get_extraction_queries()

    def _get_extraction_queries(self) -> Dict[str, str]:
        """Get SQL queries for metadata extraction"""
        return {
            "logical_units": """
                SELECT module, lu_name, lu_prompt, base_table, base_view, 
                       logical_unit_type, custom_fields
                FROM dictionary_sys_lu_active
                ORDER BY module, lu_name
            """,
            "modules": """
                SELECT module,
                       COUNT(*) as lu_count,
                       COUNT(DISTINCT base_table) as table_count,
                       LISTAGG(DISTINCT lu_type, ',') WITHIN GROUP (ORDER BY lu_type) as lu_types
                FROM dictionary_sys_lu_active 
                GROUP BY module 
                ORDER BY lu_count DESC
            """,
            "domain_mappings": """
                SELECT lu_name, package_name, db_value, client_value
                FROM dictionary_sys_domain_tab
                WHERE ROWNUM <= 10000
                ORDER BY lu_name, package_name
            """,
            "views": """
                SELECT lu_name, view_name, view_type, view_prompt, view_comment
                FROM dictionary_sys_view_tab
                WHERE view_type IN ('A', 'B', 'R', 'S')
                  AND ROWNUM <= 5000
                ORDER BY lu_name, view_name
            """,
        }

    def print_extraction_instructions(self) -> None:
        """Print step-by-step instructions for manual extraction"""
        print("\n" + "=" * 80)
        print("IFS CLOUD METADATA EXTRACTION INSTRUCTIONS")
        print("=" * 80)
        print("\nTo extract metadata for enhanced search, follow these steps:")
        print("\n1. Connect to your IFS Cloud database using MCP SQLcl server")
        print("   Example: Use the mcp_sqlcl_connect tool with your connection name")

        print("\n2. Execute the following queries and save results:")

        for query_name, sql in self.extraction_queries.items():
            print(f"\n   Query: {query_name}")
            print("   " + "-" * 40)
            print("   " + sql.replace("\n", "\n   "))

        print("\n3. Process the results using this utility:")
        print("   python extract_metadata.py --process-results results.json")

        print(
            "\n4. The processed metadata will be saved automatically for search enhancement"
        )
        print("\n" + "=" * 80)

    def process_csv_results(self, csv_files: Dict[str, Path]) -> Dict[str, List[Dict]]:
        """
        Process CSV results from MCP queries

        Args:
            csv_files: Dictionary mapping query names to CSV file paths

        Returns:
            Processed results ready for metadata extraction
        """
        results = {}

        for query_name, csv_file in csv_files.items():
            if not csv_file.exists():
                logger.warning(f"CSV file not found: {csv_file}")
                results[query_name] = []
                continue

            try:
                import csv

                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                results[query_name] = rows
                logger.info(f"Loaded {len(rows)} rows from {csv_file}")

            except Exception as e:
                logger.error(f"Failed to process CSV file {csv_file}: {e}")
                results[query_name] = []

        return results

    def process_json_results(self, json_file: Path) -> Dict[str, List[Dict]]:
        """
        Process JSON results from MCP queries

        Args:
            json_file: Path to JSON file with all query results

        Returns:
            Processed results ready for metadata extraction
        """
        if not json_file.exists():
            raise FileNotFoundError(f"JSON results file not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        for query_name in self.extraction_queries.keys():
            if query_name not in data:
                logger.warning(f"Query results missing for: {query_name}")
                data[query_name] = []

        logger.info(f"Loaded JSON results with {len(data)} query result sets")
        return data

    def extract_and_save_metadata(
        self, ifs_version: str, query_results: Dict[str, List[Dict]]
    ) -> bool:
        """
        Extract metadata from query results and save for search enhancement

        Args:
            ifs_version: IFS Cloud version (e.g., "25.1.0")
            query_results: Dictionary with query results

        Returns:
            True if successful
        """
        try:
            if self.indexer:
                # Use indexer's extraction method
                return self.indexer.extract_metadata_from_mcp_results(
                    ifs_version, query_results
                )
            else:
                # Standalone extraction
                from .metadata_extractor import (
                    DatabaseMetadataExtractor,
                    MetadataManager,
                )

                # Create extractor and manager
                extractor = DatabaseMetadataExtractor()
                manager = MetadataManager(Path.cwd() / "metadata")

                # Extract metadata
                metadata_extract = extractor.extract_from_mcp_queries(
                    ifs_version, query_results
                )

                # Save metadata
                saved_path = manager.save_metadata(metadata_extract)
                logger.info(f"Metadata saved to {saved_path}")

                return True

        except Exception as e:
            logger.error(f"Failed to extract and save metadata: {e}")
            return False

    def create_sample_results_template(self, output_file: Path) -> None:
        """Create a template JSON file for manual result entry"""
        template = {}

        for query_name, sql in self.extraction_queries.items():
            template[query_name] = [
                {
                    "_comment": f"Results for query: {query_name}",
                    "_sql": sql.strip(),
                    "_example": "Add your query results here as a list of dictionaries",
                }
            ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        logger.info(f"Sample results template created: {output_file}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract IFS Cloud metadata for search enhancement"
    )
    parser.add_argument(
        "--ifs-version", required=True, help="IFS Cloud version (e.g., 25.1.0)"
    )
    parser.add_argument(
        "--instructions", action="store_true", help="Print extraction instructions"
    )
    parser.add_argument(
        "--create-template", help="Create sample results template JSON file"
    )
    parser.add_argument("--process-json", help="Process results from JSON file")
    parser.add_argument("--process-csv-dir", help="Process CSV files from directory")

    args = parser.parse_args()

    extractor = MCPMetadataExtractor()

    if args.instructions:
        extractor.print_extraction_instructions()
        return

    if args.create_template:
        template_path = Path(args.create_template)
        extractor.create_sample_results_template(template_path)
        return

    if args.process_json:
        json_file = Path(args.process_json)
        try:
            query_results = extractor.process_json_results(json_file)
            success = extractor.extract_and_save_metadata(
                args.ifs_version, query_results
            )

            if success:
                print(f"✅ Metadata extraction successful for IFS {args.ifs_version}")
            else:
                print(f"❌ Metadata extraction failed for IFS {args.ifs_version}")

        except Exception as e:
            logger.error(f"Error processing JSON results: {e}")
            return

    if args.process_csv_dir:
        csv_dir = Path(args.process_csv_dir)
        csv_files = {
            "logical_units": csv_dir / "logical_units.csv",
            "modules": csv_dir / "modules.csv",
            "domain_mappings": csv_dir / "domain_mappings.csv",
            "views": csv_dir / "views.csv",
        }

        try:
            query_results = extractor.process_csv_results(csv_files)
            success = extractor.extract_and_save_metadata(
                args.ifs_version, query_results
            )

            if success:
                print(f"✅ Metadata extraction successful for IFS {args.ifs_version}")
            else:
                print(f"❌ Metadata extraction failed for IFS {args.ifs_version}")

        except Exception as e:
            logger.error(f"Error processing CSV results: {e}")
            return


if __name__ == "__main__":
    main()


# Example usage as a module
async def extract_metadata_example():
    """Example of how to use the extractor programmatically"""

    # Sample query results (normally from MCP SQLcl)
    sample_results = {
        "logical_units": [
            {
                "MODULE": "ORDER",
                "LU_NAME": "CustomerOrder",
                "LU_PROMPT": "Customer Order",
                "BASE_TABLE": "CUSTOMER_ORDER_TAB",
                "BASE_VIEW": "CUSTOMER_ORDER",
                "LOGICAL_UNIT_TYPE": "L",
                "CUSTOM_FIELDS": None,
            }
        ],
        "modules": [
            {"MODULE": "ORDER", "LU_COUNT": 505, "TABLE_COUNT": 283, "LU_TYPES": "L"}
        ],
        "domain_mappings": [
            {
                "LU_NAME": "CustomerOrder",
                "PACKAGE_NAME": "CUSTOMER_ORDER_API",
                "DB_VALUE": "PLANNED",
                "CLIENT_VALUE": "Planned",
            }
        ],
        "views": [
            {
                "LU_NAME": "CustomerOrder",
                "VIEW_NAME": "CUSTOMER_ORDER",
                "VIEW_TYPE": "A",
                "VIEW_PROMPT": "Customer Orders",
                "VIEW_COMMENT": "Main view for customer orders",
            }
        ],
    }

    # Extract and process
    extractor = MCPMetadataExtractor()
    success = extractor.extract_and_save_metadata("25.1.0", sample_results)

    if success:
        logger.info("✅ Example metadata extraction completed successfully")
    else:
        logger.error("❌ Example metadata extraction failed")


# Run example if executed directly
if __name__ == "__main__":
    # Check if running with arguments
    import sys

    if len(sys.argv) == 1:
        # No arguments - run example
        asyncio.run(extract_metadata_example())
    else:
        # Run main CLI
        main()
