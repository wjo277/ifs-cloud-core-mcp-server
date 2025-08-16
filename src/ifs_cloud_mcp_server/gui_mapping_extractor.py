#!/usr/bin/env python3
"""
GUI Navigation Mapping Extractor

This tool processes the SQL query results from FND_NAVIGATOR_ALL and MD_PROJECTION_ENTITYSET
to create a comprehensive GUI-to-backend mapping file for the search algorithm.

Usage during development (with SQL access):
1. Run the provided SQL query against IFS Cloud database
2. Export results to CSV
3. Run this script to process the CSV into gui_navigation_mappings.json

The generated mapping file will be used at runtime for GUI-aware search.
"""

import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def extract_gui_mappings(csv_file_path: str) -> Dict:
    """
    Process CSV export from the GUI navigation SQL query.

    Expected CSV columns:
    - projection
    - client
    - label
    - page_type
    - page
    - entry_type
    - sort_order
    - entityset_name
    - entity_name
    - definition
    """

    mappings = {
        "gui_to_entity": {},  # GUI label -> entity names
        "gui_to_projection": {},  # GUI label -> projection names
        "entity_synonyms": {},  # Entity name variations
        "projection_patterns": {},  # Projection name patterns
        "metadata": {
            "extraction_date": "2025-08-16",
            "source_query": "FND_NAVIGATOR_ALL + MD_PROJECTION_ENTITYSET",
            "total_mappings": 0,
        },
    }

    entity_variations = defaultdict(set)
    gui_entities = defaultdict(set)
    gui_projections = defaultdict(set)

    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                label = row.get("label", "").strip()
                entity_name = row.get("entity_name", "").strip()
                projection = row.get("projection", "").strip()
                entityset_name = row.get("entityset_name", "").strip()

                if not label or not entity_name:
                    continue

                # Build GUI to entity mappings
                gui_entities[label.lower()].add(entity_name)

                # Build GUI to projection mappings
                if projection:
                    gui_projections[label.lower()].add(projection)

                # Generate entity name variations
                entity_variations[entity_name.lower()].add(entity_name)
                entity_variations[entity_name.lower()].add(camel_to_snake(entity_name))

                # Add entityset variations
                if entityset_name and entityset_name != entity_name:
                    entity_variations[entity_name.lower()].add(entityset_name)
                    entity_variations[entityset_name.lower()].add(entity_name)

        # Convert to final format
        for gui_label, entities in gui_entities.items():
            mappings["gui_to_entity"][gui_label] = list(entities)

        for gui_label, projections in gui_projections.items():
            mappings["gui_to_projection"][gui_label] = list(projections)

        for base_entity, variations in entity_variations.items():
            if len(variations) > 1:
                mappings["entity_synonyms"][base_entity] = list(variations)

        mappings["metadata"]["total_mappings"] = len(mappings["gui_to_entity"])

        print(f"✅ Processed {len(gui_entities)} GUI labels")
        print(f"✅ Generated {len(entity_variations)} entity synonym groups")
        print(f"✅ Created {len(gui_projections)} GUI-to-projection mappings")

        return mappings

    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None


def create_sample_mappings() -> Dict:
    """Create sample GUI mappings for testing (when CSV is not available)."""

    return {
        "gui_to_entity": {
            "employee file": ["CompanyPerson", "Person", "Employee"],
            "expense sheet": ["ExpenseHeader", "ExpenseDetail"],
            "project transaction": ["ProjectTransaction"],
            "activity": ["Activity"],
            "payment address": ["PaymentAddress"],
            "item catalog": ["PartCatalog", "Part"],
            "inventory part": ["InventoryPart"],
            "purchase part": ["PurchasePart"],
        },
        "gui_to_projection": {
            "employee file": ["EmployeesHandling", "PersonHandling"],
            "expense sheet": ["ExpenseSheetHandling"],
            "project transaction": ["ProjectTransactionHandling"],
            "activity": ["ActivityHandling"],
            "payment address": ["PaymentHandling"],
            "item catalog": ["PartCatalogHandling"],
            "inventory part": ["InventoryPartHandling"],
            "purchase part": ["PurchasePartHandling"],
        },
        "entity_synonyms": {
            "companyperson": ["CompanyPerson", "company_person", "Employee", "Person"],
            "expenseheader": [
                "ExpenseHeader",
                "expense_header",
                "ExpenseSheet",
                "expense_sheet",
            ],
            "expensedetail": [
                "ExpenseDetail",
                "expense_detail",
                "ExpenseLine",
                "expense_line",
            ],
            "projecttransaction": ["ProjectTransaction", "project_transaction"],
            "activity": ["Activity"],
            "paymentaddress": ["PaymentAddress", "payment_address"],
            "partcatalog": [
                "PartCatalog",
                "part_catalog",
                "ItemCatalog",
                "item_catalog",
            ],
            "inventorypart": ["InventoryPart", "inventory_part"],
            "purchasepart": ["PurchasePart", "purchase_part"],
        },
        "projection_patterns": {
            "handling": "Main entity management projections",
            "overview": "Summary and list projections",
            "details": "Detail management projections",
        },
        "metadata": {
            "extraction_date": "2025-08-16",
            "source": "Sample data for testing",
            "note": "Replace with actual SQL extraction in production",
            "total_mappings": 8,
        },
    }


def main():
    """Main extraction process."""

    # Set UTF-8 encoding for Windows console
    import sys
    import os

    if os.name == "nt":  # Windows
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    print("🔍 GUI Navigation Mapping Extractor")
    print("=" * 50)

    # Try to find CSV file
    csv_path = Path("gui_navigation_export.csv")

    if csv_path.exists():
        print(f"📄 Found CSV export: {csv_path}")
        mappings = extract_gui_mappings(str(csv_path))
    else:
        print("⚠️  No CSV export found, creating sample mappings")
        print("💡 To generate full mappings:")
        print("   1. Run the SQL query against IFS Cloud database")
        print("   2. Export results to 'gui_navigation_export.csv'")
        print("   3. Re-run this script")
        print()
        mappings = create_sample_mappings()

    if not mappings:
        print("❌ Failed to create mappings")
        return 1

    # Save to data directory
    output_path = Path("data/gui_navigation_mappings.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)

    print(f"✅ GUI mappings saved to: {output_path}")
    print(f"📊 Total GUI labels: {mappings['metadata']['total_mappings']}")

    # Show sample mappings
    print("\n🔍 Sample GUI mappings:")
    for gui_label, entities in list(mappings["gui_to_entity"].items())[:5]:
        print(f"   '{gui_label}' -> {entities}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
