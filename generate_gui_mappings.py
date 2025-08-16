#!/usr/bin/env python3
"""GUI Navigation Mapping Generator - Build-time data extraction for runtime GUI-to-backend mappings."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class NavigationMapping:
    """Navigation mapping entry from IFS Cloud metadata."""

    projection: str
    client: str
    label: str
    page_type: str
    page: str
    entry_type: str
    sort_order: int
    entityset_name: str
    entity_name: str
    definition: str


@dataclass
class GUIMapping:
    """Processed GUI-to-backend mapping."""

    gui_label: str
    entity_name: str
    projection: str
    client: str
    synonyms: List[str]
    search_terms: List[str]


class NavigationMappingGenerator:
    """Generates GUI navigation mappings from IFS Cloud metadata for runtime use."""

    def __init__(self, output_path: Path):
        """Initialize the mapping generator.

        Args:
            output_path: Path where the generated mapping file should be saved
        """
        self.output_path = Path(output_path)
        self.mappings: List[GUIMapping] = []

    def process_sql_results(self, sql_results: List[Dict]) -> None:
        """Process SQL query results into GUI mappings.

        Args:
            sql_results: List of dictionaries from the navigation SQL query
        """
        # Group by label to consolidate mappings
        label_groups: Dict[str, List[NavigationMapping]] = {}

        for row in sql_results:
            nav_mapping = NavigationMapping(**row)
            if nav_mapping.label not in label_groups:
                label_groups[nav_mapping.label] = []
            label_groups[nav_mapping.label].append(nav_mapping)

        # Process each GUI label group
        for label, mappings in label_groups.items():
            self._process_label_group(label, mappings)

    def _process_label_group(
        self, label: str, mappings: List[NavigationMapping]
    ) -> None:
        """Process a group of navigation mappings for a single GUI label.

        Args:
            label: GUI label (e.g., "Employee File")
            mappings: List of navigation mappings for this label
        """
        # Extract unique entities and projections
        entities = list(
            set(mapping.entity_name for mapping in mappings if mapping.entity_name)
        )
        projections = list(
            set(mapping.projection for mapping in mappings if mapping.projection)
        )
        clients = list(set(mapping.client for mapping in mappings if mapping.client))

        # Generate search synonyms
        synonyms = self._generate_synonyms(label, entities)
        search_terms = self._generate_search_terms(label, entities)

        # Create GUI mapping (use primary entity if available)
        primary_entity = entities[0] if entities else ""
        primary_projection = projections[0] if projections else ""
        primary_client = clients[0] if clients else ""

        gui_mapping = GUIMapping(
            gui_label=label,
            entity_name=primary_entity,
            projection=primary_projection,
            client=primary_client,
            synonyms=synonyms,
            search_terms=search_terms,
        )

        self.mappings.append(gui_mapping)

    def _generate_synonyms(self, label: str, entities: List[str]) -> List[str]:
        """Generate search synonyms for a GUI label.

        Args:
            label: GUI label
            entities: Associated entity names

        Returns:
            List of search synonyms
        """
        synonyms = set()

        # Add the original label
        synonyms.add(label.lower())

        # Add entity names
        for entity in entities:
            synonyms.add(entity.lower())
            # Add camelCase variations
            synonyms.add(self._camel_to_words(entity).lower())

        # Add common word variations
        label_words = label.lower().split()
        synonyms.update(label_words)

        # Domain-specific mappings
        domain_mappings = {
            "employee": ["person", "personnel", "staff", "worker"],
            "customer": ["client", "account", "buyer"],
            "supplier": ["vendor", "provider"],
            "item": ["part", "product", "material"],
            "project": ["job", "work", "task"],
            "order": ["purchase", "request", "requisition"],
        }

        for word in label_words:
            if word in domain_mappings:
                synonyms.update(domain_mappings[word])

        return list(synonyms)

    def _generate_search_terms(self, label: str, entities: List[str]) -> List[str]:
        """Generate search terms that should match this GUI element.

        Args:
            label: GUI label
            entities: Associated entity names

        Returns:
            List of search terms that should find this element
        """
        search_terms = []

        # Exact label match
        search_terms.append(label.lower())

        # Entity name matches
        for entity in entities:
            search_terms.append(entity.lower())
            search_terms.append(self._camel_to_words(entity).lower())

        # Common search patterns
        label_lower = label.lower()
        if "file" in label_lower:
            # For "Employee File" type labels
            base_term = label_lower.replace(" file", "")
            search_terms.extend(
                [
                    base_term,
                    f"{base_term} management",
                    f"{base_term} maintenance",
                    f"{base_term} data",
                ]
            )

        return search_terms

    def _camel_to_words(self, camel_str: str) -> str:
        """Convert CamelCase to space-separated words.

        Args:
            camel_str: CamelCase string

        Returns:
            Space-separated words
        """
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", " ", camel_str)

    def save_mappings(self) -> None:
        """Save the processed mappings to a JSON file for runtime use."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        mappings_dict = {
            "version": "1.0",
            "generated_at": "2025-08-16T00:00:00Z",  # Would be current timestamp in production
            "description": "GUI-to-backend navigation mappings extracted from IFS Cloud metadata",
            "mappings": [asdict(mapping) for mapping in self.mappings],
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(mappings_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.mappings)} GUI mappings to {self.output_path}")

    def generate_from_sql_file(self, sql_results_file: Path) -> None:
        """Generate mappings from a JSON file containing SQL results.

        Args:
            sql_results_file: Path to JSON file with SQL query results
        """
        with open(sql_results_file, "r", encoding="utf-8") as f:
            sql_results = json.load(f)

        self.process_sql_results(sql_results)
        self.save_mappings()


def create_sample_navigation_data():
    """Create sample navigation data for testing (simulates SQL results)."""
    sample_data = [
        {
            "projection": "CompanyPersonHandling",
            "client": "CompanyPersons",
            "label": "Employee File",
            "page_type": "Form",
            "page": "CompanyPerson",
            "entry_type": "MainMenu",
            "sort_order": 100,
            "entityset_name": "CompanyPersonSet",
            "entity_name": "CompanyPerson",
            "definition": "COMPANY_PERSON",
        },
        {
            "projection": "ProjectTransactionHandling",
            "client": "ProjectTransactions",
            "label": "Project Transactions",
            "page_type": "List",
            "page": "ProjectTransaction",
            "entry_type": "MainMenu",
            "sort_order": 200,
            "entityset_name": "ProjectTransactionSet",
            "entity_name": "ProjectTransaction",
            "definition": "PROJECT_TRANSACTION",
        },
        {
            "projection": "ActivityHandling",
            "client": "Activities",
            "label": "Activity Management",
            "page_type": "Form",
            "page": "Activity",
            "entry_type": "MainMenu",
            "sort_order": 300,
            "entityset_name": "ActivitySet",
            "entity_name": "Activity",
            "definition": "ACTIVITY",
        },
        {
            "projection": "ExpenseSheetHandling",
            "client": "ExpenseSheets",
            "label": "Expense Sheets",
            "page_type": "List",
            "page": "ExpenseHeader",
            "entry_type": "MainMenu",
            "sort_order": 400,
            "entityset_name": "ExpenseHeaderSet",
            "entity_name": "ExpenseHeader",
            "definition": "EXPENSE_HEADER",
        },
    ]

    return sample_data


def main():
    """Main function to generate GUI mappings."""
    # For development/testing - create sample data
    sample_data = create_sample_navigation_data()

    # Generate mappings
    output_path = Path(__file__).parent / "data" / "gui_navigation_mappings.json"
    generator = NavigationMappingGenerator(output_path)
    generator.process_sql_results(sample_data)
    generator.save_mappings()

    print(f"✅ Generated GUI navigation mappings at {output_path}")
    print(f"📊 Created {len(generator.mappings)} mappings")

    # Display sample mappings
    print("\n📋 Sample mappings:")
    for mapping in generator.mappings[:3]:
        print(f"  • '{mapping.gui_label}' -> {mapping.entity_name}")
        print(
            f"    Synonyms: {', '.join(mapping.synonyms[:5])}{'...' if len(mapping.synonyms) > 5 else ''}"
        )


if __name__ == "__main__":
    main()
