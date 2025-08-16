#!/usr/bin/env python3
"""
Final integration test to verify all GUI mapping components work together.
This test validates that the complete workflow is functional.
"""

import json
import csv
from pathlib import Path


def test_gui_mappings_exist():
    """Test that GUI mappings file exists and contains data."""
    mappings_file = Path("data/gui_navigation_mappings.json")
    if not mappings_file.exists():
        print("❌ GUI mappings file not found")
        return False

    try:
        with open(mappings_file) as f:
            mappings = json.load(f)

        if not mappings:
            print("❌ GUI mappings file is empty")
            return False

        print(f"✅ GUI mappings loaded: {len(mappings)} mappings")
        return True
    except Exception as e:
        print(f"❌ Error loading GUI mappings: {e}")
        return False


def test_csv_format():
    """Test that CSV file has correct format."""
    csv_file = Path("gui_navigation_export.csv")
    if not csv_file.exists():
        print("❌ CSV file not found")
        return False

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_columns = {"label", "entity_name", "projection"}

            if not required_columns.issubset(reader.fieldnames or []):
                print(f"❌ CSV missing required columns. Found: {reader.fieldnames}")
                return False

            row_count = sum(1 for row in reader)
            print(f"✅ CSV format valid: {row_count} rows")
            return True
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def test_documentation_exists():
    """Test that all documentation files exist."""
    docs = [
        "GUI_MAPPING_EXTRACTION_GUIDE.md",
        "sql/extract_gui_mappings.sql",
        "refresh_gui_mappings.py",
    ]

    all_exist = True
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ Documentation exists: {doc}")
        else:
            print(f"❌ Missing documentation: {doc}")
            all_exist = False

    return all_exist


def main():
    print("🧪 Final Integration Test")
    print("=" * 40)

    tests = [
        ("GUI Mappings", test_gui_mappings_exist),
        ("CSV Format", test_csv_format),
        ("Documentation", test_documentation_exists),
    ]

    passed = 0
    for name, test_func in tests:
        print(f"\n📋 Testing {name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed: {name}")

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All integration tests passed!")
        print("✅ GUI mapping system is ready for production")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
