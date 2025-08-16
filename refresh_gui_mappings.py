#!/usr/bin/env python3
"""
IFS Cloud GUI Mapping Refresh Script

Quick automation script for refreshing GUI mappings after IFS Cloud updates.
This script encapsulates the most common steps from the extraction guide.

Usage:
    uv run python refresh_gui_mappings.py [csv_file_path]

If no CSV file is provided, it will look for 'gui_navigation_export.csv'
in the current directory.
"""

import sys
import subprocess
import json
from pathlib import Path


def main():
    # Set UTF-8 encoding for Windows console
    import os

    if os.name == "nt":  # Windows
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    """Main execution flow for refreshing GUI mappings."""

    print("🔄 IFS Cloud GUI Mapping Refresh Tool")
    print("=" * 50)

    # Determine CSV file path
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "gui_navigation_export.csv"
    csv_path = Path(csv_file)

    # Check if CSV file exists
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("📋 Steps to create the CSV file:")
        print("   1. Connect to IFS Cloud database using Oracle MCP tool")
        print("   2. Run the SQL query from GUI_MAPPING_EXTRACTION_GUIDE.md")
        print("   3. Export results to 'gui_navigation_export.csv'")
        print("   4. Ensure column headers are: label,entity_name,projection")
        return False

    print(f"📄 Found CSV file: {csv_path}")

    # Run the GUI mapping extractor
    print("🔍 Processing GUI mappings...")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.ifs_cloud_mcp_server.gui_mapping_extractor",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print("✅ GUI mapping extraction completed!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during extraction: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

    # Check if mapping file was created
    mapping_file = Path("data/gui_navigation_mappings.json")
    if not mapping_file.exists():
        print("❌ GUI mapping file was not created!")
        return False

    # Load and display mapping statistics
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        gui_count = len(mappings.get("gui_to_entity", {}))
        entity_count = len(mappings.get("entity_synonyms", {}))
        projection_count = len(mappings.get("gui_to_projection", {}))

        print(f"📊 Mapping Statistics:")
        print(f"   • GUI labels: {gui_count}")
        print(f"   • Entity synonyms: {entity_count}")
        print(f"   • Projections: {projection_count}")

    except Exception as e:
        print(f"⚠️  Could not read mapping statistics: {e}")

    # Run tests to verify mappings work
    print("\n🧪 Running verification tests...")
    test_commands = [
        ["uv", "run", "python", "gui_mapping_test.py"],
        ["uv", "run", "python", "benchmark_test.py"],
    ]

    for cmd in test_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {' '.join(cmd[3:])} - PASSED")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {' '.join(cmd[3:])} - FAILED")
            print(f"     Error: {e.stderr[:200]}...")

    print("\n🎉 GUI mapping refresh completed successfully!")
    print("📖 For detailed instructions, see: GUI_MAPPING_EXTRACTION_GUIDE.md")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
