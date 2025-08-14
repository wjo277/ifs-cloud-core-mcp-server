"""Simple test to check basic functionality without dependencies."""

import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test basic imports."""
    print("Testing imports...")

    try:
        from ifs_cloud_mcp_server.parsers import IFSFileParser

        print("✓ Parser imported successfully")

        parser = IFSFileParser()

        # Test with simple content
        entity_sample = (
            """<ENTITY><NAME>TestEntity</NAME><COMPONENT>TEST</COMPONENT></ENTITY>"""
        )
        result = parser.parse(entity_sample, ".entity")
        print(f"✓ Entity parsing works: {result.entities}")

    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Other error: {e}")


def test_file_structure():
    """Test if work directory has the expected structure."""
    print("\nChecking _work directory structure...")

    work_dir = Path("_work")
    if not work_dir.exists():
        print("✗ _work directory not found")
        return

    print("✓ _work directory found")

    # Count files by extension
    extensions = [
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".fragment",
        ".client",
        ".projection",
        ".plsvc",
    ]
    file_counts = {}

    for ext in extensions:
        files = list(work_dir.rglob(f"*{ext}"))
        file_counts[ext] = len(files)
        if files:
            print(f"✓ Found {len(files)} {ext} files")
            # Show first few files as examples
            for i, file in enumerate(files[:3]):
                print(f"    Example: {file.relative_to(work_dir)}")
        else:
            print(f"✗ No {ext} files found")

    return file_counts


def main():
    """Run basic tests."""
    print("IFS Cloud MCP Server - Basic Functionality Check")
    print("=" * 50)

    test_imports()
    file_counts = test_file_structure()

    print("\n" + "=" * 50)
    print("Basic checks completed!")

    if file_counts:
        total_files = sum(file_counts.values())
        print(f"Total IFS files found: {total_files}")


if __name__ == "__main__":
    main()
