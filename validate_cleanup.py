#!/usr/bin/env python3
"""
Simple validation that the conservative analyzer works after cleanup.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.projection_analyzer import ProjectionAnalyzer


def test_cleanup_validation():
    """Test that the conservative analyzer still works after removing old implementations"""

    # Test content from a real projection
    content = """
projection AccountsHandling;
component ACCRUL;
layer Core;
description "Accounts Overview";

entityset AccountSet for Account {
   context Company(Company);
}

entity Account {
   attribute Account Text;
   reference CompanyRef(Company) to CompanyFinance(Company);
}
"""

    analyzer = ProjectionAnalyzer(strict_mode=False)
    ast = analyzer.analyze(content)

    print(f"SUCCESS: Parsed projection '{ast.name}'")
    print(f"Component: {ast.component}")
    print(f"Layer: {ast.layer}")
    print(f"Description: {ast.description}")
    print(f"Errors: {len(ast.get_errors())}")
    print(f"Warnings: {len(ast.get_warnings())}")
    print(f"Valid: {ast.is_valid}")
    print(f"EntitySets: {len(ast.entitysets)}")
    print(f"Entities: {len(ast.entities)}")

    # Validate no false errors on real projection content
    if len(ast.get_errors()) == 0 and len(ast.get_warnings()) == 0:
        print(
            "PERFECT: Conservative analyzer shows 0 errors, 0 warnings on real projection!"
        )
        return True
    else:
        print("ISSUE: Found unexpected errors/warnings on real projection content")
        return False


if __name__ == "__main__":
    success = test_cleanup_validation()
    print("\n" + "=" * 50)
    if success:
        print("✅ CLEANUP VALIDATION PASSED!")
        print("✅ Conservative analyzer working perfectly!")
    else:
        print("❌ CLEANUP VALIDATION FAILED!")
    print("=" * 50)
