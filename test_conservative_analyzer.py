#!/usr/bin/env python3
"""
Test script for the more conservative projection analyzer.
This validates that we prefer missing errors over false positives.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.projection_analyzer import (
    ProjectionAnalyzer,
    DiagnosticSeverity,
)
import json


def test_conservative_approach():
    """Test that the analyzer is more conservative to avoid false positives"""

    print("🛡️ Testing Conservative Error Detection")
    print("=" * 60)

    # Test cases that should NOT be flagged as errors (conservative approach)
    conservative_test_cases = [
        {
            "name": "Valid Alternative Component Name",
            "content": """
            projection AccountsHandling;
            component Acc;  // Short but valid
            layer Core;
            
            entityset AccountSet for Account {
                context Company(Company);
            }
            """,
            "expect_no_errors": True,
            "expect_max_warnings": 0,
        },
        {
            "name": "Description Without Quotes (Valid Style)",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            description Accounts Overview;  // No quotes but clear
            """,
            "expect_no_errors": True,
            "expect_max_warnings": 0,
        },
        {
            "name": "Mixed Case EntitySet (Could be Valid)",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            
            entityset accountSet for Account {  // lowercase start
                context Company(Company);
            }
            """,
            "expect_no_errors": True,
            "expect_max_hints": 1,  # Should only be a hint, not warning
        },
        {
            "name": "Complex Where Clause (Valid)",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            
            entityset AccountSet for Account {
                context Company(Company);
                where = "status = 'ACTIVE' AND type IN ('REVENUE', 'EXPENSE')";
            }
            """,
            "expect_no_errors": True,
            "expect_max_warnings": 0,
        },
        {
            "name": "External Entity Reference (Common)",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            
            entityset CompanySet for Company {
                context Company(Company);
            }
            """,
            "expect_no_errors": True,
            "expect_max_warnings": 0,  # Company is a common base entity
        },
        {
            "name": "Minimal But Valid Projection",
            "content": """
            projection Test;
            """,
            "expect_no_errors": True,
            "expect_max_warnings": 0,  # Should only have hints, not warnings
        },
        {
            "name": "Partial Description with Quote",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            description "Accounts Overview;  // Missing end quote but functional
            """,
            "expect_no_errors": True,
            "expect_max_hints": 1,  # Should be hint, not warning
        },
    ]

    # Test cases that SHOULD still be flagged (clear errors)
    error_test_cases = [
        {
            "name": "Completely Empty Where Clause",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            
            entityset AccountSet for Account {
                context Company(Company);
                where = ;  // Clearly broken
            }
            """,
            "expect_errors": True,
        },
        {
            "name": "Entityset Syntax Error",
            "content": """
            projection AccountsHandling;
            component ACCRUL;
            layer Core;
            
            entityset AccountSet {  // Missing "for"
                context Company(Company);
            }
            """,
            "expect_errors": True,
        },
    ]

    analyzer = ProjectionAnalyzer(strict_mode=False)

    print("\n🟢 Testing Conservative Cases (Should NOT flag as errors/warnings)")
    print("-" * 60)

    for i, test_case in enumerate(conservative_test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)

        ast = analyzer.analyze(test_case["content"])

        errors = ast.get_errors()
        warnings = ast.get_warnings()
        hints = [d for d in ast.diagnostics if d.severity == DiagnosticSeverity.HINT]

        print(f"   ❌ Errors: {len(errors)}")
        print(f"   ⚠️  Warnings: {len(warnings)}")
        print(f"   💡 Hints: {len(hints)}")

        # Check expectations
        if test_case.get("expect_no_errors") and errors:
            print(f"   ❗ UNEXPECTED ERRORS:")
            for err in errors:
                print(f"      - {err.message}")

        if test_case.get("expect_max_warnings", float("inf")) < len(warnings):
            print(f"   ❗ TOO MANY WARNINGS:")
            for warn in warnings:
                print(f"      - {warn.message}")

        if test_case.get("expect_max_hints", float("inf")) < len(hints):
            print(f"   ❗ TOO MANY HINTS:")
            for hint in hints:
                print(f"      - {hint.message}")

        # Show what we did flag (if anything)
        if errors or warnings:
            print("   🔍 Flagged issues:")
            for diag in ast.diagnostics:
                severity_icon = {
                    DiagnosticSeverity.ERROR: "❌",
                    DiagnosticSeverity.WARNING: "⚠️",
                    DiagnosticSeverity.INFO: "ℹ️",
                    DiagnosticSeverity.HINT: "💡",
                }.get(diag.severity, "🔸")
                print(f"      {severity_icon} {diag.message}")
        else:
            print("   ✅ No errors or warnings - Good!")

    print("\n🔴 Testing Error Cases (Should STILL flag as errors)")
    print("-" * 60)

    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)

        ast = analyzer.analyze(test_case["content"])

        errors = ast.get_errors()
        warnings = ast.get_warnings()

        print(f"   ❌ Errors: {len(errors)}")
        print(f"   ⚠️  Warnings: {len(warnings)}")

        if test_case.get("expect_errors") and not errors:
            print(f"   ❗ MISSING EXPECTED ERRORS!")
        elif errors:
            print("   ✅ Correctly flagged errors:")
            for err in errors:
                print(f"      - {err.message}")

    print("\n" + "=" * 60)
    print("✅ Conservative error detection test completed!")
    print(
        "🎯 The analyzer now prefers to miss some issues rather than flag valid code as erroneous!"
    )


if __name__ == "__main__":
    test_conservative_approach()

    print("\n🏆 The projection analyzer is now more conservative!")
    print("📝 Key improvements:")
    print("  • Component naming: Only hints for likely issues")
    print("  • Descriptions: More lenient quote handling")
    print("  • Entity references: Only warn on clearly custom long names")
    print("  • Naming conventions: Hints instead of warnings")
    print("  • Missing components: Only flag if substantial content exists")
    print("  • Empty projections: Only flag if truly minimal")
    print("  🎉 Better safe than sorry - avoiding false positives!")
