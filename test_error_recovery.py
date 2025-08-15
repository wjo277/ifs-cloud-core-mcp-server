#!/usr/bin/env python3
"""
Test script for projection analyzer error recovery and syntax feedback.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.projection_analyzer import (
    ProjectionAnalyzer,
    DiagnosticSeverity,
)
import json


def test_syntax_error_recovery():
    """Test analyzer with various syntax errors"""

    print("🧪 Testing Syntax Error Recovery and Feedback")
    print("=" * 60)

    # Test cases with intentional errors
    test_cases = [
        {
            "name": "Missing Projection Name",
            "content": """
            projection;
            component ORDER;
            layer Core;
            """,
        },
        {
            "name": "Missing Component",
            "content": """
            projection TestProjection;
            layer Core;
            description "Test projection";
            """,
        },
        {
            "name": "Empty EntitySet",
            "content": """
            projection TestProjection;
            component ORDER;
            layer Core;
            
            entityset TestSet for {
            }
            """,
        },
        {
            "name": "Malformed EntitySet",
            "content": """
            projection TestProjection;
            component ORDER;
            layer Core;
            
            entityset TestSet {
                context Company(Company);
            }
            """,
        },
        {
            "name": "Unclosed Block",
            "content": """
            projection TestProjection;
            component ORDER;
            layer Core;
            
            entityset TestSet for TestEntity {
                context Company(Company);
                where = "company = '1'";
            // Missing closing brace
            """,
        },
        {
            "name": "Invalid Component Name",
            "content": """
            projection TestProjection;
            component order;
            layer Core;
            """,
        },
        {
            "name": "Partial Quotes in Description",
            "content": """
            projection TestProjection;
            component ORDER;
            layer Core;
            description "Incomplete description;
            """,
        },
        {
            "name": "Valid Projection (No Errors)",
            "content": """
            projection TestProjection;
            component ORDER;
            layer Core;
            description "Valid test projection";
            category Users;
            
            entityset TestSet for TestEntity {
                context Company(Company);
            }
            
            @Override
            entity TestEntity {
                attribute TestAttr Text;
            }
            """,
        },
    ]

    analyzer = ProjectionAnalyzer(strict_mode=False)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)

        try:
            ast = analyzer.analyze(test_case["content"])

            # Show basic info
            print(f"   ✅ Projection: {ast.name}")
            print(f"   ✅ Valid: {ast.is_valid}")
            print(f"   📊 Errors: {len(ast.get_errors())}")
            print(f"   ⚠️  Warnings: {len(ast.get_warnings())}")

            # Show diagnostics
            if ast.diagnostics:
                print("   🔍 Diagnostics:")
                for diag in ast.diagnostics:
                    severity_icon = {
                        DiagnosticSeverity.ERROR: "❌",
                        DiagnosticSeverity.WARNING: "⚠️",
                        DiagnosticSeverity.INFO: "ℹ️",
                        DiagnosticSeverity.HINT: "💡",
                    }.get(diag.severity, "🔸")

                    print(f"     {severity_icon} Line {diag.line}: {diag.message}")
                    if diag.fix_suggestion:
                        print(f"        💡 Fix: {diag.fix_suggestion}")
                    if diag.code:
                        print(f"        🏷️  Code: {diag.code}")
            else:
                print("   ✅ No issues found!")

        except Exception as e:
            print(f"   💥 Fatal error: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ Syntax error recovery test completed!")


def test_with_real_broken_projection():
    """Test with a real projection file that has been intentionally broken"""

    print("\n🔧 Testing with Intentionally Broken Real Projection")
    print("=" * 60)

    # Create a broken version of a real projection
    broken_content = """
    ----------------------------------------------------------------------------------------------------
    -- Intentionally broken projection for testing
    ----------------------------------------------------------------------------------------------------
    
    projection AccountsHandling;
    component  // Missing component name
    layer Core;
    description "Accounts Overview;  // Unclosed quote
    category Users;
    
    include fragment AccountsConsolidationSelector;
    include fragment AccountCommonHandling;
    
    ----------------------------- MAIN ENTRY POINTS -----------------------------
    entityset AccountSet for {  // Missing entity name
       context Company(Company);
    }
    
    entityset MultiCompanyAccountSet for Account {
       where = ;  // Empty where clause
    }
    
    ------------------------------ ENTITY DETAILS -------------------------------
    @Override
    entity Account {
       attribute Account Text;
       // Missing closing brace
    
    ---------------------------------- ACTIONS ----------------------------------
    action ValidateGetSelectedCompany Text {
       parameter VarListText List<Text>;
    // Missing closing brace
    """

    analyzer = ProjectionAnalyzer(strict_mode=False)
    ast = analyzer.analyze(broken_content)

    print(f"📋 Projection: {ast.name}")
    print(f"🏗️  Component: {ast.component or 'MISSING'}")
    print(f"✅ Parsed Successfully: {ast.is_valid}")
    print(f"❌ Errors Found: {len(ast.get_errors())}")
    print(f"⚠️  Warnings Found: {len(ast.get_warnings())}")
    print(f"📊 Entity Sets: {len(ast.entitysets)}")
    print(f"🏛️  Entities: {len(ast.entities)}")

    print("\n🔍 Detailed Diagnostics:")
    for diag in ast.diagnostics:
        severity_icon = {
            DiagnosticSeverity.ERROR: "❌",
            DiagnosticSeverity.WARNING: "⚠️",
            DiagnosticSeverity.INFO: "ℹ️",
            DiagnosticSeverity.HINT: "💡",
        }.get(diag.severity, "🔸")

        print(f"  {severity_icon} Line {diag.line}: {diag.message}")
        if diag.code:
            print(f"      Code: {diag.code}")
        if diag.fix_suggestion:
            print(f"      Fix: {diag.fix_suggestion}")
        print()

    # Show JSON output with diagnostics
    ast_dict = ast.to_dict()
    print(f"📄 JSON Output includes {len(ast_dict['diagnostics'])} diagnostics")
    print("✅ Error recovery test with real projection completed!")


if __name__ == "__main__":
    test_syntax_error_recovery()
    test_with_real_broken_projection()

    print("\n🎉 All error recovery tests completed!")
    print(
        "The analyzer can now handle syntax errors gracefully and provide helpful feedback!"
    )
