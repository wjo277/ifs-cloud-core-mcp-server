#!/usr/bin/env python3
"""
Test Client Analyzer Error Detection

Test the client analyzer's ability to detect actual syntax errors while
maintaining the conservative approach (no false positives).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.ifs_cloud_mcp_server.client_analyzer import analyze_client_file


def test_error_detection():
    """Test that the analyzer can detect actual syntax errors"""

    print("=" * 60)
    print("TESTING ERROR DETECTION")
    print("=" * 60)

    # Test cases with actual syntax errors
    error_test_cases = [
        {
            "name": "Severe Brace Imbalance",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

page Form using TestSet {
   label = "Test";
   group TestGroup {
      field TestField;
   }
   // Missing closing brace for page
""",
            "expected_severity": "HINT",  # Conservative approach - only hint
        },
        {
            "name": "Extra Closing Braces",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

page Form using TestSet {
   label = "Test";
}
}
}
""",
            "expected_severity": "ERROR",  # Clear error - extra braces
        },
        {
            "name": "Missing Essential Declarations",
            "content": """
-- No client, component, or layer declarations
projection TestHandling;

page Form using TestSet {
   label = "Test";
}
""",
            "expected_severity": "WARNING",  # Should warn about missing declarations
        },
    ]

    for test_case in error_test_cases:
        print(f"\n--- {test_case['name']} ---")
        result = analyze_client_file(
            test_case["content"], f"{test_case['name']}.client"
        )

        print(f"Valid: {result['valid']}")
        print(
            f"Errors: {result['errors']}, Warnings: {result['warnings']}, Hints: {result['hints']}"
        )

        if result["diagnostics"]:
            for diag in result["diagnostics"]:
                print(f"  {diag['severity']} (Line {diag['line']}): {diag['message']}")

            # Check if we detected the expected issue severity
            severities = [diag["severity"] for diag in result["diagnostics"]]
            if test_case["expected_severity"] in severities:
                print(f"  ✓ Expected {test_case['expected_severity']} detected")
            else:
                print(
                    f"  ⚠ Expected {test_case['expected_severity']}, got {severities}"
                )
        else:
            print("  No issues detected")


def test_edge_cases():
    """Test edge cases and boundary conditions"""

    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    edge_cases = [
        {"name": "Empty File", "content": ""},
        {
            "name": "Only Comments",
            "content": """
-- This is a comment only file
-- Date: 2024-01-01
-- Another comment
""",
        },
        {
            "name": "Complex Comment Block",
            "content": """
--------------------------------------------------------------------------------------
-- Date        Sign    History
-- ----------  ------  ---------------------------------------------------------------
-- 2024-01-01  Test    Created test file
-- 2024-01-02  Test    Modified test
--------------------------------------------------------------------------------------

client TestClient;
component TEST;
layer Core;
projection TestHandling;
""",
        },
        {
            "name": "Dynamic Component Dependencies",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

@DynamicComponentDependency ORDER
include fragment OrderFragment;

@DynamicComponentDependency PURCH  
include fragment PurchaseFragment;
""",
        },
        {
            "name": "Complex Navigator Structure",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

navigator {
   entry TestEntry parent TestNavigator.Section at index 100 {
      label = "Test Entry";
      page Form home Test;
   }
   entry AnotherEntry parent TestNavigator.OtherSection at index 200 {
      label = "Another Entry";
      page List home Tests;
   }
}
""",
        },
    ]

    for test_case in edge_cases:
        print(f"\n--- {test_case['name']} ---")
        result = analyze_client_file(
            test_case["content"], f"{test_case['name']}.client"
        )

        print(f"Valid: {result['valid']}")
        print(f"Issues: {result['errors']}E, {result['warnings']}W, {result['hints']}H")

        if result["diagnostics"]:
            for diag in result["diagnostics"]:
                print(f"  {diag['severity']}: {diag['message']}")
        else:
            print("  Clean analysis")

        if result["ast"]:
            print(f"  AST: {len(result['ast']['children'])} top-level elements")


def test_performance():
    """Test analyzer performance with a large client file"""

    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)

    # Test with the largest client file we found
    large_file = "_work/order/model/order/CustomerOrder.client"

    try:
        with open(large_file, "r", encoding="utf-8") as f:
            content = f.read()

        import time

        start_time = time.time()

        result = analyze_client_file(content, large_file)

        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"File: {large_file}")
        print(f"Size: {len(content)} characters, {len(content.split())} lines")
        print(f"Analysis time: {analysis_time:.3f} seconds")
        print(f"Valid: {result['valid']}")
        print(f"AST nodes: {len(result['ast']['children']) if result['ast'] else 0}")

        # Performance benchmark
        if analysis_time < 1.0:
            print("✓ Performance: Excellent (< 1 second)")
        elif analysis_time < 5.0:
            print("✓ Performance: Good (< 5 seconds)")
        else:
            print("⚠ Performance: Could be improved (> 5 seconds)")

    except FileNotFoundError:
        print(f"Large test file not found: {large_file}")


if __name__ == "__main__":
    print("IFS Cloud Client Analyzer - Error Detection & Edge Case Tests")

    # Test error detection
    test_error_detection()

    # Test edge cases
    test_edge_cases()

    # Test performance
    test_performance()

    print(f"\n{'='*60}")
    print("Extended test suite completed!")
