#!/usr/bin/env python3
"""
Test Conservative Client Analyzer with Real IFS Cloud Client Files

This script tests the conservative client analyzer against real IFS Cloud client files
to ensure it produces zero false positives while providing useful analysis.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.ifs_cloud_mcp_server.client_analyzer import analyze_client_file


def test_real_client_files():
    """Test the analyzer with real client files from _work directory"""

    # Test files from the _work directory
    test_files = [
        "_work/order/model/order/SalesChargeType.client",
        "_work/order/model/order/CustomerOrder.client",
        "_work/purch/model/purch/PurchaseOrder.client",
        "_work/purch/model/purch/Buyers.client",
    ]

    results = {}

    for test_file in test_files:
        print(f"\n=== Testing {test_file} ===")

        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()

            result = analyze_client_file(content, test_file)
            results[test_file] = result

            print(f"Valid: {result['valid']}")
            print(f"Errors: {result['errors']}")
            print(f"Warnings: {result['warnings']}")
            print(f"Info: {result['info']}")
            print(f"Hints: {result['hints']}")

            if result["diagnostics"]:
                print("Diagnostics:")
                for diag in result["diagnostics"]:
                    print(
                        f"  {diag['severity']} (Line {diag['line']}): {diag['message']}"
                    )
            else:
                print("No diagnostics - clean analysis!")

            # Show AST summary
            if result["ast"]:
                print(
                    f"AST: {result['ast']['type']} with {len(result['ast']['children'])} children"
                )

                # Show key structural elements found
                children_types = [child["type"] for child in result["ast"]["children"]]
                print(f"Structural elements: {', '.join(set(children_types))}")

        except FileNotFoundError:
            print(f"File not found: {test_file}")
            results[test_file] = {"error": "File not found"}
        except Exception as e:
            print(f"Error analyzing {test_file}: {e}")
            results[test_file] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful_tests = []
    failed_tests = []
    total_errors = 0
    total_warnings = 0

    for file, result in results.items():
        if "error" in result:
            failed_tests.append(file)
        else:
            successful_tests.append(file)
            total_errors += result["errors"]
            total_warnings += result["warnings"]

    print(f"Successfully analyzed: {len(successful_tests)}/{len(test_files)} files")
    print(f"Total errors found: {total_errors}")
    print(f"Total warnings found: {total_warnings}")

    if successful_tests:
        print(f"\nSuccessful analyses:")
        for file in successful_tests:
            result = results[file]
            status = (
                "✓ CLEAN"
                if result["errors"] == 0 and result["warnings"] == 0
                else f"⚠ {result['errors']}E, {result['warnings']}W"
            )
            print(f"  {status}: {file}")

    if failed_tests:
        print(f"\nFailed analyses:")
        for file in failed_tests:
            print(f"  ✗ FAILED: {file}")

    # Conservative approach success check
    if total_errors == 0:
        print(
            f"\n🎉 SUCCESS: Conservative approach achieved - 0 errors on all real client files!"
        )
        print(
            "This confirms the analyzer doesn't produce false positives on legitimate IFS Cloud code."
        )
    else:
        print(
            f"\n⚠ ATTENTION: Found {total_errors} errors - need to review for potential false positives"
        )

    return results


def test_syntax_variations():
    """Test with various client syntax patterns"""

    print("\n" + "=" * 60)
    print("TESTING SYNTAX VARIATIONS")
    print("=" * 60)

    # Test valid client patterns
    test_cases = [
        {
            "name": "Minimal Valid Client",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;
""",
        },
        {
            "name": "Client with Navigator",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

navigator {
   entry TestEntry parent TestNavigator.Section at index 100 {
      label = "Test";
   }
}
""",
        },
        {
            "name": "Client with Include Fragments",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

include fragment TestFragment;
include fragment AnotherTestFragment;
""",
        },
        {
            "name": "Client with Page and Command",
            "content": """
client TestClient;
component TEST;
layer Core;
projection TestHandling;

page Form using TestSet {
   label = "Test";
   group TestGroup;
}

command TestCommand for Test {
   enabled = [true];
   execute {
      call TestAction();
   }
}
""",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        result = analyze_client_file(
            test_case["content"], f"{test_case['name']}.client"
        )

        print(f"Valid: {result['valid']}")
        print(f"Errors: {result['errors']}, Warnings: {result['warnings']}")

        if result["diagnostics"]:
            for diag in result["diagnostics"]:
                print(f"  {diag['severity']}: {diag['message']}")
        else:
            print("  Clean - no issues detected!")


if __name__ == "__main__":
    print("IFS Cloud Conservative Client Analyzer Test Suite")
    print("=" * 60)

    # Test with real files
    test_real_client_files()

    # Test syntax variations
    test_syntax_variations()

    print(f"\n{'='*60}")
    print("Test suite completed!")
