#!/usr/bin/env python3
"""
Test script for the IFS Cloud Projection Analyzer

This script demonstrates the projection analyzer capabilities
by parsing sample projection files and displaying the AST.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifs_cloud_mcp_server.projection_analyzer import (
    ProjectionAnalyzer,
    print_ast_summary,
    NodeType,
    ProjectionType,
)


def test_sample_projection():
    """Test with a comprehensive sample projection"""
    sample_projection = """
// Sample Customer Order Projection
projection CustomerOrderHandling : BaseProjection;

import component FNDCOB;
import component ORDER;

@DynamicComponentDependency ORDER
ludependency CustomerOrder;
ludependency PartCatalog;

entity CustomerOrderHeader {
    attribute OrderNo Text {
        label = "Order Number";
        maxlength = 12;
        required = true;
    }
    attribute CustomerNo Text {
        label = "Customer";
        maxlength = 20;
        required = true;
        lovswitch = {
            CustomerInfo;
        }
    }
    attribute OrderDate Date {
        label = "Order Date";
        required = true;
    }
    attribute DeliveryDate Date {
        label = "Delivery Date";
    }
    attribute TotalAmount Number {
        label = "Total Amount";
        datatype = NUMBER;
    }
    
    reference CustomerInfo(CustomerNo) to Customer(CustomerNo) {
        label = "Customer";
    }
    
    array OrderLines(OrderNo) to CustomerOrderLine(OrderNo) {
        label = "Order Lines";
    }
    
    action ReleaseOrder {
        parameter OrderNo Text;
        parameter ReleaseDate Date;
        ludependency ReleaseOrderProcess;
    }
    
    function CalculateTotal Number {
        parameter OrderNo Text;
    }
}

entity CustomerOrderLine {
    attribute OrderNo Text {
        required = true;
    }
    attribute LineNo Number {
        required = true;
    }
    attribute PartNo Text {
        maxlength = 25;
        required = true;
    }
    attribute Qty Number {
        label = "Quantity";
        required = true;
    }
    attribute UnitPrice Number {
        label = "Unit Price";
    }
    attribute LineAmount Number {
        label = "Line Amount";
        datatype = NUMBER;
    }
    
    reference PartInfo(PartNo) to InventoryPart(PartNo);
    reference OrderHeader(OrderNo) to CustomerOrderHeader(OrderNo);
    
    action UpdateQuantity {
        parameter NewQty Number;
    }
}

entityset CustomerOrdersSet for CustomerOrderHeader;
entityset OrderLinesSet for CustomerOrderLine;

list CustomerOrdersList for CustomerOrderHeader {
    field OrderNo;
    field CustomerNo;
    field OrderDate;
    field TotalAmount;
    
    command ReleaseOrderCommand for ReleaseOrder;
}

card CustomerOrderCard for CustomerOrderHeader {
    group OrderHeaderGroup {
        field OrderNo;
        field CustomerNo;
        field OrderDate;
        field DeliveryDate;
    }
    
    group OrderTotalsGroup {
        field TotalAmount;
    }
}

page CustomerOrderPage using CustomerOrdersSet {
    list CustomerOrdersList;
    card CustomerOrderCard;
}
"""

    print("🔍 Analyzing Sample Customer Order Projection...")
    print("=" * 60)

    analyzer = ProjectionAnalyzer()
    ast = analyzer.analyze(sample_projection)

    # Print basic summary
    print_ast_summary(ast)

    print("\n" + "=" * 60)
    print("📊 Detailed Analysis")
    print("=" * 60)

    # Entity hierarchy
    hierarchy = analyzer.get_entity_hierarchy(ast)
    print(f"\n🏗️  Entity Hierarchy:")
    for entity, children in hierarchy.items():
        print(f"  📦 {entity}")
        for child in children:
            print(f"    └─ {child}")

    # Client metadata structure
    client_structure = analyzer.get_client_metadata_structure(ast)
    print(f"\n🖥️  Client Metadata Structure:")
    for name, info in client_structure.items():
        print(f"  📱 {name} ({info['type']})")
        if info["attributes"]:
            print(f"    Attributes: {list(info['attributes'].keys())}")
        if info["children"]:
            for child in info["children"]:
                print(f"    └─ {child['name']} ({child['type']})")

    # Action signatures
    signatures = analyzer.get_action_signatures(ast)
    print(f"\n⚡ Action/Function Signatures:")
    for name, sig in signatures.items():
        params = ", ".join([f"{p['name']}: {p['type']}" for p in sig["parameters"]])
        print(f"  🔧 {name}({params}) [{sig['type']}]")

    # References to specific entities
    print(f"\n🔗 References to 'Customer':")
    refs = analyzer.find_references_to(ast, "Customer")
    for ref in refs:
        print(f"  📎 {ref.name} ({ref.node_type.value}) at line {ref.line_start}")

    print("\n" + "=" * 60)
    print("🌟 AST Features Demonstrated:")
    print("=" * 60)
    print("✅ Full and partial projection support")
    print("✅ Entity definitions with attributes and metadata")
    print("✅ References and arrays parsing")
    print("✅ Actions and functions with parameters")
    print("✅ Client metadata (lists, cards, pages)")
    print("✅ LU dependencies and imports")
    print("✅ Hierarchical structure analysis")
    print("✅ Reference tracking")
    print("✅ Type-aware parsing")


def test_partial_projection():
    """Test with a partial (fragment) projection"""
    partial_projection = """
// Partial projection fragment
fragment projection CustomerOrderExtensions;

entity CustomerOrderHeader {
    attribute ExtendedField Text {
        label = "Extended Field";
        maxlength = 50;
    }
    
    action CustomAction {
        parameter InputValue Text;
    }
}

list ExtendedOrdersList for CustomerOrderHeader {
    field ExtendedField;
}
"""

    print("\n🔍 Analyzing Partial Projection Fragment...")
    print("=" * 60)

    analyzer = ProjectionAnalyzer()
    ast = analyzer.analyze(partial_projection)

    print_ast_summary(ast)

    print(f"\n📋 Fragment Details:")
    print(f"  Type: {ast.projection_type.value}")
    print(f"  Extends: {ast.base_projection or 'N/A'}")


def main():
    """Run projection analyzer tests"""
    print("🚀 IFS Cloud Projection Analyzer Test Suite")
    print("=" * 80)

    try:
        # Test full projection
        test_sample_projection()

        # Test partial projection
        test_partial_projection()

        print("\n✅ All tests completed successfully!")
        print("\n💡 The projection analyzer provides rich AST data that can help:")
        print("   • Code completion and IntelliSense")
        print("   • Dependency analysis")
        print("   • Refactoring tools")
        print("   • Code generation")
        print("   • Architecture visualization")
        print("   • Impact analysis")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
