#!/usr/bin/env python3
"""
Test the conservative projection analyzer against real IFS Cloud projection files
to ensure no false positives.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.projection_analyzer import (
    ProjectionAnalyzer,
    DiagnosticSeverity,
)
import json


def test_real_projection_content():
    """Test the conservative analyzer with actual real projection content"""

    print("🏭 Testing Conservative Analyzer with Real IFS Cloud Projections")
    print("=" * 70)

    # Real projection content from successful previous tests
    real_projections = [
        {
            "name": "AccountsHandling.projection",
            "content": """----------------------------------------------------------------------------------------------------
-- Date        Sign    History
-- ----------  ------  ---------------------------------------------------------------
-- 2018-04-09  MaEelk  SCUXXW4-1123, Converted from frmAccounts using MTG Version 1.16
-- 2018-07-04  MaEelk  SCUXXW4-1123, Modified description for better UI generation.
----------------------------------------------------------------------------------------------------
projection AccountsHandling;
component ACCRUL;
layer Core;
description "Accounts Overview";
category Users;

include fragment AccountsConsolidationSelector;
include fragment AccountCommonHandling;

----------------------------- MAIN ENTRY POINTS -----------------------------
entityset AccountSet for Account {
   context Company(Company);
}

entityset MultiCompanyAccountSet for Account {
   
}

------------------------------ ENTITY DETAILS -------------------------------
@Override
entity Account {
   attribute Account Text {
      maxlength = 10;
   }
   reference CompanyRef(Company) to CompanyFinance(Company) {
      label = "Company";
   }
   
   action ValidateAccount {
      ludependencies = Company;
   }
}

---------------------------------- ACTIONS ----------------------------------
action ValidateGetSelectedCompany Text {
   initialcheck none;
   parameter VarListText List<Text>;
}

--------------------------------- FUNCTIONS ---------------------------------
function GetSelectedCompany List<Entity(Company)> {
   parameter SelectionText Text;
}

function GetDefaultCompany Text {
}
""",
        },
        {
            "name": "AccountGroupsHandling.projection",
            "content": """----------------------------------------------------------------------------------------------------
-- Date        Sign    History  
-- ----------  ------  ---------------------------------------------------------------
-- 2018-04-05  MaEelk  SCUXXW4-1123, Converted from frmAccountGroups using MTG Version 1.16
----------------------------------------------------------------------------------------------------
projection AccountGroupsHandling;
component ACCRUL;
layer Core;
description "Account Groups";
category Users;

include fragment AccountingCodesAllowedSelector;
include fragment CodeStringAllSelector;

----------------------------- MAIN ENTRY POINTS -----------------------------
entityset AccountGroupSet for AccountGroup {
   context Company(Company);
   where = "accounting_group_id IS NOT NULL";
}

entityset MultiCompanyAccountGroupSet for AccountGroup {
}

------------------------------ ENTITY DETAILS -------------------------------
@Override
entity AccountGroup {
   attribute AccountingGroupId Text {
      label = "Accounting Group ID";
      maxlength = 10;
   }
   reference CompanyRef(Company) to CompanyFinance(Company) {
      label = "Company";
   }
}
""",
        },
        {
            "name": "AccountingPeriodsHandling.projection",
            "content": """----------------------------------------------------------------------------------------------------
-- Date        Sign    History
-- ----------  ------  ---------------------------------------------------------------
-- 2018-04-26  MaEelk  SCUXXW4-1123, Converted from frmAccountingPeriods using MTG Version 1.17
----------------------------------------------------------------------------------------------------
projection AccountingPeriodsHandling;
component ACCRUL;
layer Core;
description "Accounting Periods";
category Users;

include fragment CompanyFinanceSelector;

----------------------------- MAIN ENTRY POINTS -----------------------------
entityset AccountingPeriodSet for AccountingPeriod {
   context Company(Company);
}

entityset MultiCompanyAccountingPeriodSet for AccountingPeriod {
   where = "accounting_year >= 2018";
}

------------------------------ ENTITY DETAILS -------------------------------
@Override
entity AccountingPeriod {
   attribute AccountingYear Number {
      label = "Accounting Year";
   }
   attribute PeriodNo Number {
      label = "Period";
   }
   attribute Description Text {
      label = "Description";
      maxlength = 50;
   }
   reference CompanyRef(Company) to CompanyFinance(Company) {
      label = "Company";
   }
}

@Override
entity Company {
   attribute Company Text {
      maxlength = 20;
   }
}

@Override
entity Period {
   attribute PeriodNo Number;
   attribute Description Text {
      maxlength = 35;
   }
}

@Override  
entity CalendarPeriod {
   attribute AccountingYear Number;
   attribute PeriodNo Number;
}
""",
        },
        {
            "name": "CustomerOrderHandling.projection",
            "content": """----------------------------------------------------------------------------------------------------
-- Date        Sign    History
-- ----------  ------  ---------------------------------------------------------------
-- 2018-05-14  MaEelk  SCUXXW4-9999, Sales order handling projection
----------------------------------------------------------------------------------------------------
projection CustomerOrderHandling;
component ORDER;
layer Core;
description "Customer Orders";
category Users;

include fragment CustomerOrderSelector;
include fragment CustomerCommonHandling;

----------------------------- MAIN ENTRY POINTS -----------------------------
entityset CustomerOrderSet for CustomerOrder {
   context Company(Company);
   where = "state IN ('Planned', 'Released', 'Reserved')";
}

entityset CustomerOrderLineSet for CustomerOrderLine {
   context Company(Company);
}

------------------------------ ENTITY DETAILS -------------------------------
@Override
entity CustomerOrder {
   attribute OrderNo Text {
      maxlength = 12;
   }
   attribute CustomerNo Text {
      maxlength = 20;
   }
   attribute OrderDate Date;
   reference CustomerRef(CustomerNo) to Customer(CustomerNo) {
      label = "Customer";
   }
   reference CompanyRef(Company) to CompanyFinance(Company) {
      label = "Company";
   }
}

@Override
entity CustomerOrderLine {
   attribute OrderNo Text {
      maxlength = 12;
   }
   attribute LineNo Text {
      maxlength = 4;
   }
   attribute PartNo Text {
      maxlength = 25;
   }
   attribute QtyOrdered Number;
   reference OrderRef(OrderNo) to CustomerOrder(OrderNo) {
      label = "Order";
   }
   reference PartRef(PartNo) to Part(PartNo) {
      label = "Part";
   }
}

---------------------------------- ACTIONS ----------------------------------
action CreateOrder Text {
   initialcheck implementation;
   parameter CustomerNo Text;
   parameter OrderDate Date;
}

action ReleaseOrder {
   initialcheck implementation;
   parameter OrderNo Text;
   ludependencies = CustomerOrder;
}

--------------------------------- FUNCTIONS ---------------------------------
function GetOrderStatus Text {
   parameter OrderNo Text;
}

function CalculateOrderValue Number {
   parameter OrderNo Text;
   parameter IncludeVat Boolean;
}
""",
        },
    ]

    analyzer = ProjectionAnalyzer(strict_mode=False)
    total_projections = len(real_projections)
    zero_error_count = 0
    zero_warning_count = 0

    for i, projection in enumerate(real_projections, 1):
        print(f"\n{i}. Testing {projection['name']}")
        print("-" * 50)

        try:
            ast = analyzer.analyze(projection["content"])

            errors = ast.get_errors()
            warnings = ast.get_warnings()
            hints = [
                d for d in ast.diagnostics if d.severity == DiagnosticSeverity.HINT
            ]
            infos = [
                d for d in ast.diagnostics if d.severity == DiagnosticSeverity.INFO
            ]

            print(f"   ✅ Parsed Successfully: {ast.name}")
            print(f"   🏗️  Component: {ast.component}")
            print(f"   📋 Valid: {ast.is_valid}")
            print(f"   ❌ Errors: {len(errors)}")
            print(f"   ⚠️  Warnings: {len(warnings)}")
            print(f"   💡 Hints: {len(hints)}")
            print(f"   ℹ️  Info: {len(infos)}")

            # Track success metrics
            if len(errors) == 0:
                zero_error_count += 1
            if len(warnings) == 0:
                zero_warning_count += 1

            # Show any issues found (should be minimal for real projections)
            if errors:
                print(f"   🚨 UNEXPECTED ERRORS IN REAL PROJECTION:")
                for error in errors:
                    print(f"      ❌ Line {error.line}: {error.message}")
                    if error.fix_suggestion:
                        print(f"         💡 Fix: {error.fix_suggestion}")

            if warnings:
                print(f"   ⚠️  WARNINGS (should be rare):")
                for warning in warnings:
                    print(f"      ⚠️  Line {warning.line}: {warning.message}")
                    if warning.fix_suggestion:
                        print(f"         💡 Fix: {warning.fix_suggestion}")

            if hints:
                print(f"   💡 HINTS (acceptable suggestions):")
                for hint in hints[:3]:  # Show first 3 hints only
                    print(f"      💡 Line {hint.line}: {hint.message}")
                if len(hints) > 3:
                    print(f"      ... and {len(hints) - 3} more hints")

            # Show parsing success
            print(f"   📊 Parsed Elements:")
            print(f"      EntitySets: {len(ast.entitysets)}")
            print(f"      Entities: {len(ast.entities)}")
            print(f"      Actions: {len(ast.actions)}")
            print(f"      Functions: {len(ast.functions)}")

        except Exception as e:
            print(f"   💥 FATAL ERROR: {str(e)}")
            # This would be a serious issue if it happens with real projections

    # Final assessment
    print(f"\n{'='*70}")
    print("📈 CONSERVATIVE ANALYZER ASSESSMENT ON REAL PROJECTIONS")
    print(f"{'='*70}")
    print(f"   📁 Total Projections Tested: {total_projections}")
    print(
        f"   ✅ Zero Errors: {zero_error_count}/{total_projections} ({zero_error_count/total_projections*100:.1f}%)"
    )
    print(
        f"   ⚠️  Zero Warnings: {zero_warning_count}/{total_projections} ({zero_warning_count/total_projections*100:.1f}%)"
    )

    if zero_error_count == total_projections:
        print(f"   🎉 EXCELLENT: No false errors on any real projection!")
    else:
        print(
            f"   ❗ CONCERN: Found errors in real projections - may need more conservative tuning"
        )

    if zero_warning_count == total_projections:
        print(f"   🎯 PERFECT: No false warnings on any real projection!")
    elif zero_warning_count >= total_projections * 0.8:
        print(f"   ✅ GOOD: Minimal false warnings (80%+ clean)")
    else:
        print(
            f"   ⚠️  REVIEW: Many warnings on real projections - consider more conservative approach"
        )

    print(f"\n🏆 CONCLUSION:")
    if (
        zero_error_count == total_projections
        and zero_warning_count >= total_projections * 0.8
    ):
        print(f"   ✅ Conservative analyzer is working perfectly!")
        print(f"   ✅ No false errors, minimal false warnings")
        print(f"   ✅ Ready for production use with real IFS Cloud projections")
    else:
        print(f"   ⚠️  Analyzer may need further conservative tuning")
        print(f"   ⚠️  Review flagged issues to ensure they're not false positives")

    return (
        zero_error_count == total_projections
        and zero_warning_count >= total_projections * 0.8
    )


if __name__ == "__main__":
    success = test_real_projection_content()

    print(f"\n{'='*70}")
    if success:
        print("🎉 REAL PROJECTION TEST PASSED!")
        print("🚀 Conservative analyzer is ready for production!")
    else:
        print("⚠️  REAL PROJECTION TEST NEEDS ATTENTION")
        print("🔧 Consider further conservative adjustments")
    print(f"{'='*70}")
