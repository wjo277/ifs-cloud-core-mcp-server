#!/usr/bin/env python3
"""
Quick Search Ranking Analysis

Let's run a few key queries and analyze the results to understand the current ranking issues.
"""

import json
import requests
from collections import Counter
from typing import List, Dict

# Configuration
SEARCH_API_URL = "http://localhost:5700/api/search"


def run_search(query: str, limit: int = 20) -> List[Dict]:
    """Run a search query and return results"""
    try:
        params = {"query": query, "limit": limit}
        response = requests.get(SEARCH_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        return []


def analyze_results(query: str, results: List[Dict], expected_types: List[str] = None):
    """Analyze search results"""
    print(f"\n{'='*80}")
    print(f"QUERY: '{query}'")
    print(f"{'='*80}")

    if not results:
        print("No results found!")
        return

    # Show top 10 results
    print(f"\nTop 10 Results:")
    print(f"{'Rank':<4} {'Score':<8} {'Type':<12} {'Module':<15} {'Name'}")
    print("-" * 80)

    for i, result in enumerate(results[:10], 1):
        score = result.get("score", 0)
        file_type = result.get("type", "")
        module = result.get("module", "")[:14]
        name = result.get("name", "")[:35]
        print(f"{i:<4} {score:<8.1f} {file_type:<12} {module:<15} {name}")

    # File type distribution
    type_counts = Counter([r.get("type", "") for r in results[:10]])
    print(f"\nFile Type Distribution (Top 10):")
    for file_type, count in type_counts.most_common():
        percentage = (count / 10) * 100
        print(f"  {file_type:<12}: {count:2d} files ({percentage:4.1f}%)")

    # Expected vs actual
    if expected_types:
        relevant_count = sum(
            1 for r in results[:10] if r.get("type", "") in expected_types
        )
        print(
            f"\nRelevant file types in top 10: {relevant_count}/10 ({relevant_count*10}%)"
        )
        print(f"Expected types: {expected_types}")


def main():
    """Run analysis on key problematic queries"""

    test_queries = [
        {
            "query": "expense authorization",
            "expected_types": [".plsql", ".client", ".views"],
            "reasoning": "Business logic search - should prioritize implementation over entities",
        },
        {
            "query": "employee status",
            "expected_types": [".plsql", ".views", ".client"],
            "reasoning": "Practical search - users want to see how status works, not just entity definition",
        },
        {
            "query": "customer order",
            "expected_types": [".plsql", ".views", ".projection", ".client"],
            "reasoning": "Balanced search - entity ok but should also show business logic",
        },
        {
            "query": "invoice validation",
            "expected_types": [".plsql", ".views"],
            "reasoning": "Validation logic search - strongly favor business logic files",
        },
        {
            "query": "project management",
            "expected_types": [".plsql", ".client", ".views", ".entity"],
            "reasoning": "General topic - should be balanced across file types",
        },
    ]

    print("IFS Cloud Search Ranking Analysis")
    print("=================================")
    print("Analyzing current ranking issues...")

    overall_stats = {
        "total_queries": len(test_queries),
        "entity_dominance": 0,
        "views_dominance": 0,
        "balanced_results": 0,
    }

    for test in test_queries:
        query = test["query"]
        expected_types = test["expected_types"]

        results = run_search(query, 20)
        analyze_results(query, results, expected_types)

        # Analyze dominance issues
        top_5_types = [r.get("type", "") for r in results[:5]]
        entity_count = top_5_types.count(".entity")
        views_count = top_5_types.count(".views")

        if entity_count >= 3:
            overall_stats["entity_dominance"] += 1
            print(f"⚠️  ENTITY DOMINANCE: {entity_count}/5 entities in top 5")

        if views_count >= 3:
            overall_stats["views_dominance"] += 1
            print(f"⚠️  VIEWS DOMINANCE: {views_count}/5 views in top 5")

        relevant_in_top_5 = sum(
            1 for r in results[:5] if r.get("type", "") in expected_types
        )
        if relevant_in_top_5 >= 3:
            overall_stats["balanced_results"] += 1
            print(f"✅ GOOD BALANCE: {relevant_in_top_5}/5 relevant files in top 5")
        else:
            print(
                f"❌ POOR BALANCE: Only {relevant_in_top_5}/5 relevant files in top 5"
            )

        print(f"\nReasoning: {test['reasoning']}")

        # Wait for user to review
        input("\nPress Enter to continue to next query...")

    # Summary
    print(f"\n{'='*80}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries analyzed: {overall_stats['total_queries']}")
    print(f"Queries with entity dominance: {overall_stats['entity_dominance']}")
    print(f"Queries with views dominance: {overall_stats['views_dominance']}")
    print(f"Queries with good balance: {overall_stats['balanced_results']}")

    dominance_issues = (
        overall_stats["entity_dominance"] + overall_stats["views_dominance"]
    )
    print(
        f"\nDominance issues: {dominance_issues}/{overall_stats['total_queries']} queries"
    )

    if dominance_issues > overall_stats["total_queries"] / 2:
        print("🔴 CRITICAL: File type dominance is a major issue")
    elif dominance_issues > 0:
        print("🟡 WARNING: Some file type dominance issues detected")
    else:
        print("🟢 GOOD: No significant dominance issues")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR RANKING IMPROVEMENTS:")
    print("=" * 80)

    if overall_stats["entity_dominance"] > 0:
        print("1. REDUCE ENTITY FILE SCORING:")
        print(
            "   - Apply stronger penalties to .entity files for business logic queries"
        )
        print("   - Only boost entities for entity-name exact matches")

    if overall_stats["views_dominance"] > 0:
        print("2. BALANCE VIEWS WITH OTHER FILE TYPES:")
        print("   - Reduce views scoring slightly to make room for other file types")
        print("   - Boost .plsql and .client files more aggressively")

    if overall_stats["balanced_results"] < overall_stats["total_queries"]:
        print("3. IMPROVE BUSINESS CONTEXT UNDERSTANDING:")
        print(
            "   - Detect business logic terms (authorization, validation, calculation)"
        )
        print("   - Apply stronger boosts to implementation files (.plsql, .client)")
        print("   - Consider query intent (workflow vs data access)")

    print("\n4. SUGGESTED SCORING ADJUSTMENTS:")
    print(
        "   - Business logic queries: .plsql (*3.0), .client (*2.0), .views (*1.5), .entity (*0.5)"
    )
    print(
        "   - Entity queries: .plsql (*2.0), .views (*1.8), .entity (*1.5), .client (*1.3)"
    )
    print(
        "   - General queries: .plsql (*2.5), .views (*1.6), .client (*1.4), .entity (*1.0)"
    )


if __name__ == "__main__":
    main()
