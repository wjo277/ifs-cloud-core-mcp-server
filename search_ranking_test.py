#!/usr/bin/env python3
"""
IFS Cloud Search Ranking Test and Benchmark System

This script systematically tests search queries that IFS business users would typically make,
evaluates the results, and provides benchmarks for improving the search ranking algorithm.
"""

import json
import requests
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import time
from datetime import datetime
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SEARCH_API_URL = "http://localhost:5700/api/search"
RESULTS_LIMIT = 100
BENCHMARK_FILE = "search_ranking_benchmark.json"
TEST_RESULTS_FILE = "search_ranking_test_results.json"


@dataclass
class SearchQuery:
    """Represents a search query with business context"""

    query: str
    department: str
    use_case: str
    description: str
    expected_file_types: List[str]  # File types that would be most useful
    business_priority: str  # "high", "medium", "low"


@dataclass
class SearchResult:
    """Represents a search result for evaluation"""

    path: str
    name: str
    type: str
    score: float
    module: str
    logical_unit: str
    rank: int


@dataclass
class ExpectedResult:
    """Expected result for benchmarking"""

    path: str
    relevance_score: int  # 1-10 scale
    reason: str


@dataclass
class QueryEvaluation:
    """Evaluation of a query's results"""

    query: SearchQuery
    results: List[SearchResult]
    expected_results: List[ExpectedResult]
    timestamp: str


class SearchRankingTester:
    """Main tester class for search ranking evaluation"""

    def __init__(self):
        self.queries = self._define_test_queries()
        self.benchmark_data = self._load_benchmark()

    def _define_test_queries(self) -> List[SearchQuery]:
        """Define typical business user search queries"""
        return [
            # Finance Department Queries
            SearchQuery(
                query="expense authorization",
                department="Finance",
                use_case="Understanding expense approval workflow",
                description="Finance team needs to understand how expense authorization works",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="high",
            ),
            SearchQuery(
                query="invoice validation",
                department="Finance",
                use_case="Invoice processing rules",
                description="Understanding how invoices are validated and processed",
                expected_file_types=[".plsql", ".views", ".entity"],
                business_priority="high",
            ),
            SearchQuery(
                query="accounting rules",
                department="Finance",
                use_case="Chart of accounts and posting rules",
                description="Understanding accounting rules and automatic postings",
                expected_file_types=[".plsql", ".views", ".entity"],
                business_priority="high",
            ),
            # HR Department Queries
            SearchQuery(
                query="employee salary calculation",
                department="HR",
                use_case="Payroll processing",
                description="HR needs to understand how employee salaries are calculated",
                expected_file_types=[".plsql", ".views"],
                business_priority="high",
            ),
            SearchQuery(
                query="employee status changes",
                department="HR",
                use_case="Employee lifecycle management",
                description="Managing employee status transitions (hired, active, terminated)",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="medium",
            ),
            SearchQuery(
                query="leave approval workflow",
                department="HR",
                use_case="Leave management",
                description="Understanding how leave requests are approved",
                expected_file_types=[".plsql", ".client"],
                business_priority="medium",
            ),
            # Procurement Department Queries
            SearchQuery(
                query="purchase order approval",
                department="Procurement",
                use_case="PO approval workflow",
                description="Understanding purchase order approval process",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="high",
            ),
            SearchQuery(
                query="supplier evaluation",
                department="Procurement",
                use_case="Vendor management",
                description="How suppliers are evaluated and rated",
                expected_file_types=[".plsql", ".views", ".entity"],
                business_priority="medium",
            ),
            SearchQuery(
                query="goods receipt matching",
                department="Procurement",
                use_case="Three-way matching process",
                description="How goods receipts are matched with POs and invoices",
                expected_file_types=[".plsql", ".views"],
                business_priority="high",
            ),
            # Sales Department Queries
            SearchQuery(
                query="customer order pricing",
                department="Sales",
                use_case="Order pricing logic",
                description="Understanding how customer order prices are calculated",
                expected_file_types=[".plsql", ".views"],
                business_priority="high",
            ),
            SearchQuery(
                query="customer credit check",
                department="Sales",
                use_case="Credit management",
                description="How customer credit limits are checked",
                expected_file_types=[".plsql", ".views"],
                business_priority="high",
            ),
            SearchQuery(
                query="delivery planning",
                department="Sales",
                use_case="Order fulfillment",
                description="How deliveries are planned and scheduled",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="medium",
            ),
            # IT Department Queries (Technical)
            SearchQuery(
                query="user authentication",
                department="IT",
                use_case="Security implementation",
                description="Understanding user authentication mechanisms",
                expected_file_types=[".plsql", ".entity", ".views"],
                business_priority="high",
            ),
            SearchQuery(
                query="data archiving",
                department="IT",
                use_case="Data management",
                description="How data archiving is implemented",
                expected_file_types=[".plsql", ".entity"],
                business_priority="medium",
            ),
            # Manufacturing Department Queries
            SearchQuery(
                query="work order scheduling",
                department="Manufacturing",
                use_case="Production planning",
                description="How work orders are scheduled",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="high",
            ),
            SearchQuery(
                query="material requirements planning",
                department="Manufacturing",
                use_case="MRP calculations",
                description="How material requirements are calculated",
                expected_file_types=[".plsql", ".views"],
                business_priority="high",
            ),
            # Quality Department Queries
            SearchQuery(
                query="quality control inspection",
                department="Quality",
                use_case="QC processes",
                description="Quality control inspection procedures",
                expected_file_types=[".plsql", ".views", ".client"],
                business_priority="medium",
            ),
            # General Entity Searches (should balance different file types)
            SearchQuery(
                query="customer information",
                department="General",
                use_case="Customer data access",
                description="Accessing customer master data",
                expected_file_types=[".entity", ".views", ".projection", ".client"],
                business_priority="medium",
            ),
            SearchQuery(
                query="project management",
                department="General",
                use_case="Project tracking",
                description="Project management functionality",
                expected_file_types=[".plsql", ".views", ".entity", ".client"],
                business_priority="medium",
            ),
        ]

    def _load_benchmark(self) -> Dict[str, Any]:
        """Load existing benchmark data if available"""
        benchmark_path = Path(BENCHMARK_FILE)
        if benchmark_path.exists():
            with open(benchmark_path, "r") as f:
                return json.load(f)
        return {}

    def _save_benchmark(self, data: Dict[str, Any]):
        """Save benchmark data"""
        with open(BENCHMARK_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def run_search_query(self, query: SearchQuery) -> List[SearchResult]:
        """Execute a search query against the API"""
        try:
            params = {"query": query.query, "limit": RESULTS_LIMIT}

            response = requests.get(SEARCH_API_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = []

            for i, result in enumerate(data.get("results", [])):
                results.append(
                    SearchResult(
                        path=result.get("path", ""),
                        name=result.get("name", ""),
                        type=result.get("type", ""),
                        score=result.get("score", 0.0),
                        module=result.get("module", ""),
                        logical_unit=result.get("logical_unit", ""),
                        rank=i + 1,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error executing search query '{query.query}': {e}")
            return []

    def evaluate_results(self, query: SearchQuery, results: List[SearchResult]) -> None:
        """Manually evaluate search results and identify top 5 most relevant"""
        print(f"\n{'='*80}")
        print(f"EVALUATING QUERY: '{query.query}'")
        print(f"Department: {query.department}")
        print(f"Use Case: {query.use_case}")
        print(f"Description: {query.description}")
        print(f"Expected File Types: {query.expected_file_types}")
        print(f"{'='*80}")

        print(f"\nTop 20 Results (out of {len(results)} total):")
        print(f"{'Rank':<4} {'Score':<8} {'Type':<12} {'Module':<15} {'Name':<40}")
        print("-" * 90)

        for i, result in enumerate(results[:20], 1):
            print(
                f"{i:<4} {result.score:<8.1f} {result.type:<12} {result.module[:14]:<15} {result.name[:39]}"
            )

        # Analysis by file type
        type_counts = Counter([r.type for r in results[:20]])
        print(f"\nFile Type Distribution (Top 20):")
        for file_type, count in type_counts.most_common():
            percentage = (count / 20) * 100
            print(f"  {file_type:<12}: {count:2d} files ({percentage:4.1f}%)")

        # Analysis by expected file types
        expected_types = set(query.expected_file_types)
        relevant_results = [r for r in results[:20] if r.type in expected_types]
        print(
            f"\nRelevant File Types in Top 20: {len(relevant_results)}/20 ({len(relevant_results)/20*100:.1f}%)"
        )

        # Ask for manual evaluation
        print(f"\n" + "=" * 50)
        print("MANUAL EVALUATION NEEDED")
        print("=" * 50)
        print("Please review the results above and the actual file contents.")
        print("Which 5 results would be MOST RELEVANT for this business use case?")
        print("Consider:")
        print("- Business logic implementation (how things actually work)")
        print("- User interfaces (how users interact with the system)")
        print("- Data structures (what data is stored and how)")
        print("- Practical utility for the business user")
        print("\nTop 10 candidates:")

        for i, result in enumerate(results[:10], 1):
            print(f"{i:2d}. {result.name} ({result.type}) - Score: {result.score:.1f}")
            print(f"    Path: {result.path}")
            print(f"    Module: {result.module}, LU: {result.logical_unit}")
            print()

    def run_all_evaluations(self):
        """Run evaluations for all test queries"""
        evaluations = []

        print(f"Starting search ranking evaluation with {len(self.queries)} queries...")
        print(f"Results will be saved to {TEST_RESULTS_FILE}")

        for i, query in enumerate(self.queries, 1):
            print(f"\n[{i}/{len(self.queries)}] Processing query: '{query.query}'")

            results = self.run_search_query(query)
            if results:
                evaluation = QueryEvaluation(
                    query=query,
                    results=results,
                    expected_results=[],  # To be filled manually
                    timestamp=datetime.now().isoformat(),
                )
                evaluations.append(evaluation)

                # Evaluate results
                self.evaluate_results(query, results)

                # Wait for user input before continuing
                input("\nPress Enter to continue to next query...")
            else:
                print(f"No results found for query: '{query.query}'")

        # Save results
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.queries),
            "evaluations": [asdict(e) for e in evaluations],
        }

        with open(TEST_RESULTS_FILE, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"\nEvaluation complete. Results saved to {TEST_RESULTS_FILE}")
        print("Next steps:")
        print("1. Review the saved results and manually add expected_results")
        print("2. Use create_benchmark() to create a benchmark test")
        print("3. Use run_benchmark() to test ranking improvements")

    def create_benchmark(self):
        """Create benchmark from evaluated results (after manual review)"""
        if not Path(TEST_RESULTS_FILE).exists():
            print(
                f"Please run run_all_evaluations() first to generate {TEST_RESULTS_FILE}"
            )
            return

        with open(TEST_RESULTS_FILE, "r") as f:
            data = json.load(f)

        benchmark = {
            "created_at": datetime.now().isoformat(),
            "description": "Search ranking benchmark based on IFS business user queries",
            "queries": {},
        }

        for evaluation in data["evaluations"]:
            query_text = evaluation["query"]["query"]
            benchmark["queries"][query_text] = {
                "query": evaluation["query"],
                "expected_results": evaluation.get("expected_results", []),
                "total_results": len(evaluation["results"]),
            }

        self._save_benchmark(benchmark)
        print(f"Benchmark created and saved to {BENCHMARK_FILE}")

    def run_benchmark(self) -> Dict[str, float]:
        """Run benchmark test against current search algorithm"""
        if not self.benchmark_data:
            print("No benchmark data available. Run create_benchmark() first.")
            return {}

        print("Running benchmark test...")
        results = {}

        for query_text, benchmark_query in self.benchmark_data.get(
            "queries", {}
        ).items():
            query = SearchQuery(**benchmark_query["query"])
            search_results = self.run_search_query(query)

            # Calculate relevance score
            score = self._calculate_relevance_score(
                search_results, benchmark_query.get("expected_results", [])
            )

            results[query_text] = score
            print(f"Query: '{query_text}' - Score: {score:.2f}")

        overall_score = sum(results.values()) / len(results) if results else 0
        print(f"\nOverall Benchmark Score: {overall_score:.2f}")

        return results

    def _calculate_relevance_score(
        self, results: List[SearchResult], expected: List[ExpectedResult]
    ) -> float:
        """Calculate relevance score for a query"""
        if not expected:
            return 0.0

        expected_paths = {e["path"]: e["relevance_score"] for e in expected}
        score = 0.0
        max_possible_score = sum(exp["relevance_score"] for exp in expected)

        # Score based on position of expected results in top 10
        for i, result in enumerate(results[:10]):
            if result.path in expected_paths:
                relevance = expected_paths[result.path]
                position_weight = (10 - i) / 10  # Higher weight for higher positions
                score += relevance * position_weight

        return (score / max_possible_score) * 100 if max_possible_score > 0 else 0.0


def main():
    """Main function"""
    tester = SearchRankingTester()

    print("IFS Cloud Search Ranking Test System")
    print("=====================================")
    print("1. run_all_evaluations() - Evaluate all queries and create test results")
    print("2. create_benchmark() - Create benchmark from manual evaluations")
    print("3. run_benchmark() - Test current algorithm against benchmark")

    # For automated testing, uncomment the line below:
    # tester.run_all_evaluations()


if __name__ == "__main__":
    main()
