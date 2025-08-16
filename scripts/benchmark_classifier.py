"""
Benchmark FastAI Intent Classifier Performance: CPU vs GPU

This script compares inference speed and accuracy between CPU and GPU modes.
"""

import time
import torch
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from dataclasses import dataclass

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ifs_cloud_mcp_server.proper_fastai_classifier import (
    FastAIIntentClassifier,
    QueryIntent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    device: str
    total_queries: int
    total_time: float
    avg_time_per_query: float
    queries_per_second: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    accuracy_stats: Dict[str, int]


class ClassifierBenchmark:
    """Benchmark the FastAI intent classifier."""

    def __init__(self):
        self.test_queries = [
            # Quick queries for speed testing
            "customer order validation",
            "database schema",
            "user interface form",
            "rest api endpoint",
            "inventory report",
            "error handling",
            "business workflow",
            "data access view",
            "authorization check",
            "troubleshoot issue",
            # More comprehensive test set
            "validate purchase order workflow",
            "customer information entity definition",
            "create new page component",
            "projection service integration",
            "generate sales report view",
            "debug authentication error",
            "approval process business logic",
            "query database table",
            "user permission validation",
            "system configuration issue",
            "supplier invoice processing",
            "product catalog schema",
            "navigation menu component",
            "external api integration",
            "financial dashboard view",
            "connection timeout error",
            "pricing calculation rules",
            "export data query",
            "role-based access control",
            "performance optimization",
            # Edge cases and variations
            "complex multi-step workflow validation",
            "detailed entity relationship definition",
            "responsive ui component design",
            "microservice api orchestration",
            "real-time analytics dashboard",
            "critical system failure diagnosis",
            "advanced business rule engine",
            "optimized database query execution",
            "fine-grained permission management",
            "comprehensive system monitoring",
        ]

    def warm_up_model(self, classifier: FastAIIntentClassifier, num_warmup: int = 5):
        """Warm up the model to avoid cold start effects."""
        logger.info(f"Warming up model with {num_warmup} queries...")
        for i in range(num_warmup):
            classifier.predict(self.test_queries[i % len(self.test_queries)])
        logger.info("✅ Model warmed up")

    def benchmark_device(
        self, use_gpu: bool, num_iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark classifier on specified device."""
        device = "GPU" if use_gpu else "CPU"
        logger.info(f"🚀 Benchmarking {device} performance...")

        # Initialize classifier
        classifier = FastAIIntentClassifier(use_gpu=use_gpu)

        if not classifier.is_trained:
            logger.error(f"❌ Model not available for {device} testing")
            return None

        # Warm up
        self.warm_up_model(classifier)

        # Benchmark iterations
        all_times = []
        accuracy_stats = {"correct": 0, "total": 0}

        for iteration in range(num_iterations):
            logger.info(f"  Iteration {iteration + 1}/{num_iterations}")
            iteration_times = []

            for query in self.test_queries:
                start_time = time.perf_counter()
                result = classifier.predict(query)
                end_time = time.perf_counter()

                inference_time = end_time - start_time
                iteration_times.append(inference_time)
                all_times.append(inference_time)

                # Basic accuracy check (this would need ground truth for real accuracy)
                accuracy_stats["total"] += 1
                if result.confidence > 0.5:  # Reasonable confidence threshold
                    accuracy_stats["correct"] += 1

        # Calculate statistics
        total_time = sum(all_times)
        avg_time = total_time / len(all_times)
        qps = len(all_times) / total_time

        result = BenchmarkResult(
            device=device,
            total_queries=len(all_times),
            total_time=total_time,
            avg_time_per_query=avg_time,
            queries_per_second=qps,
            median_time=statistics.median(all_times),
            min_time=min(all_times),
            max_time=max(all_times),
            std_dev=statistics.stdev(all_times) if len(all_times) > 1 else 0,
            accuracy_stats=accuracy_stats,
        )

        return result

    def run_full_benchmark(self, num_iterations: int = 3) -> Dict[str, BenchmarkResult]:
        """Run comprehensive CPU vs GPU benchmark."""
        results = {}

        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        logger.info(f"🔍 GPU Available: {gpu_available}")
        if gpu_available:
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")

        # Benchmark CPU
        logger.info("\n" + "=" * 50)
        logger.info("🖥️  CPU BENCHMARK")
        logger.info("=" * 50)
        cpu_result = self.benchmark_device(use_gpu=False, num_iterations=num_iterations)
        if cpu_result:
            results["CPU"] = cpu_result

        # Benchmark GPU if available
        if gpu_available:
            logger.info("\n" + "=" * 50)
            logger.info("🚀 GPU BENCHMARK")
            logger.info("=" * 50)
            gpu_result = self.benchmark_device(
                use_gpu=True, num_iterations=num_iterations
            )
            if gpu_result:
                results["GPU"] = gpu_result
        else:
            logger.warning("⚠️  GPU not available, skipping GPU benchmark")

        return results

    def print_results(self, results: Dict[str, BenchmarkResult]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 70)
        print("📊 FASTAI INTENT CLASSIFIER BENCHMARK RESULTS")
        print("=" * 70)

        for device, result in results.items():
            print(f"\n🔥 {device} Performance:")
            print(f"   Total Queries: {result.total_queries}")
            print(f"   Total Time: {result.total_time:.3f}s")
            print(f"   Average Time per Query: {result.avg_time_per_query*1000:.2f}ms")
            print(f"   Queries per Second: {result.queries_per_second:.1f} QPS")
            print(f"   Median Time: {result.median_time*1000:.2f}ms")
            print(
                f"   Min/Max Time: {result.min_time*1000:.2f}ms / {result.max_time*1000:.2f}ms"
            )
            print(f"   Std Deviation: {result.std_dev*1000:.2f}ms")

            accuracy = (
                result.accuracy_stats["correct"] / result.accuracy_stats["total"] * 100
            )
            print(f"   High Confidence Predictions: {accuracy:.1f}%")

        # Comparison if both devices tested
        if len(results) == 2 and "CPU" in results and "GPU" in results:
            cpu_result = results["CPU"]
            gpu_result = results["GPU"]

            speedup = cpu_result.avg_time_per_query / gpu_result.avg_time_per_query
            qps_improvement = (
                gpu_result.queries_per_second / cpu_result.queries_per_second - 1
            ) * 100

            print(f"\n⚡ GPU vs CPU Comparison:")
            print(f"   GPU Speedup: {speedup:.2f}x faster")
            print(f"   QPS Improvement: +{qps_improvement:.1f}%")

            if speedup > 1.5:
                print(
                    f"   🎯 Recommendation: Use GPU for production (significant speedup)"
                )
            elif speedup > 1.1:
                print(f"   💡 Recommendation: GPU provides moderate improvement")
            else:
                print(f"   🤔 Recommendation: CPU performance is competitive")

        print("\n" + "=" * 70)


def main():
    """Run the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark FastAI Intent Classifier")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Which device(s) to benchmark",
    )

    args = parser.parse_args()

    benchmark = ClassifierBenchmark()

    if args.device == "both":
        results = benchmark.run_full_benchmark(args.iterations)
    elif args.device == "cpu":
        results = {
            "CPU": benchmark.benchmark_device(
                use_gpu=False, num_iterations=args.iterations
            )
        }
    elif args.device == "gpu":
        if torch.cuda.is_available():
            results = {
                "GPU": benchmark.benchmark_device(
                    use_gpu=True, num_iterations=args.iterations
                )
            }
        else:
            logger.error("❌ GPU not available")
            return

    # Filter out None results
    results = {k: v for k, v in results.items() if v is not None}

    if results:
        benchmark.print_results(results)
    else:
        logger.error("❌ No successful benchmark results")


if __name__ == "__main__":
    main()
