"""
Batch Benchmark for FastAI Intent Classifier

Test performance with different batch sizes to see if GPU shows advantages
with larger batches (GPU typically excels with parallel processing).
"""

import time
import torch
import logging
import sys
from pathlib import Path
from typing import List, Dict
import statistics

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


def generate_test_batch(size: int) -> List[str]:
    """Generate a batch of test queries."""
    base_queries = [
        "customer order validation workflow",
        "database schema definition structure",
        "user interface form component",
        "rest api endpoint integration",
        "inventory report data view",
        "error handling troubleshooting",
        "business logic rule validation",
        "data access query optimization",
        "user authorization permission check",
        "system configuration debugging",
        "supplier invoice processing workflow",
        "product catalog entity schema",
        "navigation menu ui component",
        "external api service integration",
        "financial dashboard reporting view",
        "connection timeout error diagnosis",
        "pricing calculation business rules",
        "database query performance tuning",
        "role-based access control security",
        "system monitoring performance analysis",
    ]

    # Repeat and vary queries to reach desired batch size
    batch = []
    for i in range(size):
        base_query = base_queries[i % len(base_queries)]
        # Add slight variations to make each query unique
        variations = [
            f"{base_query}",
            f"help with {base_query}",
            f"how to {base_query}",
            f"implement {base_query}",
            f"debug {base_query}",
        ]
        batch.append(variations[i % len(variations)])

    return batch


def benchmark_batch_processing(
    use_gpu: bool, batch_sizes: List[int], num_runs: int = 3
):
    """Benchmark batch processing performance."""
    device = "GPU" if use_gpu else "CPU"
    logger.info(f"🚀 Benchmarking {device} batch processing...")

    # Initialize classifier
    classifier = FastAIIntentClassifier(use_gpu=use_gpu)

    if not classifier.is_trained:
        logger.error(f"❌ Model not available for {device} testing")
        return None

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"  Testing batch size: {batch_size}")

        # Generate test batch
        test_batch = generate_test_batch(batch_size)

        # Warm up with a few queries
        for _ in range(3):
            classifier.predict(test_batch[0])

        batch_times = []

        for run in range(num_runs):
            start_time = time.perf_counter()

            # Process entire batch
            for query in test_batch:
                classifier.predict(query)

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            batch_times.append(batch_time)

        # Calculate statistics
        avg_batch_time = statistics.mean(batch_times)
        avg_per_query = avg_batch_time / batch_size
        qps = batch_size / avg_batch_time

        results[batch_size] = {
            "avg_batch_time": avg_batch_time,
            "avg_per_query": avg_per_query,
            "qps": qps,
            "batch_times": batch_times,
        }

        logger.info(f"    Avg batch time: {avg_batch_time:.3f}s")
        logger.info(f"    Avg per query: {avg_per_query*1000:.2f}ms")
        logger.info(f"    QPS: {qps:.1f}")

    return results


def main():
    """Run batch processing benchmark."""
    batch_sizes = [1, 10, 25, 50, 100, 200]

    print("=" * 70)
    print("📊 FASTAI CLASSIFIER BATCH PROCESSING BENCHMARK")
    print("=" * 70)

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    logger.info(f"🔍 GPU Available: {gpu_available}")
    if gpu_available:
        logger.info(f"   GPU: {torch.cuda.get_device_name()}")

    # Benchmark CPU
    print(f"\n🖥️  CPU BATCH PROCESSING")
    print("-" * 50)
    cpu_results = benchmark_batch_processing(use_gpu=False, batch_sizes=batch_sizes)

    # Benchmark GPU if available
    gpu_results = None
    if gpu_available:
        print(f"\n🚀 GPU BATCH PROCESSING")
        print("-" * 50)
        gpu_results = benchmark_batch_processing(use_gpu=True, batch_sizes=batch_sizes)

    # Print comparison table
    if cpu_results and (gpu_results if gpu_available else True):
        print(f"\n📈 BATCH SIZE PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Batch Size':<12} {'CPU QPS':<12} {'CPU ms/query':<15}", end="")
        if gpu_available and gpu_results:
            print(f"{'GPU QPS':<12} {'GPU ms/query':<15} {'Speedup':<10}")
        else:
            print()
        print("-" * 70)

        for batch_size in batch_sizes:
            cpu_qps = cpu_results[batch_size]["qps"]
            cpu_ms = cpu_results[batch_size]["avg_per_query"] * 1000

            print(f"{batch_size:<12} {cpu_qps:<12.1f} {cpu_ms:<15.2f}", end="")

            if gpu_available and gpu_results:
                gpu_qps = gpu_results[batch_size]["qps"]
                gpu_ms = gpu_results[batch_size]["avg_per_query"] * 1000
                speedup = gpu_qps / cpu_qps

                print(f"{gpu_qps:<12.1f} {gpu_ms:<15.2f} {speedup:<10.2f}x")
            else:
                print()

        print("=" * 70)

        # Analysis
        if gpu_available and gpu_results:
            print(f"\n🔍 ANALYSIS:")

            # Find best GPU advantage
            best_speedup = 0
            best_batch_size = 0
            for batch_size in batch_sizes:
                speedup = (
                    gpu_results[batch_size]["qps"] / cpu_results[batch_size]["qps"]
                )
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_batch_size = batch_size

            print(
                f"   Best GPU performance: {best_speedup:.2f}x speedup at batch size {best_batch_size}"
            )

            if best_speedup > 1.2:
                print(
                    f"   💡 Recommendation: Use GPU for batch sizes ≥ {best_batch_size}"
                )
            elif best_speedup > 1.05:
                print(f"   🤔 Recommendation: GPU provides marginal improvement")
            else:
                print(f"   🖥️  Recommendation: CPU performance is competitive")
                print(f"      - FastAI ULMFiT is well-optimized for CPU inference")
                print(f"      - Small model size doesn't fully utilize GPU parallelism")


if __name__ == "__main__":
    main()
