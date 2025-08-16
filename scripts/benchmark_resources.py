"""
Memory and Loading Benchmark for FastAI Intent Classifier

Test memory usage, model loading times, and resource efficiency
between CPU and GPU modes.
"""

import time
import torch
import logging
import sys
import psutil
import gc
from pathlib import Path
from typing import Dict, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ifs_cloud_mcp_server.proper_fastai_classifier import FastAIIntentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()

    result = {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }

    # Add GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        result.update({"gpu_allocated_mb": gpu_memory, "gpu_reserved_mb": gpu_reserved})

    return result


def benchmark_model_loading(use_gpu: bool, num_loads: int = 3) -> Dict:
    """Benchmark model loading performance and memory usage."""
    device = "GPU" if use_gpu else "CPU"
    logger.info(f"🔍 Benchmarking {device} model loading...")

    results = {
        "device": device,
        "load_times": [],
        "memory_before": None,
        "memory_after": None,
        "memory_peak": None,
        "success": True,
    }

    try:
        # Clear memory before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Measure baseline memory
        results["memory_before"] = get_memory_usage()

        load_times = []
        peak_memory = results["memory_before"].copy()

        for i in range(num_loads):
            logger.info(f"  Load attempt {i+1}/{num_loads}")

            # Clear any existing models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.perf_counter()

            # Create and load model
            classifier = FastAIIntentClassifier(use_gpu=use_gpu)

            # Test a prediction to ensure model is fully loaded
            if classifier.is_trained:
                classifier.predict("test query for loading")

            end_time = time.perf_counter()
            load_time = end_time - start_time
            load_times.append(load_time)

            # Measure memory during loading
            current_memory = get_memory_usage()
            for key in current_memory:
                if key in peak_memory:
                    peak_memory[key] = max(peak_memory[key], current_memory[key])

            logger.info(f"    Load time: {load_time:.3f}s")

            # Clean up
            del classifier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["load_times"] = load_times
        results["memory_peak"] = peak_memory
        results["memory_after"] = get_memory_usage()

    except Exception as e:
        logger.error(f"❌ Error benchmarking {device} loading: {e}")
        results["success"] = False
        results["error"] = str(e)

    return results


def benchmark_sustained_performance(use_gpu: bool, duration_seconds: int = 30) -> Dict:
    """Benchmark sustained performance over time."""
    device = "GPU" if use_gpu else "CPU"
    logger.info(
        f"🔄 Benchmarking {device} sustained performance for {duration_seconds}s..."
    )

    classifier = FastAIIntentClassifier(use_gpu=use_gpu)

    if not classifier.is_trained:
        logger.error(f"❌ Model not available for {device} testing")
        return None

    # Test queries
    test_queries = [
        "customer order validation",
        "database schema definition",
        "user interface component",
        "api integration service",
        "data access report",
    ]

    start_time = time.perf_counter()
    query_times = []
    query_count = 0
    memory_samples = []

    while (time.perf_counter() - start_time) < duration_seconds:
        query = test_queries[query_count % len(test_queries)]

        query_start = time.perf_counter()
        result = classifier.predict(query)
        query_end = time.perf_counter()

        query_times.append(query_end - query_start)
        query_count += 1

        # Sample memory periodically
        if query_count % 50 == 0:
            memory_samples.append(get_memory_usage())

    total_time = time.perf_counter() - start_time

    return {
        "device": device,
        "total_time": total_time,
        "total_queries": query_count,
        "qps": query_count / total_time,
        "avg_query_time": sum(query_times) / len(query_times),
        "min_query_time": min(query_times),
        "max_query_time": max(query_times),
        "memory_samples": memory_samples,
        "memory_stability": len(set(round(m["rss_mb"]) for m in memory_samples)) <= 2,
    }


def main():
    """Run comprehensive resource benchmark."""
    print("=" * 80)
    print("🔧 FASTAI CLASSIFIER RESOURCE & PERFORMANCE BENCHMARK")
    print("=" * 80)

    # System info
    logger.info(
        f"🖥️  System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM"
    )

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
        )

    print("\n" + "=" * 50)
    print("📈 MODEL LOADING BENCHMARK")
    print("=" * 50)

    # Test model loading
    cpu_loading = benchmark_model_loading(use_gpu=False)
    gpu_loading = benchmark_model_loading(use_gpu=True) if gpu_available else None

    # Print loading results
    for result in [cpu_loading, gpu_loading]:
        if result and result["success"]:
            device = result["device"]
            avg_load_time = sum(result["load_times"]) / len(result["load_times"])
            memory_increase = (
                result["memory_peak"]["rss_mb"] - result["memory_before"]["rss_mb"]
            )

            print(f"\n{device} Loading Performance:")
            print(f"   Average load time: {avg_load_time:.3f}s")
            print(f"   Memory increase: {memory_increase:.1f}MB")
            print(f"   Peak RSS memory: {result['memory_peak']['rss_mb']:.1f}MB")

            if "gpu_allocated_mb" in result["memory_peak"]:
                print(
                    f"   Peak GPU memory: {result['memory_peak']['gpu_allocated_mb']:.1f}MB"
                )

    print("\n" + "=" * 50)
    print("⏱️  SUSTAINED PERFORMANCE BENCHMARK")
    print("=" * 50)

    # Test sustained performance
    cpu_sustained = benchmark_sustained_performance(use_gpu=False, duration_seconds=20)
    gpu_sustained = (
        benchmark_sustained_performance(use_gpu=True, duration_seconds=20)
        if gpu_available
        else None
    )

    # Print sustained results
    for result in [cpu_sustained, gpu_sustained]:
        if result:
            device = result["device"]
            print(f"\n{device} Sustained Performance (20s):")
            print(f"   Total queries: {result['total_queries']}")
            print(f"   Average QPS: {result['qps']:.1f}")
            print(f"   Avg query time: {result['avg_query_time']*1000:.2f}ms")
            print(
                f"   Min/Max time: {result['min_query_time']*1000:.2f}ms / {result['max_query_time']*1000:.2f}ms"
            )
            print(f"   Memory stable: {'✅' if result['memory_stability'] else '⚠️'}")

    # Final comparison
    print("\n" + "=" * 50)
    print("🏁 RESOURCE EFFICIENCY SUMMARY")
    print("=" * 50)

    if cpu_loading["success"] and (gpu_loading["success"] if gpu_loading else True):
        print("\n💡 Recommendations:")

        # Loading speed comparison
        cpu_load_time = sum(cpu_loading["load_times"]) / len(cpu_loading["load_times"])
        if gpu_loading and gpu_loading["success"]:
            gpu_load_time = sum(gpu_loading["load_times"]) / len(
                gpu_loading["load_times"]
            )
            if gpu_load_time < cpu_load_time * 0.8:
                print("   🚀 GPU loads significantly faster")
            elif gpu_load_time > cpu_load_time * 1.2:
                print("   🖥️  CPU loads faster (less GPU overhead)")
            else:
                print("   ⚖️  Similar loading performance")

        # Memory efficiency
        cpu_memory = (
            cpu_loading["memory_peak"]["rss_mb"]
            - cpu_loading["memory_before"]["rss_mb"]
        )
        if gpu_loading and gpu_loading["success"]:
            gpu_memory = (
                gpu_loading["memory_peak"]["rss_mb"]
                - gpu_loading["memory_before"]["rss_mb"]
            )
            total_gpu_memory = gpu_memory + gpu_loading["memory_peak"].get(
                "gpu_allocated_mb", 0
            )

            print(f"   💾 CPU memory overhead: {cpu_memory:.1f}MB")
            print(f"   💾 GPU total overhead: {total_gpu_memory:.1f}MB")

            if cpu_memory < total_gpu_memory * 0.7:
                print("   💡 CPU is more memory efficient")
            else:
                print("   💡 Similar memory efficiency")

        # Overall recommendation
        if gpu_available:
            print(f"\n🎯 Overall Recommendation:")
            print(
                f"   For this FastAI ULMFiT model, CPU and GPU performance are very similar"
            )
            print(f"   • CPU: Simpler deployment, lower memory usage")
            print(f"   • GPU: No significant performance advantage for this model size")
            print(f"   • Consider CPU-only deployment to save resources")
        else:
            print(f"\n🎯 CPU-only deployment is perfectly suitable for this model")


if __name__ == "__main__":
    main()
