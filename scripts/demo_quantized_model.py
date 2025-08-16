"""
Quick Model Comparison Demo

This script demonstrates the quantized model vs original model performance
with some sample IFS Cloud queries.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ifs_cloud_mcp_server.model_config import get_model_config
from ifs_cloud_mcp_server.proper_fastai_classifier import FastAIIntentClassifier
import time


def demo_model_comparison():
    """Run a quick demo comparing original vs quantized models."""

    print("🚀 IFS Cloud Intent Classification Model Comparison")
    print("=" * 60)

    # Sample queries
    test_queries = [
        "customer order validation workflow",
        "invoice approval authorization rules",
        "purchase order data structure definition",
        "user interface form design patterns",
        "REST API projection configuration",
        "database view performance optimization",
        "error message debugging guide",
        "system architecture overview",
    ]

    print(f"\nTesting {len(test_queries)} sample queries...\n")

    # Test original model
    print("📊 Testing Original Model...")
    original_classifier = FastAIIntentClassifier(use_quantized=False)

    original_times = []
    original_predictions = []

    for query in test_queries:
        start_time = time.perf_counter()
        prediction = original_classifier.predict(query)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        original_times.append(inference_time)
        original_predictions.append(prediction)

    original_avg_time = sum(original_times) / len(original_times)

    # Test quantized model
    print("📊 Testing Quantized Model...")
    quantized_classifier = FastAIIntentClassifier(use_quantized=True)

    quantized_times = []
    quantized_predictions = []

    for query in test_queries:
        start_time = time.perf_counter()
        prediction = quantized_classifier.predict(query)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        quantized_times.append(inference_time)
        quantized_predictions.append(prediction)

    quantized_avg_time = sum(quantized_times) / len(quantized_times)

    # Compare results
    print("\n" + "=" * 80)
    print("📈 COMPARISON RESULTS")
    print("=" * 80)

    # File sizes
    original_size = Path(
        "src/ifs_cloud_mcp_server/models/fastai_intent/export.pkl"
    ).stat().st_size / (1024 * 1024)
    quantized_size = Path(
        "src/ifs_cloud_mcp_server/models/fastai_intent/export_quantized.pkl"
    ).stat().st_size / (1024 * 1024)

    print(f"\n📁 Model Sizes:")
    print(f"   Original:  {original_size:.1f} MB")
    print(f"   Quantized: {quantized_size:.1f} MB")
    print(
        f"   Reduction: {((original_size - quantized_size) / original_size * 100):.1f}% smaller"
    )

    print(f"\n⚡ Average Inference Time:")
    print(f"   Original:  {original_avg_time:.2f} ms")
    print(f"   Quantized: {quantized_avg_time:.2f} ms")
    print(f"   Speedup:   {(original_avg_time / quantized_avg_time):.2f}x faster")
    print(
        f"   Improvement: {((original_avg_time - quantized_avg_time) / original_avg_time * 100):.1f}% faster"
    )

    print(f"\n🎯 Prediction Comparison:")
    differences = 0
    for i, query in enumerate(test_queries):
        orig_intent = original_predictions[i].intent.value
        quan_intent = quantized_predictions[i].intent.value
        orig_conf = original_predictions[i].confidence
        quan_conf = quantized_predictions[i].confidence

        if orig_intent != quan_intent:
            differences += 1
            print(f"   ❗ '{query[:40]}...': {orig_intent} → {quan_intent}")
        else:
            conf_diff = abs(orig_conf - quan_conf)
            status = "✅" if conf_diff < 0.1 else "⚠️" if conf_diff < 0.2 else "❌"
            print(
                f"   {status} '{query[:40]}...': {orig_intent} (conf diff: {conf_diff:.3f})"
            )

    accuracy_retention = (len(test_queries) - differences) / len(test_queries) * 100
    print(f"\n   Accuracy Retention: {accuracy_retention:.1f}%")

    print(f"\n📊 Overall Assessment:")
    if quantized_avg_time < original_avg_time and accuracy_retention >= 80:
        print("   🟢 QUANTIZED MODEL RECOMMENDED")
        print("   ✅ Faster inference with minimal accuracy loss")
    elif accuracy_retention >= 70:
        print("   🟡 QUANTIZED MODEL ACCEPTABLE")
        print("   ⚠️ Good speed improvement, some accuracy trade-offs")
    else:
        print("   🔴 QUANTIZED MODEL NOT RECOMMENDED")
        print("   ❌ Significant accuracy loss")

    print("=" * 80 + "\n")

    # Show current configuration
    config = get_model_config()
    config.print_current_config()


if __name__ == "__main__":
    demo_model_comparison()
