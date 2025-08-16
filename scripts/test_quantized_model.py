"""
Quantized Model Integration Test

This script tests the quantized model in real search scenarios to validate
its performance in the actual search engine context.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List

# Import the search engine components
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ifs_cloud_mcp_server.proper_fastai_classifier import FastAIIntentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizedModelTester:
    """Test quantized model in real search scenarios."""

    def __init__(self):
        self.original_model_path = Path(
            "src/ifs_cloud_mcp_server/models/fastai_intent/export.pkl"
        )
        self.quantized_model_path = Path(
            "src/ifs_cloud_mcp_server/models/fastai_intent/export_quantized.pkl"
        )

        # Real-world test queries from IFS Cloud development
        self.real_world_queries = [
            "customer order authorization workflow",
            "invoice approval process validation",
            "purchase order entity definition",
            "sales order data structure schema",
            "order handling user interface",
            "customer search form design",
            "REST API customer projection",
            "web service order integration",
            "order history database view",
            "sales report generation query",
            "authorization error debugging",
            "performance issue investigation",
            "IFS Cloud general architecture",
            "system configuration overview",
            "business logic validation rules",
            "entity relationship mapping",
            "user interface navigation menu",
            "API endpoint documentation",
            "database view optimization",
            "system troubleshooting guide",
        ]

    def create_classifiers(self):
        """Create both original and quantized classifiers."""
        original_classifier = FastAIIntentClassifier(
            model_path=str(self.original_model_path.parent)
        )

        # Create quantized classifier by pointing to quantized model
        quantized_classifier = FastAIIntentClassifier(
            model_path=str(self.original_model_path.parent)
        )

        # Manually override the model path for quantized
        if self.quantized_model_path.exists():
            quantized_classifier.learner = None
            quantized_classifier._load_model()
            # Load quantized model
            from fastai.text.all import load_learner

            quantized_classifier.learner = load_learner(self.quantized_model_path)
            quantized_classifier.is_trained = True

        return original_classifier, quantized_classifier

    def benchmark_classifiers(self):
        """Benchmark both classifiers on real queries."""
        logger.info("🚀 Starting real-world classifier benchmark...")

        original_classifier, quantized_classifier = self.create_classifiers()

        results = {
            "original": {"predictions": [], "times": [], "total_time": 0},
            "quantized": {"predictions": [], "times": [], "total_time": 0},
        }

        # Test original classifier
        logger.info("Testing original classifier...")
        start_total = time.perf_counter()
        for query in self.real_world_queries:
            start = time.perf_counter()
            prediction = original_classifier.predict(query)
            end = time.perf_counter()

            results["original"]["predictions"].append(
                {
                    "query": query,
                    "intent": prediction.intent.value,
                    "confidence": prediction.confidence,
                    "time_ms": (end - start) * 1000,
                }
            )
            results["original"]["times"].append((end - start) * 1000)

        results["original"]["total_time"] = (time.perf_counter() - start_total) * 1000

        # Test quantized classifier
        logger.info("Testing quantized classifier...")
        start_total = time.perf_counter()
        for query in self.real_world_queries:
            start = time.perf_counter()
            prediction = quantized_classifier.predict(query)
            end = time.perf_counter()

            results["quantized"]["predictions"].append(
                {
                    "query": query,
                    "intent": prediction.intent.value,
                    "confidence": prediction.confidence,
                    "time_ms": (end - start) * 1000,
                }
            )
            results["quantized"]["times"].append((end - start) * 1000)

        results["quantized"]["total_time"] = (time.perf_counter() - start_total) * 1000

        return results

    def analyze_results(self, results):
        """Analyze benchmark results."""
        original = results["original"]
        quantized = results["quantized"]

        # Calculate statistics
        import numpy as np

        orig_avg_time = np.mean(original["times"])
        quan_avg_time = np.mean(quantized["times"])

        orig_median_time = np.median(original["times"])
        quan_median_time = np.median(quantized["times"])

        # Count prediction differences
        different_predictions = 0
        confidence_differences = []
        intent_changes = []

        for i, orig_pred in enumerate(original["predictions"]):
            quan_pred = quantized["predictions"][i]

            if orig_pred["intent"] != quan_pred["intent"]:
                different_predictions += 1
                intent_changes.append(
                    {
                        "query": orig_pred["query"],
                        "original": orig_pred["intent"],
                        "quantized": quan_pred["intent"],
                        "orig_confidence": orig_pred["confidence"],
                        "quan_confidence": quan_pred["confidence"],
                    }
                )

            conf_diff = abs(orig_pred["confidence"] - quan_pred["confidence"])
            confidence_differences.append(conf_diff)

        avg_confidence_diff = np.mean(confidence_differences)
        max_confidence_diff = np.max(confidence_differences)

        analysis = {
            "performance": {
                "original_avg_time_ms": round(orig_avg_time, 2),
                "quantized_avg_time_ms": round(quan_avg_time, 2),
                "original_median_time_ms": round(orig_median_time, 2),
                "quantized_median_time_ms": round(quan_median_time, 2),
                "speedup_factor": round(orig_avg_time / quan_avg_time, 2),
                "speedup_percent": round(
                    (orig_avg_time - quan_avg_time) / orig_avg_time * 100, 1
                ),
            },
            "accuracy": {
                "total_queries": len(original["predictions"]),
                "different_predictions": different_predictions,
                "accuracy_retention": round(
                    (len(original["predictions"]) - different_predictions)
                    / len(original["predictions"])
                    * 100,
                    1,
                ),
                "avg_confidence_diff": round(avg_confidence_diff, 4),
                "max_confidence_diff": round(max_confidence_diff, 4),
            },
            "intent_changes": intent_changes,
            "file_sizes": {
                "original_mb": round(
                    self.original_model_path.stat().st_size / (1024 * 1024), 2
                ),
                "quantized_mb": round(
                    self.quantized_model_path.stat().st_size / (1024 * 1024), 2
                ),
                "reduction_percent": round(
                    (
                        1
                        - (
                            self.quantized_model_path.stat().st_size
                            / self.original_model_path.stat().st_size
                        )
                    )
                    * 100,
                    1,
                ),
            },
        }

        return analysis

    def print_detailed_report(self, analysis):
        """Print a comprehensive report."""
        print("\n" + "=" * 100)
        print("🎯 QUANTIZED MODEL REAL-WORLD PERFORMANCE ANALYSIS")
        print("=" * 100)

        perf = analysis["performance"]
        acc = analysis["accuracy"]
        sizes = analysis["file_sizes"]

        print(f"\n📁 FILE SIZE COMPARISON:")
        print(f"   Original Model:  {sizes['original_mb']} MB")
        print(f"   Quantized Model: {sizes['quantized_mb']} MB")
        print(f"   Size Reduction:  {sizes['reduction_percent']}% smaller")

        print(f"\n⚡ INFERENCE SPEED COMPARISON:")
        print(f"   Original Avg:    {perf['original_avg_time_ms']:.2f} ms")
        print(f"   Quantized Avg:   {perf['quantized_avg_time_ms']:.2f} ms")
        print(f"   Original Median: {perf['original_median_time_ms']:.2f} ms")
        print(f"   Quantized Median:{perf['quantized_median_time_ms']:.2f} ms")
        print(f"   Speedup Factor:  {perf['speedup_factor']:.2f}x faster")
        print(f"   Speed Improvement: {perf['speedup_percent']:.1f}%")

        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"   Total Queries:       {acc['total_queries']}")
        print(
            f"   Identical Predictions: {acc['total_queries'] - acc['different_predictions']}"
        )
        print(f"   Different Predictions: {acc['different_predictions']}")
        print(f"   Accuracy Retention:    {acc['accuracy_retention']:.1f}%")
        print(f"   Avg Confidence Diff:  {acc['avg_confidence_diff']:.4f}")
        print(f"   Max Confidence Diff:  {acc['max_confidence_diff']:.4f}")

        if analysis["intent_changes"]:
            print(f"\n🔄 INTENT CLASSIFICATION CHANGES:")
            for i, change in enumerate(analysis["intent_changes"][:5]):  # Show first 5
                print(f"   {i+1}. '{change['query'][:50]}...'")
                print(f"      {change['original']} → {change['quantized']}")
                print(
                    f"      Confidence: {change['orig_confidence']:.3f} → {change['quan_confidence']:.3f}"
                )

            if len(analysis["intent_changes"]) > 5:
                print(f"   ... and {len(analysis['intent_changes']) - 5} more changes")
        else:
            print(f"\n✅ NO INTENT CLASSIFICATION CHANGES!")

        print(f"\n📈 OVERALL ASSESSMENT:")

        # Calculate overall score
        size_score = min(
            sizes["reduction_percent"] / 50 * 100, 100
        )  # 50% reduction = 100 points
        speed_score = min(
            perf["speedup_percent"] / 25 * 100, 100
        )  # 25% improvement = 100 points
        accuracy_score = acc["accuracy_retention"]  # Already in percentage

        overall_score = (size_score + speed_score + accuracy_score) / 3

        print(f"   Size Reduction Score:  {size_score:.1f}/100")
        print(f"   Speed Improvement Score: {speed_score:.1f}/100")
        print(f"   Accuracy Retention Score: {accuracy_score:.1f}/100")
        print(f"   Overall Score: {overall_score:.1f}/100")

        if overall_score >= 85:
            recommendation = "🟢 HIGHLY RECOMMENDED"
            details = "Excellent performance with minimal trade-offs"
        elif overall_score >= 70:
            recommendation = "🟡 RECOMMENDED"
            details = "Good performance with acceptable trade-offs"
        elif overall_score >= 55:
            recommendation = "🟠 CONSIDER CAREFULLY"
            details = "Moderate benefits, evaluate based on specific needs"
        else:
            recommendation = "🔴 NOT RECOMMENDED"
            details = "Insufficient benefits or significant drawbacks"

        print(f"\n   Recommendation: {recommendation}")
        print(f"   Details: {details}")

        print("=" * 100 + "\n")


def main():
    """Run the quantized model integration test."""
    tester = QuantizedModelTester()

    if not tester.quantized_model_path.exists():
        print("❌ Quantized model not found. Run scripts/quantize_model.py first.")
        return

    # Run benchmark
    results = tester.benchmark_classifiers()

    # Analyze results
    analysis = tester.analyze_results(results)

    # Print report
    tester.print_detailed_report(analysis)

    # Save detailed results
    output_file = Path("quantized_model_integration_test.json")
    with open(output_file, "w") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2)

    print(f"📊 Detailed test results saved to {output_file}")


if __name__ == "__main__":
    main()
