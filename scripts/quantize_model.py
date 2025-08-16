"""
Model Quantization Script for FastAI Intent Classifier

This script quantizes the existing FastAI model to reduce its size while maintaining
acceptable performance. It compares the original model with the quantized version
across multiple metrics.
"""

import torch
import time
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# FastAI imports
from fastai.text.all import *
import warnings

warnings.filterwarnings("ignore")

# Import our classifier
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ifs_cloud_mcp_server.proper_fastai_classifier import (
    FastAIIntentClassifier,
    QueryIntent,
    IntentPrediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""

    model_name: str
    file_size_mb: float
    accuracy: float
    avg_inference_time_ms: float
    memory_usage_mb: float
    predictions: List[Tuple[str, str, float]]  # query, predicted, confidence


class ModelQuantizer:
    """Handles model quantization and performance comparison."""

    def __init__(self, base_model_path: str):
        """Initialize with base model path."""
        self.base_model_path = Path(base_model_path)
        self.quantized_model_path = self.base_model_path.parent / "export_quantized.pkl"

        # Test queries for evaluation
        self.test_queries = [
            ("customer order validation workflow", "business_logic"),
            ("authorization check implementation", "business_logic"),
            ("approval process rules", "business_logic"),
            ("customer order data structure", "entity_definition"),
            ("purchase order schema definition", "entity_definition"),
            ("invoice data model", "entity_definition"),
            ("order entry form design", "ui_components"),
            ("customer search page layout", "ui_components"),
            ("navigation menu structure", "ui_components"),
            ("REST API endpoint definition", "api_integration"),
            ("web service integration", "api_integration"),
            ("projection configuration", "api_integration"),
            ("database view creation", "data_access"),
            ("report generation query", "data_access"),
            ("data extraction logic", "data_access"),
            ("error message debugging", "troubleshooting"),
            ("performance issue analysis", "troubleshooting"),
            ("system log investigation", "troubleshooting"),
            ("general system overview", "general"),
            ("documentation search", "general"),
        ]

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes."""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0

    def quantize_model(self) -> bool:
        """Quantize the FastAI model."""
        logger.info(f"🔧 Starting model quantization...")

        try:
            # Load the original FastAI learner
            logger.info("Loading original model...")
            learner = load_learner(self.base_model_path)

            # Get the underlying PyTorch model
            model = learner.model

            # Apply dynamic quantization (excluding embeddings due to FastAI compatibility)
            logger.info("Applying dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
            )

            # Replace the model in the learner
            learner.model = quantized_model

            # Export the quantized learner
            logger.info(f"Exporting quantized model to {self.quantized_model_path}")
            learner.export(self.quantized_model_path)

            logger.info("✅ Model quantization completed!")
            return True

        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            return False

    def benchmark_model(self, model_path: Path, model_name: str) -> ModelPerformance:
        """Benchmark a model's performance."""
        logger.info(f"📊 Benchmarking {model_name}...")

        # Get file size
        file_size = self.get_file_size_mb(model_path)

        # Load model
        try:
            learner = load_learner(model_path)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return ModelPerformance(
                model_name=model_name,
                file_size_mb=file_size,
                accuracy=0.0,
                avg_inference_time_ms=0.0,
                memory_usage_mb=0.0,
                predictions=[],
            )

        # Measure memory usage (rough estimate)
        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)

        # Warm up the model
        logger.info("Warming up model...")
        for _ in range(3):
            learner.predict("test query")

        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_usage = memory_after - memory_before

        # Evaluate accuracy and inference time
        correct_predictions = 0
        inference_times = []
        predictions = []

        logger.info("Running evaluation...")
        for query, expected_intent in self.test_queries:
            start_time = time.perf_counter()

            try:
                pred_class, pred_idx, probs = learner.predict(query)
                predicted_intent = str(pred_class)
                confidence = float(probs.max())

                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)

                predictions.append((query, predicted_intent, confidence))

                if predicted_intent == expected_intent:
                    correct_predictions += 1

            except Exception as e:
                logger.warning(f"Prediction failed for '{query}': {e}")
                predictions.append((query, "ERROR", 0.0))

        accuracy = correct_predictions / len(self.test_queries)
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0

        return ModelPerformance(
            model_name=model_name,
            file_size_mb=file_size,
            accuracy=accuracy,
            avg_inference_time_ms=avg_inference_time,
            memory_usage_mb=max(memory_usage, 0),  # Ensure positive
            predictions=predictions,
        )

    def compare_models(self) -> Dict:
        """Compare original and quantized models."""
        logger.info("🔍 Starting model comparison...")

        # Benchmark original model
        original_performance = self.benchmark_model(self.base_model_path, "Original")

        # Quantize model if not exists
        if not self.quantized_model_path.exists():
            if not self.quantize_model():
                return {"error": "Failed to quantize model"}

        # Benchmark quantized model
        quantized_performance = self.benchmark_model(
            self.quantized_model_path, "Quantized"
        )

        # Calculate improvements
        size_reduction = (
            (original_performance.file_size_mb - quantized_performance.file_size_mb)
            / original_performance.file_size_mb
            * 100
        )

        speed_change = (
            (
                quantized_performance.avg_inference_time_ms
                - original_performance.avg_inference_time_ms
            )
            / original_performance.avg_inference_time_ms
            * 100
        )

        accuracy_change = (
            quantized_performance.accuracy - original_performance.accuracy
        ) * 100

        memory_change = (
            quantized_performance.memory_usage_mb - original_performance.memory_usage_mb
        )

        # Detailed analysis
        comparison = {
            "original": {
                "file_size_mb": round(original_performance.file_size_mb, 2),
                "accuracy": round(original_performance.accuracy, 4),
                "avg_inference_time_ms": round(
                    original_performance.avg_inference_time_ms, 2
                ),
                "memory_usage_mb": round(original_performance.memory_usage_mb, 2),
            },
            "quantized": {
                "file_size_mb": round(quantized_performance.file_size_mb, 2),
                "accuracy": round(quantized_performance.accuracy, 4),
                "avg_inference_time_ms": round(
                    quantized_performance.avg_inference_time_ms, 2
                ),
                "memory_usage_mb": round(quantized_performance.memory_usage_mb, 2),
            },
            "improvements": {
                "size_reduction_percent": round(size_reduction, 1),
                "speed_change_percent": round(speed_change, 1),
                "accuracy_change_percent": round(accuracy_change, 2),
                "memory_change_mb": round(memory_change, 2),
            },
            "detailed_predictions": {
                "original": original_performance.predictions,
                "quantized": quantized_performance.predictions,
            },
        }

        return comparison

    def print_comparison_report(self, comparison: Dict):
        """Print a detailed comparison report."""
        print("\n" + "=" * 80)
        print("🚀 FASTAI MODEL QUANTIZATION COMPARISON REPORT")
        print("=" * 80)

        orig = comparison["original"]
        quan = comparison["quantized"]
        imp = comparison["improvements"]

        print(f"\n📊 MODEL SIZE COMPARISON:")
        print(f"   Original:  {orig['file_size_mb']:.1f} MB")
        print(f"   Quantized: {quan['file_size_mb']:.1f} MB")
        print(
            f"   Reduction: {imp['size_reduction_percent']:.1f}% smaller ({'✅' if imp['size_reduction_percent'] > 0 else '❌'})"
        )

        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"   Original:  {orig['accuracy']:.3f} ({orig['accuracy']*100:.1f}%)")
        print(f"   Quantized: {quan['accuracy']:.3f} ({quan['accuracy']*100:.1f}%)")
        print(
            f"   Change:    {imp['accuracy_change_percent']:.2f}% ({'✅' if imp['accuracy_change_percent'] >= -2 else '❌'})"
        )

        print(f"\n⚡ SPEED COMPARISON:")
        print(f"   Original:  {orig['avg_inference_time_ms']:.1f} ms")
        print(f"   Quantized: {quan['avg_inference_time_ms']:.1f} ms")
        print(
            f"   Change:    {imp['speed_change_percent']:.1f}% ({'✅' if imp['speed_change_percent'] < 0 else '❌' if imp['speed_change_percent'] > 10 else '➖'})"
        )

        print(f"\n💾 MEMORY USAGE COMPARISON:")
        print(f"   Original:  {orig['memory_usage_mb']:.1f} MB")
        print(f"   Quantized: {quan['memory_usage_mb']:.1f} MB")
        print(
            f"   Change:    {imp['memory_change_mb']:+.1f} MB ({'✅' if imp['memory_change_mb'] <= 0 else '❌'})"
        )

        print(f"\n🔍 PREDICTION DIFFERENCES:")
        orig_preds = {
            p[0]: (p[1], p[2]) for p in comparison["detailed_predictions"]["original"]
        }
        quan_preds = {
            p[0]: (p[1], p[2]) for p in comparison["detailed_predictions"]["quantized"]
        }

        different_predictions = 0
        for query in orig_preds:
            if orig_preds[query][0] != quan_preds[query][0]:
                different_predictions += 1
                print(
                    f"   '{query[:40]}...': {orig_preds[query][0]} → {quan_preds[query][0]}"
                )

        if different_predictions == 0:
            print("   ✅ All predictions identical!")
        else:
            print(
                f"   ❌ {different_predictions}/{len(orig_preds)} predictions changed"
            )

        print(f"\n📈 RECOMMENDATION:")
        if imp["size_reduction_percent"] > 50 and imp["accuracy_change_percent"] > -5:
            print(
                "   🟢 STRONGLY RECOMMENDED - Significant size reduction with minimal accuracy loss"
            )
        elif (
            imp["size_reduction_percent"] > 30 and imp["accuracy_change_percent"] > -10
        ):
            print(
                "   🟡 RECOMMENDED - Good size reduction with acceptable accuracy trade-off"
            )
        elif (
            imp["size_reduction_percent"] > 10 and imp["accuracy_change_percent"] > -15
        ):
            print(
                "   🟠 CONSIDER - Modest improvements, evaluate based on deployment needs"
            )
        else:
            print(
                "   🔴 NOT RECOMMENDED - Insufficient benefits or too much accuracy loss"
            )

        print("=" * 80 + "\n")


def main():
    """Run the quantization comparison."""
    # Path to the original model
    model_path = Path("src/ifs_cloud_mcp_server/models/fastai_intent/export.pkl")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please ensure the FastAI model is trained and exported.")
        return

    # Initialize quantizer
    quantizer = ModelQuantizer(str(model_path))

    # Run comparison
    comparison = quantizer.compare_models()

    if "error" in comparison:
        print(f"❌ {comparison['error']}")
        return

    # Print report
    quantizer.print_comparison_report(comparison)

    # Save detailed results
    results_file = Path("quantization_results.json")
    with open(results_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"📁 Detailed results saved to {results_file}")

    # Update the classifier to optionally use quantized model
    quantized_path = model_path.parent / "export_quantized.pkl"
    if quantized_path.exists():
        print(f"\n💡 To use the quantized model, set model_path to:")
        print(f"   {quantized_path}")


if __name__ == "__main__":
    main()
