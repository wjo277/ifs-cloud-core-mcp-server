"""
Final Validation Script for Quantized Model Default Implementation

This script validates that all components are working correctly with
the quantized model as the default choice.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
from ifs_cloud_mcp_server.model_config import (
    get_model_config,
    create_optimized_classifier,
)
from ifs_cloud_mcp_server.model_downloader import (
    get_model_path,
    MODEL_FILENAME,
    FALLBACK_MODEL_FILENAME,
)


def validate_quantized_default():
    """Validate that quantized model is properly set as default."""

    print("🔍 QUANTIZED MODEL DEFAULT VALIDATION")
    print("=" * 60)

    # 1. Check configuration defaults
    print("\n1️⃣ Configuration Validation:")
    config = get_model_config()
    is_quantized = config.is_quantized_enabled()
    print(
        f"   Quantized enabled by default: {is_quantized} {'✅' if is_quantized else '❌'}"
    )

    target_time = config.get_performance_config()["target_inference_time_ms"]
    print(
        f"   Target inference time: {target_time}ms {'✅' if target_time <= 15 else '⚠️'}"
    )

    # 2. Check model paths
    print("\n2️⃣ Model Path Configuration:")
    quantized_path = get_model_path(use_quantized=True)
    original_path = get_model_path(use_quantized=False)

    print(f"   Quantized model path: {quantized_path}")
    print(f"   Original model path: {original_path}")

    quantized_exists = quantized_path.exists()
    original_exists = original_path.exists()

    print(
        f"   Quantized model exists: {quantized_exists} {'✅' if quantized_exists else '❌'}"
    )
    print(
        f"   Original model exists: {original_exists} {'✅' if original_exists else '❌'}"
    )

    # 3. Check download filenames
    print("\n3️⃣ Download Configuration:")
    print(f"   Primary download file: {MODEL_FILENAME}")
    print(f"   Fallback download file: {FALLBACK_MODEL_FILENAME}")

    correct_primary = "quantized" in MODEL_FILENAME
    correct_fallback = "quantized" not in FALLBACK_MODEL_FILENAME

    print(
        f"   Primary is quantized: {correct_primary} {'✅' if correct_primary else '❌'}"
    )
    print(
        f"   Fallback is original: {correct_fallback} {'✅' if correct_fallback else '❌'}"
    )

    # 4. Check release preparation
    print("\n4️⃣ Release Files Validation:")
    dist_dir = Path(__file__).parent.parent / "dist"

    quantized_release = dist_dir / MODEL_FILENAME
    original_release = dist_dir / FALLBACK_MODEL_FILENAME

    quantized_release_exists = quantized_release.exists()
    original_release_exists = original_release.exists()

    print(
        f"   Quantized release file: {quantized_release_exists} {'✅' if quantized_release_exists else '❌'}"
    )
    print(
        f"   Original release file: {original_release_exists} {'✅' if original_release_exists else '❌'}"
    )

    if quantized_release_exists and original_release_exists:
        q_size = quantized_release.stat().st_size / (1024 * 1024)
        o_size = original_release.stat().st_size / (1024 * 1024)
        reduction = (o_size - q_size) / o_size * 100

        print(f"   Quantized size: {q_size:.1f} MB")
        print(f"   Original size: {o_size:.1f} MB")
        print(f"   Size reduction: {reduction:.1f}% {'✅' if reduction > 40 else '⚠️'}")

    # 5. Test classifier creation
    print("\n5️⃣ Classifier Creation Test:")
    try:
        classifier = create_optimized_classifier()
        print(f"   Optimized classifier created: ✅")

        # Quick prediction test
        prediction = classifier.predict("customer order validation")
        print(
            f"   Prediction test successful: ✅ ({prediction.intent.value}, {prediction.confidence:.3f})"
        )

    except Exception as e:
        print(f"   Classifier creation failed: ❌ {e}")

    # 6. Overall assessment
    print("\n6️⃣ Overall Assessment:")

    checks = [
        is_quantized,
        target_time <= 15,
        quantized_exists,
        correct_primary,
        correct_fallback,
        quantized_release_exists,
    ]

    passed = sum(checks)
    total = len(checks)
    score = (passed / total) * 100

    print(f"   Validation checks passed: {passed}/{total} ({score:.1f}%)")

    if score >= 90:
        status = "🟢 EXCELLENT - Ready for production deployment"
    elif score >= 75:
        status = "🟡 GOOD - Minor issues to address"
    elif score >= 50:
        status = "🟠 FAIR - Several issues need attention"
    else:
        status = "🔴 POOR - Major problems, not ready for deployment"

    print(f"   Status: {status}")

    print("\n" + "=" * 60)

    # 7. Next steps
    if score >= 90:
        print("\n🚀 READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Create GitHub release (e.g., v1.1.0)")
        print("2. Upload quantized model as primary asset")
        print("3. Upload original model as fallback asset")
        print("4. Update DEFAULT_TAG in model_downloader.py")
        print("5. Test download system with new release")
    else:
        print("\n⚠️ ISSUES DETECTED - Please address before deployment")

    return score >= 90


if __name__ == "__main__":
    validate_quantized_default()
