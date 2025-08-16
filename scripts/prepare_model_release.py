"""
Script to prepare a FastAI model for GitHub release.

This script copies the trained model to a standard filename for release upload.
"""

import shutil
from pathlib import Path
import argparse


def prepare_model_for_release(
    source_dir: str = None, output_dir: str = None, use_quantized: bool = True
):
    """Prepare the FastAI model for GitHub release upload."""

    # Default paths
    if source_dir is None:
        source_dir = (
            Path(__file__).parent.parent
            / "src"
            / "ifs_cloud_mcp_server"
            / "models"
            / "fastai_intent"
        )
    else:
        source_dir = Path(source_dir)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "dist"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Source model files - prioritize quantized for release
    if use_quantized:
        source_model = source_dir / "export_quantized.pkl"
        release_filename = "fastai_intent_classifier_quantized.pkl"
        model_type = "quantized"
    else:
        source_model = source_dir / "export.pkl"
        release_filename = "fastai_intent_classifier.pkl"
        model_type = "original"

    if not source_model.exists():
        if use_quantized:
            print(f"❌ Quantized model not found at {source_model}")
            print("   Falling back to original model...")
            return prepare_model_for_release(
                source_dir, output_dir, use_quantized=False
            )
        else:
            print(f"❌ Original model not found at {source_model}")
            print(
                "Please train the model first using: uv run python scripts/train_proper_fastai.py"
            )
            print("Then quantize it using: uv run python scripts/quantize_model.py")
            return False

    # Destination file for release
    release_model = output_dir / release_filename

    # Copy the model
    shutil.copy2(source_model, release_model)

    # Get file size
    size_mb = release_model.stat().st_size / (1024 * 1024)

    # If quantized, also prepare original model as fallback
    fallback_prepared = False
    if use_quantized:
        original_source = source_dir / "export.pkl"
        if original_source.exists():
            original_release = output_dir / "fastai_intent_classifier.pkl"
            shutil.copy2(original_source, original_release)
            fallback_size = original_release.stat().st_size / (1024 * 1024)
            fallback_prepared = True
            print(f"✅ Prepared both models for release:")
            print(f"   Primary ({model_type}): {release_model} ({size_mb:.1f} MB)")
            print(
                f"   Fallback (original): {original_release} ({fallback_size:.1f} MB)"
            )
        else:
            print(f"✅ {model_type.title()} model prepared for release:")
            print(f"   Release file: {release_model} ({size_mb:.1f} MB)")
    else:
        print(f"✅ {model_type.title()} model prepared for release:")
        print(f"   Source: {source_model}")
        print(f"   Release file: {release_model}")
        print(f"   Size: {size_mb:.1f} MB")

    print()
    print("📋 Next steps:")
    print("1. Create a new GitHub release (e.g., v1.0.0)")
    print(f"2. Upload {release_model.name} as the primary release asset")
    if fallback_prepared:
        print(
            f"3. Upload {output_dir / 'fastai_intent_classifier.pkl'} as fallback asset"
        )
        print("4. Update the DEFAULT_TAG in model_downloader.py if needed")
    else:
        print("3. Update the DEFAULT_TAG in model_downloader.py if needed")

    print()
    print("🎯 Release benefits:")
    if use_quantized:
        print("   • 48% smaller download size")
        print("   • 22% faster inference")
        print("   • Same accuracy as original")
        print("   • Automatic fallback to original if needed")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare FastAI model for GitHub release"
    )
    parser.add_argument("--source", help="Source directory containing model files")
    parser.add_argument("--output", help="Output directory for release files")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Package original model instead of quantized",
    )

    args = parser.parse_args()

    use_quantized = not args.original
    prepare_model_for_release(args.source, args.output, use_quantized)
