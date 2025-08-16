"""
Script to prepare a FastAI model for GitHub release.

This script copies the trained model to a standard filename for release upload.
"""

import shutil
from pathlib import Path
import argparse


def prepare_model_for_release(source_dir: str = None, output_dir: str = None):
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

    # Source model file
    source_model = source_dir / "export.pkl"
    if not source_model.exists():
        print(f"❌ Model not found at {source_model}")
        print(
            "Please train the model first using: uv run python scripts/train_proper_fastai.py"
        )
        return False

    # Destination file for release
    release_model = output_dir / "fastai_intent_classifier.pkl"

    # Copy the model
    shutil.copy2(source_model, release_model)

    # Get file size
    size_mb = release_model.stat().st_size / (1024 * 1024)

    print(f"✅ Model prepared for release:")
    print(f"   Source: {source_model}")
    print(f"   Release file: {release_model}")
    print(f"   Size: {size_mb:.1f} MB")
    print()
    print("📋 Next steps:")
    print("1. Create a new GitHub release (e.g., v1.0.0)")
    print(f"2. Upload {release_model.name} as a release asset")
    print("3. Update the DEFAULT_TAG in model_downloader.py if needed")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare FastAI model for GitHub release"
    )
    parser.add_argument("--source", help="Source directory containing export.pkl")
    parser.add_argument("--output", help="Output directory for release file")

    args = parser.parse_args()

    prepare_model_for_release(args.source, args.output)
