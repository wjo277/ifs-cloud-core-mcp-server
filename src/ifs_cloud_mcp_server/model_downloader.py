"""Model downloader for FastAI intent classifier models."""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# GitHub repository information
GITHUB_REPO = "graknol/ifs-cloud-core-mcp-server"
MODEL_FILENAME = "fastai_intent_classifier.pkl"
DEFAULT_TAG = "v1.0.0"  # Update this when you create releases


def get_model_download_url(tag: str = DEFAULT_TAG) -> str:
    """Get the download URL for the model from GitHub releases."""
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{MODEL_FILENAME}"


def get_model_path() -> Path:
    """Get the local path where the model should be stored."""
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models" / "fastai_intent"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "export.pkl"


def is_model_available() -> bool:
    """Check if the model file exists locally."""
    return get_model_path().exists()


def download_model(tag: str = DEFAULT_TAG, force: bool = False) -> bool:
    """
    Download the FastAI intent classifier model from GitHub releases.

    Args:
        tag: The release tag to download from
        force: Whether to re-download even if model exists

    Returns:
        True if download was successful, False otherwise
    """
    model_path = get_model_path()

    if model_path.exists() and not force:
        logger.info(f"Model already exists at {model_path}")
        return True

    download_url = get_model_download_url(tag)
    logger.info(f"Downloading model from {download_url}")

    try:
        # Create a temporary file for download
        temp_path = model_path.with_suffix(".tmp")

        # Download with progress (for large files)
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading model... {percent}%", end="", flush=True)

        urllib.request.urlretrieve(download_url, temp_path, progress_hook)
        print()  # New line after progress

        # Move to final location
        temp_path.rename(model_path)
        logger.info(f"Model downloaded successfully to {model_path}")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.error(
                f"Model not found at {download_url}. Please check the release tag."
            )
            logger.error(
                "Available releases: https://github.com/{GITHUB_REPO}/releases"
            )
        else:
            logger.error(f"HTTP error downloading model: {e}")
        return False

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        # Clean up temp file if it exists
        temp_path = model_path.with_suffix(".tmp")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_model_available(tag: str = DEFAULT_TAG) -> bool:
    """
    Ensure the model is available locally, downloading if necessary.

    Args:
        tag: The release tag to download from if needed

    Returns:
        True if model is available (was already present or downloaded successfully)
    """
    if is_model_available():
        return True

    logger.info("Model not found locally, attempting to download...")
    return download_model(tag)


if __name__ == "__main__":
    # Command line usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Download FastAI intent classifier model"
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    success = download_model(args.tag, args.force)
    if success:
        print(f"Model available at: {get_model_path()}")
    else:
        print("Failed to download model")
        exit(1)
