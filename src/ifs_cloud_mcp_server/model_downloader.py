"""Model downloader for FastAI intent classifier models."""

import os
import urllib.request
import urllib.error
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# GitHub repository information
GITHUB_REPO = "graknol/ifs-cloud-core-mcp-server"
MODEL_FILENAME = "fastai_intent_classifier_quantized.pkl"  # Quantized model as default
FALLBACK_MODEL_FILENAME = "fastai_intent_classifier.pkl"  # Original model as fallback
DEFAULT_TAG = "v1.0.0"  # Update this when you create releases


def get_model_download_url(tag: str = DEFAULT_TAG, use_quantized: bool = True) -> str:
    """Get the download URL for the model from GitHub releases."""
    filename = MODEL_FILENAME if use_quantized else FALLBACK_MODEL_FILENAME
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{filename}"


def get_model_path(use_quantized: bool = True) -> Path:
    """Get the local path where the model should be stored."""
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models" / "fastai_intent"
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = "export_quantized.pkl" if use_quantized else "export.pkl"
    return models_dir / filename


def is_github_cli_available() -> bool:
    """Check if GitHub CLI is available and authenticated."""
    try:
        # Check if gh is installed
        subprocess.run(["gh", "--version"], capture_output=True, check=True)

        # Check if authenticated
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)

        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_github_cli(tag: str, filename: str, output_path: Path) -> bool:
    """Download model using GitHub CLI (handles private repos)."""
    try:
        logger.info(f"Downloading {filename} using GitHub CLI...")

        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download using gh release download
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    GITHUB_REPO,
                    "--pattern",
                    filename,
                    "--dir",
                    str(temp_path),
                ],
                check=True,
                capture_output=True,
            )

            # Move file to final location
            downloaded_file = temp_path / filename
            if downloaded_file.exists():
                downloaded_file.rename(output_path)
                logger.info(f"Successfully downloaded {filename} via GitHub CLI")
                return True
            else:
                logger.error(f"Downloaded file not found: {downloaded_file}")
                return False

    except subprocess.CalledProcessError as e:
        logger.error(f"GitHub CLI download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during GitHub CLI download: {e}")
        return False


def download_with_http(url: str, output_path: Path) -> bool:
    """Download model using direct HTTP request (for public repos)."""
    try:
        logger.info(f"Downloading from {url}")

        # Create a temporary file for download
        temp_path = output_path.with_suffix(".tmp")

        # Download with progress (for large files)
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading... {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, temp_path, progress_hook)
        print()  # New line after progress

        # Move to final location
        temp_path.rename(output_path)
        logger.info(f"Successfully downloaded via HTTP")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.error(f"Model not found at {url}")
        else:
            logger.error(f"HTTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"HTTP download error: {e}")
        return False


def is_model_available(use_quantized: bool = True) -> bool:
    """Check if the model file exists locally."""
    return get_model_path(use_quantized).exists()


def download_model(
    tag: str = DEFAULT_TAG, force: bool = False, use_quantized: bool = True
) -> bool:
    """
    Download the FastAI intent classifier model from GitHub releases.

    Uses GitHub CLI if available (for private repos), falls back to HTTP.

    Args:
        tag: The release tag to download from
        force: Whether to re-download even if model exists
        use_quantized: Whether to download quantized model (default) or original

    Returns:
        True if download was successful, False otherwise
    """
    model_path = get_model_path(use_quantized)
    model_type = "quantized" if use_quantized else "original"
    filename = MODEL_FILENAME if use_quantized else FALLBACK_MODEL_FILENAME

    if model_path.exists() and not force:
        logger.info(f"{model_type.title()} model already exists at {model_path}")
        return True

    # Try GitHub CLI first (handles authentication for private repos)
    if is_github_cli_available():
        if download_with_github_cli(tag, filename, model_path):
            return True
        else:
            logger.warning("GitHub CLI download failed, trying HTTP fallback...")

    # Fallback to direct HTTP download
    download_url = get_model_download_url(tag, use_quantized)
    if download_with_http(download_url, model_path):
        return True

    # If quantized model failed and we were trying quantized, try original as fallback
    if use_quantized:
        logger.info(
            "Quantized model download failed, trying original model as fallback..."
        )
        return download_model(tag, force, use_quantized=False)

    return False


def ensure_model_available(tag: str = DEFAULT_TAG, use_quantized: bool = True) -> bool:
    """
    Ensure the model is available locally, downloading if necessary.
    Tries quantized model first, falls back to original if needed.

    Args:
        tag: The release tag to download from if needed
        use_quantized: Whether to prefer quantized model (default: True)

    Returns:
        True if model is available (was already present or downloaded successfully)
    """
    # First try the requested model type
    if is_model_available(use_quantized):
        return True

    logger.info(
        f"{'Quantized' if use_quantized else 'Original'} model not found locally, attempting to download..."
    )
    success = download_model(tag, use_quantized=use_quantized)

    if success:
        return True

    # If quantized model failed and we were trying quantized, try original as fallback
    if use_quantized:
        logger.info(
            "Quantized model download failed, trying original model as fallback..."
        )
        return download_model(tag, use_quantized=False)

    return False


if __name__ == "__main__":
    # Command line usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Download FastAI intent classifier model"
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Download original model instead of quantized",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    use_quantized = not args.original
    success = download_model(args.tag, args.force, use_quantized)
    if success:
        print(f"Model available at: {get_model_path(use_quantized)}")
    else:
        print("Failed to download model")
        exit(1)
