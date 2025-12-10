#!/usr/bin/env python3
"""Utility script to download the SAM model if missing"""

import os
import requests
from pathlib import Path
from src.utils.logger import get_logger
from src.pipeline.config import get_config

logger = get_logger(__name__)

def download_sam_model(model_path: str = None, force_download: bool = False):
    """
    Download the SAM model if it doesn't exist

    Args:
        model_path: Path where the model should be saved (default: uses network volume path)
        force_download: Whether to download even if file exists
    """
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # If no model_path specified, use the network volume path
    if model_path is None:
        config = get_config()
        serverless_config = config.get("serverless", {})
        volume_path = serverless_config.get("model_volume_path", "/models")
        model_file = Path(volume_path) / "sam" / "sam_vit_h_4b8939.pth"
    else:
        model_file = Path(model_path)

    # If file exists and we're not forcing download, skip
    if model_file.exists() and not force_download:
        logger.info(f"Model already exists at {model_file}, skipping download")
        return True

    # Create parent directories if they don't exist
    model_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading SAM model from {model_url} to {model_file}")

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        # Download with progress indication
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"\rDownloading... {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)

        print()  # New line after progress
        logger.info(f"Successfully downloaded SAM model to {model_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to download SAM model: {str(e)}")
        if model_file.exists():
            # Clean up partial download
            model_file.unlink()
        return False

def main():
    """Main function to run the SAM model downloader"""
    import argparse

    parser = argparse.ArgumentParser(description="Download SAM model for image generation pipeline")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path where to save the model (default: uses network volume path /models/sam/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists"
    )

    args = parser.parse_args()

    # Set up logging
    from src.utils.logger import setup_logging
    setup_logging()

    success = download_sam_model(model_path=args.model_path, force_download=args.force)

    if success:
        model_path = args.model_path or "/models/sam/sam_vit_h_4b8939.pth"
        print(f"✓ SAM model successfully downloaded to {model_path}")
        return 0
    else:
        model_path = args.model_path or "/models/sam/sam_vit_h_4b8939.pth"
        print(f"✗ Failed to download SAM model to {model_path}")
        return 1

if __name__ == "__main__":
    exit(main())