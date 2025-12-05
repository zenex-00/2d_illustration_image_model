#!/usr/bin/env python3
"""Setup script to download and organize models on network volume"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Model definitions with their download sources
MODELS = {
    "grounding_dino": {
        "model_id": "IDEA-Research/grounding-dino-base",
        "type": "huggingface",
        "subdir": "grounding_dino"
    },
    "sam": {
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "type": "direct",
        "subdir": "sam"
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "type": "huggingface",
        "subdir": "sdxl"
    },
    "controlnet_depth": {
        "model_id": "diffusers/controlnet-depth-sdxl-1.0",
        "type": "huggingface",
        "subdir": "controlnet/depth"
    },
    "controlnet_canny": {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "type": "huggingface",
        "subdir": "controlnet/canny"
    },
    "realesrgan": {
        "model_name": "RealESRGAN_x4plus_anime",
        "type": "realesrgan",
        "subdir": "realesrgan"
    }
}


def download_huggingface_model(model_id: str, output_dir: Path, token: str = None):
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        
        logger.info("downloading_huggingface_model", model_id=model_id, output_dir=str(output_dir))
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            token=token,
            local_dir_use_symlinks=False
        )
        
        logger.info("huggingface_model_downloaded", model_id=model_id)
        return True
    except Exception as e:
        logger.error("huggingface_download_failed", model_id=model_id, error=str(e))
        return False


def download_direct_file(url: str, output_path: Path):
    """Download file directly from URL"""
    import requests
    
    logger.info("downloading_file", url=url, output_path=str(output_path))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info("file_downloaded", output_path=str(output_path))
    return True


def setup_model_volume(volume_path: str, models: List[str] = None, hf_token: str = None):
    """
    Setup model volume with all required models
    
    Args:
        volume_path: Path to network volume mount point
        models: List of model names to download (None = all)
        hf_token: HuggingFace token for private models
    """
    volume_dir = Path(volume_path)
    volume_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_download = models or list(MODELS.keys())
    
    logger.info("setting_up_model_volume", volume_path=volume_path, models=models_to_download)
    
    for model_name in models_to_download:
        if model_name not in MODELS:
            logger.warning("unknown_model", model_name=model_name)
            continue
        
        model_config = MODELS[model_name]
        model_dir = volume_dir / model_config["subdir"]
        
        if model_config["type"] == "huggingface":
            success = download_huggingface_model(
                model_config["model_id"],
                model_dir,
                token=hf_token
            )
        elif model_config["type"] == "direct":
            output_path = model_dir / model_config["checkpoint"]
            success = download_direct_file(
                model_config["url"],
                output_path
            )
        elif model_config["type"] == "realesrgan":
            # RealESRGAN models are downloaded automatically by the library
            # Just create the directory
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("realesrgan_dir_created", path=str(model_dir))
            success = True
        else:
            logger.warning("unknown_model_type", model_name=model_name, model_type=model_config["type"])
            success = False
        
        if success:
            logger.info("model_setup_complete", model_name=model_name, path=str(model_dir))
        else:
            logger.error("model_setup_failed", model_name=model_name)
    
    logger.info("model_volume_setup_complete", volume_path=volume_path)


def main():
    parser = argparse.ArgumentParser(description="Setup model volume for serverless deployment")
    parser.add_argument(
        "--volume-path",
        type=str,
        default="/models",
        help="Path to network volume mount point"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to download (default: all)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token for private models"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    setup_model_volume(
        volume_path=args.volume_path,
        models=args.models,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()






