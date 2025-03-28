#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model downloader for the Speech-to-Text application.
This script is used to download the whisper model on first run.
"""

import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_downloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelDownloader")

def download_model(model_name, cache_dir=None):
    """
    Download a model from the Hugging Face Hub.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to store the model
    
    Returns:
        Path to the downloaded model
    """
    try:
        # Import here to avoid issues with PyInstaller
        from huggingface_hub import snapshot_download
        import torch
        
        # Get model repo ID
        repo_id = f"guillaumekln/{model_name}"
        
        # Download the model
        logger.info(f"Downloading model {model_name} from {repo_id}...")
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        logger.info(f"Model downloaded to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: model_downloader.py <model_name> [<cache_dir>]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Download the model
    model_path = download_model(model_name, cache_dir)
    
    if model_path:
        print(f"Model downloaded to {model_path}")
        sys.exit(0)
    else:
        print("Failed to download model")
        sys.exit(1)
