#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Faster Whisper model
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path
import torch

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.faster_whisper_model import FasterWhisperModel


def test_model_loading(model_path, model_name="large-v3"):
    """Test model loading"""
    print(f"\n=== Testing Faster Whisper Model Loading ({model_name}) ===")
    
    # Determine device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    
    try:
        # Create model
        whisper_model = FasterWhisperModel(
            model_dir=model_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type
        )
        
        # Load model and measure time
        start_time = time.time()
        result = whisper_model.load_model()
        load_time = time.time() - start_time
        
        if result["success"]:
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            return True
        else:
            print(f"❌ Failed to load model: {result['error']}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_transcription(model_path, audio_file, model_name="large-v3"):
    """Test transcription"""
    print(f"\n=== Testing Faster Whisper Transcription ({model_name}) ===")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    # Determine device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    try:
        # Create model
        whisper_model = FasterWhisperModel(
            model_dir=model_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type
        )
        
        # Load model
        result = whisper_model.load_model()
        if not result["success"]:
            print(f"❌ Failed to load model: {result['error']}")
            return False
        
        # Transcribe audio
        print(f"Transcribing audio file: {audio_file}")
        start_time = time.time()
        result = whisper_model.transcribe(
            audio_file=audio_file,
            language="en",
            word_timestamps=True,
            vad_filter=True
        )
        transcribe_time = time.time() - start_time
        
        if result["success"]:
            print(f"✅ Transcription completed in {transcribe_time:.2f} seconds")
            print(f"\nTranscribed text: {result['text']}")
            
            # Print segment information
            print("\nSegments:")
            for i, segment in enumerate(result["segments"]):
                print(f"  {i+1}. [{segment['start']:.2f}s -> {segment['end']:.2f}s]: {segment['text']}")
                
                # Print word-level timestamps for the first segment only
                if i == 0 and "words" in segment:
                    print("\n  Word timestamps (first segment):")
                    for word in segment["words"][:5]:  # Show first 5 words
                        print(f"    [{word['start']:.2f}s -> {word['end']:.2f}s]: {word['word']}")
                    
                    if len(segment["words"]) > 5:
                        print(f"    ... and {len(segment['words']) - 5} more words")
            
            return True
        else:
            print(f"❌ Transcription failed: {result['error']}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Faster Whisper model")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory",
        default=os.path.expanduser("~/whisper-models")
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name",
        default="large-v3",
        choices=[
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en", 
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3",
            "distil-large-v3"
        ]
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for transcription test",
        default=None
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get absolute path
    model_path = os.path.abspath(args.model_path)
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"Creating model directory: {model_path}")
        os.makedirs(model_path, exist_ok=True)
    
    # Test model loading
    loading_success = test_model_loading(model_path, args.model_name)
    
    # Test transcription if audio file is provided and model loading succeeded
    if loading_success and args.audio_file:
        audio_path = os.path.abspath(args.audio_file)
        test_transcription(model_path, audio_path, args.model_name)


if __name__ == "__main__":
    main() 