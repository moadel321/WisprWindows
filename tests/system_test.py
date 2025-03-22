#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System test for the Speech-to-Text application
Tests the complete workflow from microphone to text insertion
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

from src.config.settings import AppSettings
from src.audio.microphone import MicrophoneManager
from src.audio.audio_processor import AudioProcessor
from src.audio.vad_processor import VADProcessor
from src.models.vad_model import SileroVAD
from src.models.faster_whisper_model import FasterWhisperModel
from src.text_insertion.text_inserter import TextInserter


def test_microphone_enumeration():
    """Test microphone enumeration"""
    print("\n=== Testing Microphone Enumeration ===")
    mic_manager = MicrophoneManager()
    mics = mic_manager.get_available_microphones()
    
    if not mics:
        print("❌ No microphones found")
        return False
    
    print(f"✅ Found {len(mics)} microphone(s):")
    for i, mic in enumerate(mics):
        print(f"  {i+1}. {mic['name']} (ID: {mic['id']})")
    
    return True


def test_vad_model_loading():
    """Test VAD model loading"""
    print("\n=== Testing VAD Model Loading ===")
    try:
        vad_model = SileroVAD()
        print("✅ VAD model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load VAD model: {str(e)}")
        return False


def test_whisper_model_loading(model_path):
    """Test Whisper model loading"""
    print("\n=== Testing Whisper Model Loading ===")
    try:
        whisper_model = FasterWhisperModel(
            model_dir=model_path, 
            model_name="large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        result = whisper_model.load_model()
        
        if result["success"]:
            print(f"✅ Faster Whisper model loaded successfully")
            print(f"  Device: {whisper_model.device}")
            print(f"  Compute type: {whisper_model.compute_type}")
            print(f"  Language: {whisper_model.language}")
            return True
        else:
            print(f"❌ Failed to load Faster Whisper model: {result['error']}")
            return False
    except Exception as e:
        print(f"❌ Failed to load Faster Whisper model: {str(e)}")
        return False


def test_text_insertion():
    """Test text insertion functionality"""
    print("\n=== Testing Text Insertion ===")
    try:
        text_inserter = TextInserter()
        
        print("Please focus a text field (e.g., Notepad, browser input) within 5 seconds...")
        time.sleep(5)
        
        element_info = text_inserter.get_focused_element()
        if not element_info:
            print("❌ No focused element detected")
            return False
        
        print(f"Detected focused element in: {element_info['app']}")
        print(f"Element type: {element_info['control_type']}")
        
        if not element_info['editable']:
            print("❌ Focused element is not editable")
            return False
        
        test_text = "This is a test of the Speech-to-Text system. [System Test]"
        print(f"Inserting text: \"{test_text}\"...")
        
        success = text_inserter.insert_text(test_text)
        if success:
            print("✅ Text inserted successfully")
            return True
        else:
            print("❌ Failed to insert text")
            return False
    except Exception as e:
        print(f"❌ Error during text insertion test: {str(e)}")
        return False


def test_end_to_end(model_path, test_duration=10):
    """
    Test end-to-end workflow with live audio
    
    Args:
        model_path: Path to Whisper model directory
        test_duration: Duration of the test in seconds
    """
    print("\n=== Testing End-to-End Workflow ===")
    
    # Initialize components
    settings = AppSettings()
    
    # Initialize audio components
    mic_manager = MicrophoneManager()
    audio_processor = AudioProcessor(
        sample_rate=16000,
        channels=1
    )
    
    # Initialize VAD processor
    vad_processor = VADProcessor(
        sample_rate=16000,
        vad_threshold=0.5,
        window_size_ms=30,
        audio_processor=audio_processor
    )
    
    # Initialize Whisper model
    whisper_model = FasterWhisperModel(
        model_dir=model_path,
        model_name="large-v3",
        language="en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
    )
    
    # Initialize text inserter
    text_inserter = TextInserter()
    
    # Load model
    result = whisper_model.load_model()
    if not result["success"]:
        print(f"❌ Failed to load Faster Whisper model: {result['error']}")
        return False
    
    # Get available microphones
    mics = mic_manager.get_available_microphones()
    if not mics:
        print("❌ No microphones available")
        return False
    
    # Select first microphone
    mic_manager.select_microphone(mics[0]["id"])
    print(f"Selected microphone: {mics[0]['name']}")
    
    # Setup callbacks
    def on_speech_detected(is_speech):
        status = "SPEECH DETECTED" if is_speech else "No speech"
        print(f"\r{status}", end="")
    
    def on_speech_segment(audio_file, segment_info):
        print(f"\nTranscribing segment ({segment_info['duration']:.1f}s)...")
        
        # Transcribe audio
        result = whisper_model.transcribe(audio_file)
        
        if result["success"]:
            transcription = result["text"]
            print(f"Transcription: \"{transcription}\"")
            
            # Try to insert text
            print("Checking for editable text field...")
            if text_inserter.is_text_editable():
                if text_inserter.insert_text(transcription):
                    print("✅ Text inserted successfully")
                else:
                    print("❌ Failed to insert text")
            else:
                print("❌ No editable text field focused")
        else:
            print(f"❌ Transcription failed: {result['error']}")
    
    # Set callbacks
    vad_processor.set_callbacks(
        on_speech_detected=on_speech_detected
    )
    vad_processor.set_speech_segment_callback(on_speech_segment)
    
    # Start processing
    print("\nPreparing to record. Please focus a text field and speak clearly.")
    print(f"Recording will last for {test_duration} seconds...")
    time.sleep(2)
    
    try:
        # Start processing
        vad_processor.start_processing()
        
        # Start recording
        audio_processor.start_recording(mic_manager)
        
        # Wait for test duration
        for i in range(test_duration, 0, -1):
            time.sleep(1)
        
        # Stop recording
        audio_processor.stop_recording()
        
        # Stop processing
        vad_processor.stop_processing()
        
        print("\n✅ End-to-end test completed")
        return True
    except Exception as e:
        print(f"\n❌ Error during end-to-end test: {str(e)}")
        return False
    finally:
        # Clean up
        try:
            audio_processor.stop_recording()
            vad_processor.stop_processing()
        except:
            pass


def run_system_tests(model_path):
    """
    Run all system tests
    
    Args:
        model_path: Path to Whisper model directory
    """
    # Configure logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Speech-to-Text System Tests ===")
    print("Testing core components and end-to-end workflow")
    
    tests = {
        "Microphone Enumeration": test_microphone_enumeration,
        "VAD Model Loading": test_vad_model_loading,
        "Whisper Model Loading": lambda: test_whisper_model_loading(model_path),
        "Text Insertion": test_text_insertion
    }
    
    results = {}
    
    # Run individual tests
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {str(e)}")
            results[test_name] = False
    
    # Run end-to-end test if all component tests passed
    if all(results.values()):
        print("\nAll component tests passed. Running end-to-end test...")
        results["End-to-End Test"] = test_end_to_end(model_path)
    
    # Print summary
    print("\n=== Test Results Summary ===")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    # Overall result
    if all(results.values()):
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


def main(model_path=None):
    """Main function"""
    if model_path is None:
        # Only parse arguments if called directly
        parser = argparse.ArgumentParser(description="Run system tests for the Speech-to-Text application")
        parser.add_argument(
            "--model-path",
            type=str,
            help="Path to Whisper model directory",
            default=os.path.expanduser("~/whisper-models")
        )
        
        args = parser.parse_args()
        model_path = args.model_path
    
    # Make sure model path exists
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        print("Please specify a valid model directory with --model-path")
        return 1
    
    return run_system_tests(model_path)


if __name__ == "__main__":
    sys.exit(main()) 