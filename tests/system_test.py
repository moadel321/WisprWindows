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
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import AppSettings
from src.audio.microphone import MicrophoneManager
from src.audio.audio_processor import AudioProcessor
from src.audio.vad_processor import VADProcessor
from src.models.vad_model import SileroVAD
from src.models.faster_whisper_model import FasterWhisperModel
from src.text_insertion.text_inserter import TextInserter
from src.gui.app_controller import AppController


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


def test_whisper_model_loading():
    """Test loading the Whisper model"""
    # Skip the actual model loading for automated tests
    if os.environ.get('SKIP_MODEL_TESTS', 'false').lower() == 'true':
        print("Skipping actual model loading test")
        return
        
    # Mock the model loading to avoid actual downloads and initialization
    with patch('src.models.faster_whisper_model.download_model', return_value="/mock/path/to/model"):
        with patch('src.models.faster_whisper_model.WhisperModel') as mock_whisper:
            # Set up the mock
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = iter([MagicMock(text="Test transcription")])
            mock_whisper.return_value = mock_instance
        
            # Create model
            model = FasterWhisperModel(
                model_dir=tempfile.mkdtemp(),
                model_name="distil-large-v3",
                language="en",
                device="cpu",
                compute_type="int8"
            )
            
            # Test loading
            result = model.load_model()
            assert result, "Model loading failed"
            assert model.is_loaded, "Model not marked as loaded"
            assert model._model is not None, "Model instance not set"


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


def test_end_to_end():
    """Test the end-to-end process"""
    # Skip this test as it requires real microphone and model
    if 'CI' in os.environ or os.environ.get('SKIP_E2E_TESTS', 'false').lower() == 'true':
        print("Skipping end-to-end test in CI environment")
        return
        
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create settings
        settings = AppSettings()
        settings.set("model.directory", temp_dir)
        settings.set("model.name", "distil-large-v3")
        settings.set("model.compute_type", "int8")
        settings.set("audio.sample_rate", 16000)
        settings.set("audio.channels", 1)
        
        # Create controller with extensive mocking
        with patch('src.models.faster_whisper_model.download_model', return_value="/mock/path/to/model"):
            with patch('src.models.faster_whisper_model.WhisperModel') as mock_whisper:
                with patch('src.models.vad_model.SileroVAD.load_model', return_value=True):
                    with patch('src.audio.microphone.MicrophoneManager.get_available_microphones', 
                               return_value=[{"id": 0, "name": "Test Mic", "channels": 1}]):
                        with patch('src.text_insertion.text_inserter.win32gui'):
                        
                            # Set up the mock model
                            mock_instance = MagicMock()
                            mock_instance.transcribe.return_value = iter([MagicMock(text="Test transcription")])
                            mock_whisper.return_value = mock_instance
                            
                            # Create controller
                            controller = AppController(settings)
                            
                            # Mock microphone and VAD
                            controller.mic_manager.start_recording = MagicMock(return_value=True)
                            controller.mic_manager.stop_recording = MagicMock(return_value=True)
                            controller.vad_processor.start_processing = MagicMock(return_value=True)
                            controller.vad_processor.stop_processing = MagicMock(return_value=True)
                            
                            # Track callback results
                            model_loaded = False
                            transcription_text = None
                            
                            def model_status_callback(is_loaded, error=None):
                                nonlocal model_loaded
                                model_loaded = is_loaded
                            
                            def transcription_callback(text, success=False):
                                nonlocal transcription_text
                                transcription_text = text
                            
                            # Set callbacks
                            controller.set_model_status_callback(model_status_callback)
                            controller.set_transcription_callback(transcription_callback)
                            
                            # Test model loading
                            controller.load_model()
                            assert model_loaded, "Model not loaded"
                            
                            # Test start/stop transcription
                            result = controller.start_transcription()
                            assert result, "Failed to start transcription"
                            assert controller.is_transcribing, "Controller not in transcribing state"
                            
                            # Simulate speech detection and processing
                            # Create audio segment
                            segment = {
                                "audio": np.zeros(16000, dtype=np.float32),
                                "start_time": time.time() - 1,
                                "end_time": time.time(),
                                "duration": 1.0,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "trace_id": "test_trace"
                            }
                            
                            # Force the model into a loaded state
                            controller.whisper_model._model = mock_instance
                            controller.whisper_model.is_loaded = True
                            controller.is_model_loaded = True
                            
                            # Call the speech end callback directly
                            controller.vad_processor._on_speech_end(segment)
                            
                            # Wait for processing
                            time.sleep(0.5)
                            
                            # Stop transcription
                            result = controller.stop_transcription()
                            assert result, "Failed to stop transcription"
                            assert not controller.is_transcribing, "Controller still in transcribing state"
    
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            os.rmdir(temp_dir)


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
        "Whisper Model Loading": test_whisper_model_loading,
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
        results["End-to-End Test"] = test_end_to_end()
    
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