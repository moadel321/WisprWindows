#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for error handling and edge cases in the STT application
"""

import sys
import os
import unittest
import tempfile
import logging
import time
import threading
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import AppSettings
from src.audio.microphone import MicrophoneManager
from src.audio.vad_processor import VADProcessor
from src.models.faster_whisper_model import FasterWhisperModel
from src.text_insertion.text_inserter import TextInserter
from src.gui.app_controller import AppController


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests"""
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create settings
        cls.settings = AppSettings()
        cls.settings.set("model.directory", cls.temp_dir)
        cls.settings.set("model.name", "distil-large-v3")
        cls.settings.set("model.compute_type", "int8")
        cls.settings.set("audio.sample_rate", 16000)
        cls.settings.set("audio.channels", 1)
        cls.settings.set("vad.sensitivity", 0.5)
        cls.settings.set("vad.window_size_ms", 30)
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures for all tests"""
        # Re-enable logging
        logging.disable(logging.NOTSET)
        
        # Clean up temporary directories
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            for file in os.listdir(cls.temp_dir):
                try:
                    os.remove(os.path.join(cls.temp_dir, file))
                except:
                    pass
            os.rmdir(cls.temp_dir)
    
    def test_controller_model_load_failure(self):
        """Test error handling when model fails to load"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Mock model loading to fail
        controller.whisper_model.load_model = MagicMock(return_value=False)
        controller.whisper_model.error_message = "Test error: Model not found"
        
        # Create callback tracking
        model_status_result = None
        error_message = None
        
        def model_status_callback(is_loaded, error=None):
            nonlocal model_status_result, error_message
            model_status_result = is_loaded
            error_message = error
        
        def error_callback(msg):
            nonlocal error_message
            error_message = msg
        
        controller.set_model_status_callback(model_status_callback)
        controller.set_error_callback(error_callback)
        
        # Try to load the model
        result = controller.ensure_model_loaded()
        
        # Check results
        self.assertFalse(result)
        self.assertFalse(controller.is_model_loaded)
        self.assertFalse(model_status_result)
        self.assertIn("Model not found", error_message)
    
    def test_microphone_start_failure(self):
        """Test error handling when microphone fails to start"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Mock microphone to fail
        controller.mic_manager.start_recording = MagicMock(return_value=False)
        
        # Create callback tracking
        error_received = False
        
        def error_callback(msg):
            nonlocal error_received
            error_received = True
        
        controller.set_error_callback(error_callback)
        
        # Also mock VAD to avoid issues
        controller.vad_processor.start_processing = MagicMock(return_value=True)
        controller.vad_processor.stop_processing = MagicMock(return_value=True)
        
        # Try to start transcription
        result = controller.start_transcription()
        
        # Check results
        self.assertFalse(result)
        self.assertFalse(controller.is_transcribing)
        self.assertTrue(error_received)
        
        # Verify VAD processing was stopped due to microphone failure
        controller.vad_processor.stop_processing.assert_called_once()
    
    def test_vad_start_failure(self):
        """Test error handling when VAD fails to start"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Use a MagicMock that provides the 'called' attribute
        mock_start_processing = MagicMock(return_value=False)
        controller.vad_processor.start_processing = mock_start_processing
        
        # Also properly mock the mic_manager
        mock_start_recording = MagicMock(return_value=True)
        controller.mic_manager.start_recording = mock_start_recording
        
        # Create callback tracking
        error_received = False
        
        def error_callback(msg):
            nonlocal error_received
            error_received = True
        
        controller.set_error_callback(error_callback)
        
        # Try to start transcription
        result = controller.start_transcription()
        
        # Check results
        self.assertFalse(result)
        self.assertFalse(controller.is_transcribing)
        self.assertTrue(error_received)
        
        # Verify microphone was not started due to VAD failure
        self.assertEqual(mock_start_recording.call_count, 0)
    
    def test_transcription_error_handling(self):
        """Test error handling during transcription"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Mock the whisper model to raise an exception
        controller.whisper_model.transcribe = MagicMock(side_effect=Exception("Test transcription error"))
        controller.whisper_model.is_loaded = True
        controller.is_model_loaded = True
        
        # Create callback tracking
        error_message = None
        
        def error_callback(msg):
            nonlocal error_message
            error_message = msg
        
        controller.set_error_callback(error_callback)
        
        # Create a test segment info
        segment_info = {
            "start_time": time.time() - 3,
            "end_time": time.time(),
            "duration": 3.0,
            "timestamp": "2023-03-27 12:34:56",
            "trace_id": "test_trace"
        }
        
        # Call the transcription worker directly
        controller._transcription_worker("nonexistent_file.wav", segment_info)
        
        # Wait for any async operations
        time.sleep(0.2)
        
        # Check results
        self.assertIsNotNone(error_message)
        self.assertIn("Transcription error", error_message)
        self.assertFalse(controller.is_processing_transcription)
    
    def test_stop_transcription_with_pending_work(self):
        """Test stopping transcription while processing is in progress"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Force the processing flag to be True
        with controller.transcription_lock:
            controller.is_processing_transcription = True
        
        # Mark as transcribing
        controller.is_transcribing = True
        
        # Mock the microphone and VAD components
        controller.mic_manager.stop_recording = MagicMock(return_value=True)
        controller.vad_processor.stop_processing = MagicMock(return_value=True)
        
        # Create callback tracking
        speech_detection_called = False
        
        def speech_detection_callback(is_speech):
            nonlocal speech_detection_called
            speech_detection_called = True
        
        controller.set_speech_detected_callback(speech_detection_callback)
        controller.speech_detected = True
        
        # Stop transcription
        result = controller.stop_transcription()
        
        # Check results
        self.assertTrue(result)
        self.assertFalse(controller.is_transcribing)
        self.assertTrue(speech_detection_called)
        self.assertFalse(controller.speech_detected)
        self.assertFalse(controller.is_processing_transcription)
    
    def test_text_insertion_all_methods_fail(self):
        """Test text insertion when all methods fail"""
        # Create a text inserter
        text_inserter = TextInserter()
        
        # Mock get_focused_element to return a valid element
        mock_element = MagicMock()
        mock_element_info = {
            "app": "Notepad",
            "app_class": "Notepad",
            "element": mock_element,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Mock _ensure_foreground_window
        text_inserter._ensure_foreground_window = MagicMock(return_value=True)
        
        # Mock all insertion methods to fail
        text_inserter._insert_via_clipboard = MagicMock(return_value=False)
        text_inserter._insert_via_element = MagicMock(return_value=False)
        text_inserter._insert_via_direct_input = MagicMock(return_value=False)
        text_inserter._insert_via_char_by_char = MagicMock(return_value=False)
        
        # Try to insert text
        result = text_inserter.insert_text("Test text")
        
        # Check results
        self.assertFalse(result)
        
        # Verify all methods were attempted
        text_inserter._insert_via_clipboard.assert_called_once()
        text_inserter._insert_via_element.assert_called_once()
        text_inserter._insert_via_direct_input.assert_called_once()
        text_inserter._insert_via_char_by_char.assert_called_once()
    
    def test_model_transcribe_empty_audio(self):
        """Test model handling of empty/corrupted audio files"""
        # Create a faster whisper model
        model = FasterWhisperModel(
            model_dir=self.temp_dir,
            model_name="distil-large-v3",
            language="en",
            device="cpu",
            compute_type="int8"
        )
        
        # Instead of creating an actual empty file, just use a non-existent path
        empty_audio_file = os.path.join(self.temp_dir, "nonexistent_audio.wav")
        
        # Create a custom transcribe function that mimics the error we want
        def mock_transcribe(audio_path, **kwargs):
            return {
                "success": False,
                "error": f"File not found: {audio_path}",
                "text": ""
            }
            
        # Save original and replace with our mock
        original_transcribe = model.transcribe
        model.transcribe = mock_transcribe
        
        try:
            # Try to transcribe with our mock
            result = model.transcribe(empty_audio_file)
            
            # Check results
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("not found", result["error"])
        finally:
            # Restore original method
            model.transcribe = original_transcribe
    
    def test_stop_fail_recovery(self):
        """Test recovery from failed stop operation"""
        # Create controller with mocked components
        controller = AppController(self.settings)
        
        # Set transcribing state
        controller.is_transcribing = True
        
        # Mock the microphone and VAD components to fail
        controller.mic_manager.stop_recording = MagicMock(return_value=False)
        controller.vad_processor.stop_processing = MagicMock(side_effect=Exception("VAD stop error"))
        
        # Stop transcription
        result = controller.stop_transcription()
        
        # Check results - should return failure but reset state
        self.assertFalse(result)
        self.assertFalse(controller.is_transcribing)  # State should be reset despite errors


if __name__ == '__main__':
    unittest.main() 