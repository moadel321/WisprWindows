#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for audio and VAD components
"""

import sys
import os
import time
import unittest
import logging
from unittest.mock import MagicMock, patch
import numpy as np
import threading

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio.microphone import MicrophoneManager
from src.audio.audio_processor import AudioProcessor
from src.audio.vad_processor import VADProcessor
from src.models.vad_model import SileroVAD


class TestAudioVADIntegration(unittest.TestCase):
    """Test cases for audio and VAD integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create a mock VAD model
        self.mock_vad_model = MagicMock(spec=SileroVAD)
        self.mock_vad_model.process_audio.return_value = 0.2  # No speech by default
        
        # Create AudioProcessor with simulated audio data
        self.audio_processor = AudioProcessor(
            sample_rate=16000,
            channels=1
        )
        
        # Create VAD processor with mocked VAD model
        self.vad_processor = VADProcessor(
            sample_rate=16000,
            vad_threshold=0.5,
            window_size_ms=30,
            audio_processor=self.audio_processor
        )
        
        # Replace real VAD model with mock
        self.vad_processor.vad_model = self.mock_vad_model
        
        # Setup callback mocks
        self.speech_start_callback = MagicMock()
        self.speech_end_callback = MagicMock()
        self.speech_detected_callback = MagicMock()
        
        # Set callbacks
        self.vad_processor.set_callbacks(
            on_speech_start=self.speech_start_callback,
            on_speech_end=self.speech_end_callback,
            on_speech_detected=self.speech_detected_callback
        )
        
        # Create mock microphone manager
        self.mic_manager = MagicMock(spec=MicrophoneManager)
        self.audio_processor.mic_manager = self.mic_manager
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop VAD processing if running
        if self.vad_processor.is_processing:
            self.vad_processor.stop_processing()
            
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_vad_start_stop_processing(self):
        """Test starting and stopping VAD processing"""
        # Start VAD processing
        result = self.vad_processor.start_processing()
        
        # Assert processing started
        self.assertTrue(result)
        self.assertTrue(self.vad_processor.is_processing)
        
        # Stop VAD processing
        result = self.vad_processor.stop_processing()
        
        # Assert processing stopped
        self.assertTrue(result)
        self.assertFalse(self.vad_processor.is_processing)
    
    def test_vad_detects_speech(self):
        """Test VAD correctly detects speech"""
        # Start VAD processing
        self.vad_processor.start_processing()
        
        # Simulate speech detection
        # First set VAD to return low probability (no speech)
        self.mock_vad_model.process_audio.return_value = 0.2
        
        # Call audio callback with some simulated audio data
        audio_data = np.zeros(16000 // 10, dtype=np.float32)  # 100ms of silence
        self.audio_processor._process_audio(audio_data)
        
        # Check speech not detected yet
        self.speech_detected_callback.assert_called_with(False)
        
        # Now change VAD to return high probability (speech)
        self.mock_vad_model.process_audio.return_value = 0.8
        
        # Call audio callback with some simulated audio data
        audio_data = np.random.rand(16000 // 10).astype(np.float32)  # 100ms of "speech"
        self.audio_processor._process_audio(audio_data)
        
        # Check speech detected
        self.speech_detected_callback.assert_called_with(True)
        self.speech_start_callback.assert_called_once()
        
        # Now back to silence
        self.mock_vad_model.process_audio.return_value = 0.2
        
        # Call audio callback with some simulated audio data
        audio_data = np.zeros(16000 // 10, dtype=np.float32)  # 100ms of silence
        self.audio_processor._process_audio(audio_data)
        
        # Check speech ended detected
        self.speech_detected_callback.assert_called_with(False)
        self.speech_end_callback.assert_called_once()
        
        # Stop VAD processing
        self.vad_processor.stop_processing()
    
    def test_audio_segment_creation(self):
        """Test audio segments are created correctly"""
        # Set up a callback to capture audio segments
        segment_callback = MagicMock()
        self.vad_processor.set_speech_segment_callback(segment_callback)
        
        # Start VAD processing
        self.vad_processor.start_processing()
        
        # Simulate speech detection
        # First, speech begins
        self.mock_vad_model.process_audio.return_value = 0.8
        audio_data = np.random.rand(16000 // 5).astype(np.float32)  # 200ms of "speech"
        self.audio_processor._process_audio(audio_data)
        
        # More speech
        audio_data = np.random.rand(16000 // 5).astype(np.float32)  # 200ms more "speech"
        self.audio_processor._process_audio(audio_data)
        
        # Now speech ends
        self.mock_vad_model.process_audio.return_value = 0.2
        audio_data = np.zeros(16000 // 10, dtype=np.float32)  # 100ms of silence
        self.audio_processor._process_audio(audio_data)
        
        # Since real-time processing is multi-threaded, wait briefly for processing
        time.sleep(0.1)
        
        # Verify segment callback was called
        segment_callback.assert_called()
        
        # Verify the segment info contains expected fields
        call_args = segment_callback.call_args[0]
        self.assertGreaterEqual(len(call_args), 2)
        
        # Check audio file and segment info
        audio_file = call_args[0]
        segment_info = call_args[1]
        
        self.assertIsInstance(audio_file, str)
        self.assertTrue(os.path.exists(audio_file))
        self.assertIn("duration", segment_info)
        self.assertIn("timestamp", segment_info)
        
        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        # Stop VAD processing
        self.vad_processor.stop_processing()
    
    def test_error_handling(self):
        """Test error handling during processing"""
        # Set up audio processor to raise an exception during processing
        self.audio_processor._process_audio = MagicMock(side_effect=Exception("Test exception"))
        
        # Start VAD processing
        self.vad_processor.start_processing()
        
        # Try to process audio, which should raise an exception but be caught
        try:
            # Manually trigger the processing method
            callback = self.audio_processor.callback
            audio_data = np.zeros(16000 // 10, dtype=np.float32)
            callback(audio_data, None)
            
            # If we get here, exception was caught
            self.assertTrue(True)
        except Exception:
            self.fail("Exception was not properly caught")
            
        # Stop VAD processing
        self.vad_processor.stop_processing()


if __name__ == '__main__':
    unittest.main() 