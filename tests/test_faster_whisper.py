#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Faster Whisper model
"""

import sys
import os
import time
import tempfile
import unittest
import logging
from pathlib import Path
import torch
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.faster_whisper_model import FasterWhisperModel


class TestFasterWhisper(unittest.TestCase):
    """Test cases for the Faster Whisper model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests"""
        # Skip tests if running in CI environment without GPU
        if os.environ.get('CI') and not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping Whisper tests in CI environment without GPU")
            
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create a temporary directory for the model
        cls.temp_dir = tempfile.mkdtemp()
        
        # Check if we should skip tests due to model size
        cls.skip_model_tests = os.environ.get('SKIP_MODEL_TESTS', 'false').lower() == 'true'
        if cls.skip_model_tests:
            print("Skipping actual model tests due to SKIP_MODEL_TESTS=true")
    
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
    
    def setUp(self):
        """Set up test fixtures for each test"""
        # Skip actual model tests if needed
        if self.skip_model_tests:
            self.skipTest("Skipping model tests due to SKIP_MODEL_TESTS=true")
            
        # Determine device and compute type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        # Create the model instance
        self.model_name = "large-v3"
        self.model = FasterWhisperModel(
            model_dir=self.temp_dir,
            model_name=self.model_name,
            language="en",
            device=self.device,
            compute_type=self.compute_type
        )
    
    @patch('src.models.faster_whisper_model.download_model')
    @patch('src.models.faster_whisper_model.WhisperModel')
    def test_model_loading(self, mock_whisper_model, mock_download):
        """Test model loading"""
        # Mock the download_model function
        mock_download.return_value = "/path/to/model"
        
        # Create a mock for the WhisperModel
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = iter([
            MagicMock(text="Test transcription")
        ])
        mock_whisper_model.return_value = mock_model_instance
        
        # Load the model
        result = self.model.load_model()
        
        # Assert results
        self.assertTrue(result["success"])
        self.assertTrue(self.model.is_loaded)
        mock_download.assert_called_once_with(
            model=self.model_name,
            cache_dir=self.model.model_dir
        )
        mock_whisper_model.assert_called_once()
    
    @patch('src.models.faster_whisper_model.WhisperModel')
    def test_transcription(self, mock_whisper_model):
        """Test transcription"""
        # Create a mock for the WhisperModel
        mock_model_instance = MagicMock()
        
        # Create mock segments
        mock_segment1 = MagicMock()
        mock_segment1.text = "This is the first segment."
        mock_segment1.start = 0.0
        mock_segment1.end = 2.0
        
        mock_segment2 = MagicMock()
        mock_segment2.text = "This is the second segment."
        mock_segment2.start = 2.1
        mock_segment2.end = 4.0
        
        # Set up mock words
        mock_segment1.words = [
            MagicMock(word="This", start=0.0, end=0.3),
            MagicMock(word="is", start=0.35, end=0.5),
            MagicMock(word="the", start=0.55, end=0.7),
            MagicMock(word="first", start=0.75, end=1.0),
            MagicMock(word="segment", start=1.1, end=2.0)
        ]
        
        # Return an iterator with the mock segments
        mock_model_instance.transcribe.return_value = iter([mock_segment1, mock_segment2])
        mock_whisper_model.return_value = mock_model_instance
        
        # Set model as loaded with our mock
        self.model._model = mock_model_instance
        self.model.is_loaded = True
        
        # Create a temporary audio file path (don't actually create the file)
        temp_file = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a small test file
        with open(temp_file, 'wb') as f:
            f.write(b'dummy audio data')
        
        # Mock os.path.exists to ensure the audio file is found
        with patch('os.path.exists', return_value=True):
            # Call transcribe
            result = self.model.transcribe(
                audio_file=temp_file,
                language="en",
                word_timestamps=True,
                vad_filter=True
            )
            
            # Assert results
            self.assertTrue(result["success"])
            self.assertEqual(result["text"], "This is the first segment. This is the second segment.")
            self.assertEqual(len(result["segments"]), 2)
            
            # Check segment information
            self.assertEqual(result["segments"][0]["text"], "This is the first segment.")
            self.assertEqual(result["segments"][0]["start"], 0.0)
            self.assertEqual(result["segments"][0]["end"], 2.0)
            
            # Check word timestamps
            self.assertEqual(len(result["segments"][0]["words"]), 5)
            self.assertEqual(result["segments"][0]["words"][0]["word"], "This")


if __name__ == '__main__':
    unittest.main() 