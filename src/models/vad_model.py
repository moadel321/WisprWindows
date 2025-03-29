#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silero Voice Activity Detection (VAD) model implementation
"""

import logging
import os
import torch
import numpy as np
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from src.utils.constants import (
    DEFAULT_VAD_THRESHOLD,
    DEFAULT_VAD_WINDOW,
    DEFAULT_VAD_SPEECH_PAD_MS,
    DEFAULT_SAMPLE_RATE
)


class SileroVAD:
    """
    Handles Voice Activity Detection using the Silero VAD model
    """
    
    def __init__(self, 
                sensitivity: float = DEFAULT_VAD_THRESHOLD,
                sample_rate: int = DEFAULT_SAMPLE_RATE,
                window_size_ms: int = DEFAULT_VAD_WINDOW,
                speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
                model_path: Optional[str] = None):
        """
        Initialize the Silero VAD model
        
        Args:
            sensitivity: Sensitivity level for voice detection (0-1)
            sample_rate: Audio sample rate in Hz
            window_size_ms: Window size for voice detection in milliseconds
            speech_pad_ms: Padding around speech segments in milliseconds
            model_path: Optional path to a pre-downloaded model
        """
        self.logger = logging.getLogger(__name__)
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.window_size_ms = window_size_ms
        self.speech_pad_ms = speech_pad_ms
        self.model_path = model_path
        
        # Window size in samples
        self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
        
        # Speech padding in samples
        self.speech_pad_samples = int(self.sample_rate * self.speech_pad_ms / 1000)
        
        # Model components
        self.model = None
        self.utils = None
        self.get_speech_timestamps = None
        self.read_audio = None
        
        # Processing state
        self.is_ready = False
        self.last_speech_end = 0
        
        # Performance metrics
        self.avg_process_time = 0
        self.num_processes = 0
        
        self.logger.info(f"SileroVAD initialized with sensitivity={sensitivity}, window_size_ms={window_size_ms}")
    
    def load_model(self) -> bool:
        """
        Load the Silero VAD model
        
        Returns:
            bool: Whether the model was successfully loaded
        """
        start_time = time.time()
        
        try:
            # Force CPU usage for Silero VAD as requested
            self.device = torch.device('cpu')
            self.logger.info("Forcing CPU for Silero VAD")
            
            # Load the model from PyTorch Hub
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            
            # Move model to the appropriate device
            self.model = model.to(self.device)
            self.utils = utils
            
            # Get utility functions
            self.get_speech_timestamps = utils[0]
            self.read_audio = utils[2]
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Mark as ready
            self.is_ready = True
            load_time = time.time() - start_time
            
            self.logger.info(f"Silero VAD model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Silero VAD model: {str(e)}")
            self.is_ready = False
            return False
    
    def ensure_model_loaded(self) -> bool:
        """
        Ensure the model is loaded before use
        
        Returns:
            bool: Whether the model is ready
        """
        if not self.is_ready:
            return self.load_model()
        return True
    
    def is_speech(self, audio_data: np.ndarray, sample_rate: Optional[int] = None, threshold_override: Optional[float] = None) -> bool:
        """
        Detect whether the audio contains speech
        
        Args:
            audio_data: Audio data to analyze
            sample_rate: Sample rate of the audio data (default: use instance default)
            threshold_override: Override the sensitivity threshold (used for faster response)
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        # Start timing
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.ensure_model_loaded():
            self.logger.error("Cannot detect speech, VAD model not loaded")
            return False
        
        # Use instance sample rate if not specified
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Use threshold override if provided
        threshold = threshold_override if threshold_override is not None else self.sensitivity
        
        try:
            # Convert numpy array to PyTorch tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.tensor(audio_data.astype(np.float32))
            else:
                audio_tensor = audio_data
            
            # Ensure the tensor is 1D
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                self.logger.debug(f"Resampling audio from {sample_rate} to {self.sample_rate}")
                # Use simple resampling for now - we could add more sophisticated methods later
                resampled_len = int(len(audio_tensor) * self.sample_rate / sample_rate)
                indices = torch.linspace(0, len(audio_tensor) - 1, resampled_len)
                audio_tensor = torch.nn.functional.interpolate(
                    audio_tensor.view(1, 1, -1),
                    size=resampled_len,
                    mode='linear'
                ).view(-1)
            
            # Normalize audio
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Check if audio is too short
            if len(audio_tensor) < self.window_size_samples:
                self.logger.debug(f"Audio too short ({len(audio_tensor)} samples), padding")
                # Pad short audio to minimum length
                padding = torch.zeros(self.window_size_samples - len(audio_tensor))
                audio_tensor = torch.cat([audio_tensor, padding])
            
            # Move to the same device as the model
            audio_tensor = audio_tensor.to(self.device)
            
            # Get speech timestamps
            # Optimize sensitivity parameters for faster response
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=50,   # Reduced from 100ms to 50ms for faster detection
                min_silence_duration_ms=50,  # Reduced from 100ms to 50ms for faster transitions
                return_seconds=False         # Return sample indices for faster processing
            )
            
            # Update processing metrics
            end_time = time.time()
            process_time = end_time - start_time
            self.avg_process_time = (self.avg_process_time * self.num_processes + process_time) / (self.num_processes + 1)
            self.num_processes += 1
            
            # Log timing periodically
            if self.num_processes % 100 == 0:
                self.logger.debug(f"VAD average processing time: {self.avg_process_time:.4f}s")
            
            # Return True if any speech detected
            has_speech = len(speech_timestamps) > 0
            return has_speech
            
        except Exception as e:
            self.logger.error(f"Error detecting speech: {str(e)}")
            return False
    
    def get_speech_timestamps(self, audio_data: np.ndarray, 
                            sample_rate: Optional[int] = None) -> List[Dict[str, int]]:
        """
        Get timestamps for speech segments in the audio
        
        Args:
            audio_data: Audio data to analyze
            sample_rate: Sample rate of the audio data (default: use instance default)
            
        Returns:
            List[Dict[str, int]]: List of speech segments with start and end timestamps
        """
        # Ensure model is loaded
        if not self.ensure_model_loaded():
            self.logger.error("Cannot get speech timestamps, VAD model not loaded")
            return []
        
        # Use instance sample rate if not specified
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Convert numpy array to PyTorch tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.tensor(audio_data.astype(np.float32))
            else:
                audio_tensor = audio_data
            
            # Ensure the tensor is 1D
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                self.logger.debug(f"Resampling audio from {sample_rate} to {self.sample_rate}")
                # Use simple resampling for now - we could add more sophisticated methods later
                resampled_len = int(len(audio_tensor) * self.sample_rate / sample_rate)
                indices = torch.linspace(0, len(audio_tensor) - 1, resampled_len)
                audio_tensor = torch.nn.functional.interpolate(
                    audio_tensor.view(1, 1, -1),
                    size=resampled_len,
                    mode='linear'
                ).view(-1)
            
            # Normalize audio
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Move to the same device as the model
            audio_tensor = audio_tensor.to(self.device)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.sensitivity,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=100,  # Minimum speech duration in ms
                min_silence_duration_ms=100  # Minimum silence duration in ms
            )
            
            # Convert timestamps to Python dictionaries
            timestamps = []
            for ts in speech_timestamps:
                timestamps.append({
                    "start": int(ts["start"]),
                    "end": int(ts["end"])
                })
            
            return timestamps
            
        except Exception as e:
            self.logger.error(f"Error getting speech timestamps: {str(e)}")
            return []
    
    def get_speech_segments(self, audio_data: np.ndarray, 
                          sample_rate: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract speech segments from audio data
        
        Args:
            audio_data: Audio data to process
            sample_rate: Sample rate of the audio data (default: use instance default)
            
        Returns:
            List[np.ndarray]: List of speech audio segments
        """
        # Get timestamps
        timestamps = self.get_speech_timestamps(audio_data, sample_rate)
        
        # Extract segments
        segments = []
        for ts in timestamps:
            # Add padding around speech (but don't go out of bounds)
            start = max(0, ts["start"] - self.speech_pad_samples)
            end = min(len(audio_data), ts["end"] + self.speech_pad_samples)
            
            # Extract the segment
            segment = audio_data[start:end]
            segments.append(segment)
        
        return segments
    
    def filter_non_speech(self, audio_data: np.ndarray, 
                         sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Filter out non-speech parts of audio data
        
        Args:
            audio_data: Audio data to filter
            sample_rate: Sample rate of the audio data (default: use instance default)
            
        Returns:
            np.ndarray: Audio data with only speech segments
        """
        # Get timestamps
        timestamps = self.get_speech_timestamps(audio_data, sample_rate)
        
        if not timestamps:
            self.logger.debug("No speech detected in audio")
            return np.array([])
        
        # Create mask of the same length as audio_data, with 0 for non-speech and 1 for speech
        mask = np.zeros(len(audio_data), dtype=np.float32)
        
        for ts in timestamps:
            # Add padding around speech (but don't go out of bounds)
            start = max(0, ts["start"] - self.speech_pad_samples)
            end = min(len(audio_data), ts["end"] + self.speech_pad_samples)
            
            # Mark as speech
            mask[start:end] = 1.0
        
        # Apply the mask
        filtered_audio = audio_data * mask
        
        return filtered_audio
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set the VAD sensitivity
        
        Args:
            sensitivity: New sensitivity value (0-1)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.logger.info(f"VAD sensitivity set to {self.sensitivity}")
    
    def is_continuous_speech(self, audio_data: np.ndarray, 
                           max_silence_ms: int = 500) -> bool:
        """
        Check if the audio contains continuous speech without long silences
        
        Args:
            audio_data: Audio data to analyze
            max_silence_ms: Maximum allowed silence duration in milliseconds
            
        Returns:
            bool: True if the audio has continuous speech, False otherwise
        """
        # Get timestamps
        timestamps = self.get_speech_timestamps(audio_data)
        
        if not timestamps:
            return False
        
        # Maximum silence in samples
        max_silence_samples = int(self.sample_rate * max_silence_ms / 1000)
        
        # Check gaps between speech segments
        for i in range(1, len(timestamps)):
            prev_end = timestamps[i-1]["end"]
            curr_start = timestamps[i]["start"]
            
            # If silence gap is too large
            if curr_start - prev_end > max_silence_samples:
                return False
        
        return True 