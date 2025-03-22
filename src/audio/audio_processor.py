#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio processing module for the Speech-to-Text application
"""

import logging
import numpy as np
import wave
import os
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path


class AudioProcessor:
    """
    Handles audio processing tasks like preprocessing, saving, and converting
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger.info(f"AudioProcessor initialized (sample_rate={sample_rate}, channels={channels})")
    
    def preprocess_audio(self, audio_data: np.ndarray, 
                         normalize: bool = True,
                         remove_dc: bool = True) -> np.ndarray:
        """
        Preprocess audio data for better recognition
        
        Args:
            audio_data: Raw audio data
            normalize: Whether to normalize the audio
            remove_dc: Whether to remove DC offset
            
        Returns:
            np.ndarray: Processed audio data
        """
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("Empty audio data provided for preprocessing")
            return np.array([], dtype=np.int16)
        
        try:
            # Convert to float for processing
            audio_float = audio_data.astype(np.float32)
            
            # Remove DC offset (mean value) if requested
            if remove_dc:
                audio_float = audio_float - np.mean(audio_float)
            
            # Normalize if requested (to range -1.0 to 1.0)
            if normalize:
                max_val = np.max(np.abs(audio_float))
                if max_val > 0:
                    audio_float = audio_float / max_val
            
            # Convert back to int16
            audio_processed = (audio_float * 32767).astype(np.int16)
            
            self.logger.debug(f"Preprocessed audio: shape={audio_processed.shape}, dtype={audio_processed.dtype}")
            return audio_processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing audio: {str(e)}")
            return audio_data  # Return original on error
    
    def save_audio_to_wave(self, audio_data: np.ndarray, 
                          file_path: str, 
                          sample_width: int = 2) -> bool:
        """
        Save audio data to a WAV file
        
        Args:
            audio_data: Audio data to save
            file_path: Path to save the file
            sample_width: Sample width in bytes (2 for 16-bit)
            
        Returns:
            bool: Whether the file was successfully saved
        """
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("Empty audio data provided for saving")
            return False
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Open file for writing
            with wave.open(file_path, 'wb') as wave_file:
                wave_file.setnchannels(self.channels)
                wave_file.setsampwidth(sample_width)
                wave_file.setframerate(self.sample_rate)
                wave_file.writeframes(audio_data.tobytes())
            
            self.logger.info(f"Saved audio to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving audio to {file_path}: {str(e)}")
            return False
    
    def read_audio_from_wave(self, file_path: str) -> Optional[np.ndarray]:
        """
        Read audio data from a WAV file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[np.ndarray]: Audio data or None if unsuccessful
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Audio file {file_path} does not exist")
            return None
        
        try:
            with wave.open(file_path, 'rb') as wave_file:
                # Get file properties
                channels = wave_file.getnchannels()
                sample_width = wave_file.getsampwidth()
                sample_rate = wave_file.getframerate()
                frames = wave_file.getnframes()
                
                # Read raw frames
                raw_data = wave_file.readframes(frames)
                
                # Convert to numpy array
                if sample_width == 2:  # 16-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.int32)
                elif sample_width == 1:  # 8-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.uint8) - 128  # Convert to signed
                else:
                    self.logger.error(f"Unsupported sample width: {sample_width}")
                    return None
                
                # Reshape for multiple channels
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                
                self.logger.info(f"Read audio from {file_path}: {frames} frames, {channels} channels, {sample_rate} Hz")
                
                # Store sample rate and channels from file
                self.sample_rate = sample_rate
                self.channels = channels
                
                return audio_data
                
        except Exception as e:
            self.logger.error(f"Error reading audio from {file_path}: {str(e)}")
            return None
    
    def convert_sample_rate(self, audio_data: np.ndarray, 
                           original_rate: int, 
                           target_rate: int) -> np.ndarray:
        """
        Convert audio data from one sample rate to another
        
        Args:
            audio_data: Audio data to convert
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            np.ndarray: Resampled audio data
        """
        if original_rate == target_rate:
            return audio_data
            
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.int16)
            
        try:
            # Simple resampling using linear interpolation
            # For more advanced resampling, consider using scipy.signal.resample
            # or librosa.resample
            
            # Calculate resampling ratio
            ratio = target_rate / original_rate
            
            # Calculate new length
            new_length = int(len(audio_data) * ratio)
            
            # Create time points
            original_times = np.arange(len(audio_data))
            new_times = np.linspace(0, len(audio_data) - 1, new_length)
            
            # Perform linear interpolation
            resampled_data = np.interp(new_times, original_times, audio_data).astype(np.int16)
            
            self.logger.debug(f"Converted sample rate from {original_rate} to {target_rate} Hz")
            return resampled_data
            
        except Exception as e:
            self.logger.error(f"Error converting sample rate: {str(e)}")
            return audio_data  # Return original on error
    
    def split_audio(self, audio_data: np.ndarray, 
                   chunk_duration_ms: int = 1000,
                   overlap_ms: int = 0) -> List[np.ndarray]:
        """
        Split audio data into chunks
        
        Args:
            audio_data: Audio data to split
            chunk_duration_ms: Duration of each chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
            
        Returns:
            List[np.ndarray]: List of audio chunks
        """
        if audio_data is None or len(audio_data) == 0:
            return []
            
        try:
            # Calculate chunk size and overlap in samples
            chunk_size = int(self.sample_rate * chunk_duration_ms / 1000)
            overlap_size = int(self.sample_rate * overlap_ms / 1000)
            step_size = chunk_size - overlap_size
            
            # Initialize list for chunks
            chunks = []
            
            # Split the audio
            for i in range(0, len(audio_data) - overlap_size, step_size):
                end = min(i + chunk_size, len(audio_data))
                chunk = audio_data[i:end]
                
                # Only add if chunk is full size or it's the last chunk
                if len(chunk) == chunk_size or end == len(audio_data):
                    chunks.append(chunk)
            
            self.logger.debug(f"Split audio into {len(chunks)} chunks of {chunk_duration_ms}ms")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting audio: {str(e)}")
            return [audio_data]  # Return original as single chunk on error 