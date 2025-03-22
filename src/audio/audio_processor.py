#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio processing module for the Speech-to-Text application
"""

import logging
import numpy as np  # Import numpy globally so it's available throughout the class
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
        self.frames = []
        self.stream = None
        self.pyaudio_instance = None
        self.is_recording = False
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
    
    def start_recording(self, mic_manager):
        """
        Start recording audio from the selected microphone
        
        Args:
            mic_manager: MicrophoneManager instance with selected microphone
        """
        import pyaudio  # Import here to avoid dependency if not used
        
        try:
            self.logger.info("Starting audio recording")
            self.is_recording = True
            
            # Initialize PyAudio if needed
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()
            
            # Get the microphone ID from the current_mic attribute instead of calling a method
            if not hasattr(mic_manager, 'current_mic') or mic_manager.current_mic is None:
                raise ValueError("No microphone selected in MicrophoneManager")
            
            mic_id = mic_manager.current_mic["id"]
            
            # Clear previous frames
            self.frames = []
            
            # Open audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=mic_id,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            self.logger.info("Audio recording started")
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            self.stop_recording()
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Stream status
            
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        import pyaudio  # Import here to avoid dependency if not used
        
        if self.is_recording:
            self.frames.append(in_data)
            return (None, pyaudio.paContinue)
        return (None, pyaudio.paComplete)
    
    def stop_recording(self):
        """
        Stop recording audio
        
        Returns:
            np.ndarray: Recorded audio data
        """
        self.logger.info("Stopping audio recording")
        self.is_recording = False
        
        try:
            # Stop and close the stream
            if self.stream is not None:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Cleanup PyAudio
            if self.pyaudio_instance is not None:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            # Convert frames to numpy array
            if self.frames:
                audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                self.logger.info(f"Recording stopped, collected {len(audio_data)} samples")
                return audio_data
            else:
                self.logger.warning("No audio data recorded")
                return np.array([], dtype=np.int16)
                
        except Exception as e:
            self.logger.error(f"Error stopping recording: {str(e)}")
            return np.array([], dtype=np.int16) 