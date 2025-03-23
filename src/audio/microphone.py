#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Microphone handling for audio capture
"""

import logging
import pyaudio
import numpy as np
import threading
import time
import queue
from typing import Optional, List, Dict, Any, Callable, Tuple

from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FORMAT
)


class MicrophoneManager:
    """
    Handles microphone enumeration and audio capture
    """
    
    def __init__(self):
        """Initialize the microphone manager"""
        self.logger = logging.getLogger(__name__)
        self.available_mics = []
        self.current_mic = None
        self.is_recording = False
        self.stream = None
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Audio settings
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.channels = DEFAULT_CHANNELS
        self.chunk_size = DEFAULT_CHUNK_SIZE
        
        # Format mapping (convert string format to PyAudio format)
        self.format_map = {
            "int16": pyaudio.paInt16,
            "int24": pyaudio.paInt24,
            "int32": pyaudio.paInt32,
            "float32": pyaudio.paFloat32
        }
        self.format = self.format_map.get(DEFAULT_FORMAT, pyaudio.paInt16)
        
        # Recording resources
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recording_thread = None
        
        # Buffer for saving audio chunks
        self.audio_buffer = []
        
        self.logger.info("MicrophoneManager initialized")
    
    def __del__(self):
        """Clean up resources on object destruction"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
            self.logger.debug("PyAudio terminated")
    
    def get_available_microphones(self) -> List[Dict[str, Any]]:
        """
        Get a list of available microphones
        
        Returns:
            List[Dict[str, Any]]: List of microphone information
        """
        self.available_mics = []
        unique_mic_names = set()  # Track unique microphone names
        
        try:
            # Get number of available input devices
            device_count = self.audio.get_device_count()
            self.logger.debug(f"Found {device_count} audio devices")
            
            # Get default device info
            try:
                default_device = self.audio.get_default_input_device_info()
                default_id = default_device['index']
            except:
                default_id = -1  # No default device found
            
            # First pass - collect all input devices
            all_mics = []
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                
                # Check if device has input channels
                if device_info['maxInputChannels'] > 0:
                    mic_info = {
                        "id": device_info['index'],
                        "name": device_info['name'],
                        "channels": device_info['maxInputChannels'],
                        "default_sample_rate": device_info['defaultSampleRate'],
                        "is_default": default_id == device_info['index']
                    }
                    all_mics.append(mic_info)
            
            # Sort by default status (default first) and then by name
            all_mics.sort(key=lambda x: (not x["is_default"], x["name"]))
            
            # Second pass - filter out duplicates with similar names
            for mic in all_mics:
                # Clean the name for comparison (remove parentheses and numbers that often cause duplicates)
                clean_name = ''.join([c for c in mic["name"] if not c.isdigit() and c not in '()[]{}'])
                clean_name = clean_name.strip().lower()
                
                # If we haven't seen this microphone before, add it
                if clean_name not in unique_mic_names:
                    unique_mic_names.add(clean_name)
                    self.available_mics.append(mic)
                    self.logger.debug(f"Added microphone: {mic['name']} (ID: {mic['id']})")
            
            # If no microphones found, add a placeholder for the default
            if not self.available_mics:
                if default_id >= 0:
                    default_info = self.audio.get_device_info_by_index(default_id)
                    self.available_mics.append({
                        "id": default_id,
                        "name": f"Default ({default_info['name']})",
                        "channels": default_info['maxInputChannels'],
                        "default_sample_rate": default_info['defaultSampleRate'],
                        "is_default": True
                    })
                    self.logger.debug(f"Using default microphone: {default_info['name']} (ID: {default_id})")
                else:
                    # Fallback if no default device
                    self.available_mics.append({
                        "id": 0,
                        "name": "Default Microphone",
                        "channels": 1,
                        "default_sample_rate": DEFAULT_SAMPLE_RATE,
                        "is_default": True
                    })
        
        except Exception as e:
            self.logger.error(f"Error enumerating microphones: {str(e)}")
            # Add a fallback default device when enumeration fails
            self.available_mics = [{
                "id": 0,
                "name": "Default Microphone",
                "channels": 1,
                "default_sample_rate": DEFAULT_SAMPLE_RATE,
                "is_default": True
            }]
        
        return self.available_mics
    
    def select_microphone(self, mic_id: int) -> bool:
        """
        Select a microphone for recording
        
        Args:
            mic_id: ID of the microphone to select
            
        Returns:
            bool: Whether the microphone was successfully selected
        """
        try:
            # Stop any active recording
            if self.is_recording:
                self.stop_recording()
            
            # Verify the microphone exists
            device_info = self.audio.get_device_info_by_index(mic_id)
            if device_info['maxInputChannels'] <= 0:
                self.logger.error(f"Device with ID {mic_id} is not a valid input device")
                return False
            
            # Store the selected microphone information
            self.current_mic = {
                "id": mic_id,
                "name": device_info['name'],
                "channels": min(device_info['maxInputChannels'], DEFAULT_CHANNELS),  # Use default or max available
                "sample_rate": int(device_info['defaultSampleRate'])
            }
            
            self.logger.info(f"Selected microphone: {self.current_mic['name']} (ID: {self.current_mic['id']})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error selecting microphone {mic_id}: {str(e)}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flags
            
        Returns:
            tuple: (None, paContinue)
        """
        if status:
            self.logger.debug(f"Audio callback status: {status}")
        
        if not self.stop_event.is_set():
            # Convert audio data to numpy array for processing
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            try:
                # Put the audio data in the queue for processing
                self.audio_queue.put_nowait((audio_data, time_info))
                
                # Also store in buffer if needed
                self.audio_buffer.append(audio_data)
                
            except queue.Full:
                self.logger.warning("Audio queue is full, dropping audio chunk")
            
        return (None, pyaudio.paContinue)
    
    def _audio_processing_thread(self, callback: Optional[Callable[[np.ndarray, Dict], None]] = None):
        """
        Thread function to process audio data from the queue
        
        Args:
            callback: Function to call with processed audio data
        """
        self.logger.debug("Audio processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from the queue with timeout
                audio_data, time_info = self.audio_queue.get(timeout=0.5)
                
                # Call the callback function if provided
                if callback:
                    try:
                        callback(audio_data, time_info)
                    except Exception as e:
                        self.logger.error(f"Error in audio callback: {str(e)}")
                
                # Mark the task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Timeout occurred, check if we should stop
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio data: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop in case of recurring errors
        
        self.logger.debug("Audio processing thread stopped")
    
    def start_recording(self, callback: Optional[Callable[[np.ndarray, Dict], None]] = None) -> bool:
        """
        Start recording from the selected microphone
        
        Args:
            callback: Function to call with audio data and time info
            
        Returns:
            bool: Whether recording was successfully started
        """
        # Ensure we have a selected microphone
        if not self.current_mic:
            try:
                # Try to select the default microphone
                default_id = self.audio.get_default_input_device_info()['index']
                if not self.select_microphone(default_id):
                    self.logger.error("No microphone selected and couldn't select default")
                    return False
            except Exception as e:
                self.logger.error(f"Error selecting default microphone: {str(e)}")
                return False
        
        # Check if already recording
        if self.is_recording:
            self.logger.warning("Already recording, stopping previous recording first")
            self.stop_recording()
        
        try:
            # Reset state
            self.stop_event.clear()
            self.audio_buffer = []
            
            # Clear the audio queue
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            
            # Create the audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.current_mic['channels'],
                rate=self.sample_rate,
                input=True,
                input_device_index=self.current_mic['id'],
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            
            # Start the audio processing thread
            self.recording_thread = threading.Thread(
                target=self._audio_processing_thread,
                args=(callback,),
                daemon=True
            )
            self.recording_thread.start()
            
            self.is_recording = True
            self.logger.info("Started recording from microphone")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            self.stop_recording()  # Clean up any partial setup
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording from the microphone
        
        Returns:
            bool: Whether recording was successfully stopped
        """
        if not self.is_recording:
            return True  # Already stopped
        
        # Signal the processing thread to stop
        self.stop_event.set()
        
        # Stop and close the audio stream
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {str(e)}")
            finally:
                self.stream = None
        
        # Wait for the processing thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            try:
                self.recording_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error waiting for recording thread: {str(e)}")
        
        self.is_recording = False
        self.logger.info("Stopped recording from microphone")
        return True
    
    def get_recorded_audio(self) -> Optional[np.ndarray]:
        """
        Get the recorded audio data
        
        Returns:
            Optional[np.ndarray]: The recorded audio data or None if no data
        """
        if not self.audio_buffer:
            return None
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            return audio_data
        except Exception as e:
            self.logger.error(f"Error concatenating audio data: {str(e)}")
            return None 