#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Activity Detection processor for audio streams
"""

import logging
import numpy as np
import time
import os
import queue
import threading
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque
from datetime import datetime

from src.models.vad_model import SileroVAD
from src.audio.audio_processor import AudioProcessor
from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VAD_THRESHOLD,
    DEFAULT_VAD_WINDOW,
    DEFAULT_CHUNK_SIZE
)


class VADProcessor:
    """
    Process audio streams with Voice Activity Detection
    """
    
    def __init__(self, 
                sample_rate: int = DEFAULT_SAMPLE_RATE,
                vad_threshold: float = DEFAULT_VAD_THRESHOLD,
                window_size_ms: int = DEFAULT_VAD_WINDOW,
                audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the VAD processor
        
        Args:
            sample_rate: Audio sample rate in Hz
            vad_threshold: VAD sensitivity threshold (0-1)
            window_size_ms: VAD analysis window size in milliseconds
            audio_processor: Optional AudioProcessor instance for preprocessing
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.window_size_ms = window_size_ms
        
        # Initialize the VAD model
        self.vad_model = SileroVAD(
            sensitivity=vad_threshold,
            sample_rate=sample_rate,
            window_size_ms=window_size_ms
        )
        
        # Initialize or use provided audio processor
        self.audio_processor = audio_processor or AudioProcessor(sample_rate=sample_rate)
        
        # VAD state variables
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_end_time = None
        self.current_speech_buffer = []
        self.speech_segments = []
        
        # Processing buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 seconds buffer
        self.window_size_samples = int(sample_rate * window_size_ms / 1000)
        
        # Processing thread and queue
        self.processing_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # Callbacks
        self.on_speech_start_callback = None
        self.on_speech_end_callback = None
        self.on_speech_detected_callback = None
        
        # Performance metrics
        self.avg_process_time = 0
        self.num_processed = 0
        
        self.logger.info(f"VADProcessor initialized (sample_rate={sample_rate}, threshold={vad_threshold})")
    
    def start_processing(self) -> bool:
        """
        Start the VAD processing thread
        
        Returns:
            bool: Whether processing was successfully started
        """
        # Ensure VAD model is loaded
        if not self.vad_model.ensure_model_loaded():
            self.logger.error("Failed to load VAD model, cannot start processing")
            return False
        
        # Clear state
        self.stop_event.clear()
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_end_time = None
        self.current_speech_buffer = []
        self.speech_segments = []
        self.audio_buffer.clear()
        
        # Clear the queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.task_done()
            except queue.Empty:
                break
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_thread_func,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("VAD processing started")
        return True
    
    def stop_processing(self) -> bool:
        """
        Stop the VAD processing thread
        
        Returns:
            bool: Whether processing was successfully stopped
        """
        if not self.processing_thread:
            return True  # Already stopped
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Reset thread
        self.processing_thread = None
        
        # End any active speech segment
        if self.is_speech_active:
            self._handle_speech_end()
        
        self.logger.info("VAD processing stopped")
        return True
    
    def process_audio(self, audio_data: np.ndarray, time_info: Optional[Dict] = None) -> bool:
        """
        Process an audio chunk with VAD
        
        Args:
            audio_data: Audio data to process
            time_info: Optional timing information
            
        Returns:
            bool: Whether the audio was successfully queued for processing
        """
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.logger.warning("VAD processing thread not running, cannot process audio")
            return False
        
        try:
            # Preprocess audio data (normalize, etc.)
            processed_audio = self.audio_processor.preprocess_audio(audio_data)
            
            # Add to processing queue
            self.processing_queue.put((processed_audio, time_info or {}))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error queuing audio for VAD processing: {str(e)}")
            return False
    
    def _processing_thread_func(self) -> None:
        """Thread function for audio processing"""
        self.logger.debug("VAD processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue with timeout
                audio_data, time_info = self.processing_queue.get(timeout=0.1)
                
                # Process the audio data
                self._process_audio_chunk(audio_data, time_info)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                # Timeout occurred, continue checking stop event
                continue
                
            except Exception as e:
                self.logger.error(f"Error in VAD processing thread: {str(e)}")
                time.sleep(0.01)  # Prevent tight loop in case of recurring errors
        
        self.logger.debug("VAD processing thread stopped")
    
    def _process_audio_chunk(self, audio_data: np.ndarray, time_info: Dict) -> None:
        """
        Process a chunk of audio data with VAD
        
        Args:
            audio_data: Audio data to process
            time_info: Timing information
        """
        start_time = time.time()
        
        try:
            # Extend the buffer with new audio data
            self.audio_buffer.extend(audio_data)
            
            # Convert buffer to numpy array for processing
            buffer_array = np.array(list(self.audio_buffer))
            
            # Check for speech
            is_speech = self.vad_model.is_speech(buffer_array)
            
            # Process state changes
            if is_speech and not self.is_speech_active:
                self._handle_speech_start(time_info)
            elif not is_speech and self.is_speech_active:
                # Only end speech after a silence threshold to avoid choppy detection
                silence_duration = time.time() - (self.speech_start_time or 0)
                if silence_duration > 0.5:  # 500ms silence threshold
                    self._handle_speech_end()
            
            # If speech is active, add to current speech buffer
            if self.is_speech_active:
                self.current_speech_buffer.append(audio_data)
            
            # Update performance metrics
            end_time = time.time()
            process_time = end_time - start_time
            self.avg_process_time = (self.avg_process_time * self.num_processed + process_time) / (self.num_processed + 1)
            self.num_processed += 1
            
            # Log performance periodically
            if self.num_processed % 100 == 0:
                self.logger.debug(f"VAD average processing time: {self.avg_process_time:.4f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing audio chunk with VAD: {str(e)}")
    
    def _handle_speech_start(self, time_info: Dict) -> None:
        """
        Handle the start of a speech segment
        
        Args:
            time_info: Timing information
        """
        self.is_speech_active = True
        self.speech_start_time = time.time()
        self.current_speech_buffer = []
        
        self.logger.debug(f"Speech start detected at {self.speech_start_time}")
        
        # Call callback if registered
        if self.on_speech_start_callback:
            try:
                self.on_speech_start_callback(self.speech_start_time, time_info)
            except Exception as e:
                self.logger.error(f"Error in speech start callback: {str(e)}")
        
        # Call speech detection callback
        if self.on_speech_detected_callback:
            try:
                self.on_speech_detected_callback(True)
            except Exception as e:
                self.logger.error(f"Error in speech detection callback: {str(e)}")
    
    def _handle_speech_end(self) -> None:
        """Handle the end of a speech segment"""
        if not self.is_speech_active:
            return
            
        self.is_speech_active = False
        speech_end_time = time.time()
        speech_duration = speech_end_time - (self.speech_start_time or 0)
        
        self.logger.debug(f"Speech end detected, duration: {speech_duration:.2f}s")
        
        # Concatenate speech buffer
        if self.current_speech_buffer:
            try:
                speech_audio = np.concatenate(self.current_speech_buffer)
                
                # Create segment info
                segment = {
                    "audio": speech_audio,
                    "start_time": self.speech_start_time,
                    "end_time": speech_end_time,
                    "duration": speech_duration,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to segments list
                self.speech_segments.append(segment)
                
                # Call callback if registered
                if self.on_speech_end_callback:
                    try:
                        self.on_speech_end_callback(segment)
                    except Exception as e:
                        self.logger.error(f"Error in speech end callback: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error processing speech segment: {str(e)}")
        
        self.last_speech_end_time = speech_end_time
        self.current_speech_buffer = []
        
        # Call speech detection callback
        if self.on_speech_detected_callback:
            try:
                self.on_speech_detected_callback(False)
            except Exception as e:
                self.logger.error(f"Error in speech detection callback: {str(e)}")
    
    def get_speech_segments(self) -> List[Dict]:
        """
        Get all detected speech segments
        
        Returns:
            List[Dict]: List of speech segments
        """
        return self.speech_segments
    
    def clear_speech_segments(self) -> None:
        """Clear all stored speech segments"""
        self.speech_segments = []
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set the VAD sensitivity
        
        Args:
            sensitivity: Sensitivity threshold (0-1)
        """
        self.vad_threshold = max(0.0, min(1.0, sensitivity))
        self.vad_model.set_sensitivity(self.vad_threshold)
        self.logger.info(f"VAD sensitivity set to {self.vad_threshold}")
    
    def set_callbacks(self, 
                    on_speech_start: Optional[Callable] = None,
                    on_speech_end: Optional[Callable] = None,
                    on_speech_detected: Optional[Callable] = None) -> None:
        """
        Set callback functions
        
        Args:
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends
            on_speech_detected: Callback for speech detection state
        """
        self.on_speech_start_callback = on_speech_start
        self.on_speech_end_callback = on_speech_end
        self.on_speech_detected_callback = on_speech_detected 