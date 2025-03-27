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
        
        # Processing buffers - reduce buffer size for faster response
        self.audio_buffer = deque(maxlen=int(sample_rate * 0.75))  # 0.75 seconds buffer (reduced from 5s)
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
    
    @property
    def is_processing(self) -> bool:
        """Check if the VAD processor is currently running"""
        return self.processing_thread is not None and self.processing_thread.is_alive()
    
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
        try:
            if not self.processing_thread:
                return True  # Already stopped
            
            # Signal thread to stop
            self.stop_event.set()
            
            # Wait for thread to finish with timeout
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
                
                # If thread is still alive after timeout, log a warning
                if self.processing_thread.is_alive():
                    self.logger.warning("VAD processing thread did not terminate within timeout")
                    # We'll still consider this successful since we set the stop event
            
            # End any active speech segment
            if self.is_speech_active:
                try:
                    self._handle_speech_end()
                except Exception as e:
                    self.logger.error(f"Error ending active speech segment: {str(e)}")
            
            # Clear the processing queue
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.task_done()
                except queue.Empty:
                    break
            
            # Reset thread reference
            self.processing_thread = None
            
            self.logger.info("VAD processing stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping VAD processing: {str(e)}")
            # Reset thread reference even if there was an error
            self.processing_thread = None
            return False
    
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
            # Start tracing for performance logging
            frame_id = int(time.time() * 1000)
            self.logger.debug(f"[FRAME:{frame_id}] Processing audio frame")
            buffer_start = time.time()
            
            # Use a smaller buffer for more immediate activation
            # Only keep the most recent data needed for analysis (reduces lag)
            max_buffer_size = int(self.sample_rate * 0.5)  # 0.5 seconds of audio
            
            # Add new data to buffer
            self.audio_buffer.extend(audio_data)
            
            # Trim buffer if it's gotten too large
            if len(self.audio_buffer) > max_buffer_size:
                excess = len(self.audio_buffer) - max_buffer_size
                for _ in range(excess):
                    self.audio_buffer.popleft()
                    
            buffer_time = time.time() - buffer_start
            
            # Convert buffer to numpy array for processing
            vad_start = time.time()
            buffer_array = np.array(list(self.audio_buffer))
            
            # Check for speech - use more aggressive threshold for faster activation
            # Increase sensitivity to speech by 10% to detect speech faster
            adjusted_threshold = max(0.1, self.vad_threshold * 0.9)  # Lower threshold = higher sensitivity
            is_speech = self.vad_model.is_speech(buffer_array, threshold_override=adjusted_threshold)
            vad_time = time.time() - vad_start
            
            self.logger.debug(f"[FRAME:{frame_id}] VAD check: is_speech={is_speech}, buffer_time={buffer_time:.4f}s, vad_time={vad_time:.4f}s")
            
            # Track consecutive non-speech frames for better end detection
            if not hasattr(self, 'consecutive_silence_frames'):
                self.consecutive_silence_frames = 0
                
            # Process state changes
            if is_speech:
                self.consecutive_silence_frames = 0
                if not self.is_speech_active:
                    self.logger.debug(f"[FRAME:{frame_id}] Speech starting")
                    self._handle_speech_start(time_info)
            else:
                # Increment silence counter
                self.consecutive_silence_frames += 1
                
                # End speech after a very short period of silence (reduced from 8 to 4 frames)
                # ~0.2 seconds of silence (varies based on processing speed)
                if self.is_speech_active and self.consecutive_silence_frames >= 4:
                    self.logger.debug(f"[FRAME:{frame_id}] Speech ending after {self.consecutive_silence_frames} silence frames")
                    self._handle_speech_end()
                    self.consecutive_silence_frames = 0
            
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
        
        trace_id = f"speech_{int(speech_end_time * 1000)}"
        self.logger.info(f"[TRACE:{trace_id}] Speech end detected, duration: {speech_duration:.2f}s")
        
        # Concatenate speech buffer
        if self.current_speech_buffer:
            try:
                processing_start = time.time()
                self.logger.info(f"[TRACE:{trace_id}] Starting speech buffer processing")
                
                speech_audio = np.concatenate(self.current_speech_buffer)
                
                # Create segment info
                segment = {
                    "audio": speech_audio,
                    "start_time": self.speech_start_time,
                    "end_time": speech_end_time,
                    "duration": speech_duration,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "trace_id": trace_id
                }
                
                # Add to segments list
                self.speech_segments.append(segment)
                
                processing_time = time.time() - processing_start
                self.logger.info(f"[TRACE:{trace_id}] Speech buffer processing completed in {processing_time:.3f}s")
                
                # Call callback if registered
                if self.on_speech_end_callback:
                    try:
                        callback_start = time.time()
                        self.logger.info(f"[TRACE:{trace_id}] Calling speech end callback")
                        self.on_speech_end_callback(segment)
                        callback_time = time.time() - callback_start
                        self.logger.info(f"[TRACE:{trace_id}] Speech end callback completed in {callback_time:.3f}s")
                    except Exception as e:
                        self.logger.error(f"[TRACE:{trace_id}] Error in speech end callback: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"[TRACE:{trace_id}] Error processing speech segment: {str(e)}")
        
        self.last_speech_end_time = speech_end_time
        self.current_speech_buffer = []
        
        # Call speech detection callback
        if self.on_speech_detected_callback:
            try:
                self.on_speech_detected_callback(False)
            except Exception as e:
                self.logger.error(f"[TRACE:{trace_id}] Error in speech detection callback: {str(e)}")
    
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