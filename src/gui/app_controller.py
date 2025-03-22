#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application controller to connect GUI with business logic
"""

import logging
import tempfile
import time
import os
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import torch

from config.settings import AppSettings
from audio.microphone import MicrophoneManager
from audio.audio_processor import AudioProcessor
from audio.vad_processor import VADProcessor
from models.vad_model import SileroVAD
from models.faster_whisper_model import FasterWhisperModel
from text_insertion.text_inserter import TextInserter


class AppController:
    """
    Controller for connecting the GUI with the application logic
    Provides high-level methods for the GUI to call
    """
    
    def __init__(self, settings: AppSettings):
        """
        Initialize the application controller
        
        Args:
            settings: Application settings instance
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Initialize audio components
        sample_rate = settings.get("audio.sample_rate", 16000)
        channels = settings.get("audio.channels", 1)
        
        self.mic_manager = MicrophoneManager()
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            channels=channels
        )
        
        # Initialize VAD component
        vad_sensitivity = settings.get("vad.sensitivity", 0.5)
        vad_window_size = settings.get("vad.window_size_ms", 30)
        
        self.vad_processor = VADProcessor(
            sample_rate=sample_rate,
            vad_threshold=vad_sensitivity,
            window_size_ms=vad_window_size,
            audio_processor=self.audio_processor
        )
        
        # Set VAD callbacks
        self.vad_processor.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_speech_detected=self._on_speech_detected
        )
        
        # Initialize Whisper model
        model_dir = settings.get("model.directory")
        model_name = settings.get("model.name", "large-v3")
        compute_type = settings.get("model.compute_type", "auto")
        
        # If compute_type is auto, determine based on device
        if compute_type == "auto":
            if torch.cuda.is_available():
                compute_type = "float16"
            else:
                compute_type = "int8"
                
        self.whisper_model = FasterWhisperModel(
            model_dir=model_dir,
            model_name=model_name,
            language="en",  # Default to English per PRD
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=compute_type
        )
        
        # Initialize text inserter
        self.text_inserter = TextInserter()
        
        # State variables
        self.is_transcribing = False
        self.transcription_history: List[Dict[str, Any]] = []
        self.speech_detected = False
        self.temp_dir = tempfile.mkdtemp(prefix="stt_audio_")
        
        # Transcription pool and queue
        self.transcription_lock = threading.Lock()
        self.is_processing_transcription = False
        self.transcription_thread = None
        
        # Callbacks for UI updates
        self.on_speech_detected_callback = None
        self.on_transcription_callback = None
        self.on_error_callback = None
        self.on_model_status_callback = None
        
        # Model loading flag
        self.is_model_loaded = False
        
        self.logger.info("AppController initialized")
    
    def __del__(self):
        """Clean up resources on destruction"""
        try:
            # Stop transcription if active
            if self.is_transcribing:
                self.stop_transcription()
                
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
                os.rmdir(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {str(e)}")
    
    def get_microphones(self) -> List[Dict[str, Any]]:
        """
        Get list of available microphones
        
        Returns:
            List[Dict[str, Any]]: List of microphone information
        """
        return self.mic_manager.get_available_microphones()
    
    def select_microphone(self, mic_id: int) -> bool:
        """
        Select microphone for recording
        
        Args:
            mic_id: ID of the microphone to select
            
        Returns:
            bool: Whether the microphone was successfully selected
        """
        result = self.mic_manager.select_microphone(mic_id)
        if result:
            self.settings.set("audio.microphone_id", mic_id)
        return result
    
    def set_vad_sensitivity(self, sensitivity: float) -> None:
        """
        Set the VAD sensitivity
        
        Args:
            sensitivity: Sensitivity value (0-1)
        """
        self.vad_processor.set_sensitivity(sensitivity)
        self.logger.info(f"VAD sensitivity set to {sensitivity}")
    
    def ensure_model_loaded(self) -> bool:
        """
        Ensure the Whisper model is loaded
        
        Returns:
            bool: Whether the model is loaded
        """
        if not self.is_model_loaded:
            self.logger.info("Loading Whisper model")
            
            # Update model directory from settings
            self.whisper_model.model_dir = self.settings.get("model.directory")
            
            # Attempt to load the model
            if self.whisper_model.load_model():
                self.is_model_loaded = True
                self.logger.info("Whisper model loaded successfully")
                
                # Notify UI of model status change
                if self.on_model_status_callback:
                    self.on_model_status_callback(True)
                    
                return True
            else:
                self.logger.error(f"Failed to load Whisper model: {self.whisper_model.error_message}")
                
                # Notify UI of model status change
                if self.on_model_status_callback:
                    self.on_model_status_callback(False, self.whisper_model.error_message)
                    
                if self.on_error_callback:
                    self.on_error_callback(f"Failed to load Whisper model: {self.whisper_model.error_message}")
                    
                return False
        
        return True
    
    def _process_audio(self, audio_data: np.ndarray, time_info: Dict[str, Any]) -> None:
        """
        Process audio data from the microphone
        This is called by the audio callback
        
        Args:
            audio_data: Audio data to process
            time_info: Timing information from PyAudio
        """
        try:
            # Pass audio data to VAD processor
            self.vad_processor.process_audio(audio_data, time_info)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            if self.on_error_callback:
                self.on_error_callback(f"Audio processing error: {str(e)}")
    
    def _on_speech_start(self, timestamp: float, time_info: Dict[str, Any]) -> None:
        """
        Handle speech start event from VAD
        
        Args:
            timestamp: Time when speech started
            time_info: Timing information
        """
        self.logger.debug(f"Speech started at {timestamp}")
    
    def _on_speech_end(self, segment: Dict[str, Any]) -> None:
        """
        Handle speech end event from VAD
        
        Args:
            segment: Speech segment information including audio data
        """
        self.logger.debug(f"Speech ended, duration: {segment['duration']:.2f}s")
        
        try:
            # Save the speech segment for processing
            timestamp = segment['timestamp'].replace(':', '-').replace(' ', '_')
            segment_file = os.path.join(self.temp_dir, f"speech_{timestamp}.wav")
            
            # Save to WAV for further processing
            self.audio_processor.save_audio_to_wave(segment['audio'], segment_file)
            self.logger.debug(f"Speech segment saved to {segment_file}")
            
            # Process with Whisper model
            self._process_speech_segment(segment_file, segment)
                
        except Exception as e:
            self.logger.error(f"Error processing speech segment: {str(e)}")
            if self.on_error_callback:
                self.on_error_callback(f"Error processing speech: {str(e)}")
    
    def _process_speech_segment(self, audio_file: str, segment_info: Dict[str, Any]) -> None:
        """
        Process a speech segment with the Whisper model
        
        Args:
            audio_file: Path to the audio file
            segment_info: Information about the speech segment
        """
        # Start processing in a thread to avoid blocking the audio pipeline
        if not self.transcription_thread or not self.transcription_thread.is_alive():
            self.transcription_thread = threading.Thread(
                target=self._transcription_worker,
                args=(audio_file, segment_info),
                daemon=True
            )
            self.transcription_thread.start()
    
    def _transcription_worker(self, audio_file: str, segment_info: Dict[str, Any]) -> None:
        """
        Worker thread for processing transcription
        
        Args:
            audio_file: Path to the audio file
            segment_info: Information about the speech segment
        """
        with self.transcription_lock:
            if self.is_processing_transcription:
                self.logger.debug("Already processing a transcription, skipping")
                return
                
            self.is_processing_transcription = True
        
        try:
            # Make sure the model is loaded
            if not self.ensure_model_loaded():
                # Store placeholder text if model loading failed
                transcription_text = f"(Model not loaded - Speech segment {segment_info['duration']:.1f}s)"
                self.add_to_history(transcription_text)
                
                # Notify UI
                if self.on_transcription_callback:
                    self.on_transcription_callback(transcription_text)
                    
                return
            
            # Transcribe the audio
            self.logger.info(f"Transcribing speech segment: {audio_file}")
            result = self.whisper_model.transcribe(audio_file)
            
            if result["success"]:
                transcription_text = result["text"]
                self.logger.info(f"Transcription result: {transcription_text}")
                
                # Add to history
                self.add_to_history(transcription_text)
                
                # Insert text into focused application
                insert_success = False
                try:
                    # Check if element is editable
                    if self.text_inserter.is_text_editable():
                        # Insert the text
                        insert_success = self.text_inserter.insert_text(transcription_text)
                        if insert_success:
                            self.logger.info("Text successfully inserted into focused element")
                        else:
                            error_msg = "Failed to insert text into focused element"
                            self.logger.warning(error_msg)
                            if self.on_error_callback:
                                self.on_error_callback(error_msg)
                    else:
                        error_msg = "Focused element is not editable"
                        self.logger.warning(error_msg)
                        if self.on_error_callback:
                            self.on_error_callback(error_msg)
                except Exception as e:
                    self.logger.error(f"Error inserting text: {str(e)}")
                    if self.on_error_callback:
                        self.on_error_callback(f"Text insertion error: {str(e)}")
                
                # Notify UI
                if self.on_transcription_callback:
                    self.on_transcription_callback(transcription_text, insert_success)
            else:
                error_message = result["error"]
                self.logger.error(f"Transcription failed: {error_message}")
                
                # Store error in history
                transcription_text = f"(Transcription failed: {error_message})"
                self.add_to_history(transcription_text)
                
                # Notify UI
                if self.on_transcription_callback:
                    self.on_transcription_callback(transcription_text)
                    
                if self.on_error_callback:
                    self.on_error_callback(f"Transcription failed: {error_message}")
                
        except Exception as e:
            self.logger.error(f"Error in transcription worker: {str(e)}")
            if self.on_error_callback:
                self.on_error_callback(f"Transcription error: {str(e)}")
                
        finally:
            with self.transcription_lock:
                self.is_processing_transcription = False
    
    def _on_speech_detected(self, is_speech: bool) -> None:
        """
        Handle speech detection state change
        
        Args:
            is_speech: Whether speech is currently detected
        """
        # Only notify if state has changed
        if is_speech != self.speech_detected:
            self.speech_detected = is_speech
            
            # Notify UI
            if self.on_speech_detected_callback:
                self.on_speech_detected_callback(is_speech)
    
    def start_transcription(self) -> bool:
        """
        Start the transcription process
        
        Returns:
            bool: Whether transcription was successfully started
        """
        self.logger.info("Starting transcription process")
        
        # Start VAD processing
        if not self.vad_processor.start_processing():
            self.logger.error("Failed to start VAD processing")
            if self.on_error_callback:
                self.on_error_callback("Failed to start VAD processing")
            return False
        
        # Start microphone recording with our audio processing callback
        result = self.mic_manager.start_recording(callback=self._process_audio)
        if result:
            self.is_transcribing = True
            self.speech_detected = False
            self.logger.info("Transcription started successfully")
            
            # Try to ensure model is loaded (but don't block UI)
            threading.Thread(
                target=self.ensure_model_loaded,
                daemon=True
            ).start()
            
        else:
            # Stop VAD processing if microphone failed
            self.vad_processor.stop_processing()
            
            self.logger.error("Failed to start transcription")
            if self.on_error_callback:
                self.on_error_callback("Failed to start recording from microphone")
        
        return result
    
    def stop_transcription(self) -> bool:
        """
        Stop the transcription process
        
        Returns:
            bool: Whether transcription was successfully stopped
        """
        self.logger.info("Stopping transcription process")
        
        # Stop microphone recording
        mic_result = self.mic_manager.stop_recording()
        
        # Stop VAD processing
        vad_result = self.vad_processor.stop_processing()
        
        # Reset state
        self.is_transcribing = False
        
        # Reset speech detection state and notify UI if necessary
        if self.speech_detected and self.on_speech_detected_callback:
            self.speech_detected = False
            self.on_speech_detected_callback(False)
        
        return mic_result and vad_result
    
    def add_to_history(self, text: str) -> None:
        """
        Add transcribed text to history
        
        Args:
            text: Transcribed text to add
        """
        # Create history entry with timestamp
        entry = {
            "text": text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.mic_manager.current_mic["name"] if self.mic_manager.current_mic else "Unknown",
        }
        self.transcription_history.append(entry)
        
        # Trim history if needed
        max_entries = self.settings.get("ui.max_history_entries", 100)
        if len(self.transcription_history) > max_entries:
            self.transcription_history = self.transcription_history[-max_entries:]
        
        self.logger.debug(f"Added to history: {text[:30]}...")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the transcription history
        
        Returns:
            List[Dict[str, Any]]: List of transcription history entries
        """
        return self.transcription_history
    
    def clear_history(self) -> None:
        """Clear the transcription history"""
        self.transcription_history = []
        self.logger.info("Transcription history cleared")
    
    def get_current_timestamp(self) -> str:
        """
        Get the current timestamp as a string
        
        Returns:
            str: Current timestamp
        """
        return datetime.now().strftime("%H:%M:%S")
    
    def get_whisper_model(self) -> FasterWhisperModel:
        """
        Get the Whisper model instance
        
        Returns:
            FasterWhisperModel: The Whisper model instance
        """
        return self.whisper_model
    
    def set_speech_detected_callback(self, callback) -> None:
        """
        Set callback for speech detection events
        
        Args:
            callback: Function to call when speech is detected/ended
        """
        self.on_speech_detected_callback = callback
    
    def set_transcription_callback(self, callback) -> None:
        """
        Set callback for new transcriptions
        
        Args:
            callback: Function to call with new transcription text
        """
        self.on_transcription_callback = callback
    
    def set_error_callback(self, callback) -> None:
        """
        Set callback for error events
        
        Args:
            callback: Function to call with error messages
        """
        self.on_error_callback = callback
        
    def set_model_status_callback(self, callback) -> None:
        """
        Set callback for model status events
        
        Args:
            callback: Function to call with model status updates
        """
        self.on_model_status_callback = callback 