#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Faster Whisper model implementation for speech recognition
Using CUDA 12.1 for GPU acceleration (no fallbacks)
"""

import os
import logging
import time
import torch
import numpy as np
import ctypes
import sys
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from threading import Lock

# Function to load CUDA 12.1 libraries
def load_cuda_libraries():
    """Load CUDA 12.1 libraries required for Faster Whisper"""
    if sys.platform != "win32":
        return True  # Only needed on Windows
        
    try:
        # CUDA 12.1 paths (specifically)
        cuda_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
            os.path.join(os.environ.get("CUDA_PATH", ""), "bin"),
            *os.environ.get("PATH", "").split(";")
        ]
        
        # CUDA 12.1 libraries
        cuda_libraries = [
            "cublas64_12", "cublasLt64_12", "cudart64_12", "cudnn64_8"
        ]
        
        # Try to load each library
        loaded_any = False
        for lib in cuda_libraries:
            for path in cuda_paths:
                if not os.path.exists(path):
                    continue
                    
                lib_path = os.path.join(path, f"{lib}.dll")
                if os.path.exists(lib_path):
                    try:
                        ctypes.WinDLL(lib_path)
                        logging.getLogger(__name__).info(f"Loaded {lib}.dll from {path}")
                        loaded_any = True
                        break
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Could not load {lib}.dll: {e}")
            
        return loaded_any
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error loading CUDA libraries: {e}")
        return False

# Check if CUDA should be used
USE_CUDA = os.environ.get("STT_USE_CUDA", "0").lower() in ("1", "true", "yes", "on")

if USE_CUDA:
    logging.getLogger(__name__).info("CUDA mode enabled via STT_USE_CUDA environment variable")
    # Set CUDA 12.1 compatibility
    os.environ["CT2_CUDA_COMPATIBILITY"] = "12.1"
    # Load CUDA libraries
    if load_cuda_libraries():
        logging.getLogger(__name__).info("Successfully loaded CUDA 12.1 libraries")
    else:
        logging.getLogger(__name__).warning("Failed to load CUDA 12.1 libraries")
else:
    # Force CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logging.getLogger(__name__).info("Using CPU mode (set STT_USE_CUDA=1 to enable CUDA)")

# Import after setting environment variables
from faster_whisper import WhisperModel

from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_TIMEOUT
)


class FasterWhisperModel:
    """
    Handles speech recognition using the Faster Whisper model with CUDA 12.1
    """
    
    def __init__(
        self,
        model_dir: str = None,
        model_name: str = "large-v3",
        language: str = DEFAULT_LANGUAGE,
        device: str = None,
        compute_type: str = None,
        cpu_threads: int = 4,
        num_workers: int = 1,
        download_root: str = None,
        local_files_only: bool = False
    ):
        """
        Initialize the Faster Whisper model
        
        Args:
            model_dir: Directory containing the model files
            model_name: Name of the model (e.g., "large-v3")
            language: Language code for transcription (e.g., "en")
            device: Device to use for inference (e.g., "cuda", "cpu")
            compute_type: Compute type for inference (e.g., "float16", "int8")
            cpu_threads: Number of CPU threads to use
            num_workers: Number of workers for parallel processing
            download_root: Directory to download models to
            local_files_only: Only use local files, don't download
        """
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.model_dir = model_dir
        self.model_name = model_name
        self.language = language
        self.download_root = download_root
        self.local_files_only = local_files_only
        
        # Device selection (CUDA or CPU)
        if USE_CUDA and torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"Using CUDA 12.1 for transcription (CUDA: {torch.version.cuda})")
            # Default to float16 for CUDA
            self.compute_type = compute_type or "float16"
        else:
            self.device = "cpu"
            self.logger.info("Using CPU for transcription")
            # Default to int8 for CPU
            self.compute_type = compute_type or "int8"
        
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        
        # Model instance
        self.model = None
        self.model_lock = Lock()
        
        self.logger.info(f"FasterWhisperModel initialized (model={model_name}, device={self.device}, compute_type={self.compute_type})")
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the Faster Whisper model
        
        Returns:
            Dict[str, Any]: Result with success status and error message if any
        """
        with self.model_lock:
            if self.model is not None:
                return {"success": True}
            
            try:
                self.logger.info(f"Loading Faster Whisper model {self.model_name} on {self.device}...")
                start_time = time.time()
                
                # Find the model path
                model_path = None
                
                # Check if model_dir points to a specific model with model files
                if self.model_dir and (os.path.exists(os.path.join(self.model_dir, "model.bin")) or 
                                       os.path.exists(os.path.join(self.model_dir, "ct2_model.bin"))):
                    model_path = self.model_dir
                    self.logger.info(f"Using model at {model_path}")
                
                # Check if model_dir/model_name exists with model files
                elif self.model_dir and self.model_name:
                    potential_path = os.path.join(self.model_dir, self.model_name)
                    if os.path.exists(os.path.join(potential_path, "model.bin")) or \
                       os.path.exists(os.path.join(potential_path, "ct2_model.bin")):
                        model_path = potential_path
                        self.logger.info(f"Using model at {model_path}")
                    
                # Fall back to model_name as a HuggingFace model ID
                if not model_path:
                    model_path = self.model_name
                    self.logger.info(f"Using model name as identifier: {model_path}")
                
                # Load the model
                self.model = WhisperModel(
                    model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                    download_root=self.download_root,
                    local_files_only=self.local_files_only
                )
                
                load_time = time.time() - start_time
                self.logger.info(f"Faster Whisper model loaded in {load_time:.2f} seconds")
                
                return {"success": True}
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error loading Faster Whisper model: {error_msg}")
                return {"success": False, "error": error_msg}
    
    def transcribe(
        self,
        audio_file: str,
        initial_prompt: str = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Dict[str, Any] = None,
        language: str = None,
        task: str = "transcribe",
        beam_size: int = 5,
        patience: float = 1.0,
        temperature: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        best_of: int = None,
        compression_ratio_threshold: float = 2.4,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        max_initial_timestamp: float = 1.0,
        timeout: float = DEFAULT_WHISPER_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Transcribe audio using the Faster Whisper model
        
        Args:
            audio_file: Path to the audio file
            initial_prompt: Optional prompt to guide the transcription
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use VAD filtering to remove silence
            vad_parameters: Parameters for the VAD filter
            language: Language code (e.g., "en")
            task: Task type ("transcribe" or "translate")
            beam_size: Beam size for beam search
            patience: Beam search patience factor
            temperature: Temperature for sampling
            best_of: Number of samples to generate and select the best from
            compression_ratio_threshold: Compression ratio threshold
            no_speech_threshold: No speech threshold
            condition_on_previous_text: Whether to condition on previous text
            max_initial_timestamp: Maximum initial timestamp
            timeout: Timeout in seconds (not used in Faster Whisper)
            
        Returns:
            Dict[str, Any]: Result with transcription and metadata
        """
        # Extract trace ID from filename if present
        trace_id = "whisper"
        if "_ptt_" in audio_file:
            trace_id = f"ptt_{os.path.basename(audio_file).split('_ptt_')[1].split('.')[0]}"
        elif "speech_" in audio_file:
            trace_id = f"speech_{os.path.basename(audio_file).split('speech_')[1].split('.')[0]}"
            
        self.logger.info(f"[TRACE:{trace_id}] Starting Whisper transcription")
            
        if self.model is None:
            self.logger.info(f"[TRACE:{trace_id}] Model not loaded, attempting to load")
            load_start = time.time()
            load_result = self.load_model()
            self.logger.info(f"[TRACE:{trace_id}] Model load attempt took {time.time() - load_start:.3f}s")
            
            if not load_result["success"]:
                self.logger.error(f"[TRACE:{trace_id}] Failed to load model: {load_result['error']}")
                return {"success": False, "error": load_result["error"]}
        
        try:
            audio_size_kb = os.path.getsize(audio_file) / 1024
            self.logger.info(f"[TRACE:{trace_id}] Transcribing audio file: {audio_file} ({audio_size_kb:.1f} KB)")
            start_time = time.time()
            
            # Set language or default
            lang = language or self.language
            
            # Configure VAD parameters if needed
            vad_params = vad_parameters if vad_parameters else {}
            
            # Detailed timing for model phases
            model_prep_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Preparing model for transcription")
            
            # Transcribe the audio
            self.logger.info(f"[TRACE:{trace_id}] Starting model.transcribe at {time.time() - start_time:.3f}s")
            transcribe_start = time.time()
            segments, info = self.model.transcribe(
                audio_file,
                language=lang,
                task=task,
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                patience=patience,
                temperature=temperature,
                best_of=best_of,
                compression_ratio_threshold=compression_ratio_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                max_initial_timestamp=max_initial_timestamp
            )
            model_time = time.time() - transcribe_start
            self.logger.info(f"[TRACE:{trace_id}] model.transcribe completed in {model_time:.3f}s")
            
            # Convert generator to list
            self.logger.info(f"[TRACE:{trace_id}] Processing segments")
            segments_start = time.time()
            segments_list = list(segments)
            
            # Extract text and metadata
            text_parts = []
            segment_data = []
            
            for segment in segments_list:
                text_parts.append(segment.text)
                
                segment_info = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                
                # Add word-level data if available
                if word_timestamps and hasattr(segment, "words"):
                    segment_info["words"] = [
                        {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                        for word in segment.words
                    ]
                
                segment_data.append(segment_info)
            
            # Concatenate all text parts
            full_text = " ".join(text_parts)
            segments_time = time.time() - segments_start
            self.logger.info(f"[TRACE:{trace_id}] Processed {len(segments_list)} segments in {segments_time:.3f}s")
            
            transcribe_time = time.time() - start_time
            text_length = len(full_text)
            chars_per_sec = text_length / transcribe_time if transcribe_time > 0 else 0
            self.logger.info(f"[TRACE:{trace_id}] Transcription completed in {transcribe_time:.3f}s, {text_length} chars ({chars_per_sec:.1f} chars/sec)")
            
            # Performance breakdown
            self.logger.info(f"[TRACE:{trace_id}] Performance breakdown:")
            self.logger.info(f"[TRACE:{trace_id}] - Core model time: {model_time:.3f}s ({model_time/transcribe_time*100:.1f}%)")
            self.logger.info(f"[TRACE:{trace_id}] - Segment processing: {segments_time:.3f}s ({segments_time/transcribe_time*100:.1f}%)")
            
            # Return complete results
            return {
                "success": True,
                "text": full_text,
                "segments": segment_data,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "processing_time": transcribe_time,
                "trace_id": trace_id
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error transcribing audio: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        # Check if model is loaded
        if self.model is None:
            return {
                "model_name": self.model_name,
                "status": "Not loaded",
                "device": self.device,
                "compute_type": self.compute_type,
                "language": self.language
            }
        
        # Get device info
        if self.device == "cuda":
            device_info = f"CUDA 12.1 ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CUDA (Not available)"
            memory_allocated = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            memory_reserved = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        else:
            device_info = "CPU"
            memory_allocated = "N/A"
            memory_reserved = "N/A"
        
        return {
            "model_name": self.model_name,
            "status": "Loaded",
            "device": device_info,
            "precision": self.compute_type,
            "language": self.language,
            "memory_allocated": memory_allocated,
            "memory_reserved": memory_reserved,
            "is_ready": True
        }
    
    def clean_up(self):
        """Clean up resources"""
        with self.model_lock:
            self.model = None
            torch.cuda.empty_cache()
            self.logger.info("Model resources released")
    
    @staticmethod
    def find_available_models(directory: str) -> Dict[str, str]:
        """
        Find available Whisper models in the given directory
        
        Args:
            directory: Directory to search for models
            
        Returns:
            Dict[str, str]: Dictionary of model name -> model path
        """
        models = {}
        
        if not directory or not os.path.exists(directory):
            return models
            
        # Check if the directory itself contains a model file
        has_model_file = (
            os.path.exists(os.path.join(directory, "model.bin")) or
            os.path.exists(os.path.join(directory, "ct2_model.bin"))
        )
        
        if has_model_file:
            # The directory itself is a model
            model_name = os.path.basename(directory)
            models[model_name] = directory
            return models
            
        # Check subdirectories for model files
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isdir(item_path):
                has_model_file = (
                    os.path.exists(os.path.join(item_path, "model.bin")) or
                    os.path.exists(os.path.join(item_path, "ct2_model.bin"))
                )
                
                if has_model_file:
                    models[item] = item_path
                    
        return models 