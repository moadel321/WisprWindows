#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper model implementation for speech recognition
"""

import os
import logging
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from threading import Lock
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    AutoFeatureExtractor,
    pipeline
)

from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_TIMEOUT
)


class WhisperModel:
    """
    Handles loading and inference for the Whisper V3 Multi-Large model
    """
    
    def __init__(self, 
                model_dir: Optional[str] = None, 
                model_name: Optional[str] = None,
                language: str = DEFAULT_LANGUAGE,
                device: Optional[str] = None,
                use_fp16: bool = True,
                timeout: int = DEFAULT_WHISPER_TIMEOUT):
        """
        Initialize the Whisper model manager
        
        Args:
            model_dir: Directory containing the model files (default: None, uses settings)
            model_name: Name of the model to load (default: None, uses whisper-large-v3)
            language: Language code for transcription (default: 'en')
            device: Device to use for inference (default: None, auto-detect)
            use_fp16: Whether to use FP16 precision (default: True)
            timeout: Timeout for model inference in seconds (default: 60)
        """
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.model_name = model_name or "openai/whisper-large-v3"
        self.language = language
        self.timeout = timeout
        
        # Detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.logger.info("CUDA not available, using CPU")
        else:
            self.device = device
        
        # Set precision
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Model components
        self.model = None
        self.processor = None
        self.feature_extractor = None
        self.pipe = None
        
        # Thread safety and state
        self.lock = Lock()
        self.is_ready = False
        self.error_message = ""
        
        # Performance metrics
        self.avg_inference_time = 0
        self.num_inferences = 0
        
        self.logger.info(f"WhisperModel initialized (model={self.model_name}, device={self.device}, use_fp16={self.use_fp16})")
    
    def load_model(self) -> bool:
        """
        Load the Whisper model from the specified directory
        
        Returns:
            bool: Whether the model was successfully loaded
        """
        with self.lock:
            if self.is_ready:
                return True
                
            start_time = time.time()
            
            try:
                self.logger.info(f"Loading Whisper model: {self.model_name}")
                
                # Set cache directory if provided
                local_files_only = False
                if self.model_dir and os.path.exists(self.model_dir):
                    os.environ["TRANSFORMERS_CACHE"] = self.model_dir
                    local_files_only = True
                    self.logger.info(f"Using model directory: {self.model_dir}")
                
                # Load model with appropriate configurations
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    local_files_only=local_files_only
                )
                
                # Move model to appropriate device
                model.to(self.device)
                
                # Load processor and feature extractor
                processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=local_files_only
                )
                
                # Create pipeline for inference
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16 if self.device == "cuda" else 1,
                    return_timestamps=True,
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                )
                
                # Store model components
                self.model = model
                self.processor = processor
                self.pipe = pipe
                
                # Mark as ready
                self.is_ready = True
                self.error_message = ""
                
                load_time = time.time() - start_time
                self.logger.info(f"Whisper model loaded successfully in {load_time:.2f} seconds")
                
                # Log model info
                model_size = sum(p.numel() for p in model.parameters())
                self.logger.info(f"Model size: {model_size/1e6:.2f}M parameters")
                
                return True
                
            except Exception as e:
                self.is_ready = False
                self.error_message = str(e)
                self.logger.error(f"Error loading Whisper model: {str(e)}")
                return False
    
    def ensure_model_loaded(self) -> bool:
        """
        Ensure the model is loaded before use
        
        Returns:
            bool: Whether the model is loaded and ready
        """
        if not self.is_ready:
            return self.load_model()
        return True
    
    def transcribe(self, 
                  audio_data: Union[np.ndarray, str], 
                  sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, Any]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data to transcribe or path to audio file
            sample_rate: Sample rate of the audio data in Hz
            
        Returns:
            Dict[str, Any]: Transcription result including text and metadata
        """
        # Start timing
        start_time = time.time()
        
        # Default result in case of error
        default_result = {
            "text": "",
            "chunks": [],
            "language": self.language,
            "success": False,
            "error": "Transcription failed"
        }
        
        # Ensure model is loaded
        if not self.ensure_model_loaded():
            default_result["error"] = f"Model not loaded: {self.error_message}"
            return default_result
        
        try:
            # Prepare inference timeout
            inference_start = time.time()
            timeout_at = inference_start + self.timeout
            
            # Process audio (different handling for file path vs numpy array)
            if isinstance(audio_data, str):
                # It's a file path
                if not os.path.exists(audio_data):
                    self.logger.error(f"Audio file not found: {audio_data}")
                    default_result["error"] = f"Audio file not found: {audio_data}"
                    return default_result
                    
                # Use the file path directly
                inputs = audio_data
                
            else:
                # It's a numpy array, ensure it's float32
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize int16 to float32
                
                # Ensure shape is correct (flatten if needed)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # Use the numpy array
                inputs = {"array": audio_data, "sampling_rate": sample_rate}
            
            # Check if time's already up
            if time.time() > timeout_at:
                self.logger.warning("Transcription timed out during audio preprocessing")
                default_result["error"] = "Transcription timed out during preprocessing"
                return default_result
            
            # Run inference with specified language
            generation_kwargs = {"language": self.language}
            if self.model_name.endswith("large-v3"):
                generation_kwargs["task"] = "transcribe"
            
            # Perform transcription
            result = self.pipe(
                inputs,
                generate_kwargs=generation_kwargs,
                max_new_tokens=256
            )
            
            # Check if timed out
            if time.time() > timeout_at:
                self.logger.warning("Transcription timed out during processing")
                default_result["error"] = "Transcription timed out during processing"
                return default_result
            
            # Get the transcribed text
            transcription = result["text"]
            
            # Process any timestamp chunks if available
            chunks = []
            if "chunks" in result:
                chunks = [{
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                    "text": chunk["text"]
                } for chunk in result["chunks"]]
            
            # Update performance metrics
            inference_time = time.time() - inference_start
            self.avg_inference_time = (self.avg_inference_time * self.num_inferences + inference_time) / (self.num_inferences + 1)
            self.num_inferences += 1
            
            # Log timing periodically
            if self.num_inferences % 10 == 0:
                self.logger.info(f"Average transcription time: {self.avg_inference_time:.2f}s")
            
            # Prepare success result
            return {
                "text": transcription,
                "chunks": chunks,
                "language": self.language,
                "success": True,
                "error": None,
                "inference_time": inference_time
            }
            
        except Exception as e:
            # Log error and return default result
            self.logger.error(f"Error transcribing audio: {str(e)}")
            default_result["error"] = str(e)
            return default_result
    
    def set_language(self, language: str) -> None:
        """
        Set the language for transcription
        
        Args:
            language: Language code (e.g., 'en', 'fr', etc.)
        """
        self.language = language
        self.logger.info(f"Transcription language set to: {language}")
    
    def set_timeout(self, timeout: int) -> None:
        """
        Set timeout for transcription
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = max(1, timeout)  # Ensure minimum 1 second
        self.logger.info(f"Transcription timeout set to: {self.timeout}s")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_ready:
            return {
                "model_name": self.model_name,
                "status": "Not loaded",
                "error": self.error_message
            }
        
        # Get device info
        if self.device == "cuda":
            device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
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
            "precision": "FP16" if self.use_fp16 else "FP32",
            "language": self.language,
            "avg_inference_time": f"{self.avg_inference_time:.2f}s",
            "num_inferences": self.num_inferences,
            "memory_allocated": memory_allocated,
            "memory_reserved": memory_reserved
        }
    
    def unload_model(self) -> None:
        """Unload the model from memory"""
        with self.lock:
            if not self.is_ready:
                return
                
            # Delete model components
            del self.pipe
            del self.processor
            del self.model
            
            # Clear memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Reset state
            self.pipe = None
            self.processor = None
            self.model = None
            self.is_ready = False
            
            self.logger.info("Whisper model unloaded") 