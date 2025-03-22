#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Faster Whisper model implementation for speech recognition
"""

import os
import logging
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel

from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_TIMEOUT
)


class FasterWhisperModel:
    """
    Handles speech recognition using the Faster Whisper model
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
        
        # Inference parameters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine compute type based on device if not specified
        if compute_type is None:
            if self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
        
        self.compute_type = compute_type
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
                
                # If model_dir is provided, use it as the model path
                # Otherwise, use the model_name (which can be a Hugging Face model ID)
                model_path = self.model_dir if self.model_dir else self.model_name
                
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
        temperature_increment_on_fallback: float = 0.2,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        max_initial_timestamp: float = 1.0,
        timeout: float = DEFAULT_WHISPER_TIMEOUT,
        suppress_tokens: List[int] = [-1],
        batch_size: int = 16
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
            temperature_increment_on_fallback: Temperature increment when falling back
            compression_ratio_threshold: Compression ratio threshold
            logprob_threshold: Log probability threshold
            no_speech_threshold: No speech threshold
            condition_on_previous_text: Whether to condition on previous text
            max_initial_timestamp: Maximum initial timestamp
            timeout: Timeout in seconds
            suppress_tokens: List of token IDs to suppress
            batch_size: Batch size for processing
            
        Returns:
            Dict[str, Any]: Result with transcription and metadata
        """
        if self.model is None:
            load_result = self.load_model()
            if not load_result["success"]:
                return {"success": False, "error": load_result["error"]}
        
        try:
            self.logger.info(f"Transcribing audio file: {audio_file}")
            start_time = time.time()
            
            # Set language or default
            lang = language or self.language
            
            # Configure VAD parameters if needed
            vad_params = vad_parameters if vad_parameters else {}
            
            # Transcribe the audio
            segments, info = self.model.transcribe(
                audio_file,
                language=lang,
                task=task,
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                patience=patience,
                temperature=temperature,
                best_of=best_of,
                temperature_increment_on_fallback=temperature_increment_on_fallback,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                max_initial_timestamp=max_initial_timestamp,
                batch_size=batch_size
            )
            
            # Convert generator to list to fully process the transcription
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
            
            transcribe_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")
            
            # Return complete results
            return {
                "success": True,
                "text": full_text,
                "segments": segment_data,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "processing_time": transcribe_time
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error transcribing audio: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Faster Whisper models
        
        Returns:
            List[str]: List of available model names
        """
        return [
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3",
            "distil-large-v3"
        ]
    
    def clean_up(self):
        """Release resources"""
        with self.model_lock:
            self.model = None
            torch.cuda.empty_cache()
            self.logger.info("Model resources released") 