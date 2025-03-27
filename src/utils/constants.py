#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constants for the Speech-to-Text application
"""

# Application information
APP_NAME = "Speech-to-Text Tool"
APP_VERSION = "0.1.0"
APP_AUTHOR = "STT Team"

# Audio settings
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_CHANNELS = 1  # Mono
DEFAULT_CHUNK_SIZE = 512  # Samples per buffer (reduced from 1024 for faster processing)
DEFAULT_FORMAT = "int16"  # Audio format
DEFAULT_TEMP_DIR = "temp_audio"  # Temporary audio storage
DEFAULT_WAVE_EXTENSION = ".wav"  # File extension for wave files

# Audio processing
DEFAULT_NORMALIZE = True  # Whether to normalize audio
DEFAULT_REMOVE_DC = True  # Whether to remove DC offset
DEFAULT_CHUNK_DURATION_MS = 30  # Duration of audio chunks in ms

# VAD settings
DEFAULT_VAD_THRESHOLD = 0.45  # Default VAD sensitivity (reduced from 0.5 for faster detection)
DEFAULT_VAD_WINDOW = 20  # Window size in ms (reduced from 30 for faster processing)
DEFAULT_VAD_SPEECH_PAD_MS = 150  # Padding for speech detection in ms (reduced from 300)

# Whisper model settings
DEFAULT_MODEL_NAME = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "en"  # English
DEFAULT_WHISPER_TIMEOUT = 60  # Timeout for Whisper inference in seconds
DEFAULT_USE_FP16 = True  # Whether to use FP16 precision
DEFAULT_MAX_NEW_TOKENS = 256  # Maximum number of new tokens to generate
DEFAULT_CHUNK_LENGTH_S = 30  # Length of audio chunks in seconds
DEFAULT_BATCH_SIZE = 16  # Batch size for model inference

# UI settings
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_MAX_HISTORY_ENTRIES = 100

# Colors
COLOR_RED = "#FF0000"
COLOR_GREEN = "#00FF00"
COLOR_GRAY = "#808080"
COLOR_ERROR = "#FF6347"  # Tomato red
COLOR_WARNING = "#FFA500"  # Orange
COLOR_INFO = "#4682B4"  # Steel blue 