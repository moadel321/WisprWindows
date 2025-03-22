#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application settings management
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class AppSettings:
    """Application settings manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings: Dict[str, Any] = {}
        self.config_dir = self._get_config_dir()
        self.config_file = self.config_dir / "settings.json"
        
        # Ensure config directory exists
        self._ensure_config_dir()
        
        # Load settings or create defaults
        self._load_settings()
    
    def _get_config_dir(self) -> Path:
        """Get the configuration directory path"""
        # Use AppData for Windows
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "SpeechToTextTool"
        
        # Fallback to user home directory
        return Path.home() / ".speech_to_text_tool"
    
    def _ensure_config_dir(self):
        """Ensure the configuration directory exists"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created config directory: {self.config_dir}")
    
    def _load_settings(self):
        """Load settings from config file or create defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                self.logger.info("Settings loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load settings: {str(e)}")
                self._create_default_settings()
        else:
            self._create_default_settings()
    
    def _create_default_settings(self):
        """Create default settings"""
        self.settings = {
            "model": {
                "directory": str(Path.home() / "WhisperModels"),
                "name": "whisper-large-v3",
            },
            "audio": {
                "microphone_id": None,
                "microphone_name": "Default",
                "sample_rate": 16000,
            },
            "vad": {
                "enabled": True,
                "sensitivity": 0.5,
            },
            "ui": {
                "theme": "system",
                "save_history": True,
                "max_history_entries": 100,
            }
        }
        self.save()
        self.logger.info("Created default settings")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value using dot notation (e.g., 'model.directory')"""
        parts = key.split(".")
        current = self.settings
        
        for part in parts:
            if part not in current:
                return default
            current = current[part]
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value using dot notation (e.g., 'model.directory')"""
        parts = key.split(".")
        current = self.settings
        
        # Navigate to the deepest level
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
        
        # Save the settings file
        self.save()
    
    def save(self) -> bool:
        """Save settings to the config file"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            self.logger.info("Settings saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save settings: {str(e)}")
            return False 