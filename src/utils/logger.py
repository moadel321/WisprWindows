#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging utilities for the application
"""

import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger():
    """Configure application logging"""
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    
    # Create file handler
    log_dir = _get_log_directory()
    log_file = log_dir / f"stt_app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    logging.info(f"Logging initialized. Log file: {log_file}")


def _get_log_directory() -> Path:
    """Get the log directory path"""
    # Use AppData for Windows
    app_data = os.environ.get("APPDATA")
    if app_data:
        log_dir = Path(app_data) / "SpeechToTextTool" / "logs"
    else:
        # Fallback to user home directory
        log_dir = Path.home() / ".speech_to_text_tool" / "logs"
    
    # Ensure log directory exists
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    return log_dir


class UILogHandler:
    """Handler to capture logs and display them in the UI"""
    
    def __init__(self, error_log_widget):
        """Initialize with a reference to the QTextEdit widget"""
        self.error_log_widget = error_log_widget
        self.handler = None
        self.setup()
    
    def setup(self):
        """Set up the log handler"""
        self.handler = logging.Handler()
        self.handler.setLevel(logging.WARNING)  # Only capture warnings and above
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.handler.emit = self.emit
        
        # Add handler to root logger
        logging.getLogger().addHandler(self.handler)
    
    def emit(self, record):
        """Emit a log record to the UI widget"""
        if self.error_log_widget is None:
            return
            
        msg = self.handler.format(record)
        # This will be properly implemented in Phase 6 with signals/slots
        # For now, it's a placeholder
        print(f"UI Log: {msg}") 