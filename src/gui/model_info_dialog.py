#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog to display model information and status
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QGroupBox, QFormLayout,
    QProgressBar, QFileDialog, QMessageBox,
    QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.models.faster_whisper_model import FasterWhisperModel
from src.config.settings import AppSettings


class ModelInfoDialog(QDialog):
    """Dialog to display model information and status"""
    
    def __init__(self, whisper_model: FasterWhisperModel, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = whisper_model
        self.settings = settings
        
        # Set window properties
        self.setWindowTitle("Model Information")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Use try/except to handle different PyQt6 versions and flag names
        try:
            # Try newer PyQt6 way first
            from PyQt6.QtCore import Qt
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        except AttributeError:
            try:
                # Fall back to older PyQt6 versions
                self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            except AttributeError:
                # If all else fails, don't modify the flags
                self.logger.warning("Could not modify window flags - context help button may be visible")
        
        # Initialize UI
        self._init_ui()
        
        # Update info when dialog opens
        self._update_model_info()
        
        # Set up timer for periodic updates if model is running
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_model_info)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        self.logger.debug("ModelInfoDialog initialized")
    
    def _init_ui(self):
        """Initialize the user interface components"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Model information group
        info_group = QGroupBox("Model Information")
        info_layout = QFormLayout(info_group)
        
        # Model name
        self.model_name_label = QLabel("-")
        info_layout.addRow("Model Name:", self.model_name_label)
        
        # Status
        self.status_label = QLabel("-")
        info_layout.addRow("Status:", self.status_label)
        
        # Device
        self.device_label = QLabel("-")
        info_layout.addRow("Device:", self.device_label)
        
        # Precision
        self.precision_label = QLabel("-")
        info_layout.addRow("Precision:", self.precision_label)
        
        # Language
        self.language_label = QLabel("-")
        info_layout.addRow("Language:", self.language_label)
        
        # Add to main layout
        layout.addWidget(info_group)
        
        # Performance group
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout(perf_group)
        
        # Average inference time
        self.inf_time_label = QLabel("-")
        perf_layout.addRow("Avg. Inference Time:", self.inf_time_label)
        
        # Number of inferences
        self.num_inf_label = QLabel("-")
        perf_layout.addRow("Inferences Performed:", self.num_inf_label)
        
        # Memory usage
        self.memory_label = QLabel("-")
        perf_layout.addRow("Memory Usage:", self.memory_label)
        
        # Add to main layout
        layout.addWidget(perf_group)
        
        # Model directory group
        dir_group = QGroupBox("Model Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        # Current directory
        self.dir_label = QLabel(self.settings.get("model.directory", "-"))
        self.dir_label.setWordWrap(True)
        
        # Change directory button
        change_dir_button = QPushButton("Change...")
        change_dir_button.clicked.connect(self._on_change_dir)
        
        dir_layout.addWidget(self.dir_label, 1)  # Stretch factor 1
        dir_layout.addWidget(change_dir_button)
        
        # Add to main layout
        layout.addWidget(dir_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        # Load model button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self._on_load_model)
        
        # Unload model button
        self.unload_button = QPushButton("Unload Model")
        self.unload_button.clicked.connect(self._on_unload_model)
        self.unload_button.setEnabled(False)  # Disabled until model is loaded
        
        # Test model button
        self.test_button = QPushButton("Test Model")
        self.test_button.clicked.connect(self._on_test_model)
        self.test_button.setEnabled(False)  # Disabled until model is loaded
        
        actions_layout.addWidget(self.load_button)
        actions_layout.addWidget(self.unload_button)
        actions_layout.addWidget(self.test_button)
        
        # Add to main layout
        layout.addWidget(actions_group)
        
        # Close button
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        button_layout.addStretch(1)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def _update_model_info(self):
        """Update the model information display"""
        try:
            # Get model info
            info = self.model.get_model_info()
            
            # Update labels
            self.model_name_label.setText(info.get("model_name", "-"))
            
            # Status with appropriate color
            status = info.get("status", "Unknown")
            if status == "Loaded":
                self.status_label.setText('<span style="color:green">Loaded</span>')
                self.load_button.setEnabled(False)
                self.unload_button.setEnabled(True)
                self.test_button.setEnabled(True)
            else:
                error = info.get("error", "")
                if error:
                    status_text = f'<span style="color:red">Not Loaded: {error}</span>'
                else:
                    status_text = '<span style="color:orange">Not Loaded</span>'
                self.status_label.setText(status_text)
                self.load_button.setEnabled(True)
                self.unload_button.setEnabled(False)
                self.test_button.setEnabled(False)
            
            # Other info fields
            self.device_label.setText(info.get("device", "-"))
            self.precision_label.setText(info.get("precision", "-"))
            self.language_label.setText(info.get("language", "-"))
            
            # Performance metrics
            self.inf_time_label.setText(info.get("avg_inference_time", "-"))
            self.num_inf_label.setText(str(info.get("num_inferences", "-")))
            
            # Memory usage
            if "memory_allocated" in info and info["memory_allocated"] != "N/A":
                memory_text = f"{info['memory_allocated']} / {info['memory_reserved']}"
            else:
                memory_text = "N/A"
            self.memory_label.setText(memory_text)
            
            # Current model directory
            self.dir_label.setText(self.settings.get("model.directory", "-"))
            
        except Exception as e:
            self.logger.error(f"Error updating model info: {str(e)}")
    
    def _on_change_dir(self):
        """Handle change directory button click"""
        # Get the current directory
        current_dir = self.settings.get("model.directory", "")
        
        # Open directory selection dialog
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            current_dir
        )
        
        if dir_path:
            # Update settings and display
            self.settings.set("model.directory", dir_path)
            self.dir_label.setText(dir_path)
            
            # Ask if user wants to reload model
            if self.model.model is not None:
                result = QMessageBox.question(
                    self,
                    "Reload Model",
                    "Model directory has changed. Would you like to reload the model?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if result == QMessageBox.StandardButton.Yes:
                    self._on_unload_model()
                    self._on_load_model()
    
    def _on_load_model(self):
        """Handle load model button click"""
        # Disable buttons during loading
        self.load_button.setEnabled(False)
        self.load_button.setText("Loading...")
        QApplication.processEvents()  # Update UI
        
        try:
            # Set model directory from settings
            self.model.model_dir = self.settings.get("model.directory")
            
            # Load the model
            result = self.model.load_model()
            success = result.get("success", False) if isinstance(result, dict) else result
            error = result.get("error", "") if isinstance(result, dict) else ""
            
            if success:
                QMessageBox.information(
                    self,
                    "Model Loaded",
                    "Whisper model has been successfully loaded."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Model Load Failed",
                    f"Failed to load Whisper model: {error}"
                )
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading the model: {str(e)}"
            )
            
        finally:
            # Update buttons and info
            self._update_model_info()
            self.load_button.setText("Load Model")
    
    def _on_unload_model(self):
        """Handle unload model button click"""
        try:
            self.model.unload_model()
            self._update_model_info()
            
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while unloading the model: {str(e)}"
            )
    
    def _on_test_model(self):
        """Handle test model button click"""
        if self.model.model is None:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "Please load the model before testing."
            )
            return
            
        # Select audio file for testing
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File for Testing",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # Disable button during transcription
            self.test_button.setEnabled(False)
            self.test_button.setText("Transcribing...")
            QApplication.processEvents()  # Update UI
            
            # Transcribe the audio file
            result = self.model.transcribe(file_path)
            
            if result["success"]:
                QMessageBox.information(
                    self,
                    "Transcription Result",
                    f"Transcription completed in {result.get('inference_time', 0):.2f}s:\n\n{result['text']}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Transcription Failed",
                    f"Failed to transcribe audio: {result['error']}"
                )
                
        except Exception as e:
            self.logger.error(f"Error testing model: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while testing the model: {str(e)}"
            )
            
        finally:
            # Update buttons and info
            self._update_model_info()
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Model")
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        # Stop the update timer
        self.update_timer.stop()
        
        # Accept the close event
        event.accept() 