#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings dialog for the application
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QTabWidget, QWidget, QLineEdit,
    QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

from config.settings import AppSettings
from src.utils.constants import (
    DEFAULT_VAD_THRESHOLD,
    DEFAULT_VAD_WINDOW,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MAX_HISTORY_ENTRIES,
    DEFAULT_WHISPER_TIMEOUT
)


class SettingsDialog(QDialog):
    """Settings dialog for the application"""
    
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Set window properties
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Initialize UI
        self._init_ui()
        
        # Load current settings
        self._load_settings()
        
        self.logger.debug("Settings dialog initialized")
    
    def _init_ui(self):
        """Initialize the user interface components"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # General settings tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Model directory group
        model_group = QGroupBox("Model Directory")
        model_layout = QHBoxLayout(model_group)
        
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setReadOnly(True)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._on_browse_model_dir)
        
        model_layout.addWidget(self.model_dir_edit)
        model_layout.addWidget(browse_button)
        
        general_layout.addWidget(model_group)
        
        # Whisper Model settings group
        whisper_group = QGroupBox("Whisper Model Settings")
        whisper_layout = QFormLayout(whisper_group)
        
        # Model name/identifier - change to dropdown
        self.model_name_combo = QComboBox()
        self.model_name_combo.addItems([
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en", 
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3",
            "distil-large-v3"
        ])
        whisper_layout.addRow("Model Name:", self.model_name_combo)
        
        # Compute type options
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems([
            "auto",  # Default, will be determined based on device
            "float16",  # Fastest on GPU with appropriate hardware
            "float32",  # Standard precision
            "int8",  # Quantized, reduced memory usage
            "int8_float16"  # Mixed precision for GPU
        ])
        whisper_layout.addRow("Compute Type:", self.compute_type_combo)
        
        # Timeout setting
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 300)
        self.timeout_spin.setSingleStep(10)
        self.timeout_spin.setSuffix(" seconds")
        whisper_layout.addRow("Inference Timeout:", self.timeout_spin)
        
        # Add to general layout
        general_layout.addWidget(whisper_group)
        
        # History settings group
        history_group = QGroupBox("Transcription History")
        history_layout = QFormLayout(history_group)
        
        self.save_history_check = QCheckBox("Save transcription history")
        self.max_history_entries = QSpinBox()
        self.max_history_entries.setRange(10, 1000)
        self.max_history_entries.setSingleStep(10)
        
        history_layout.addRow(self.save_history_check)
        history_layout.addRow("Maximum entries:", self.max_history_entries)
        
        general_layout.addWidget(history_group)
        
        # Add spacer
        general_layout.addStretch(1)
        
        # Audio settings tab
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        
        # Audio settings group
        audio_group = QGroupBox("Audio Settings")
        audio_form = QFormLayout(audio_group)
        
        self.sample_rate_combo = QComboBox()
        sample_rates = ["8000", "16000", "22050", "44100", "48000"]
        self.sample_rate_combo.addItems(sample_rates)
        
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["1 (Mono)", "2 (Stereo)"])
        
        audio_form.addRow("Sample Rate (Hz):", self.sample_rate_combo)
        audio_form.addRow("Channels:", self.channels_combo)
        
        audio_layout.addWidget(audio_group)
        
        # VAD settings group
        vad_group = QGroupBox("Voice Activity Detection")
        vad_layout = QVBoxLayout(vad_group)
        
        # Sensitivity slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Sensitivity:")
        self.sensitivity_value_label = QLabel("0.5")
        self.sensitivity_value_label.setMinimumWidth(30)
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_value_label)
        
        # Window size
        window_layout = QHBoxLayout()
        window_label = QLabel("Window Size (ms):")
        
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(10, 1000)
        self.window_size_spin.setSingleStep(10)
        
        window_layout.addWidget(window_label)
        window_layout.addWidget(self.window_size_spin)
        
        # Add to VAD group
        vad_layout.addLayout(sensitivity_layout)
        vad_layout.addLayout(window_layout)
        
        audio_layout.addWidget(vad_group)
        
        # Add spacer
        audio_layout.addStretch(1)
        
        # Add tabs to tab widget
        tabs.addTab(general_tab, "General")
        tabs.addTab(audio_tab, "Audio & VAD")
        
        # Add tabs to main layout
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Defaults button
        defaults_button = QPushButton("Restore Defaults")
        defaults_button.clicked.connect(self._on_restore_defaults)
        
        # OK/Cancel buttons
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(defaults_button)
        button_layout.addStretch(1)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def _on_sensitivity_changed(self, value):
        """Handle sensitivity slider value change"""
        sensitivity = value / 100.0
        self.sensitivity_value_label.setText(f"{sensitivity:.2f}")
    
    def _on_browse_model_dir(self):
        """Handle browse button for model directory"""
        current_dir = self.model_dir_edit.text()
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            current_dir
        )
        
        if dir_path:
            self.model_dir_edit.setText(dir_path)
    
    def _on_restore_defaults(self):
        """Handle restore defaults button"""
        # Confirm with user
        result = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            # Load default values
            self.model_dir_edit.setText(self.settings.get("model.directory"))
            self.model_name_combo.setCurrentText("large-v3")
            self.compute_type_combo.setCurrentText("auto")
            self.timeout_spin.setValue(DEFAULT_WHISPER_TIMEOUT)
            self.save_history_check.setChecked(True)
            self.max_history_entries.setValue(DEFAULT_MAX_HISTORY_ENTRIES)
            
            # Audio settings
            index = self.sample_rate_combo.findText(str(DEFAULT_SAMPLE_RATE))
            self.sample_rate_combo.setCurrentIndex(max(0, index))
            self.channels_combo.setCurrentIndex(0)  # Mono
            
            # VAD settings
            self.sensitivity_slider.setValue(int(DEFAULT_VAD_THRESHOLD * 100))
            self.window_size_spin.setValue(DEFAULT_VAD_WINDOW)
    
    def _on_ok(self):
        """Handle OK button click"""
        try:
            # Save settings
            self._save_settings()
            self.accept()
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.warning(
                self,
                "Error",
                f"An error occurred while saving settings: {str(e)}"
            )
    
    def _load_settings(self):
        """Load settings into the UI components"""
        try:
            if not self.settings.exists():
                # Load default values
                self.model_dir_edit.setText(self.settings.get("model.directory"))
                self.model_name_combo.setCurrentText("large-v3")
                self.compute_type_combo.setCurrentText("auto")
                self.timeout_spin.setValue(DEFAULT_WHISPER_TIMEOUT)
                self.save_history_check.setChecked(True)
                
                # Default VAD settings
                self.sensitivity_slider.setValue(int(DEFAULT_VAD_THRESHOLD * 100))
                self.window_size_spin.setValue(DEFAULT_VAD_WINDOW)
                
                # Default audio settings
                self.sample_rate_combo.setCurrentText(str(DEFAULT_SAMPLE_RATE))
                self.channels_combo.setCurrentIndex(0)  # Mono
                
            else:
                # General settings
                self.model_dir_edit.setText(self.settings.get("model.directory"))
                self.model_name_combo.setCurrentText(self.settings.get("model.name", "large-v3"))
                self.compute_type_combo.setCurrentText(self.settings.get("model.compute_type", "auto"))
                self.timeout_spin.setValue(self.settings.get("model.timeout", DEFAULT_WHISPER_TIMEOUT))
                self.save_history_check.setChecked(self.settings.get("ui.save_history", True))
                
                # VAD settings
                self.sensitivity_slider.setValue(int(self.settings.get("vad.sensitivity", DEFAULT_VAD_THRESHOLD) * 100))
                self.window_size_spin.setValue(self.settings.get("vad.window_size_ms", DEFAULT_VAD_WINDOW))
                
                # Audio settings
                self.sample_rate_combo.setCurrentText(str(self.settings.get("audio.sample_rate", DEFAULT_SAMPLE_RATE)))
                self.channels_combo.setCurrentIndex(self.settings.get("audio.channels", 1) - 1)
                
            self._on_sensitivity_changed(self.sensitivity_slider.value())  # Update sensitivity label
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {str(e)}")
    
    def _save_settings(self):
        """Save settings from the UI components"""
        try:
            # General settings
            self.settings.set("model.directory", self.model_dir_edit.text())
            self.settings.set("model.name", self.model_name_combo.currentText())
            self.settings.set("model.compute_type", self.compute_type_combo.currentText())
            self.settings.set("model.timeout", self.timeout_spin.value())
            self.settings.set("ui.save_history", self.save_history_check.isChecked())
            self.settings.set("ui.max_history_entries", self.max_history_entries.value())
            
            # Audio settings
            self.settings.set("audio.sample_rate", int(self.sample_rate_combo.currentText()))
            channels = 1 if self.channels_combo.currentIndex() == 0 else 2
            self.settings.set("audio.channels", channels)
            
            # VAD settings
            sensitivity = self.sensitivity_slider.value() / 100.0
            self.settings.set("vad.sensitivity", sensitivity)
            self.settings.set("vad.window_size_ms", self.window_size_spin.value())
            
            self.logger.info("Settings saved")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")
            raise 