#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog for handling permission requests
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont

from audio.windows_permission import (
    check_microphone_permission,
    request_microphone_permission,
    ensure_microphone_permission
)


class PermissionDialog(QDialog):
    """Dialog for handling permission requests"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Set window properties
        self.setWindowTitle("Microphone Permission Required")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Initialize UI
        self._init_ui()
        
        # Check initial permission
        self.has_permission = check_microphone_permission()
        self._update_status()
        
        self.logger.debug(f"PermissionDialog initialized, has_permission={self.has_permission}")
    
    def _init_ui(self):
        """Initialize the user interface components"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Icon and title
        title_layout = QHBoxLayout()
        title_icon = QLabel("🎤")
        title_icon.setFont(QFont("", 24))
        title_text = QLabel("Microphone Access Required")
        title_text.setFont(QFont("", 14, QFont.Weight.Bold))
        
        title_layout.addWidget(title_icon)
        title_layout.addWidget(title_text)
        title_layout.addStretch(1)
        
        layout.addLayout(title_layout)
        
        # Explanation
        explanation = QLabel(
            "This application needs permission to access your microphone. "
            "Without this permission, the speech-to-text functionality cannot work."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Status
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Instructions
        self.instructions = QTextEdit()
        self.instructions.setReadOnly(True)
        self.instructions.setMaximumHeight(150)
        self.instructions.setPlainText(
            "To grant microphone access:\n\n"
            "1. Click the 'Open Settings' button below\n"
            "2. In Windows Privacy Settings, ensure 'Microphone access' is turned ON\n"
            "3. Make sure this application is allowed to use the microphone\n"
            "4. Close the settings window and click 'Check Again'"
        )
        layout.addWidget(self.instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.settings_button = QPushButton("Open Settings")
        self.settings_button.clicked.connect(self._on_settings_clicked)
        
        self.check_button = QPushButton("Check Again")
        self.check_button.clicked.connect(self._on_check_clicked)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.check_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def _update_status(self):
        """Update the status label and buttons based on permission state"""
        if self.has_permission:
            self.status_label.setText("✅ Microphone access is granted.")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.settings_button.setEnabled(False)
            self.check_button.setEnabled(False)
            self.close_button.setText("Continue")
        else:
            self.status_label.setText("❌ Microphone access is not granted.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.settings_button.setEnabled(True)
            self.check_button.setEnabled(True)
            self.close_button.setText("Close")
    
    def _on_settings_clicked(self):
        """Handle settings button click"""
        self.logger.info("Opening microphone privacy settings")
        success, message = request_microphone_permission()
        
        if not success:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open settings: {message}\n\n"
                "Please open Windows Settings manually and grant microphone access."
            )
    
    def _on_check_clicked(self):
        """Handle check again button click"""
        self.logger.info("Checking microphone permission")
        self.has_permission = check_microphone_permission()
        self._update_status()
        
        if self.has_permission:
            QMessageBox.information(
                self,
                "Permission Granted",
                "Microphone access has been granted. You can now continue using the application."
            )
    
    @staticmethod
    def check_and_request_permission(parent=None) -> bool:
        """
        Static method to check and request permission if needed
        
        Args:
            parent: Parent widget
            
        Returns:
            bool: Whether permission is granted
        """
        logger = logging.getLogger(__name__)
        
        # First check if we already have permission
        if check_microphone_permission():
            logger.info("Microphone permission already granted")
            return True
        
        # If not, show the dialog
        logger.info("Showing permission dialog")
        dialog = PermissionDialog(parent)
        result = dialog.exec()
        
        # Check final permission state
        has_permission = check_microphone_permission()
        logger.info(f"Dialog closed, has_permission={has_permission}")
        
        return has_permission 