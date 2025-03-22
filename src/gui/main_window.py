#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main window implementation for the Speech-to-Text application
"""

import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QLabel, QTabWidget,
    QTextEdit, QStatusBar, QListWidget, QListWidgetItem,
    QSplitter, QApplication, QMessageBox, QMenu, QMenuBar, QAction, QGroupBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor

from config.settings import AppSettings
from gui.app_controller import AppController
from gui.permission_dialog import PermissionDialog
from gui.settings_dialog import SettingsDialog
from gui.model_info_dialog import ModelInfoDialog
from utils.logger import UILogHandler
from src.utils.constants import APP_VERSION


class MainWindow(QMainWindow):
    """Main application window"""
    
    # Define signals for threading safety
    speech_detected_signal = pyqtSignal(bool)
    transcription_signal = pyqtSignal(str, bool)  # Added bool for text insertion status
    error_signal = pyqtSignal(str)
    model_status_signal = pyqtSignal(bool, str)
    
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Create controller
        self.controller = AppController(settings)
        
        # Connect controller callbacks to our signals
        self.controller.set_speech_detected_callback(self._on_speech_detected)
        self.controller.set_transcription_callback(self._on_transcription)
        self.controller.set_error_callback(self._on_error)
        self.controller.set_model_status_callback(self._on_model_status)
        
        # Connect our signals to UI update methods
        self.speech_detected_signal.connect(self._update_speech_indicator)
        self.transcription_signal.connect(self._update_transcription_history)
        self.error_signal.connect(self._update_error_log)
        self.model_status_signal.connect(self._update_model_status)
        
        # Setup window properties
        self.setWindowTitle("Speech-to-Text Tool")
        self.setMinimumSize(800, 600)
        
        # Initialize UI
        self._init_ui()
        self._create_menu()
        
        # Visual feedback for recording
        self.recording_indicator_timer = QTimer(self)
        self.recording_indicator_timer.timeout.connect(self._update_recording_indicator)
        self.recording_indicator_active = False
        
        # Connect UI logger
        self.ui_logger = UILogHandler(self.error_log)
        
        # Check microphone permission
        QTimer.singleShot(500, self._check_permission)
        
        self.logger.info("Main window initialized")
    
    def _init_ui(self):
        """Initialize the user interface components"""
        # Set application stylesheet for consistent appearance
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #a0a0a0;
            }
            QListWidget {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: white;
            }
            QComboBox {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: white;
                min-width: 200px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create top bar with controls
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)
        
        # Create a grouped box for controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Microphone selection
        mic_label = QLabel("Microphone:")
        self.mic_combo = QComboBox()
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        self.mic_combo.setMinimumWidth(250)
        mic_label.setBuddy(self.mic_combo)
        controls_layout.addWidget(mic_label)
        controls_layout.addWidget(self.mic_combo)
        
        # Add some spacing
        controls_layout.addSpacing(20)
        
        # Start/stop buttons
        self.start_button = QPushButton("Start Transcription")
        self.start_button.setIcon(QIcon.fromTheme("media-record"))
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self._on_start_clicked)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.setMinimumWidth(100)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        
        # Add to top bar
        top_bar.addWidget(controls_group)
        
        # Create status group
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)
        
        # Status indicators
        self.recording_indicator = QLabel("●")
        self.recording_indicator.setStyleSheet("color: gray; font-size: 16px;")
        self.recording_indicator.setFixedWidth(20)
        status_layout.addWidget(self.recording_indicator)
        
        self.speech_indicator = QLabel("Speech: Inactive")
        self.speech_indicator.setStyleSheet("color: gray;")
        status_layout.addWidget(self.speech_indicator)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-style: italic;")
        status_layout.addWidget(self.status_label)
        
        # Model status
        status_layout.addSpacing(20)
        self.model_status_label = QLabel("Model: Not loaded")
        status_layout.addWidget(self.model_status_label)
        
        self.model_info_button = QPushButton("Model Info")
        self.model_info_button.clicked.connect(self._on_model_info_clicked)
        status_layout.addWidget(self.model_info_button)
        
        # Add status group to top bar
        top_bar.addWidget(status_group)
        
        # Add to main layout
        main_layout.addLayout(top_bar)
        
        # Create tabs for transcription and logs
        self.tabs = QTabWidget()
        
        # Transcription history tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        history_header = QLabel("Recent Transcriptions")
        history_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        history_layout.addWidget(history_header)
        
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setWordWrap(True)
        self.history_list.setMinimumHeight(200)
        
        history_layout.addWidget(self.history_list)
        
        # Create history controls
        history_controls = QHBoxLayout()
        history_controls.setContentsMargins(0, 10, 0, 0)
        
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self._on_clear_history_clicked)
        history_controls.addWidget(self.clear_history_button)
        
        # Add export button
        self.export_history_button = QPushButton("Export History")
        self.export_history_button.clicked.connect(self._on_export_history_clicked)
        history_controls.addWidget(self.export_history_button)
        
        history_controls.addStretch(1)
        
        history_layout.addLayout(history_controls)
        
        # Error log tab
        error_tab = QWidget()
        error_layout = QVBoxLayout(error_tab)
        
        error_header = QLabel("Error Log")
        error_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        error_layout.addWidget(error_header)
        
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setFont(QFont("Monospace", 9))
        error_layout.addWidget(self.error_log)
        
        # VAD info tab
        vad_tab = QWidget()
        vad_layout = QVBoxLayout(vad_tab)
        
        vad_header = QLabel("Voice Activity Detection")
        vad_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        vad_layout.addWidget(vad_header)
        
        vad_description = QLabel(
            "This tab shows real-time information about voice activity detection. "
            "Green indicates speech, red indicates silence or non-speech audio."
        )
        vad_description.setWordWrap(True)
        vad_description.setStyleSheet("color: #666666; margin-bottom: 10px;")
        vad_layout.addWidget(vad_description)
        
        self.vad_info_text = QTextEdit()
        self.vad_info_text.setReadOnly(True)
        vad_layout.addWidget(self.vad_info_text)
        
        # Add tabs
        self.tabs.addTab(history_tab, "Transcription History")
        self.tabs.addTab(error_tab, "Error Log")
        self.tabs.addTab(vad_tab, "VAD Info")
        
        # Add to main layout
        main_layout.addWidget(self.tabs)
        
        # Create status bar
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Add VAD sensitivity indicator to status bar
        self.vad_status_label = QLabel("VAD: Ready")
        status_bar.addWidget(self.vad_status_label)
        
        # Add permanent text insertion status to status bar
        self.text_insertion_status = QLabel("Text Insertion: Ready")
        self.text_insertion_status.setStyleSheet("color: gray;")
        status_bar.addPermanentWidget(self.text_insertion_status)
        
        # Add application version to status bar
        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setStyleSheet("color: #888888;")
        status_bar.addPermanentWidget(version_label)
        
        # Load microphones
        self._load_microphones()
        
        # Load initial state based on settings
        self._update_model_status(False)
        
        # Show initial info in the error log
        self.logger.info("Application initialized and ready")
    
    def _create_menu(self):
        """Create the application menu"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        
        # Export history action
        export_action = QAction("Export Transcription History...", self)
        export_action.triggered.connect(self._on_export_history_clicked)
        export_action.setShortcut("Ctrl+E")
        file_menu.addAction(export_action)
        
        # Clear history action
        clear_action = QAction("Clear Transcription History", self)
        clear_action.triggered.connect(self._on_clear_history_clicked)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        # Settings action
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._on_settings_clicked)
        settings_action.setShortcut("Ctrl+,")
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Alt+F4")
        file_menu.addAction(exit_action)
        
        # Transcription menu
        transcription_menu = QMenu("Transcription", self)
        menu_bar.addMenu(transcription_menu)
        
        # Start transcription action
        start_action = QAction("Start Transcription", self)
        start_action.triggered.connect(self._on_start_clicked)
        start_action.setShortcut("F5")
        transcription_menu.addAction(start_action)
        
        # Stop transcription action
        stop_action = QAction("Stop Transcription", self)
        stop_action.triggered.connect(self._on_stop_clicked)
        stop_action.setShortcut("F6")
        transcription_menu.addAction(stop_action)
        
        # Model menu
        model_menu = QMenu("Model", self)
        menu_bar.addMenu(model_menu)
        
        # Model info action
        model_info_action = QAction("Model Information...", self)
        model_info_action.triggered.connect(self._on_model_info_clicked)
        model_menu.addAction(model_info_action)
        
        # Load model action
        self.load_model_action = QAction("Load Model", self)
        self.load_model_action.triggered.connect(self._on_load_model_clicked)
        model_menu.addAction(self.load_model_action)
        
        # Unload model action
        self.unload_model_action = QAction("Unload Model", self)
        self.unload_model_action.triggered.connect(self._on_unload_model_clicked)
        self.unload_model_action.setEnabled(False)
        model_menu.addAction(self.unload_model_action)
        
        # View menu
        view_menu = QMenu("View", self)
        menu_bar.addMenu(view_menu)
        
        # Show transcription history tab
        show_history_action = QAction("Transcription History", self)
        show_history_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        show_history_action.setShortcut("Ctrl+1")
        view_menu.addAction(show_history_action)
        
        # Show error log tab
        show_error_log_action = QAction("Error Log", self)
        show_error_log_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        show_error_log_action.setShortcut("Ctrl+2")
        view_menu.addAction(show_error_log_action)
        
        # Show VAD info tab
        show_vad_info_action = QAction("VAD Info", self)
        show_vad_info_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        show_vad_info_action.setShortcut("Ctrl+3")
        view_menu.addAction(show_vad_info_action)
        
        # Help menu
        help_menu = QMenu("Help", self)
        menu_bar.addMenu(help_menu)
        
        # Help action
        help_action = QAction("User Guide", self)
        help_action.triggered.connect(self._on_help_clicked)
        help_action.setShortcut("F1")
        help_menu.addAction(help_action)
        
        # Check for updates action
        updates_action = QAction("Check for Updates", self)
        updates_action.triggered.connect(self._on_check_updates_clicked)
        help_menu.addAction(updates_action)
        
        help_menu.addSeparator()
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about_clicked)
        help_menu.addAction(about_action)
    
    def _check_permission(self):
        """Check microphone permission and request if needed"""
        self.logger.info("Checking microphone permission")
        
        # Use the dialog to check and request permission
        has_permission = PermissionDialog.check_and_request_permission(self)
        
        if has_permission:
            self.logger.info("Microphone permission granted")
            # Refresh microphone list
            self._load_microphones()
        else:
            self.logger.warning("Microphone permission not granted")
            QMessageBox.warning(
                self,
                "Permission Required",
                "Microphone access is required for this application to function.\n\n"
                "The application will continue to run, but transcription will not work "
                "until microphone access is granted.\n\n"
                "You can grant permission later by clicking 'Start Transcription'."
            )
            
            # Disable the start button if no permission
            self.start_button.setText("Request Permission")
    
    def _load_microphones(self):
        """Load available microphones into combo box"""
        # Clear existing items
        self.mic_combo.clear()
        
        # Get microphones from controller
        microphones = self.controller.get_microphones()
        
        # Add to combo box
        for mic in microphones:
            self.mic_combo.addItem(mic["name"], mic["id"])
        
        self.logger.debug(f"Loaded {len(microphones)} microphones")
        
        # Select default microphone if available
        for i in range(self.mic_combo.count()):
            mic_id = self.mic_combo.itemData(i)
            mic_name = self.mic_combo.itemText(i)
            if "default" in mic_name.lower() or any(mic.get("is_default", False) for mic in microphones if mic["id"] == mic_id):
                self.mic_combo.setCurrentIndex(i)
                break
    
    def _on_mic_changed(self, index):
        """Handle microphone selection change"""
        # Get selected microphone ID
        mic_id = self.mic_combo.currentData()
        mic_name = self.mic_combo.currentText()
        
        # Set in controller
        if mic_id is not None:
            self.controller.select_microphone(mic_id)
            self.logger.info(f"Microphone changed to: {mic_name} (ID: {mic_id})")
    
    def _on_start_clicked(self):
        """Handle start button click"""
        # Check permission again if the button is in "Request Permission" state
        if self.start_button.text() == "Request Permission":
            has_permission = PermissionDialog.check_and_request_permission(self)
            
            if has_permission:
                self.logger.info("Microphone permission granted")
                # Refresh microphone list
                self._load_microphones()
                self.start_button.setText("Start Transcription")
            else:
                self.logger.warning("Microphone permission still not granted")
                return
        
        # Start transcription via controller
        if self.controller.start_transcription():
            self.status_label.setText("Listening...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # Update VAD status label
            self.vad_status_label.setText("VAD: Active")
            
            # Start the recording indicator animation
            self.recording_indicator_active = True
            self.recording_indicator_timer.start(500)  # Update every 500ms
            
            self.logger.info("Transcription started")
        else:
            self.logger.error("Failed to start transcription")
            self._update_error_log("Failed to start transcription. Check microphone access.")
            
            # Check if permission might be the issue and prompt again
            has_permission = PermissionDialog.check_and_request_permission(self)
            if has_permission:
                self._load_microphones()
                self.start_button.setText("Start Transcription")
            else:
                self.start_button.setText("Request Permission")
    
    def _on_stop_clicked(self):
        """Handle stop button click"""
        # Stop transcription via controller
        if self.controller.stop_transcription():
            self.status_label.setText("Ready")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            # Update VAD status label
            self.vad_status_label.setText("VAD: Not Active")
            
            # Stop the recording indicator animation
            self.recording_indicator_active = False
            self.recording_indicator_timer.stop()
            self.recording_indicator.setStyleSheet("color: gray; font-size: 16px;")
            
            self.logger.info("Transcription stopped")
        else:
            self.logger.error("Failed to stop transcription")
            self._update_error_log("Failed to stop transcription.")
    
    def _on_clear_history_clicked(self):
        """Handle clear history button click"""
        # Clear history in controller
        self.controller.clear_history()
        
        # Clear UI list
        self.history_list.clear()
        
        self.logger.info("Transcription history cleared")
    
    def _on_model_info_clicked(self):
        """Handle model info menu item click"""
        whisper_model = self.controller.get_whisper_model()
        dialog = ModelInfoDialog(whisper_model, self.settings, self)
        dialog.exec()
    
    def _on_load_model_clicked(self):
        """Handle load model menu item click"""
        self.logger.info("Loading Whisper model")
        
        # Show "loading" status
        old_status = self.model_status_label.text()
        self.model_status_label.setText("Model: Loading...")
        self.model_status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()  # Update UI
        
        # Try to load the model
        if self.controller.ensure_model_loaded():
            # Model loaded successfully
            self.logger.info("Whisper model loaded successfully")
        else:
            # Model loading failed, UI will be updated via callback
            self.logger.error("Failed to load Whisper model")
            self.model_status_label.setText(old_status)
    
    def _on_unload_model_clicked(self):
        """Handle unload model menu item click"""
        whisper_model = self.controller.get_whisper_model()
        whisper_model.unload_model()
        
        # Update UI
        self.model_status_label.setText("Model: Not Loaded")
        self.model_status_label.setStyleSheet("color: orange;")
        self.load_model_action.setEnabled(True)
        self.unload_model_action.setEnabled(False)
        
        self.logger.info("Whisper model unloaded")
    
    def _update_recording_indicator(self):
        """Update the recording indicator animation"""
        if not self.recording_indicator_active:
            return
            
        # Toggle between red and transparent
        current_color = self.recording_indicator.styleSheet()
        if "red" in current_color:
            self.recording_indicator.setStyleSheet("color: rgba(255, 0, 0, 0.3); font-size: 16px;")
        else:
            self.recording_indicator.setStyleSheet("color: red; font-size: 16px;")
    
    def _update_speech_indicator(self, is_speech):
        """Update the speech detection indicator"""
        if is_speech:
            # Green indicator for speech
            self.recording_indicator.setStyleSheet("color: green; font-size: 16px;")
            self.status_label.setText("Speech detected...")
            self.speech_indicator.setStyleSheet("color: green;")
            self.speech_indicator.setText("Speech: Active")
            
            # Update VAD info
            self._update_vad_info("Speech detected", True)
        else:
            # Red indicator for no speech/recording
            self.recording_indicator.setStyleSheet("color: red; font-size: 16px;")
            self.status_label.setText("Listening...")
            self.speech_indicator.setStyleSheet("color: gray;")
            self.speech_indicator.setText("Speech: Inactive")
            
            # Update VAD info
            self._update_vad_info("No speech detected", False)
    
    def _update_model_status(self, is_loaded, error_message=""):
        """
        Update the model status in the UI
        
        Args:
            is_loaded: Whether the model is loaded
            error_message: Optional error message if loading failed
        """
        if is_loaded:
            self.model_status_label.setText("Model: Loaded")
            self.model_status_label.setStyleSheet("color: green;")
            self.load_model_action.setEnabled(False)
            self.unload_model_action.setEnabled(True)
        else:
            if error_message:
                self.model_status_label.setText("Model: Error")
                self.model_status_label.setStyleSheet("color: red;")
                self._update_error_log(f"Model error: {error_message}")
            else:
                self.model_status_label.setText("Model: Not Loaded")
                self.model_status_label.setStyleSheet("color: orange;")
            
            self.load_model_action.setEnabled(True)
            self.unload_model_action.setEnabled(False)
    
    def _update_vad_info(self, message: str, is_speech: bool):
        """
        Update the VAD info text
        
        Args:
            message: Message to display
            is_speech: Whether speech is detected
        """
        timestamp = self.controller.get_current_timestamp()
        
        # Format with color based on speech detection
        color = "green" if is_speech else "red"
        html_message = f"<span style='color: {color};'>[{timestamp}] {message}</span><br/>"
        
        # Add to VAD info text
        self.vad_info_text.insertHtml(html_message)
        
        # Scroll to bottom
        vscroll = self.vad_info_text.verticalScrollBar()
        vscroll.setValue(vscroll.maximum())
    
    def _update_transcription_history(self, text, insert_success=False):
        """
        Update the transcription history list
        
        Args:
            text: The transcribed text
            insert_success: Whether the text was successfully inserted
        """
        # Format the text with timestamp
        # Get the latest entry from the controller
        history = self.controller.get_history()
        if history:
            latest_entry = history[-1]
            timestamp = latest_entry.get("timestamp", "")
            source = latest_entry.get("source", "")
            
            # Create a better formatted item
            item_text = f"[{timestamp}] {text}"
        else:
            item_text = text
            
        # Add to history list
        item = QListWidgetItem(item_text)
        
        # Color coding based on insertion success
        if insert_success:
            item.setForeground(QColor("darkgreen"))
            item.setToolTip("Successfully inserted into application")
        else:
            item.setForeground(QColor("darkred"))
            item.setToolTip("Failed to insert into application")
            
        # Bold the item if it contains specific keywords indicating errors
        if "failed" in text.lower() or "error" in text.lower() or "not loaded" in text.lower():
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            
        # Add the item to the list
        self.history_list.addItem(item)
        self.history_list.scrollToBottom()
        
        # Update text insertion status
        if insert_success:
            self.text_insertion_status.setText("Text Insertion: Success")
            self.text_insertion_status.setStyleSheet("color: green;")
        else:
            self.text_insertion_status.setText("Text Insertion: Failed")
            self.text_insertion_status.setStyleSheet("color: red;")
            
        # Reset status after 3 seconds
        QTimer.singleShot(3000, self._reset_text_insertion_status)
    
    def _reset_text_insertion_status(self):
        """Reset the text insertion status indicator to ready state"""
        self.text_insertion_status.setText("Text Insertion: Ready")
        self.text_insertion_status.setStyleSheet("color: gray;")
    
    def _update_error_log(self, error_text):
        """Update the error log with new error text"""
        self.error_log.append(f"ERROR: {error_text}")
        
        # Make sure the error tab is visible
        self.tabs.setCurrentIndex(0)
    
    def _on_speech_detected(self, is_speech):
        """Thread-safe handler for speech detection updates"""
        self.speech_detected_signal.emit(is_speech)
    
    def _on_transcription(self, text, insert_success=False):
        """Thread-safe handler for transcription updates"""
        self.transcription_signal.emit(text, insert_success)
    
    def _on_error(self, error_text):
        """Thread-safe handler for error updates"""
        self.error_signal.emit(error_text)
    
    def _on_model_status(self, is_loaded, error_message=""):
        """Handle model status updates from controller (called from another thread)"""
        self.model_status_signal.emit(is_loaded, error_message)
    
    def _on_settings_clicked(self):
        """Handle settings menu item click"""
        dialog = SettingsDialog(self.settings, self)
        
        # If settings are accepted, update the controller
        if dialog.exec():
            self.logger.info("Settings updated")
            
            # Update the VAD sensitivity if changed
            sensitivity = self.settings.get("vad.sensitivity", 0.5)
            self.controller.set_vad_sensitivity(sensitivity)
    
    def _on_about_clicked(self):
        """Handle about menu item click"""
        # Create styled about message
        about_text = f"""
        <div style="text-align: center;">
            <h1 style="color: #3c78d8;">Speech-to-Text Tool</h1>
            <p style="font-size: 14px;">Version {APP_VERSION}</p>
            <hr style="width: 80%;">
            <p>A productivity tool for converting speech to text using local processing.</p>
            <p>All processing is performed locally on your device, ensuring privacy and security.</p>
            
            <h3 style="margin-top: 20px;">Features</h3>
            <ul style="text-align: left; margin-left: 50px;">
                <li>Real-time speech-to-text transcription</li>
                <li>Local processing - no data leaves your device</li>
                <li>Voice activity detection to filter out non-speech sounds</li>
                <li>Text insertion directly into any focused text field</li>
                <li>Microphone selection from available devices</li>
                <li>Transcription history with export capability</li>
            </ul>
            
            <h3 style="margin-top: 20px;">Technologies Used</h3>
            <p>Whisper V3 Multi-Large for speech recognition</p>
            <p>Silero VAD for voice activity detection</p>
            <p>PyQt6 for the graphical user interface</p>
            <p>PyWinAuto for text insertion capabilities</p>
            
            <p style="margin-top: 30px; font-style: italic; color: #666;">
                Created according to the PRD requirements
            </p>
        </div>
        """
        
        QMessageBox.about(self, "About Speech-to-Text Tool", about_text)
    
    def _on_export_history_clicked(self):
        """Handle export history button click"""
        from datetime import datetime
        import os
        
        # Get the transcription history from the controller
        history = self.controller.get_history()
        
        if not history:
            QMessageBox.information(self, "Export History", "No transcription history to export.")
            return
            
        # Create file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_history_{timestamp}.txt"
        
        # Ask the user for the file name and location
        from PyQt6.QtWidgets import QFileDialog
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transcription History", 
            os.path.join(os.path.expanduser("~"), filename),
            "Text Files (*.txt);;All Files (*)"
        )
        
        if not export_path:
            return  # User cancelled
            
        try:
            # Write history to file
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write("Speech-to-Text Transcription History\n")
                f.write("=" * 40 + "\n\n")
                
                for entry in history:
                    timestamp = entry.get("timestamp", "Unknown time")
                    source = entry.get("source", "Unknown source")
                    text = entry.get("text", "")
                    
                    f.write(f"[{timestamp}] (Source: {source})\n")
                    f.write(f"{text}\n\n")
                    
            self.logger.info(f"Transcription history exported to {export_path}")
            QMessageBox.information(
                self, "Export Successful", 
                f"Transcription history has been exported to:\n{export_path}"
            )
                
        except Exception as e:
            self.logger.error(f"Error exporting history: {str(e)}")
            QMessageBox.critical(
                self, "Export Failed", 
                f"Failed to export transcription history:\n{str(e)}"
            )
    
    def _on_help_clicked(self):
        """Handle help menu item click"""
        QMessageBox.information(
            self,
            "User Guide",
            """
            <h2>Speech-to-Text Tool User Guide</h2>
            
            <h3>Basic Usage</h3>
            <ol>
                <li>Select your microphone from the dropdown</li>
                <li>Click "Start Transcription" to begin</li>
                <li>Speak into your microphone</li>
                <li>Text will be transcribed and inserted into the focused text field</li>
                <li>Click "Stop" when finished</li>
            </ol>
            
            <h3>Tips</h3>
            <ul>
                <li>Make sure you click into the text field where you want the text inserted before speaking</li>
                <li>The application detects when you're speaking, so it won't transcribe silence</li>
                <li>You can export your transcription history to a text file from the File menu</li>
                <li>Adjust VAD sensitivity in Settings if you find it's not detecting your speech properly</li>
            </ul>
            
            <h3>Keyboard Shortcuts</h3>
            <table>
                <tr><td><b>F5</b></td><td>Start Transcription</td></tr>
                <tr><td><b>F6</b></td><td>Stop Transcription</td></tr>
                <tr><td><b>Ctrl+E</b></td><td>Export Transcription History</td></tr>
                <tr><td><b>Ctrl+,</b></td><td>Open Settings</td></tr>
                <tr><td><b>Ctrl+1-3</b></td><td>Switch between tabs</td></tr>
                <tr><td><b>F1</b></td><td>Open User Guide</td></tr>
            </table>
            """
        )
    
    def _on_check_updates_clicked(self):
        """Handle check for updates menu item click"""
        QMessageBox.information(
            self,
            "Check for Updates",
            "You are running the latest version of the Speech-to-Text Tool.\n\n"
            f"Current version: {APP_VERSION}"
        )
    
    def closeEvent(self, event):
        """Handle window close event to clean up resources"""
        # Stop any active recording
        if self.controller.is_transcribing:
            self.controller.stop_transcription()
            
        # Stop the timer
        self.recording_indicator_timer.stop()
        
        # Accept the close event
        event.accept() 