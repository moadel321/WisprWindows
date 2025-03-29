#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech-to-Text Productivity Tool for Windows
Main application entry point
"""

import sys
import os
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QFont, QIcon

from src.config.settings import AppSettings
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger
from src.utils.constants import APP_NAME, APP_VERSION
from src.audio.windows_permission import is_admin, check_microphone_permission

# --- START DIAGNOSTIC PRINT ---
import torch
print(f"MAIN.PY START: CUDA available? {torch.cuda.is_available()}")
# --- END DIAGNOSTIC PRINT ---

def exception_hook(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and show error dialog"""
    # Log the exception
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Format traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)
    
    # Show error dialog (if QApplication exists)
    if QApplication.instance():
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle(f"{APP_NAME} - Error")
        error_box.setText("An unexpected error occurred:")
        error_box.setInformativeText(str(exc_value))
        error_box.setDetailedText(tb_text)
        error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_box.exec()


def main():
    """Main application entry point"""
    # Set up exception handling
    sys.excepthook = exception_hook
    
    # Set application information
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setOrganizationName("STT")
    QCoreApplication.setApplicationVersion(APP_VERSION)
    
    # Configure Qt for high DPI displays (using a more compatible approach)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # Set up logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info(f"Application starting... (v{APP_VERSION})")
    
    # Use try/except to handle different PyQt6 versions
    try:
        # Newer PyQt6 versions
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        # High DPI attributes are enabled by default in newer versions
    except AttributeError:
        # Fallback for older PyQt6 versions
        try:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        except AttributeError:
            logger.warning("Could not set high DPI attributes - may be using newer PyQt6 version")
    
    # Log system information
    logger.info(f"Running as administrator: {is_admin()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Initialize settings
    settings = AppSettings()
    
    # Create and launch the application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")  # Cross-platform consistent look
    
    # Set default font
    default_font = QFont("Segoe UI", 9)  # Modern font that works well on Windows
    app.setFont(default_font)
    
    # Try to load application icon if available
    try:
        app_icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        if os.path.exists(app_icon_path):
            app_icon = QIcon(app_icon_path)
            app.setWindowIcon(app_icon)
    except Exception as e:
        logger.warning(f"Could not load application icon: {str(e)}")
    
    # Create and show main window
    window = MainWindow(settings)
    window.show()
    
    # Execute application and return exit code
    exit_code = app.exec()
    
    # Clean up on exit
    logger.info("Application exiting...")
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 