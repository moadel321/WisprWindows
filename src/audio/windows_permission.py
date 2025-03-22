#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows microphone permission handling
"""

import logging
import os
import subprocess
import sys
import ctypes
from typing import Tuple


def is_admin() -> bool:
    """
    Check if the current process has administrator privileges
    
    Returns:
        bool: True if running as admin, False otherwise
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def check_microphone_permission() -> bool:
    """
    Check if the application has permission to access the microphone on Windows
    This is a simple check that doesn't guarantee access
    
    Returns:
        bool: True if permission appears to be granted, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # On Windows 10/11, microphone permission is stored in the registry
        # We can use PowerShell to query it
        ps_command = (
            'Get-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone" '
            '-Name "Value" | Select-Object -ExpandProperty "Value"'
        )
        
        # Run PowerShell command and capture output
        process = subprocess.Popen(
            ["powershell", "-Command", ps_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"Failed to check microphone permission: {stderr.strip()}")
            # Fall back to assuming permission is granted
            return True
        
        # Check the result - "Allow" means permission is granted
        result = stdout.strip().lower()
        has_permission = result == "allow"
        
        logger.info(f"Microphone permission status: {result}")
        return has_permission
        
    except Exception as e:
        logger.error(f"Error checking microphone permission: {str(e)}")
        # Fall back to assuming permission is granted
        return True


def request_microphone_permission() -> Tuple[bool, str]:
    """
    Request microphone permission on Windows by opening the privacy settings
    
    Returns:
        Tuple[bool, str]: Success status and message
    """
    logger = logging.getLogger(__name__)
    
    try:
        # On Windows 10/11, we can open the microphone privacy settings directly
        settings_command = "start ms-settings:privacy-microphone"
        
        # Run the command to open settings
        process = subprocess.Popen(
            ["cmd", "/c", settings_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        process.communicate()
        
        if process.returncode != 0:
            logger.error("Failed to open microphone privacy settings")
            return False, "Failed to open microphone privacy settings"
        
        message = (
            "Please grant microphone access in the privacy settings that have been opened.\n"
            "1. Make sure 'Microphone access' is turned ON\n"
            "2. Ensure this application is allowed to use the microphone\n"
            "3. Close the settings window and try again"
        )
        
        logger.info("Opened microphone privacy settings")
        return True, message
        
    except Exception as e:
        logger.error(f"Error requesting microphone permission: {str(e)}")
        return False, f"Error requesting microphone permission: {str(e)}"


def ensure_microphone_permission() -> Tuple[bool, str]:
    """
    Check and request microphone permission if needed
    
    Returns:
        Tuple[bool, str]: Success status and message
    """
    logger = logging.getLogger(__name__)
    
    # First check if we already have permission
    if check_microphone_permission():
        logger.info("Microphone permission already granted")
        return True, "Microphone permission already granted"
    
    # If not, request permission
    logger.info("Requesting microphone permission")
    return request_microphone_permission() 