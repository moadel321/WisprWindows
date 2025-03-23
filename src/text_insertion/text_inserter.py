#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text insertion into focused applications using pywinauto
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
import win32gui
import win32api
import win32con
import pywinauto
from pywinauto import Desktop
from pywinauto.application import Application


class TextInserter:
    """
    Handles detecting the focused text box and inserting text
    """
    
    def __init__(self):
        """Initialize the text inserter"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("TextInserter initialized")
        self.last_active_element = None
        
    def get_focused_element(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently focused UI element
        
        Returns:
            Optional[Dict[str, Any]]: Information about the focused element, or None if not found
        """
        try:
            # Get handle of the foreground window
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                self.logger.warning("No foreground window found")
                return None
                
            # Get the window class and text
            window_class = win32gui.GetClassName(hwnd)
            window_text = win32gui.GetWindowText(hwnd)
            
            self.logger.info(f"Focused window: {window_text} (Class: {window_class})")
            
            # Try to connect to the window with pywinauto
            try:
                app = Application(backend="uia").connect(handle=hwnd)
                window = app.window(handle=hwnd)
                
                # Get the control that has focus
                try:
                    focused_element = window.get_focus()
                    control_type = getattr(focused_element, "control_type", "unknown")
                    
                    # Check if this is likely to be an editable control
                    editable = self._is_likely_editable(focused_element, control_type)
                    
                    element_info = {
                        "app": window_text,
                        "element": focused_element,
                        "editable": editable,
                        "control_type": control_type,
                        "hwnd": hwnd
                    }
                    
                    self.last_active_element = element_info
                    return element_info
                except (pywinauto.findwindows.ElementNotFoundError, 
                        AttributeError, RuntimeError) as e:
                    self.logger.warning(f"Could not get focused element: {str(e)}")
            except (pywinauto.application.AppStartError, 
                    pywinauto.findwindows.ElementNotFoundError) as e:
                self.logger.warning(f"Could not connect to application: {str(e)}")
                
            # Fallback method: try Desktop().from_point
            try:
                # Get cursor position
                cursor_pos = win32gui.GetCursorPos()
                x, y = cursor_pos
                element = Desktop(backend="uia").from_point(x, y)
                control_type = getattr(element, "control_type", "unknown")
                
                # Check if this is likely to be an editable control
                editable = self._is_likely_editable(element, control_type)
                
                element_info = {
                    "app": window_text,
                    "element": element,
                    "editable": editable,
                    "control_type": control_type,
                    "hwnd": hwnd
                }
                
                self.last_active_element = element_info
                return element_info
            except Exception as e:
                self.logger.warning(f"Fallback method failed: {str(e)}")
                
            return None
        except Exception as e:
            self.logger.error(f"Error getting focused element: {str(e)}")
            return None
            
    def _is_likely_editable(self, element, control_type: str) -> bool:
        """
        Determine if an element is likely to be editable
        
        Args:
            element: The UI element
            control_type: The control type of the element
            
        Returns:
            bool: Whether the element is likely to be editable
        """
        # For testing/debugging purposes, make Cursor editor always editable
        hwnd = win32gui.GetForegroundWindow()
        window_text = win32gui.GetWindowText(hwnd)
        if "Cursor" in window_text:
            self.logger.info("Detected Cursor editor - treating as editable")
            return True
            
        # List of control types that are typically editable
        editable_types = ["Edit", "Document", "Text", "DataItem", "edit", "document", "text"]
        
        # Check control type
        if control_type and control_type.lower() in [t.lower() for t in editable_types]:
            self.logger.info(f"Control type {control_type} is likely editable")
            return True
            
        # Check for common editor window classes
        try:
            if hasattr(element, "class_name"):
                class_name = element.class_name()
                common_editor_classes = [
                    "Edit", "RichEdit", "RichEdit20", "TextBox",
                    "RICHEDIT50W", "Chrome_RenderWidgetHostHWND"
                ]
                if any(cls.lower() in class_name.lower() for cls in common_editor_classes):
                    self.logger.info(f"Element class {class_name} is likely editable")
                    return True
        except Exception as e:
            self.logger.debug(f"Error checking class name: {str(e)}")
            
        # Check window title for common editor applications
        try:
            editor_keywords = ["notepad", "word", "editor", "text", "document", "code"]
            if any(keyword in window_text.lower() for keyword in editor_keywords):
                self.logger.info(f"Window title contains editor keyword: {window_text}")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking window title: {str(e)}")
        
        # Try more intensive checks
        try:
            # Check if it has an editable pattern
            if hasattr(element, "is_editable") and element.is_editable():
                self.logger.info("Element has editable pattern")
                return True
                
            # Check if it has a value pattern
            if hasattr(element, "get_value") and hasattr(element, "set_value"):
                self.logger.info("Element has value pattern")
                return True
                
            # Check if it has a text pattern
            if hasattr(element, "get_text") and callable(getattr(element, "get_text", None)):
                self.logger.info("Element has text pattern")
                return True
                
            # Check if it's a generic keyboard-focusable element
            if hasattr(element, "has_keyboard_focus") and element.has_keyboard_focus():
                self.logger.info("Element has keyboard focus")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking element patterns: {str(e)}")
            
        # Check element properties if available
        try:
            if hasattr(element, "get_properties"):
                properties = element.get_properties()
                if "editable" in properties and properties["editable"]:
                    self.logger.info("Element property 'editable' is True")
                    return True
                if "is_content_element" in properties and properties["is_content_element"]:
                    self.logger.info("Element is a content element")
                    return True
                if "has_keyboard_focus" in properties and properties["has_keyboard_focus"]:
                    self.logger.info("Element has keyboard focus property")
                    return True
        except Exception as e:
            self.logger.debug(f"Error checking element properties: {str(e)}")
            
        # Final fallback: just try to insert text anyway if the window title suggests a text editor
        if any(term in window_text.lower() for term in [".txt", ".py", ".md", ".js", ".html", ".css", ".java"]):
            self.logger.info(f"Window appears to be a code or text editor: {window_text}")
            return True
            
        return False
    
    def is_text_editable(self) -> bool:
        """
        Check if the focused element can accept text input
        
        Returns:
            bool: Whether the focused element is editable
        """
        element_info = self.get_focused_element()
        if not element_info:
            self.logger.warning("No focused element found to check editability")
            return False
            
        return element_info["editable"]
    
    def insert_text(self, text: str) -> bool:
        """
        Insert text into the focused element
        
        Args:
            text: Text to insert
            
        Returns:
            bool: Whether the text was successfully inserted
        """
        if not text:
            self.logger.warning("Empty text provided for insertion")
            return False
            
        try:
            element_info = self.get_focused_element()
            if not element_info:
                self.logger.warning("No focused element found for text insertion")
                return False
                
            if not element_info["editable"]:
                self.logger.warning("Focused element is not editable")
                return False
                
            element = element_info["element"]
            hwnd = element_info["hwnd"]
            
            # Method 1: Try to use the element's type_keys method
            try:
                element.type_keys(text, with_spaces=True, with_tabs=True, with_newlines=True)
                self.logger.info(f"Text inserted using type_keys: {text[:20]}...")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to insert text with type_keys: {str(e)}")
                
            # Method 2: Try to set_text directly if the element supports it
            try:
                if hasattr(element, "set_text") and callable(getattr(element, "set_text", None)):
                    element.set_text(text)
                    self.logger.info(f"Text inserted using set_text: {text[:20]}...")
                    return True
            except Exception as e:
                self.logger.warning(f"Failed to insert text with set_text: {str(e)}")
                
            # Method 3: Try clipboard approach
            try:
                # Initialize COM for clipboard operations on Windows
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    com_initialized = True
                except (ImportError, Exception) as e:
                    self.logger.warning(f"Could not initialize COM: {str(e)}")
                    com_initialized = False
                
                # Remember original clipboard content
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    original_clipboard = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                except:
                    original_clipboard = None
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text)
                win32clipboard.CloseClipboard()
                
                # Make sure the window is in the foreground
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.1)  # Small delay to ensure window activation
                
                # Send Ctrl+V to paste
                pywinauto.keyboard.send_keys('^v')
                time.sleep(0.2)  # Give time for paste to complete
                
                # Restore original clipboard if there was one
                if original_clipboard:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(original_clipboard)
                    win32clipboard.CloseClipboard()
                
                self.logger.info(f"Text inserted using clipboard: {text[:20]}...")
                
                # Uninitialize COM if we initialized it
                if com_initialized:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
                
                return True
            except Exception as e:
                # Uninitialize COM if we initialized it
                if 'com_initialized' in locals() and com_initialized:
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
                
                self.logger.warning(f"Failed to insert text with clipboard: {str(e)}")
                
            # Method 4: Fallback to character-by-character keypress simulation
            try:
                # Make sure the window is in the foreground
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.1)  # Small delay to ensure window activation
                
                # Simulate keystrokes for the text
                for char in text:
                    if char == '\n':
                        pywinauto.keyboard.send_keys('{ENTER}')
                    elif char == '\t':
                        pywinauto.keyboard.send_keys('{TAB}')
                    else:
                        pywinauto.keyboard.send_keys(char, with_spaces=True)
                    time.sleep(0.01)  # Small delay between keystrokes
                
                self.logger.info(f"Text inserted using character-by-character input: {text[:20]}...")
                return True
            except Exception as e:
                self.logger.error(f"Failed to insert text with character-by-character input: {str(e)}")
                
            return False
        except Exception as e:
            self.logger.error(f"Error inserting text: {str(e)}")
            return False 