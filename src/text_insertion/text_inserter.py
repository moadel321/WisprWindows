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
    
    def insert_text(self, text: str, trace_id: str = None) -> bool:
        """
        Insert text into the focused element
        
        Args:
            text: Text to insert
            trace_id: Optional trace ID for logging
            
        Returns:
            bool: Whether the text was successfully inserted
        """
        if not text:
            self.logger.warning("Empty text provided for insertion")
            return False
        
        # Use provided trace ID or generate one
        if not trace_id:
            trace_id = f"insert_{int(time.time() * 1000)}"
            
        start_time = time.time()
        self.logger.info(f"[TRACE:{trace_id}] Starting text insertion ({len(text)} chars)")
            
        try:
            # Get focused element
            element_check_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Getting focused element")
            element_info = self.get_focused_element()
            if not element_info:
                self.logger.warning(f"[TRACE:{trace_id}] No focused element found for text insertion")
                return False
                
            if not element_info["editable"]:
                self.logger.warning(f"[TRACE:{trace_id}] Focused element is not editable")
                return False
                
            element = element_info["element"]
            hwnd = element_info["hwnd"]
            app_name = element_info["app"]
            
            element_check_time = time.time() - element_check_start
            self.logger.info(f"[TRACE:{trace_id}] Found editable element in '{app_name}' in {element_check_time:.3f}s")
            
            # Method 1: Try to use the element's type_keys method
            method1_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Attempting insertion method 1: type_keys")
            try:
                element.type_keys(text, with_spaces=True, with_tabs=True, with_newlines=True)
                method1_time = time.time() - method1_start
                self.logger.info(f"[TRACE:{trace_id}] Text successfully inserted using type_keys in {method1_time:.3f}s")
                return True
            except Exception as e:
                method1_time = time.time() - method1_start
                self.logger.warning(f"[TRACE:{trace_id}] Failed to insert text with type_keys after {method1_time:.3f}s: {str(e)}")
                
            # Method 2: Try to set_text directly if the element supports it
            method2_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Attempting insertion method 2: set_text")
            try:
                if hasattr(element, "set_text") and callable(getattr(element, "set_text", None)):
                    element.set_text(text)
                    method2_time = time.time() - method2_start
                    self.logger.info(f"[TRACE:{trace_id}] Text successfully inserted using set_text in {method2_time:.3f}s")
                    return True
                else:
                    self.logger.info(f"[TRACE:{trace_id}] Element does not support set_text method")
            except Exception as e:
                method2_time = time.time() - method2_start
                self.logger.warning(f"[TRACE:{trace_id}] Failed to insert text with set_text after {method2_time:.3f}s: {str(e)}")
                
            # Method 3: Try clipboard approach
            method3_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Attempting insertion method 3: clipboard")
            try:
                # Initialize COM for clipboard operations on Windows
                com_init_start = time.time()
                try:
                    # Make sure we properly import and initialize COM
                    import pythoncom
                    # Force COM initialization even if it's already initialized
                    # This addresses the "CoInitialize has not been called" error
                    try:
                        pythoncom.CoUninitialize()  # Clean any previous state
                    except:
                        pass
                    pythoncom.CoInitialize()  # Initialize fresh
                    com_initialized = True
                    com_init_time = time.time() - com_init_start
                    self.logger.info(f"[TRACE:{trace_id}] COM initialized in {com_init_time:.3f}s")
                except (ImportError, Exception) as e:
                    self.logger.warning(f"[TRACE:{trace_id}] Could not initialize COM: {str(e)}")
                    com_initialized = False
                
                # Remember original clipboard content
                clipboard_start = time.time()
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    original_clipboard = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                    self.logger.info(f"[TRACE:{trace_id}] Original clipboard content saved")
                except:
                    original_clipboard = None
                    self.logger.info(f"[TRACE:{trace_id}] No original clipboard content to save")
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text)
                win32clipboard.CloseClipboard()
                clipboard_time = time.time() - clipboard_start
                self.logger.info(f"[TRACE:{trace_id}] Clipboard prepared in {clipboard_time:.3f}s")
                
                # Make sure the window is in the foreground
                focus_start = time.time()
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.1)  # Small delay to ensure window activation
                focus_time = time.time() - focus_start
                self.logger.info(f"[TRACE:{trace_id}] Window focused in {focus_time:.3f}s")
                
                # Send Ctrl+V to paste
                paste_start = time.time()
                pywinauto.keyboard.send_keys('^v')
                time.sleep(0.1)  # Reduced from 0.2 to 0.1 to improve performance
                paste_time = time.time() - paste_start
                self.logger.info(f"[TRACE:{trace_id}] Paste command sent in {paste_time:.3f}s")
                
                # Restore original clipboard if there was one
                restore_start = time.time()
                if original_clipboard:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(original_clipboard)
                    win32clipboard.CloseClipboard()
                    self.logger.info(f"[TRACE:{trace_id}] Original clipboard content restored")
                restore_time = time.time() - restore_start
                
                method3_time = time.time() - method3_start
                self.logger.info(f"[TRACE:{trace_id}] Text inserted using clipboard in {method3_time:.3f}s")
                
                # Uninitialize COM if we initialized it
                if com_initialized:
                    try:
                        pythoncom.CoUninitialize()
                        self.logger.info(f"[TRACE:{trace_id}] COM uninitialized")
                    except Exception:
                        pass
                
                return True
            except Exception as e:
                # Uninitialize COM if we initialized it
                if 'com_initialized' in locals() and com_initialized:
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                        self.logger.info(f"[TRACE:{trace_id}] COM uninitialized after error")
                    except Exception:
                        pass
                
                method3_time = time.time() - method3_start
                self.logger.warning(f"[TRACE:{trace_id}] Failed to insert text with clipboard after {method3_time:.3f}s: {str(e)}")
                
            # Method 4: Fallback to character-by-character keypress simulation
            method4_start = time.time()
            self.logger.info(f"[TRACE:{trace_id}] Attempting insertion method 4: character-by-character")
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
                
                method4_time = time.time() - method4_start
                self.logger.info(f"[TRACE:{trace_id}] Text inserted character-by-character in {method4_time:.3f}s")
                return True
            except Exception as e:
                method4_time = time.time() - method4_start
                self.logger.error(f"[TRACE:{trace_id}] Failed to insert text with character-by-character input after {method4_time:.3f}s: {str(e)}")
            
            total_time = time.time() - start_time
            self.logger.error(f"[TRACE:{trace_id}] All text insertion methods failed after {total_time:.3f}s")
            return False
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"[TRACE:{trace_id}] Error inserting text after {total_time:.3f}s: {str(e)}")
            return False 