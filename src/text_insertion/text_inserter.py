#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text insertion into focused applications using multiple methods
"""

import logging
import time
import sys
import ctypes
from typing import Optional, Dict, Any, Tuple, List
import win32gui
import win32api
import win32con
import win32clipboard
import pywinauto
from pywinauto import Desktop
from pywinauto.application import Application
from pywinauto.keyboard import send_keys


class TextInserter:
    """
    Handles detecting the focused text box and inserting text using multiple methods
    """
    
    def __init__(self):
        """Initialize the text inserter"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("TextInserter initialized")
        self.last_active_element = None
        self.last_insertion_method = None
        self.insertion_stats = {
            "type_keys": {"attempts": 0, "successes": 0},
            "set_text": {"attempts": 0, "successes": 0},
            "clipboard": {"attempts": 0, "successes": 0},
            "char_by_char": {"attempts": 0, "successes": 0},
            "direct_input": {"attempts": 0, "successes": 0},
            "win32_input": {"attempts": 0, "successes": 0}
        }
    
    def get_focused_element(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently focused element for text insertion
        
        Returns:
            Optional[Dict[str, Any]]: Information about the focused element, or None if not found
        """
        try:
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            
            # Get window title and class
            window_text = win32gui.GetWindowText(hwnd)
            window_class = win32gui.GetClassName(hwnd)
            
            self.logger.info(f"Focused window: {window_text} (Class: {window_class})")
            
            # Skip if no window is focused
            if not hwnd or not window_text:
                self.logger.warning("No foreground window detected")
                return None
            
            # First try to get the focused element using pywinauto
            try:
                app = Desktop(backend="uia").connect(handle=hwnd)
                element = app.get_focus()
                
                # Check if likely editable
                control_type = getattr(element, "control_type", "unknown")
                editable = self._is_likely_editable(element, control_type)
                
                element_info = {
                    "app": window_text,
                    "app_class": window_class,
                    "element": element,
                    "editable": editable,
                    "control_type": control_type,
                    "hwnd": hwnd
                }
                
                self.logger.info(f"Found focused element in {window_text}, control type: {control_type}, editable: {editable}")
                self.last_active_element = element_info
                return element_info
                
            except Exception as e:
                self.logger.info(f"Could not get focused element with pywinauto: {str(e)}")
            
            # Fallback: use cursor position
            try:
                cursor_pos = win32gui.GetCursorPos()
                x, y = cursor_pos
                
                # Get window at cursor position
                hwnd_at_cursor = win32gui.WindowFromPoint(cursor_pos)
                window_text_at_cursor = win32gui.GetWindowText(hwnd_at_cursor)
                
                # Try to get element at cursor position
                try:
                    element = Desktop(backend="uia").from_point(x, y)
                    control_type = getattr(element, "control_type", "unknown")
                    editable = self._is_likely_editable(element, control_type)
                except Exception:
                    element = None
                    control_type = "unknown"
                    # Consider it editable if it's a known text editor window
                    editable = self._is_known_text_editor(window_text_at_cursor, win32gui.GetClassName(hwnd_at_cursor))
                
                element_info = {
                    "app": window_text_at_cursor,
                    "app_class": win32gui.GetClassName(hwnd_at_cursor),
                    "element": element,
                    "editable": editable,
                    "control_type": control_type,
                    "hwnd": hwnd_at_cursor,
                    "cursor_pos": cursor_pos
                }
                
                self.logger.info(f"Found element at cursor in {window_text_at_cursor}, editable: {editable}")
                self.last_active_element = element_info
                return element_info
                
            except Exception as e:
                self.logger.info(f"Could not get element at cursor: {str(e)}")
            
            # Last resort: just use the foreground window
            element_info = {
                "app": window_text,
                "app_class": window_class,
                "element": None,
                "editable": self._is_known_text_editor(window_text, window_class),
                "control_type": "window",
                "hwnd": hwnd
            }
            
            self.logger.info(f"Using foreground window fallback: {window_text}, assuming editable: {element_info['editable']}")
            self.last_active_element = element_info
            return element_info
            
        except Exception as e:
            self.logger.error(f"Error getting focused element: {str(e)}")
            return None
    
    def _is_known_text_editor(self, window_title: str, window_class: str) -> bool:
        """Check if window is a known text editor based on title or class"""
        # Known text editor window classes
        editor_classes = [
            "Notepad", "TextEditorWindow", "RICHEDIT", "EDIT", 
            "Chrome_RenderWidgetHostHWND", "MozillaWindowClass",
            "OpusApp", "EXCEL", "bosa_sdm", "CabinetWClass"
        ]
        
        # Known text editor window title keywords
        editor_keywords = [
            "notepad", "word", "editor", "text", "document", "excel", 
            "code", "visual studio", "vscode", "sublime", "atom", 
            "chrome", "firefox", "edge", "safari", "browser",
            "obsidian", "evernote", "onenote", "slack", "discord",
            "terminal", "powershell", "command", "prompt"
        ]
        
        # Check class
        if any(cls.lower() in window_class.lower() for cls in editor_classes):
            return True
            
        # Check title
        if any(kw.lower() in window_title.lower() for kw in editor_keywords):
            return True
            
        # Check for file extensions in title that suggest text editing
        file_extensions = [".txt", ".md", ".py", ".js", ".html", ".css", ".c", ".cpp", ".java", ".json", ".xml", ".csv"]
        if any(ext in window_title for ext in file_extensions):
            return True
            
        return False
            
    def _is_likely_editable(self, element, control_type: str) -> bool:
        """
        Determine if an element is likely to be editable
        
        Args:
            element: The UI element
            control_type: The control type of the element
            
        Returns:
            bool: Whether the element is likely to be editable
        """
        # List of control types that are typically editable
        editable_types = [
            "Edit", "Document", "Text", "DataItem", "edit", "document", 
            "text", "RichEdit", "RICHEDIT", "TextBox", "Editable"
        ]
        
        # Check control type
        if control_type and any(t.lower() in control_type.lower() for t in editable_types):
            self.logger.info(f"Control type {control_type} is likely editable")
            return True
            
        # Check element properties if available
        try:
            # Check if it has editable pattern
            if hasattr(element, "is_editable") and element.is_editable():
                return True
                
            # Check if it can be typed into
            if hasattr(element, "can_type_keys") and element.can_type_keys():
                return True
                
            # Check if it has a value pattern
            if hasattr(element, "get_value") and hasattr(element, "set_value"):
                return True
                
            # Check if it has a text pattern
            if hasattr(element, "get_text") and callable(getattr(element, "get_text", None)):
                return True
                
            # Check if keyboard focusable
            if hasattr(element, "has_keyboard_focus") and element.has_keyboard_focus():
                return True
        except Exception:
            pass
            
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
        Insert text into the focused element using multiple methods
        
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
        
        # Get focused element
        element_info = self.get_focused_element()
        if not element_info:
            self.logger.warning(f"[TRACE:{trace_id}] No focused element found for text insertion")
            return False
        
        # Ensure window is in foreground before attempting insertion
        hwnd = element_info["hwnd"]
        self._ensure_foreground_window(hwnd)
        
        # Method 1: Try direct clipboard method (most reliable)
        result = self._insert_via_clipboard(text, trace_id)
        if result:
            self.last_insertion_method = "clipboard"
            return True
            
        # Method 2: Try PyWinAuto element methods if we have a valid element
        if element_info["element"] is not None:
            result = self._insert_via_element(element_info["element"], text, trace_id)
            if result:
                self.last_insertion_method = "element"
                return True
        
        # Method 3: Try direct input simulation
        result = self._insert_via_direct_input(text, trace_id)
        if result:
            self.last_insertion_method = "direct_input"
            return True
            
        # Method 4: Try character-by-character as last resort
        result = self._insert_via_char_by_char(text, trace_id)
        if result:
            self.last_insertion_method = "char_by_char"
            return True
        
        # If we get here, all methods failed
        total_time = time.time() - start_time
        self.logger.error(f"[TRACE:{trace_id}] All text insertion methods failed after {total_time:.3f}s")
        return False
    
    def _ensure_foreground_window(self, hwnd: int) -> bool:
        """Ensure the window is in the foreground before insertion"""
        try:
            # Check if already foreground
            if win32gui.GetForegroundWindow() == hwnd:
                return True
                
            # Try to bring window to foreground
            win32gui.SetForegroundWindow(hwnd)
            
            # Small delay to ensure window activation
            time.sleep(0.1)
            
            return win32gui.GetForegroundWindow() == hwnd
        except Exception as e:
            self.logger.warning(f"Failed to bring window to foreground: {str(e)}")
            return False
    
    def _insert_via_element(self, element, text: str, trace_id: str) -> bool:
        """Try to insert text using pywinauto element methods"""
        # Method 2a: Try type_keys
        self.insertion_stats["type_keys"]["attempts"] += 1
        try:
            if hasattr(element, "type_keys") and callable(getattr(element, "type_keys", None)):
                element.type_keys(text, with_spaces=True, with_tabs=True, with_newlines=True)
                self.logger.info(f"[TRACE:{trace_id}] Text successfully inserted using type_keys")
                self.insertion_stats["type_keys"]["successes"] += 1
                return True
        except Exception as e:
            self.logger.info(f"[TRACE:{trace_id}] Failed to insert text with type_keys: {str(e)}")
        
        # Method 2b: Try set_text
        self.insertion_stats["set_text"]["attempts"] += 1
        try:
            if hasattr(element, "set_text") and callable(getattr(element, "set_text", None)):
                element.set_text(text)
                self.logger.info(f"[TRACE:{trace_id}] Text successfully inserted using set_text")
                self.insertion_stats["set_text"]["successes"] += 1
                return True
        except Exception as e:
            self.logger.info(f"[TRACE:{trace_id}] Failed to insert text with set_text: {str(e)}")
            
        return False
    
    def _insert_via_clipboard(self, text: str, trace_id: str) -> bool:
        """Insert text using clipboard (copy/paste)"""
        self.insertion_stats["clipboard"]["attempts"] += 1
        
        try:
            # Remember original clipboard content
            original_clipboard = None
            try:
                win32clipboard.OpenClipboard()
                try:
                    original_clipboard = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                except (TypeError, win32clipboard.error):
                    pass
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
                win32clipboard.CloseClipboard()
            except Exception as e:
                self.logger.info(f"[TRACE:{trace_id}] Clipboard preparation failed: {str(e)}")
                try:
                    win32clipboard.CloseClipboard()
                except:
                    pass
                return False
            
            # Send Ctrl+V to paste
            time.sleep(0.05)  # Small delay
            send_keys('^v')  # Ctrl+V
            time.sleep(0.05)  # Wait for paste to complete
            
            # Restore original clipboard content if there was one
            if original_clipboard:
                try:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(original_clipboard, win32clipboard.CF_UNICODETEXT)
                    win32clipboard.CloseClipboard()
                except:
                    try:
                        win32clipboard.CloseClipboard()
                    except:
                        pass
            
            self.logger.info(f"[TRACE:{trace_id}] Text inserted using clipboard method")
            self.insertion_stats["clipboard"]["successes"] += 1
            return True
            
        except Exception as e:
            self.logger.info(f"[TRACE:{trace_id}] Failed to insert text with clipboard: {str(e)}")
            try:
                win32clipboard.CloseClipboard()
            except:
                pass
            return False
    
    def _insert_via_direct_input(self, text: str, trace_id: str) -> bool:
        """Insert text using direct input simulation"""
        self.insertion_stats["direct_input"]["attempts"] += 1
        
        try:
            # Send raw text as keystrokes (send_keys can handle larger chunks)
            sanitized_text = text.replace('^', '{^}').replace('%', '{%}').replace('+', '{+}').replace('~', '{~}')
            send_keys(sanitized_text, pause=0.01)
            
            self.logger.info(f"[TRACE:{trace_id}] Text inserted using direct input")
            self.insertion_stats["direct_input"]["successes"] += 1
            return True
            
        except Exception as e:
            self.logger.info(f"[TRACE:{trace_id}] Failed to insert text with direct input: {str(e)}")
            return False
    
    def _insert_via_char_by_char(self, text: str, trace_id: str) -> bool:
        """Insert text character by character as last resort"""
        self.insertion_stats["char_by_char"]["attempts"] += 1
        
        try:
            # Simulate keystrokes for the text one character at a time
            for char in text:
                if char == '\n':
                    send_keys('{ENTER}')
                elif char == '\t':
                    send_keys('{TAB}')
                else:
                    sanitized_char = char
                    if char in '^%+~()[]{}':
                        sanitized_char = '{' + char + '}'
                    send_keys(sanitized_char)
                time.sleep(0.01)  # Small delay between keystrokes
            
            self.logger.info(f"[TRACE:{trace_id}] Text inserted character-by-character")
            self.insertion_stats["char_by_char"]["successes"] += 1
            return True
            
        except Exception as e:
            self.logger.info(f"[TRACE:{trace_id}] Failed to insert text character-by-character: {str(e)}")
            return False
    
    def get_insertion_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about which insertion methods have been successful"""
        return self.insertion_stats 