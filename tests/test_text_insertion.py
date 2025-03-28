#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the text insertion component
"""

import sys
import os
import unittest
import logging
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_insertion.text_inserter import TextInserter


class TestTextInsertion(unittest.TestCase):
    """Test cases for text insertion functionality"""
    
    def setUp(self):
        """Set up test fixtures, if any"""
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create instance with mocked dependencies
        self.text_inserter = TextInserter()
    
    def tearDown(self):
        """Tear down test fixtures, if any"""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_get_focused_element_no_window(self, mock_win32gui):
        """Test get_focused_element when no window is focused"""
        # Set up mock to return 0 (no foreground window)
        mock_win32gui.GetForegroundWindow.return_value = 0
        
        # Call the method
        result = self.text_inserter.get_focused_element()
        
        # Assert results
        self.assertIsNone(result)
        mock_win32gui.GetForegroundWindow.assert_called_once()
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_get_focused_element_with_window(self, mock_win32gui):
        """Test get_focused_element with a focused window"""
        # Set up mock to return a valid window handle and window information
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetClassName.return_value = "Notepad"
        mock_win32gui.GetWindowText.return_value = "Untitled - Notepad"
        
        # Patch Desktop to simulate window connection failure
        with patch('src.text_insertion.text_inserter.Desktop') as mock_desktop:
            # Set up the mock to raise an exception, triggering the fallback path
            mock_desktop.return_value.connect.side_effect = Exception("Connection failed")
            
            # Patch Desktop.from_point to return a mock element
            mock_desktop.return_value.from_point.side_effect = Exception("from_point failed")
            
            # Call the method (should fall back to window-level detection)
            result = self.text_inserter.get_focused_element()
            
            # Assert results
            self.assertIsNotNone(result)
            self.assertEqual(result["app"], "Untitled - Notepad")
            self.assertEqual(result["hwnd"], 12345)
    
    def test_is_known_text_editor(self):
        """Test known text editor detection"""
        # Test with known editor window title
        self.assertTrue(self.text_inserter._is_known_text_editor("Untitled - Notepad", "Notepad"))
        self.assertTrue(self.text_inserter._is_known_text_editor("example.py - Visual Studio Code", "Chrome_WidgetWin_1"))
        self.assertTrue(self.text_inserter._is_known_text_editor("Obsidian v1.4.5", "Chrome_WidgetWin_1"))
        
        # Test with file extension in title
        self.assertTrue(self.text_inserter._is_known_text_editor("test.py - Editor", "Unknown"))
        
        # Test non-editor window
        self.assertFalse(self.text_inserter._is_known_text_editor("Task Manager", "Unknown"))
    
    def test_is_text_editable_no_element(self):
        """Test is_text_editable when no element is found"""
        # Patch get_focused_element to return None
        self.text_inserter.get_focused_element = MagicMock(return_value=None)
        
        # Call the method
        result = self.text_inserter.is_text_editable()
        
        # Assert results
        self.assertFalse(result)
        self.text_inserter.get_focused_element.assert_called_once()
    
    def test_is_text_editable_with_editable_element(self):
        """Test is_text_editable with an editable element"""
        # Patch get_focused_element to return an editable element
        mock_element_info = {
            "app": "Notepad",
            "app_class": "Notepad",
            "element": MagicMock(),
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Call the method
        result = self.text_inserter.is_text_editable()
        
        # Assert results
        self.assertTrue(result)
        self.text_inserter.get_focused_element.assert_called_once()
    
    def test_insert_text_empty_text(self):
        """Test insert_text with empty text"""
        # Call the method with empty text
        result = self.text_inserter.insert_text("")
        
        # Assert results
        self.assertFalse(result)
    
    def test_insert_text_no_focused_element(self):
        """Test insert_text when no element is focused"""
        # Patch get_focused_element to return None
        self.text_inserter.get_focused_element = MagicMock(return_value=None)
        
        # Call the method
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertFalse(result)
        self.text_inserter.get_focused_element.assert_called_once()
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_insert_text_via_clipboard(self, mock_win32gui):
        """Test text insertion via clipboard method"""
        # Create mock element info
        mock_element_info = {
            "app": "Notepad",
            "app_class": "Notepad",
            "element": None,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Mock _ensure_foreground_window
        self.text_inserter._ensure_foreground_window = MagicMock(return_value=True)
        
        # Mock the instance method directly
        self.text_inserter._insert_via_clipboard = MagicMock(return_value=True)
        
        # Call the method
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertTrue(result)
        self.text_inserter._insert_via_clipboard.assert_called_once()
        self.assertEqual(self.text_inserter.last_insertion_method, "clipboard")
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_insertion_method_fallbacks(self, mock_win32gui):
        """Test that all insertion methods are attempted in sequence"""
        # Create mock element info with a mock element
        mock_element = MagicMock()
        mock_element_info = {
            "app": "Notepad",
            "app_class": "Notepad",
            "element": mock_element,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Mock _ensure_foreground_window
        self.text_inserter._ensure_foreground_window = MagicMock(return_value=True)
        
        # Mock all insertion methods to fail except the last one
        self.text_inserter._insert_via_clipboard = MagicMock(return_value=False)
        self.text_inserter._insert_via_element = MagicMock(return_value=False)
        self.text_inserter._insert_via_direct_input = MagicMock(return_value=False)
        self.text_inserter._insert_via_char_by_char = MagicMock(return_value=True)
        
        # Call insert_text
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertTrue(result)
        self.text_inserter._insert_via_clipboard.assert_called_once()
        self.text_inserter._insert_via_element.assert_called_once()
        self.text_inserter._insert_via_direct_input.assert_called_once()
        self.text_inserter._insert_via_char_by_char.assert_called_once()
        self.assertEqual(self.text_inserter.last_insertion_method, "char_by_char")
    
    def test_all_methods_fail(self):
        """Test behavior when all insertion methods fail"""
        # Create mock element info
        mock_element_info = {
            "app": "Notepad",
            "app_class": "Notepad",
            "element": MagicMock(),
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Mock _ensure_foreground_window
        self.text_inserter._ensure_foreground_window = MagicMock(return_value=True)
        
        # Mock all insertion methods to fail
        self.text_inserter._insert_via_clipboard = MagicMock(return_value=False)
        self.text_inserter._insert_via_element = MagicMock(return_value=False)
        self.text_inserter._insert_via_direct_input = MagicMock(return_value=False)
        self.text_inserter._insert_via_char_by_char = MagicMock(return_value=False)
        
        # Call insert_text
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertFalse(result)
        # Ensure all methods were called
        self.text_inserter._insert_via_clipboard.assert_called_once()
        self.text_inserter._insert_via_element.assert_called_once()
        self.text_inserter._insert_via_direct_input.assert_called_once()
        self.text_inserter._insert_via_char_by_char.assert_called_once()
    
    def test_insertion_stats_tracking(self):
        """Test that insertion statistics are tracked correctly"""
        # Reset stats
        self.text_inserter.insertion_stats = {
            "type_keys": {"attempts": 0, "successes": 0},
            "set_text": {"attempts": 0, "successes": 0},
            "clipboard": {"attempts": 0, "successes": 0},
            "char_by_char": {"attempts": 0, "successes": 0},
            "direct_input": {"attempts": 0, "successes": 0},
            "win32_input": {"attempts": 0, "successes": 0}
        }
        
        # Setup text_inserter for testing
        mock_element = MagicMock()
        self.text_inserter.get_focused_element = MagicMock(return_value={
            "app": "Notepad",
            "app_class": "Notepad",
            "element": mock_element,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        })
        self.text_inserter._ensure_foreground_window = MagicMock(return_value=True)
        
        # Test clipboard method (success)
        with patch('src.text_insertion.text_inserter.win32clipboard'):
            with patch('src.text_insertion.text_inserter.send_keys'):
                # Mock internal methods
                original_clipboard_method = self.text_inserter._insert_via_clipboard
                self.text_inserter._insert_via_clipboard = MagicMock(side_effect=lambda text, trace_id: original_clipboard_method(text, trace_id))
                
                # Perform a test clipboard insertion
                result = self.text_inserter._insert_via_clipboard("Test", "test_trace")
                
                # Check stats
                self.assertEqual(self.text_inserter.insertion_stats["clipboard"]["attempts"], 1)
                if result:  # Only check success if the mock succeeded
                    self.assertEqual(self.text_inserter.insertion_stats["clipboard"]["successes"], 1)


if __name__ == '__main__':
    unittest.main() 