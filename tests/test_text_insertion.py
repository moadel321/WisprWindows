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
        
        # Patch Application to simulate window connection failure
        with patch('src.text_insertion.text_inserter.Application') as mock_app:
            # Set up the mock to raise an exception, triggering the fallback path
            mock_app.return_value.connect.side_effect = Exception("Connection failed")
            
            # Patch Desktop.from_point to return a mock element
            with patch('src.text_insertion.text_inserter.Desktop') as mock_desktop:
                # Set up the mock desktop
                mock_element = MagicMock()
                mock_element.control_type = "Edit"
                mock_desktop.return_value.from_point.return_value = mock_element
                
                # Call the method
                result = self.text_inserter.get_focused_element()
                
                # Assert results
                self.assertIsNotNone(result)
                self.assertEqual(result["app"], "Untitled - Notepad")
                self.assertEqual(result["control_type"], "Edit")
                self.assertEqual(result["hwnd"], 12345)
                self.assertTrue(result["editable"])
    
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
    
    def test_is_text_editable_with_non_editable_element(self):
        """Test is_text_editable with a non-editable element"""
        # Patch get_focused_element to return a non-editable element
        mock_element_info = {
            "app": "Notepad",
            "element": MagicMock(),
            "editable": False,
            "control_type": "Button",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Call the method
        result = self.text_inserter.is_text_editable()
        
        # Assert results
        self.assertFalse(result)
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
    
    def test_insert_text_non_editable_element(self):
        """Test insert_text with a non-editable element"""
        # Patch get_focused_element to return a non-editable element
        mock_element_info = {
            "app": "Notepad",
            "element": MagicMock(),
            "editable": False,
            "control_type": "Button",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Call the method
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertFalse(result)
        self.text_inserter.get_focused_element.assert_called_once()
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_insert_text_successful(self, mock_win32gui):
        """Test successful text insertion"""
        # Create a mock element with working type_keys method
        mock_element = MagicMock()
        
        # Patch get_focused_element to return an editable element
        mock_element_info = {
            "app": "Notepad",
            "element": mock_element,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Call the method
        result = self.text_inserter.insert_text("Test text")
        
        # Assert results
        self.assertTrue(result)
        self.text_inserter.get_focused_element.assert_called_once()
        mock_element.type_keys.assert_called_once_with("Test text", with_spaces=True, with_tabs=True, with_newlines=True)
    
    @patch('src.text_insertion.text_inserter.win32gui')
    def test_insert_text_fallback_methods(self, mock_win32gui):
        """Test fallback methods for text insertion"""
        # Create a mock element with failing type_keys method
        mock_element = MagicMock()
        mock_element.type_keys.side_effect = Exception("type_keys failed")
        mock_element.set_text.side_effect = Exception("set_text failed")
        
        # Patch get_focused_element to return an editable element
        mock_element_info = {
            "app": "Notepad",
            "element": mock_element,
            "editable": True,
            "control_type": "Edit",
            "hwnd": 12345
        }
        self.text_inserter.get_focused_element = MagicMock(return_value=mock_element_info)
        
        # Patch win32clipboard for the clipboard fallback
        with patch('src.text_insertion.text_inserter.win32clipboard') as mock_clipboard:
            # Patch pywinauto.keyboard.send_keys
            with patch('src.text_insertion.text_inserter.pywinauto.keyboard.send_keys') as mock_send_keys:
                # Call the method
                result = self.text_inserter.insert_text("Test text")
                
                # Assert results
                self.assertTrue(result)
                mock_element.type_keys.assert_called_once()
                mock_clipboard.OpenClipboard.assert_called()
                mock_clipboard.EmptyClipboard.assert_called()
                mock_clipboard.SetClipboardText.assert_called_with("Test text")
                mock_clipboard.CloseClipboard.assert_called()
                mock_send_keys.assert_called_with('^v')


if __name__ == '__main__':
    unittest.main() 