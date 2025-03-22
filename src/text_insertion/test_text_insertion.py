#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for text insertion functionality
"""

import sys
import time
import logging
import os
import traceback
from pathlib import Path

# Add parent directory to path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text_insertion.text_inserter import TextInserter
from src.utils.logger import setup_logger


def main():
    """Test the TextInserter class"""
    # Set up logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Text insertion test script starting...")
    
    try:
        # Create a TextInserter instance
        inserter = TextInserter()
        logger.info("TextInserter instance created")
        
        # Wait for user to focus a text field
        print("\n" + "-" * 60)
        print("TEXT INSERTION TEST")
        print("-" * 60)
        print("1. Open any application with a text field (e.g., Notepad, browser)")
        print("2. Click into the text field to focus it")
        print("3. Return to this window and press Enter to continue")
        print("-" * 60)
        input("Press Enter when ready...")
        
        # Small delay to allow the user to switch focus back
        time.sleep(0.5)
        
        # First, check the focused element
        print("\nChecking focused element...")
        element_info = inserter.get_focused_element()
        if element_info:
            print(f"Focused application: {element_info['app']}")
            print(f"Control type: {element_info['control_type']}")
            print(f"Editable: {element_info['editable']}")
        else:
            print("No focused element detected")
            
        # Check if the element is editable
        is_editable = inserter.is_text_editable()
        print(f"Is focused element editable: {is_editable}")
        
        if is_editable:
            # Ask for text to insert
            test_text = input("\nEnter text to insert (or press Enter for default test text): ")
            if not test_text:
                test_text = "This is a test of the Speech-to-Text text insertion functionality."
                
            print(f"\nAttempting to insert text: \"{test_text}\"")
            print("Please switch focus to your text field now...")
            
            # Give user time to switch focus
            for i in range(3, 0, -1):
                print(f"Inserting in {i}...")
                time.sleep(1)
                
            # Try to insert the text
            success = inserter.insert_text(test_text)
            
            if success:
                print("\nText insertion successful!")
            else:
                print("\nText insertion failed")
        else:
            print("\nCannot insert text - focused element is not editable")
            
        print("\nTest completed")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        traceback.print_exc()
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main() 