#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for text insertion functionality with simulated audio input
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.text_insertion.text_inserter import TextInserter
from src.utils.logger import setup_logger


def main():
    """Run the text insertion test with simulated transcription"""
    # Set up logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Text insertion simulation starting")
    
    # Create a TextInserter
    inserter = TextInserter()
    
    print("\n" + "-" * 60)
    print("TEXT INSERTION SIMULATION")
    print("-" * 60)
    print("This script tests text insertion across different applications.")
    print("Steps:")
    print("1. Open any application with a text field (e.g., Notepad, browser, etc.)")
    print("2. Click into the text field to focus it")
    print("3. Return to this window and press Enter to continue")
    print("-" * 60)
    
    # Create sample transcriptions of increasing complexity
    transcriptions = [
        "This is a simple test of the text insertion functionality.",
        "The quick brown fox jumps over the lazy dog. This includes punctuation!",
        "This includes special characters: @#$%^&*()\nAnd a new line character.",
        "Speech to text conversion with multiple  spaces   and longer content that might break simpler insertion methods."
    ]
    
    # Ask which test to run
    print("\nSelect a test to run:")
    print("1. Basic test - Insert a single line of text")
    print("2. Multiple tests - Test different text types sequentially")
    print("3. Comprehensive test - Test all methods with detailed feedback")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        # Basic test
        custom_text = input("\nEnter text to insert (or press Enter for default test): ")
        text_to_insert = custom_text or "This is a test of the Speech-to-Text insertion functionality."
        run_basic_test(inserter, text_to_insert)
    elif choice == '2':
        # Multiple tests
        run_multiple_tests(inserter, transcriptions)
    elif choice == '3':
        # Comprehensive test
        run_comprehensive_test(inserter)
    else:
        print("Invalid choice. Running basic test.")
        run_basic_test(inserter, "This is a default test of the text insertion functionality.")
    
    # Display statistics
    print("\n" + "-" * 60)
    print("TEXT INSERTION STATISTICS")
    print("-" * 60)
    stats = inserter.get_insertion_stats()
    print("Method               | Attempts | Successes | Success Rate")
    print("-" * 60)
    for method, data in stats.items():
        attempts = data["attempts"]
        successes = data["successes"]
        success_rate = (successes / attempts * 100) if attempts > 0 else 0
        print(f"{method.ljust(20)} | {str(attempts).ljust(8)} | {str(successes).ljust(9)} | {success_rate:.1f}%")
    
    print("\nTest completed.")
    input("\nPress Enter to exit...")


def run_basic_test(inserter, text):
    """Run a basic insertion test"""
    print(f"\nText to insert: \"{text}\"")
    print("Please focus your text field now...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Inserting in {i}...")
        time.sleep(1)
    
    # Check if element is editable
    element_info = inserter.get_focused_element()
    if not element_info:
        print("\nNo focused element detected. Make sure you've clicked into a text field.")
        return
    
    print(f"\nFocused application: {element_info['app']}")
    print(f"Control type: {element_info['control_type']}")
    print(f"Editable: {element_info['editable']}")
    
    # Try to insert the text
    start_time = time.time()
    success = inserter.insert_text(text)
    end_time = time.time()
    
    if success:
        print(f"\n✓ Text insertion SUCCESSFUL ({(end_time - start_time):.2f}s)")
        print(f"Method used: {inserter.last_insertion_method}")
    else:
        print("\n✗ Text insertion FAILED")
        print("Please try a different application or text field.")


def run_multiple_tests(inserter, transcriptions):
    """Run multiple insertion tests with different text types"""
    print("\nRunning multiple text insertion tests...")
    
    for i, text in enumerate(transcriptions):
        print(f"\n--- Test {i+1}/{len(transcriptions)} ---")
        print(f"Text to insert: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print("Please focus your text field now...")
        
        # Countdown
        for j in range(3, 0, -1):
            print(f"Inserting in {j}...")
            time.sleep(1)
        
        # Try to insert the text
        start_time = time.time()
        success = inserter.insert_text(text)
        end_time = time.time()
        
        if success:
            print(f"\n✓ Test {i+1}: SUCCESSFUL ({(end_time - start_time):.2f}s)")
            print(f"Method used: {inserter.last_insertion_method}")
        else:
            print(f"\n✗ Test {i+1}: FAILED")
        
        if i < len(transcriptions) - 1:
            input("\nPress Enter for next test...")


def run_comprehensive_test(inserter):
    """Run a comprehensive test of all insertion methods"""
    print("\nComprehensive Text Insertion Test")
    print("This test will attempt to identify the most reliable insertion method for your application.")
    print("Please focus your text field and keep it focused during the test.")
    
    input("\nPress Enter when ready...")
    
    # Basic info about target
    element_info = inserter.get_focused_element()
    if not element_info:
        print("\nNo focused element detected. Make sure you've clicked into a text field.")
        return
    
    print(f"\nTarget Application: {element_info['app']}")
    print(f"Application Class: {element_info['app_class']}")
    print(f"Control Type: {element_info['control_type']}")
    print(f"Element Detected: {'Yes' if element_info['element'] is not None else 'No'}")
    print(f"Editable: {element_info['editable']}")
    
    # Define test texts
    test_texts = {
        "clipboard": "Text inserted using clipboard method. [CLIPBOARD TEST]",
        "type_keys": "Text inserted using type_keys method. [TYPE_KEYS TEST]",
        "set_text": "Text inserted using set_text method. [SET_TEXT TEST]",
        "direct_input": "Text inserted using direct input method. [DIRECT_INPUT TEST]",
        "char_by_char": "Text inserted using character-by-character method. [CHAR_BY_CHAR TEST]"
    }
    
    # Test each method
    print("\nTesting insertion methods...\n")
    successful_methods = []
    
    # Test clipboard method
    print("Testing clipboard method...")
    if element_info['element'] is None or _test_clipboard_method(inserter, test_texts["clipboard"]):
        print("✓ Clipboard method: SUCCESSFUL")
        successful_methods.append("clipboard")
    else:
        print("✗ Clipboard method: FAILED")
    
    # Only test element-based methods if we have an element
    if element_info['element'] is not None:
        time.sleep(1)
        print("\nTesting type_keys method...")
        if _test_type_keys_method(inserter, element_info['element'], test_texts["type_keys"]):
            print("✓ type_keys method: SUCCESSFUL")
            successful_methods.append("type_keys")
        else:
            print("✗ type_keys method: FAILED")
        
        time.sleep(1)
        print("\nTesting set_text method...")
        if _test_set_text_method(inserter, element_info['element'], test_texts["set_text"]):
            print("✓ set_text method: SUCCESSFUL")
            successful_methods.append("set_text")
        else:
            print("✗ set_text method: FAILED")
    
    # Test direct input
    time.sleep(1)
    print("\nTesting direct input method...")
    if _test_direct_input_method(inserter, test_texts["direct_input"]):
        print("✓ Direct input method: SUCCESSFUL")
        successful_methods.append("direct_input")
    else:
        print("✗ Direct input method: FAILED")
    
    # Test char by char
    time.sleep(1)
    print("\nTesting character-by-character method...")
    if _test_char_by_char_method(inserter, test_texts["char_by_char"]):
        print("✓ Character-by-character method: SUCCESSFUL")
        successful_methods.append("char_by_char")
    else:
        print("✗ Character-by-character method: FAILED")
    
    # Summary
    print("\n" + "-" * 60)
    print("RESULTS SUMMARY")
    print("-" * 60)
    if successful_methods:
        print(f"Successful methods: {', '.join(successful_methods)}")
        print(f"Recommended method for {element_info['app']}: {successful_methods[0]}")
    else:
        print("No successful methods found for this application/control.")
        print("Try using a different text field or application.")


# Helper methods for comprehensive testing
def _test_clipboard_method(inserter, text):
    try:
        return inserter._insert_via_clipboard(text, "test_clipboard")
    except Exception:
        return False

def _test_type_keys_method(inserter, element, text):
    try:
        return inserter._insert_via_element(element, text, "test_type_keys")
    except Exception:
        return False

def _test_set_text_method(inserter, element, text):
    try:
        # This test is included in _insert_via_element but we're isolating it
        if hasattr(element, "set_text") and callable(getattr(element, "set_text", None)):
            element.set_text(text)
            return True
        return False
    except Exception:
        return False

def _test_direct_input_method(inserter, text):
    try:
        return inserter._insert_via_direct_input(text, "test_direct_input")
    except Exception:
        return False

def _test_char_by_char_method(inserter, text):
    try:
        return inserter._insert_via_char_by_char(text, "test_char_by_char")
    except Exception:
        return False


if __name__ == "__main__":
    main() 