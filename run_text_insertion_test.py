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
    print("This script simulates the text insertion from speech recognition")
    print("1. Open any application with a text field (e.g., Notepad, browser)")
    print("2. Click into the text field to focus it")
    print("3. Return to this window and press Enter to continue")
    print("-" * 60)
    input("Press Enter when ready...")
    
    # Create sample transcriptions
    transcriptions = [
        "This is a test of the speech to text insertion functionality.",
        "The quick brown fox jumps over the lazy dog.",
        "Speech to text conversion can significantly improve productivity.",
        "All transcriptions are processed locally with no data leaving your device."
    ]
    
    print("\nBeginning text insertion test...")
    
    for i, text in enumerate(transcriptions):
        print(f"\nTranscription {i+1}/{len(transcriptions)}:")
        print(f"Text: \"{text}\"")
        print("Please focus your text field now...")
        
        # Countdown
        for j in range(3, 0, -1):
            print(f"Inserting in {j}...")
            time.sleep(1)
            
        # Check if the element is editable
        is_editable = inserter.is_text_editable()
        print(f"Target field is editable: {is_editable}")
        
        if is_editable:
            # Try to insert the text
            success = inserter.insert_text(text)
            print(f"Text insertion {'successful' if success else 'failed'}")
        else:
            print("Cannot insert text - target is not editable")
            
        # Wait before next insertion
        if i < len(transcriptions) - 1:
            input("\nPress Enter for next transcription...")
    
    print("\nText insertion test completed")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main() 