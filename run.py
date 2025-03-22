#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run script for the Speech-to-Text application
"""

import sys
import os
import argparse

def main():
    """Run the Speech-to-Text application"""
    # Create parser
    parser = argparse.ArgumentParser(description="Run the Speech-to-Text application")
    
    # Add arguments
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to Whisper model directory", 
        default=os.path.expanduser("~/whisper-models")
    )
    parser.add_argument(
        "--no-gui", 
        action="store_true", 
        help="Run in command-line mode (no GUI)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run tests instead of the application"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="Run performance tests instead of the application"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up environment variables
    os.environ["STT_MODEL_PATH"] = args.model_path
    
    if args.debug:
        os.environ["STT_DEBUG"] = "1"
    
    # Run the appropriate module
    if args.test:
        print("Running system tests...")
        from tests.system_test import main as test_main
        return test_main(args.model_path)
    elif args.performance:
        print("Running performance tests...")
        from tests.performance_test import main as perf_main
        return perf_main(args.model_path)
    else:
        print("Starting Speech-to-Text application...")
        if args.no_gui:
            print("Command-line mode not implemented yet")
            return 1
        else:
            from src.main import main as app_main
            return app_main()


if __name__ == "__main__":
    sys.exit(main()) 