#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap script to run the Speech-to-Text application
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """Main entry point to launch the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Text Application")
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to Whisper model directory",
        default=os.path.expanduser("~/whisper-models")
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run system tests instead of the application"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="Run performance tests instead of the application"
    )
    
    args = parser.parse_args()
    
    # Set model directory in environment
    os.environ["WHISPER_MODEL_PATH"] = args.model_path
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    os.environ["STT_LOG_LEVEL"] = str(log_level)
    
    try:
        # Run tests if requested
        if args.test:
            from tests.system_test import run_system_tests
            run_system_tests(args.model_path)
            return
            
        if args.performance:
            from tests.performance_test import run_performance_tests
            run_performance_tests(args.model_path)
            return
        
        # Launch application
        from src.main import main as app_main
        return app_main()
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all dependencies are installed. See requirements.txt")
        return 1
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 