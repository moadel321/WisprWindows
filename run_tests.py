#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run tests for the STT application
"""

import os
import sys
import subprocess
import argparse

def main():
    """Run tests for the STT application"""
    parser = argparse.ArgumentParser(description='Run tests for the STT application')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--error', action='store_true', help='Run error handling tests')
    parser.add_argument('--text-insertion', action='store_true', help='Run text insertion tests')
    parser.add_argument('--model', action='store_true', help='Run model tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--system', action='store_true', help='Run system tests')
    parser.add_argument('--mock', action='store_true', default=True, help='Mock external dependencies')
    parser.add_argument('--no-mock', action='store_false', dest='mock', help='Do not mock external dependencies')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--k', metavar='KEYWORD', type=str, help='Filter tests by keyword')
    parser.add_argument('--no-warnings', action='store_true', help='Hide warnings')
    
    args = parser.parse_args()
    
    # Set environment variables for test configuration
    os.environ['SKIP_MODEL_TESTS'] = 'true' if args.mock else 'false'
    os.environ['SKIP_E2E_TESTS'] = 'true' if args.mock else 'false'
    os.environ['SKIP_PERF_TESTS'] = 'true' if args.mock else 'false'
    
    # Build command
    cmd = [sys.executable, '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
    
    if args.no_warnings:
        cmd.append('-W ignore')
    
    # Add test selection
    if args.k:
        cmd.extend(['-k', args.k])
    elif args.unit:
        cmd.extend(['tests/test_text_insertion.py', 'tests/test_distil_whisper.py'])
    elif args.integration:
        cmd.extend(['tests/test_integration.py', 'tests/test_audio_vad_integration.py'])
    elif args.error:
        cmd.extend(['tests/test_error_handling.py'])
    elif args.text_insertion:
        cmd.extend(['tests/test_text_insertion.py'])
    elif args.model:
        cmd.extend(['tests/test_distil_whisper.py', 'tests/test_faster_whisper.py'])
    elif args.performance:
        cmd.extend(['tests/performance_test.py'])
    elif args.system:
        cmd.extend(['tests/system_test.py'])
    else:
        # Default to all
        cmd.append('tests/')
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    print(f"Mock settings: SKIP_MODEL_TESTS={os.environ.get('SKIP_MODEL_TESTS')}")
    
    # Run command
    result = subprocess.run(cmd)
    
    # Return exit code
    return result.returncode

if __name__ == '__main__':
    sys.exit(main()) 