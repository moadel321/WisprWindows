#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build script to create a standalone Windows executable for the Speech-to-Text application.
This script uses PyInstaller to package the application with all dependencies.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build a standalone executable for the Speech-to-Text application.")
    parser.add_argument("--output-dir", default="dist", help="Output directory for the executable.")
    parser.add_argument("--name", default="SpeechToText", help="Name of the executable.")
    parser.add_argument("--icon", default="src/resources/icon.ico", help="Path to the icon file.")
    parser.add_argument("--one-file", action="store_true", help="Build a single executable file.")
    parser.add_argument("--no-console", action="store_true", help="Hide the console window.")
    parser.add_argument("--include-models", action="store_true", help="Include Whisper models in the package.")
    parser.add_argument("--model-path", help="Path to the Whisper model directory to include.")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before building.")
    parser.add_argument("--debug", action="store_true", help="Build with debug information.")
    return parser.parse_args()


def ensure_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("PyInstaller is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller installed successfully.")


def ensure_icon():
    """Ensure the icon file exists and convert if needed."""
    icon_path = Path("src/resources/icon.ico")
    
    # If ICO file already exists, we're good
    if icon_path.exists():
        print(f"Using existing icon: {icon_path}")
        return str(icon_path)
    
    # Look for PNG or SVG alternatives
    png_path = Path("src/resources/icon.png")
    svg_path = Path("src/resources/icon.svg")
    
    if png_path.exists() or svg_path.exists():
        print("Icon needs to be converted to ICO format.")
        try:
            from PIL import Image
            if png_path.exists():
                img = Image.open(png_path)
                img.save(icon_path)
                print(f"Icon converted from PNG to ICO: {icon_path}")
                return str(icon_path)
        except ImportError:
            print("Pillow library not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            print("Trying conversion again...")
            return ensure_icon()
        except Exception as e:
            print(f"Failed to convert icon: {e}")
            print("Building without custom icon.")
            return None
    
    print("No icon file found. Building without custom icon.")
    return None


def clean_build_dirs():
    """Clean up build and dist directories."""
    print("Cleaning build directories...")
    
    for directory in ["build", "dist"]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed {directory} directory.")
    
    # Remove spec files
    for spec_file in Path().glob("*.spec"):
        os.remove(spec_file)
        print(f"Removed {spec_file}.")


def collect_data_files(args):
    """Collect data files to be included in the package."""
    data_files = [
        ("src/resources/*", "resources"),
        ("docs/*", "docs"),
        ("README.md", "."),
        ("CLAUDE.md", "."),
    ]
    
    if args.include_models and args.model_path:
        model_dir = Path(args.model_path)
        if model_dir.exists() and model_dir.is_dir():
            # Get the model name from the directory
            model_name = model_dir.name
            data_files.append((f"{args.model_path}/*", f"models/{model_name}"))
            print(f"Including model from {args.model_path}")
        else:
            print(f"Warning: Model path {args.model_path} not found or is not a directory.")
    
    return data_files


def build_executable(args):
    """Build the executable with PyInstaller."""
    # Ensure PyInstaller is installed
    ensure_pyinstaller()
    
    # Clean build directories if requested
    if args.clean:
        clean_build_dirs()
    
    # Ensure icon exists
    icon_path = ensure_icon()
    
    # Collect data files
    data_files = collect_data_files(args)
    
    # Create PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", args.name,
        "--distpath", args.output_dir,
    ]
    
    # Add icon if available
    if icon_path:
        cmd.extend(["--icon", icon_path])
    
    # Add mode options
    if args.one_file:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    if args.no_console:
        cmd.append("--windowed")
    
    # Add debug option
    if args.debug:
        cmd.append("--debug=all")
    
    # Add data files
    for src, dst in data_files:
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # Add hidden imports for VAD and GUI
    cmd.extend([
        "--hidden-import", "torch",
        "--hidden-import", "numpy",
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtGui", 
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "faster_whisper",
        "--hidden-import", "ctranslate2",
        "--hidden-import", "pywinauto",
        "--hidden-import", "win32clipboard",
    ])
    
    # Add main script
    cmd.append("run.py")
    
    # Run PyInstaller
    print("Running PyInstaller with command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    
    print(f"\nBuild completed successfully!\nExecutable saved to: {os.path.join(args.output_dir, args.name)}")


if __name__ == "__main__":
    args = parse_args()
    build_executable(args)