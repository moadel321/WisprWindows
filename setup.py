#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for the Speech-to-Text application.
This allows installing the application as a package and helps with dependency management.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from constants
version = "0.1.0"  # Default version
constants_path = os.path.join("src", "utils", "constants.py")
if os.path.exists(constants_path):
    with open(constants_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("APP_VERSION"):
                version = line.split("=")[1].strip().strip('"\'')
                break

# Define dependencies
install_requires = [
    "numpy>=1.20.0",
    "torch>=2.0.0",
    "torchaudio",
    "PyQt6>=6.4.0",
    "pyaudio>=0.2.11",
    "faster-whisper>=0.9.0",
    "ctranslate2>=3.17.1",
    "huggingface-hub",
    "pywinauto>=0.6.8",
    "pywin32>=301; platform_system=='Windows'",
    "pillow>=9.0.0",
]

# Development dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "black>=21.5b2",
        "flake8>=3.9.2",
        "isort>=5.9.1",
        "mypy>=0.812",
        "PyInstaller>=5.0.0",
    ],
    "cuda": [
        "nvidia-cudnn-cu11",
        "nvidia-cuda-runtime-cu11",
        "nvidia-cublas-cu11",
    ],
}

setup(
    name="speech-to-text-tool",
    version=version,
    author="Speech-to-Text Team",
    author_email="your_email@example.com",
    description="A Windows application for real-time speech-to-text transcription",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/speech-to-text-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Environment :: Win32 (MS Windows)",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "speech-to-text=src.main:main",
        ],
        "gui_scripts": [
            "speech-to-text-gui=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["resources/*"],
    },
)