#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert SVG icon to PNG
"""

import os
import sys
import io
import cairosvg
from PIL import Image

def convert_svg_to_png(svg_path, png_path, size=(256, 256)):
    """
    Convert SVG file to PNG
    
    Args:
        svg_path: Path to SVG file
        png_path: Path to output PNG file
        size: Size of the output PNG
    """
    print(f"Converting {svg_path} to {png_path}")
    
    # Read SVG file
    with open(svg_path, 'rb') as f:
        svg_data = f.read()
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_data, output_width=size[0], output_height=size[1])
    
    # Save PNG
    with open(png_path, 'wb') as f:
        f.write(png_data)
    
    print(f"Saved PNG: {png_path}")

def main():
    """Main function"""
    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths
    svg_path = os.path.join(script_dir, 'icon.svg')
    png_path = os.path.join(script_dir, 'icon.png')
    
    # Check if SVG exists
    if not os.path.exists(svg_path):
        print(f"Error: SVG file not found: {svg_path}")
        sys.exit(1)
    
    # Convert SVG to PNG
    try:
        convert_svg_to_png(svg_path, png_path)
        print("Conversion successful!")
    except Exception as e:
        print(f"Error converting SVG to PNG: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 