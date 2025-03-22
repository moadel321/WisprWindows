#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate a simple PNG icon
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_icon(output_path, size=(256, 256)):
    """
    Create a simple icon for the application
    
    Args:
        output_path: Path to save the icon
        size: Size of the icon
    """
    # Create a new image with a blue background
    img = Image.new('RGBA', size, (60, 120, 216, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a white circle in the center (microphone)
    center_x, center_y = size[0] // 2, size[1] // 2 - 20
    radius = 40
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=(255, 255, 255, 255)
    )
    
    # Draw microphone stand
    stand_width = 14
    stand_height = 40
    draw.rectangle(
        (center_x - stand_width // 2, center_y + radius, 
         center_x + stand_width // 2, center_y + radius + stand_height),
        fill=(255, 255, 255, 255)
    )
    
    # Draw microphone base
    base_width = 80
    base_height = 10
    draw.rounded_rectangle(
        (center_x - base_width // 2, center_y + radius + stand_height,
         center_x + base_width // 2, center_y + radius + stand_height + base_height),
        radius=5,
        fill=(255, 255, 255, 255)
    )
    
    # Draw text at the bottom
    try:
        # Try to use Arial font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = "Speech-to-Text"
    text_width = draw.textlength(text, font=font)
    draw.text(
        (center_x - text_width // 2, size[1] - 60),
        text,
        font=font,
        fill=(255, 255, 255, 255)
    )
    
    # Save the image
    img.save(output_path)
    print(f"Icon saved to: {output_path}")

def main():
    """Main function"""
    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set output path
    output_path = os.path.join(script_dir, 'icon.png')
    
    # Create the icon
    create_icon(output_path)
    print("Icon created successfully!")

if __name__ == "__main__":
    main() 