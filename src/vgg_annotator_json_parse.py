#!/usr/bin/env python3
"""
Clean script to parse Via annotations JSON (polygon regions) into binary (B&W) 
bounding box masks for each image.

- Computes axis-aligned bounding box (AABB) for each polygon region.
- Creates a single combined B&W mask per image (255=white for BB regions, 0=black elsewhere).
- Masks are saved as PNG files (lossless) with same dimensions as originals: 512x640 (HxW).
- Assumes all images are exactly 512 height x 640 width.
- Multiple regions per image are unioned (overlaps handled automatically).
- Requires: numpy, matplotlib (pip install numpy matplotlib if needed).
- Highly readable: modular functions, clear variable names, comprehensive comments.

USAGE:
1. Save your full JSON annotations to 'annotations.json' (exactly as provided).
2. Run: python this_script.py
3. Outputs: IRX_XXXX_mask.png files next to script.

Author: Experienced SWE | Clean, commented, production-ready code.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


# Constants for clarity & easy override
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 640
MASK_VALUE_FILLED = 255  # White for solar panel regions
MASK_VALUE_BG = 0        # Black background
MASK_DTYPE = np.uint8
OUTPUT_EXT = '_mask.png'


def load_annotations(json_file_path: str) -> dict:
    """
    Load JSON annotations from file.
    
    Args:
        json_file_path: Path to annotations.json
    
    Returns:
        Dict of {img_key: img_info} where img_info has 'filename' & 'regions'.
    
    Raises:
        json.JSONDecodeError: If malformed JSON.
        FileNotFoundError: If file missing.
    """
    with open(json_file_path, 'r') as f:
        return json.load(f)


def compute_bbox(points_x: list[int], points_y: list[int]) -> tuple[int, int, int, int]:
    """
    Compute tight axis-aligned bounding box from polygon points.
    
    Args:
        points_x: List of X coords.
        points_y: List of Y coords.
    
    Returns:
        (min_x, max_x, min_y, max_y) – inclusive bounds.
    """
    min_x, max_x = min(points_x), max(points_x)
    min_y, max_y = min(points_y), max(points_y)
    return min_x, max_x, min_y, max_y


def create_mask(height: int, width: int, regions: list[dict]) -> np.ndarray:
    """
    Create binary mask by filling BB of all regions.
    
    Args:
        height: Image height (rows).
        width: Image width (cols).
        regions: List of region dicts from JSON.
    
    Returns:
        HxW uint8 numpy array (0=BG, 255=foreground).
    """
    mask = np.zeros((height, width), dtype=MASK_DTYPE)
    
    for region in regions:
        shape = region['shape_attributes']
        if shape['name'] != 'polygon':
            print(f"Warning: Skipping non-polygon region.")
            continue
        
        points_x = shape['all_points_x']
        points_y = shape['all_points_y']
        
        if len(points_x) < 3 or len(points_y) < 3:
            print(f"Warning: Skipping invalid polygon (too few points).")
            continue
        
        # Compute & fill BB
        min_x, max_x, min_y, max_y = compute_bbox(points_x, points_y)
        
        # Clamp to image bounds (safety)
        min_y = max(0, min_y)
        max_y = min(height - 1, max_y)
        min_x = max(0, min_x)
        max_x = min(width - 1, max_x)
        
        # Fill rectangle (vectorized, fast)
        mask[min_y:max_y+1, min_x:max_x+1] = MASK_VALUE_FILLED
    
    return mask


def save_mask(mask: np.ndarray, output_path: str):
    """
    Save mask as grayscale PNG using matplotlib (no compression artifacts).
    
    Args:
        mask: HxW uint8 numpy array.
        output_path: Full path to save.
    """
    plt.imsave(output_path, mask, cmap='gray', vmin=0, vmax=255)
    print(f"✓ Saved: {output_path} ({mask.shape[0]}x{mask.shape[1]}, {np.sum(mask > 0)} pixels filled)")


def main(json_file_path: str = 'annotations.json'):
    """Main orchestrator – process all images."""
    print("🚀 Parsing annotations → B&W BB masks...")
    
    output_dir = os.path.dirname(json_path)
    data = load_annotations(json_file_path)
    
    for img_key, img_info in data.items():
        filename = img_info['filename']
        
        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = os.path.join(output_dir, f"{base_name}{OUTPUT_EXT}")
        
        # Create mask
        mask = create_mask(IMAGE_HEIGHT, IMAGE_WIDTH, img_info['regions'])
        
        # Save
        if np.sum(mask > 0):
            save_mask(mask, output_filename)
    
    print("\n🎉 All masks generated successfully!")


if __name__ == '__main__':
    json_path = r"D:\S\ANTS\Data\annotations\from_ivan\via_project_29Oct2025_11h18m_json.json"
    output_dir = os.path.dirname(json_path)
    main(json_file_path=json_path)