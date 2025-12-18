# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 
"""
run this script to generate augmented frames for the appendix
    python -m videoseal.augmentation.doaugs
"""

import os
import re
from collections import defaultdict

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from videoseal.augmentation.geometric import (
    Crop, HorizontalFlip, Identity, Perspective, Resize, Rotate
)
from videoseal.augmentation.valuemetric import (
    JPEG, Brightness, Contrast, GaussianBlur, Hue, MedianFilter, Saturation
)
from videoseal.augmentation.video import (
    H264, H265, H264rgb
)
from videoseal.augmentation.sequential import Sequential
from videoseal.data.loader import load_video
from videoseal.data.transforms import default_transform


def main():
    # Define the video path
    video_path = "assets/videos/1.mp4"
    
    # Define the output directory
    output_dir = "figs/appendix/augs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the video
    frames = load_video(video_path)[:48]
    
    # Define the transformations with their strengths
    transformations = [
        # Existing transformations
        (Identity(), None, "Identity"),
        (Contrast(), 0.5, "Contrast_strength_0.5"),
        (Contrast(), 1.5, "Contrast_strength_1.5"),
        (Brightness(), 0.5, "Brightness_strength_0.5"),
        (Brightness(), 1.5, "Brightness_strength_1.5"),
        (Hue(), -0.1, "Hue_strength_-0.1"),
        (Saturation(), 1.5, "Saturation_strength_1.5"),
        (GaussianBlur(), 17, "GaussianBlur_strength_17"),
        (JPEG(), 40, "JPEG_strength_40"),
        (Crop(), 0.33, "Crop_strength_0.33"),
        (Resize(), 0.5, "Resize_strength_0.5"),
        (Rotate(), 10, "Rotate_strength_10"),
        (Rotate(), 90, "Rotate_strength_90"),
        (Perspective(), 0.5, "Perspective_strength_0.5"),
        (HorizontalFlip(), None, "HorizontalFlip"),
        
        # Video transformations
        (H264(min_crf=40, max_crf=40), None, "H264_strength_40"),
        (H265(min_crf=40, max_crf=40), None, "H265_strength_40"),
        (H264rgb(min_crf=40, max_crf=40), None, "H264rgb_strength_40"),
        
        # Combined augmentation
        (Sequential(
            H264(min_crf=30, max_crf=30), 
            Brightness(), 
            Crop()
        ), (None, 0.5, 0.71), "Combined_H264_30_Brightness_0.5_Crop_0.71")
    ]
    
    # Apply each transformation to the video and save middle frame
    for transform, strength, filename in transformations:
        print(f"Applying {filename}...")
        
        # Clone the frames for each transformation
        frames_copy = frames.clone()
        mask = torch.ones_like(frames_copy)[:, 0:1, :, :]  # Create dummy mask
        
        try:
            # Apply the transformation
            if isinstance(transform, Sequential):
                augmented_frames, _ = transform(frames_copy, mask, strength)
            elif strength is None:
                augmented_frames, _ = transform(frames_copy, mask)
            else:
                augmented_frames, _ = transform(frames_copy, mask, strength)
            
            # Get middle frame
            middle_idx = len(augmented_frames[0]) // 2
            augmented_frame = augmented_frames[middle_idx]
            
            # Clamp values
            augmented_frame = torch.clamp(augmented_frame, 0, 1)
            
            # Save as JPEG for the table
            output_path = os.path.join(output_dir, f"{filename}.jpg")
            pil_img = to_pil_image(augmented_frame.squeeze(0))
            pil_img.save(output_path, quality=80)
            pil_img = pil_img.resize((pil_img.width // 2, pil_img.height // 2))
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error applying {filename}: {e}")
            
    print("Done!")

def get_augmentation_files(directory):
    """Get all augmentation image files in the directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found. Run doaugs.py first.")
    
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    return sorted(files)

def parse_filename(filename):
    """Parse the filename to get augmentation type and strength."""
    if filename == "Identity.jpg":
        return "Identity", None
    elif filename == "HorizontalFlip.jpg":
        return "Horizontal flipping", None
    elif "Combined" in filename:
        return "Combined Aug", None
    else:
        # Extract the base augmentation name and strength
        match = re.match(r"([A-Za-z0-9]+)_strength_([\-0-9\.]+)\.jpg", filename)
        if match:
            aug_name, strength = match.groups()
            
            # Format augmentation name for display
            name_map = {
                "Rotate": "Rotation",
                "MedianFilter": "Median filter",
                "GaussianBlur": "Gaussian blur"
            }
            
            display_name = name_map.get(aug_name, aug_name)
            return display_name, strength
        
        return filename.replace('.jpg', ''), None

def generate_table(files, output_file, cols_per_row=5):
    """Generate the LaTeX table code."""
    # Group files into rows
    rows = []
    current_row = []
    
    # Dictionary to store specific ordering preferences
    # Lower values appear first
    ordering_preferences = {
        'Identity': 0,
        'Horizontal flipping': 10,
        'Crop': 20,
        'Rotation': 30,
        'Perspective': 40,
        'Contrast': 50,
        'Brightness': 60,
        'Hue': 70,
        'Saturation': 80,
        'Median filter': 90,
        'Gaussian blur': 100,
        'Resize': 110,
        'JPEG': 120,
        'H264': 130,
        'H265': 140,
        'H264rgb': 150,
        'Combined Aug': 200,
    }
    
    # Parse all files and sort them
    augmentations = []
    for file in files:
        aug_name, strength = parse_filename(file)
        base_name = aug_name.split(' ')[0]
        order_key = ordering_preferences.get(base_name, 999)
        
        # For same type of augmentations, sort by strength
        if strength is not None:
            try:
                strength_value = float(strength)
                sort_key = (order_key, strength_value)
            except ValueError:
                sort_key = (order_key, strength)
        else:
            sort_key = (order_key, 0)
        
        augmentations.append((aug_name, strength, file, sort_key))
    
    # Sort augmentations by order key
    augmentations.sort(key=lambda x: x[3])
    
    # Group into rows of cols_per_row
    rows = []
    current_row = []
    current_row_labels = []
    
    for aug_name, strength, file, _ in augmentations:
        # Create label for this augmentation
        if strength is not None:
            label = f"{aug_name} {strength}"
        else:
            label = aug_name
        
        current_row.append(file)
        current_row_labels.append(label)
        
        if len(current_row) == cols_per_row:
            rows.append((current_row_labels, current_row))
            current_row = []
            current_row_labels = []
    
    # Add any remaining items as the last row
    if current_row:
        rows.append((current_row_labels, current_row))
    
    # Generate the LaTeX code
    table_code = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  % \\captionsetup{font=small}",
        "  \\caption{Illustration of transformations on which we train and evaluate the models.}",
        "  \\label{fig:app-augs}",
        "  \\scriptsize",
        "  \\begin{tabular}{*{" + str(cols_per_row) + "}{l}}",
    ]
    
    for row_idx, (row_labels, row_files) in enumerate(rows):
        # Add the row labels
        table_code.append("       " + " & ".join(row_labels) + " \\\\")
        
        # Add the images
        image_row = []
        for file in row_files:
            image_code = f"\\begin{{minipage}}{{.16\\linewidth}}\\includegraphics[width=\\linewidth]{{figs/appendix/augs/{file}}}\\end{{minipage}}"
            image_row.append(image_code)
        
        table_code.append("       " + " &  \n       ".join(image_row))
        
        # Add spacing between rows
        if row_idx < len(rows) - 1:
            table_code.append("       \\\\ \\\\")
        else:
            table_code.append("       \\\\")
    
    # Close the table
    table_code.extend([
        "  \\end{tabular}",
        "\\end{table*}"
    ])
    
    # Write the table to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(table_code))
    
    print(f"LaTeX table code written to {output_file}")
    return "\n".join(table_code)

def main_table():
    # Directory containing the augmented frames
    img_dir = "figs/appendix/augs"
    
    # Output file for the LaTeX table
    output_file = "figs/appendix/augs/aug_table.tex"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all augmentation files
    files = get_augmentation_files(img_dir)
    
    # Generate the table
    table_code = generate_table(files, output_file)
    
    # Also print the table code
    print("\nGenerated LaTeX Table Code:")
    print(table_code)

if __name__ == "__main__":
    main()
    main_table()
