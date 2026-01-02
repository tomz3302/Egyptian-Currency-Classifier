"""
Egyptian Banknote Feature Extraction Script
This script extracts simple statistical features from banknote images
It processes train, validation, and test folders separately.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# ========== CONFIGURATION ==========
# Set your base directory path here
BASE_DIR = "C:/Users/user/Desktop/ML Project/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Maximum number of images to process per class (to save time)
MAX_IMAGES_PER_CLASS = 100000

# Output file names
TRAIN_OUTPUT = "banknote_features_train.csv"
VALID_OUTPUT = "banknote_features_valid.csv"
TEST_OUTPUT = "banknote_features_test.csv"
# ====================================

def extract_simple_features(image_path):
    """
    Extract simple statistical features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary of extracted features or None if error occurs
    """
    try:
        # Load image
        with Image.open(image_path) as img:
            img_array = np.array(img)
        
        # Get image dimensions and channels
        if len(img_array.shape) == 3:
            height, width, channels = img_array.shape
        else:
            height, width = img_array.shape
            channels = 1
            
        features = {}
        
        # ========== 1. BASIC DIMENSION FEATURES ==========
        features['width'] = width
        features['height'] = height
        features['aspect_ratio'] = width / height if height > 0 else 0
        features['total_pixels'] = width * height
        
        # ========== 2. COLOR STATISTICS (if color image) ==========
        if channels == 3:
            # Extract red, green, blue channels
            red = img_array[:, :, 0].flatten()
            green = img_array[:, :, 1].flatten()
            blue = img_array[:, :, 2].flatten()
            
            # Calculate statistical measures for each channel
            features['red_mean'] = np.mean(red)
            features['red_std'] = np.std(red)
            features['red_median'] = np.median(red)
            
            features['green_mean'] = np.mean(green)
            features['green_std'] = np.std(green)
            features['green_median'] = np.median(green)
            
            features['blue_mean'] = np.mean(blue)
            features['blue_std'] = np.std(blue)
            features['blue_median'] = np.median(blue)
            
            # Calculate color ratios (useful for distinguishing banknote colors)
            features['rg_ratio'] = features['red_mean'] / (features['green_mean'] + 1e-6)
            features['rb_ratio'] = features['red_mean'] / (features['blue_mean'] + 1e-6)
            features['gb_ratio'] = features['green_mean'] / (features['blue_mean'] + 1e-6)
        
        # ========== 3. BRIGHTNESS AND CONTRAST FEATURES ==========
        # Convert to grayscale for brightness analysis
        if channels == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['brightness_min'] = np.min(gray)
        features['brightness_max'] = np.max(gray)
        
        # ========== 4. BACKGROUND ESTIMATION FEATURES ==========
        # Egyptian banknotes have colorful backgrounds, not pure white
        white_threshold = 220  # Pixel value threshold for "white"
        non_white = np.sum(gray < white_threshold)
        features['non_white_pct'] = non_white / gray.size
        
        # ========== 5. TILT ESTIMATION FEATURES ==========
        # Simple corner analysis to detect if image is tilted
        h, w = gray.shape
        if h > 20 and w > 20:
            corner_size = 20
            corners = [
                gray[:corner_size, :corner_size],          # top-left
                gray[:corner_size, -corner_size:],         # top-right
                gray[-corner_size:, :corner_size],         # bottom-left
                gray[-corner_size:, -corner_size:]         # bottom-right
            ]
            corner_means = [np.mean(c) for c in corners]
            features['corner_variance'] = np.var(corner_means)
        else:
            features['corner_variance'] = 0
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_folder(folder_path, max_images_per_class):
    """
    Process all images in a folder and extract features.
    
    Args:
        folder_path: Path to the folder containing class subfolders
        max_images_per_class: Maximum number of images to process per class
        
    Returns:
        List of dictionaries containing features for all processed images
    """
    all_features = []
    
    # Supported image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.jfif']
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return all_features
    
    # Get list of class folders (subdirectories)
    class_folders = [f for f in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, f))]
    
    if not class_folders:
        print(f"No class folders found in {folder_path}")
        return all_features
    
    print(f"Found {len(class_folders)} classes in {folder_path}")
    
    # Process each class folder
    for class_name in class_folders:
        class_path = os.path.join(folder_path, class_name)
        print(f"  Processing class: {class_name}")
        
        # Collect all image paths in this class folder
        image_paths = []
        for ext in image_extensions:
            # Look for files with this extension (case-insensitive)
            image_paths.extend(list(Path(class_path).glob(f"*{ext}")))
            image_paths.extend(list(Path(class_path).glob(f"*{ext.upper()}")))
        
        # Limit to maximum number of images
        if len(image_paths) > max_images_per_class:
            # Randomly sample if there are too many images
            import random
            image_paths = random.sample(image_paths, max_images_per_class)
            print(f"    Randomly selected {max_images_per_class} out of {len(image_paths) + (max_images_per_class - len(image_paths))} images")
        
        # Process each image
        processed_count = 0
        for img_path in image_paths:
            features = extract_simple_features(str(img_path))
            if features:
                features['label'] = class_name
                features['filename'] = img_path.name
                all_features.append(features)
                processed_count += 1
        
        print(f"    Extracted features from {processed_count} images")
    
    return all_features


def main():
    """
    Main function to orchestrate feature extraction from all folders.
    """
    print("=" * 70)
    print("EGYPTIAN BANKNOTE FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print(f"Max images per class: {MAX_IMAGES_PER_CLASS}")
    print()
    
    # ========== PROCESS TRAINING DATA ==========
    print("1. PROCESSING TRAINING DATA")
    print("-" * 40)
    train_features = process_folder(TRAIN_DIR, MAX_IMAGES_PER_CLASS)
    
    if train_features:
        train_df = pd.DataFrame(train_features)
        train_df.to_csv(TRAIN_OUTPUT, index=False)
        print(f"✅ Saved {len(train_df)} training samples to {TRAIN_OUTPUT}")
    else:
        print("❌ No training features extracted")
    
    print()
    
    # ========== PROCESS VALIDATION DATA ==========
    print("2. PROCESSING VALIDATION DATA")
    print("-" * 40)
    valid_features = process_folder(VALID_DIR, MAX_IMAGES_PER_CLASS)
    
    if valid_features:
        valid_df = pd.DataFrame(valid_features)
        valid_df.to_csv(VALID_OUTPUT, index=False)
        print(f"✅ Saved {len(valid_df)} validation samples to {VALID_OUTPUT}")
    else:
        print("❌ No validation features extracted")
    
    print()
    
    # ========== PROCESS TEST DATA ==========
    print("3. PROCESSING TEST DATA")
    print("-" * 40)
    test_features = process_folder(TEST_DIR, MAX_IMAGES_PER_CLASS)
    
    if test_features:
        test_df = pd.DataFrame(test_features)
        test_df.to_csv(TEST_OUTPUT, index=False)
        print(f"✅ Saved {len(test_df)} test samples to {TEST_OUTPUT}")
    else:
        print("❌ No test features extracted")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    
    if train_features:
        print(f"Training samples: {len(train_features)}")
        print(f"Training features per image: {len(train_features[0]) - 2}")  # minus label and filename
        
    if valid_features:
        print(f"Validation samples: {len(valid_features)}")
        
    if test_features:
        print(f"Test samples: {len(test_features)}")
    
    print("\n✅ Feature extraction complete!")


if __name__ == "__main__":
    main()