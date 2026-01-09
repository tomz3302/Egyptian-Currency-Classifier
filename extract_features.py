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
import cv2
import time
from scipy.stats import skew, kurtosis

# ========== CONFIGURATION ==========
# Set your base directory path here
BASE_DIR = "C:/Users/user/Desktop/ML Project/dataset_cleaned_frontback_split"
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

def crop_banknote(img_array, image_path):
    # ========== AUTO-CROP BANKNOTE REGION ==========
        h, w = img_array.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find brightest regions (likely background)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for banknote-sized contours
        banknote_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            
            # Banknote criteria:
            # 1. Reasonable size (not too small, not entire image)
            # 2. Aspect ratio ~1.5-2.5 (banknote shape)
            # 3. Not too close to edges (background usually at edges)
            if (area > w*h*0.05 and area < w*h*0.8 and 
                1.2 < aspect < 3.0 and
                x > w*0.05 and y > h*0.05 and
                x + cw < w*0.95 and y + ch < h*0.95):
                
                if area > max_area:
                    max_area = area
                    banknote_contour = contour
        
        cropped_img_array = img_array
        was_cropped = False

        # If found, crop to banknote
        if banknote_contour is not None:
            x, y, cw, ch = cv2.boundingRect(banknote_contour)
            # Add 10% padding
            pad_x = int(cw * 0.1)
            pad_y = int(ch * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            
            was_cropped = True
            cropped_img_array = img_array[y1:y2, x1:x2]
            print(f"  ✓ Cropped: {image_path}: {w}x{h} → {cropped_img_array.shape[1]}x{cropped_img_array.shape[0]}")

        # ========== SAVE CROPPED IMAGE TO FOLDER ==========
        if was_cropped:
            # Create cropped images folder
            cropped_dir = "cropped_banknotes"
            os.makedirs(cropped_dir, exist_ok=True)
            
            # Get class from path (path/to/train/20/filename.jpg)
            path_parts = image_path.split(os.sep)
            class_name = "unknown"
            if len(path_parts) >= 2:
                class_name = path_parts[-2]
            
            # Create unique filename with class and original name
            original_name = os.path.splitext(os.path.basename(image_path))[0]
            cropped_filename = f"{class_name}_{original_name}_cropped.jpg"
            cropped_path = os.path.join(cropped_dir, cropped_filename)
            
            # Save cropped image
            cropped_img_pil = Image.fromarray(cropped_img_array)
            cropped_img_pil.save(cropped_path)
            
            # Also save comparison image (original vs cropped)
            save_comparison_image(img_array, cropped_img_array, image_path, class_name)

        return cropped_img_array

def extract_features(image_path):
    """
    Extract HSV color histogram features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary of extracted HSV features or None if error occurs
    """
    try:
        # Load image
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
        
        # ========== BANKNOTE CROPPING ==========
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        cropped_bgr = crop_banknote(img_bgr, image_path)

        # Convert back to RGB for feature extraction
        cropped_img_array = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

        # Convert from RGB to HSV
        img_hsv = cv2.cvtColor(cropped_img_array, cv2.COLOR_RGB2HSV)
        
        if len(cropped_img_array.shape) == 3:
            height, width, channels = cropped_img_array.shape
        else:
            height, width = cropped_img_array.shape
            channels = 1

        features = {}
        
        cropped_img_array = cv2.resize(cropped_img_array, (512, 256))
        img_hsv = cv2.resize(img_hsv, (512, 256))


        # ========== 1. HISTOGRAM FEATURES ==========
        # Calculate histograms for H, S, and V channels
        # Hue has a range of 0-180 in OpenCV
        h_hist = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist / np.sum(h_hist) if np.sum(h_hist) > 0 else h_hist
        s_hist = s_hist / np.sum(s_hist) if np.sum(s_hist) > 0 else s_hist
        v_hist = v_hist / np.sum(v_hist) if np.sum(v_hist) > 0 else v_hist
        
        # Flatten histograms and add to features dictionary
        for i in range(32):
            features[f'h_hist_bin_{i}'] = float(h_hist[i][0])
            features[f's_hist_bin_{i}'] = float(s_hist[i][0])
            features[f'v_hist_bin_{i}'] = float(v_hist[i][0])
        
        # ========== 2. CHANNEL STATISTICS ==========
        # Calculate statistical measures for each channel
        h_channel = img_hsv[:, :, 0].flatten()
        s_channel = img_hsv[:, :, 1].flatten()
        v_channel = img_hsv[:, :, 2].flatten()
        
        features['hue_mean'] = np.mean(h_channel)
        features['hue_std'] = np.std(h_channel)
        features['hue_median'] = np.median(h_channel)
        
        features['saturation_mean'] = np.mean(s_channel)
        features['saturation_std'] = np.std(s_channel)
        features['saturation_median'] = np.median(s_channel)
        
        features['value_mean'] = np.mean(v_channel)
        features['value_std'] = np.std(v_channel)
        features['value_median'] = np.median(v_channel)
        
        # ========== 3. COLOR DOMINANCE RATIOS ==========
        features['hs_ratio'] = features['hue_mean'] / (features['saturation_mean'] + 1e-6)
        features['hv_ratio'] = features['hue_mean'] / (features['value_mean'] + 1e-6)
        features['sv_ratio'] = features['saturation_mean'] / (features['value_mean'] + 1e-6)
        
        # ========== 4. HISTOGRAM STATISTICS ==========
        # Dominant color bin indices
        features['hue_dominant_bin'] = np.argmax(h_hist)
        features['saturation_dominant_bin'] = np.argmax(s_hist)
        features['value_dominant_bin'] = np.argmax(v_hist)
        
        # ========== 5. COLOR STATISTICS (if color image) ==========
        if channels == 3:
            # Extract red, green, blue channels
            red = cropped_img_array[:, :, 0].flatten()
            green = cropped_img_array[:, :, 1].flatten()
            blue = cropped_img_array[:, :, 2].flatten()
            
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

        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save_comparison_image(original, cropped, image_path, class_name):
    """
    Save a side-by-side comparison of original vs cropped images.
    """
    try:
        comparison_dir = "crop_comparisons"
        os.makedirs(comparison_dir, exist_ok=True)
        
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Create comparison image
        original_h, original_w = original.shape[:2]
        cropped_h, cropped_w = cropped.shape[:2]
        
        # Resize cropped for display if needed
        display_h = min(300, cropped_h)
        scale = display_h / cropped_h
        display_w = int(cropped_w * scale)
        cropped_display = cv2.resize(cropped, (display_w, display_h))
        
        # Resize original for display
        orig_display_h = min(300, original_h)
        orig_scale = orig_display_h / original_h
        orig_display_w = int(original_w * orig_scale)
        original_display = cv2.resize(original, (orig_display_w, orig_display_h))
        
        # Create white background
        total_height = max(orig_display_h, display_h) + 50  # Extra space for labels
        total_width = orig_display_w + display_w + 30  # Gap between images
        
        comparison = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place original image
        comparison[:orig_display_h, :orig_display_w] = original_display
        
        # Place cropped image
        comparison[:display_h, orig_display_w+30:orig_display_w+30+display_w] = cropped_display
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, orig_display_h + 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, f"{original_w}x{original_h}", (10, orig_display_h + 50), 
                   font, 0.5, (0, 0, 0), 1)
        
        cv2.putText(comparison, "Cropped", (orig_display_w + 40, display_h + 30), 
                   font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, f"{cropped_w}x{cropped_h}", (orig_display_w + 40, display_h + 50), 
                   font, 0.5, (0, 0, 0), 1)
        
        # Add class label
        cv2.putText(comparison, f"Class: {class_name}", (total_width//2 - 50, total_height - 10), 
                   font, 0.5, (0, 0, 0), 1)
        
        # Save comparison
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        comparison_filename = f"{class_name}_{original_name}_comparison.jpg"
        comparison_path = os.path.join(comparison_dir, comparison_filename)
        
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Could not save comparison image: {e}")


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
            # image_paths.extend(list(Path(class_path).glob(f"*{ext.upper()}")))
        
        # Limit to maximum number of images
        if len(image_paths) > max_images_per_class:
            # Randomly sample if there are too many images
            import random
            image_paths = random.sample(image_paths, max_images_per_class)
            print(f"    Randomly selected {max_images_per_class} out of {len(image_paths) + (max_images_per_class - len(image_paths))} images")
        
        # Process each image
        processed_count = 0
        for img_path in image_paths:
            features = extract_features(str(img_path))
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