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

# ========== CONFIGURATION ==========
# Set your base directory path here
BASE_DIR = "C:/Users/user/Desktop/ML Project/dataset_split"
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


def extract_simple_features_openCV(image_path):
    """
    Extract features with tilt correction and smart cropping.
    1. Correct tilt using Hough Lines
    2. Crop to banknote
    3. Save visual comparisons to a single folder
    """
    try:
        # ========== STEP 1: LOAD IMAGE ==========
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Could not load: {image_path}")
            return None
        
        original_h, original_w = original_img.shape[:2]
        working_img = original_img.copy()
        
        # Create SINGLE output directory for all visualizations
        viz_dir = "tilt_correction_comparisons"
        os.makedirs(viz_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        
        # Extract class from path for better organization
        # Path format: .../train/10/filename.jpg
        path_parts = image_path.split(os.sep)
        class_name = "unknown"
        if len(path_parts) >= 2:
            class_name = path_parts[-2]  # Get folder name (class)
        
        # ========== STEP 2: DETECT TILT USING HOUGH LINES ==========
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Use Hough Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        
        tilt_angle = 0
        if lines is not None and len(lines) > 10:
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Convert theta to degrees
                angle = np.degrees(theta)
                
                # Normalize angle to -45 to 45 degrees
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                
                # Only consider near-horizontal lines (banknote edges are usually horizontal)
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                # Use median to avoid outliers
                tilt_angle = np.median(angles)
                print(f"  Detected tilt: {tilt_angle:.1f}° in {base_name}")
                
                # ========== STEP 3: ROTATE TO CORRECT TILT ==========
                if abs(tilt_angle) > 1.0:  # Only rotate if more than 1 degree
                    # Get rotation matrix
                    (h, w) = working_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)
                    
                    # Calculate new bounding dimensions
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    
                    # Adjust rotation matrix to avoid cropping
                    M[0, 2] += (new_w / 2) - center[0]
                    M[1, 2] += (new_h / 2) - center[1]
                    
                    # Perform rotation
                    working_img = cv2.warpAffine(working_img, M, (new_w, new_h), 
                                                borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=(255, 255, 255))
        
        # ========== STEP 4: DETECT BANKNOTE AFTER TILT CORRECTION ==========
        # Now find the banknote in the (possibly) rotated image
        h, w = working_img.shape[:2]
        
        # Convert to grayscale for analysis
        gray_corrected = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Find by edges (now more reliable after tilt correction)
        edges_corrected = cv2.Canny(gray_corrected, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_corrected, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        banknote_contour = None
        if contours:
            # Filter contours by size and aspect ratio
            min_area = w * h * 0.1  # At least 10% of image
            max_area = w * h * 0.95  # Not more than 95%
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    # Get bounding rectangle
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect = cw / ch if ch > 0 else 0
                    
                    # Banknote aspect ratio is typically 1.5-2.5
                    if 1.2 <= aspect <= 3.0:
                        valid_contours.append((contour, area, aspect))
            
            if valid_contours:
                # Sort by area (largest first)
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                banknote_contour = valid_contours[0][0]
        
        # ========== STEP 5: CROP THE BANKNOTE ==========
        final_img = working_img.copy()
        crop_info = None
        
        if banknote_contour is not None:
            # Get rotated rectangle (minimum area rectangle)
            rect = cv2.minAreaRect(banknote_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(banknote_contour)
            
            # Add padding (5%)
            pad_x = int(cw * 0.05)
            pad_y = int(ch * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            
            # Crop
            cropped = working_img[y1:y2, x1:x2]
            final_img = cropped
            
            crop_info = {
                'original_size': (original_w, original_h),
                'tilt_angle': tilt_angle,
                'crop_size': (cropped.shape[1], cropped.shape[0]),
                'was_cropped': True
            }
            
            print(f"  Cropped after tilt correction: {base_name} - "
                  f"{cropped.shape[1]}x{cropped.shape[0]}")
        
        # ========== STEP 6: SAVE VISUAL COMPARISON ==========
        # Create comparison image showing all steps
        comparison_height = max(original_h, h, final_img.shape[0])
        comparison_width = original_w + w + final_img.shape[1] + 20
        
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Place images side by side
        # 1. Original
        comparison[:original_h, :original_w, :] = original_img
        
        # 2. After tilt correction
        comparison[:h, original_w+10:original_w+10+w, :] = working_img
        
        # 3. Final cropped
        h_final, w_final = final_img.shape[:2]
        start_x = original_w + w + 20
        comparison[:h_final, start_x:start_x+w_final, :] = final_img
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINAL", (10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, f"{original_w}x{original_h}", 
                   (10, 60), font, 0.5, (0, 0, 0), 1)
        
        cv2.putText(comparison, "TILT CORRECTED", (original_w+20, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, f"{w}x{h}, angle={tilt_angle:.1f}°", 
                   (original_w+20, 60), font, 0.5, (0, 0, 0), 1)
        
        cv2.putText(comparison, "FINAL CROPPED", (start_x+10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, f"{w_final}x{h_final}", 
                   (start_x+10, 60), font, 0.5, (0, 0, 0), 1)
        
        # Draw contour on tilt-corrected image
        if banknote_contour is not None:
            # Draw contour on a copy of the working image
            img_with_contour = working_img.copy()
            cv2.drawContours(img_with_contour, [banknote_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(img_with_contour, [box], -1, (0, 0, 255), 2)
            
            # Replace the middle image with contour version
            comparison[:h, original_w+10:original_w+10+w, :] = img_with_contour
            
            # Add contour info
            cv2.putText(comparison, f"Contour area: {cv2.contourArea(banknote_contour):.0f}", 
                       (original_w+20, 90), font, 0.5, (0, 0, 0), 1)
        
        # Save comparison to SINGLE folder
        # Format: class_filename.jpg
        save_filename = f"{class_name}_{base_name}"
        save_path = os.path.join(viz_dir, save_filename)
        cv2.imwrite(save_path, comparison)
        
        # Also save individual steps if you want
        # Save original
        cv2.imwrite(os.path.join(viz_dir, f"{class_name}_{base_name.replace('.', '_original.')}"), original_img)
        
        # Save final cropped (useful for checking quality)
        cv2.imwrite(os.path.join(viz_dir, f"{class_name}_{base_name.replace('.', '_cropped.')}"), final_img)
        
        # ========== STEP 7: CONVERT TO RGB FOR FEATURE EXTRACTION ==========
        # Convert final image to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        
        # ========== STEP 8: EXTRACT FEATURES ==========
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
        
        # ========== 2. COLOR STATISTICS ==========
        if channels == 3:
            red = img_array[:, :, 0].flatten()
            green = img_array[:, :, 1].flatten()
            blue = img_array[:, :, 2].flatten()
            
            features['red_mean'] = np.mean(red)
            features['red_std'] = np.std(red)
            features['red_median'] = np.median(red)
            
            features['green_mean'] = np.mean(green)
            features['green_std'] = np.std(green)
            features['green_median'] = np.median(green)
            
            features['blue_mean'] = np.mean(blue)
            features['blue_std'] = np.std(blue)
            features['blue_median'] = np.median(blue)
            
            features['rg_ratio'] = features['red_mean'] / (features['green_mean'] + 1e-6)
            features['rb_ratio'] = features['red_mean'] / (features['blue_mean'] + 1e-6)
            features['gb_ratio'] = features['green_mean'] / (features['blue_mean'] + 1e-6)
        
        # ========== 3. BRIGHTNESS AND CONTRAST ==========
        if channels == 3:
            gray_features = np.mean(img_array, axis=2)
        else:
            gray_features = img_array
        
        features['brightness_mean'] = np.mean(gray_features)
        features['brightness_std'] = np.std(gray_features)
        features['brightness_min'] = np.min(gray_features)
        features['brightness_max'] = np.max(gray_features)
        
        # ========== 4. BACKGROUND ESTIMATION ==========
        white_threshold = 220
        non_white = np.sum(gray_features < white_threshold)
        features['non_white_pct'] = non_white / gray_features.size
        
        # ========== 5. TILT ESTIMATION ==========
        h_feat, w_feat = gray_features.shape
        if h_feat > 20 and w_feat > 20:
            corner_size = 20
            corners = [
                gray_features[:corner_size, :corner_size],
                gray_features[:corner_size, -corner_size:],
                gray_features[-corner_size:, :corner_size],
                gray_features[-corner_size:, -corner_size:]
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