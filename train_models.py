"""
Egyptian Banknote Classification - Model Training Script
This script trains traditional ML models on extracted features
using proper train/validation/test splits.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Traditional ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import os
import shutil
from PIL import Image
from xgboost import XGBClassifier
import cv2

# ========== CONFIGURATION ==========
BASE_DIR = "C:/Users/user/Desktop/ML Project/dataset_cleaned_frontback_split"
# File paths for extracted features
TRAIN_FEATURES_FILE = "banknote_features_train.csv"
VALID_FEATURES_FILE = "banknote_features_valid.csv"
TEST_FEATURES_FILE = "banknote_features_test.csv"

# Output files for saving models and results
BEST_MODEL_FILE = "best_banknote_model.pkl"
SCALER_FILE = "feature_scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
RESULTS_PLOT_FILE = "model_performance.png"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"

# Confidence visualization
CONFIDENCE_IMAGES_DIR = "confidence_visualization"
CONFIDENCE_SUMMARY_FILE = "confidence_summary.csv"
# ====================================

def load_and_prepare_data():
    """
    Load and prepare the train, validation, and test datasets.
    
    Returns:
        Tuple containing prepared data arrays and label encoder
    """
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load datasets
    print("\n1. Loading feature datasets...")
    try:
        train_df = pd.read_csv(TRAIN_FEATURES_FILE)
        valid_df = pd.read_csv(VALID_FEATURES_FILE)
        test_df = pd.read_csv(TEST_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run extract_features.py first to create feature files.")
        return None, None, None, None, None, None, None
    
    print(f"✅ Training set: {train_df.shape[0]} samples")
    print(f"✅ Validation set: {valid_df.shape[0]} samples")
    print(f"✅ Test set: {test_df.shape[0]} samples")
    
    # ========== CHECK FOR NAN VALUES ==========
    print("\n2. Checking for missing values...")
    
    # Check NaN in each dataset
    train_nan = train_df.isna().sum().sum()
    valid_nan = valid_df.isna().sum().sum()
    test_nan = test_df.isna().sum().sum()
    
    print(f"   Training set NaN values: {train_nan}")
    print(f"   Validation set NaN values: {valid_nan}")
    print(f"   Test set NaN values: {test_nan}")
    
    # Check which columns have NaN
    nan_columns = train_df.columns[train_df.isna().any()].tolist()
    if nan_columns:
        print(f"   Columns with NaN: {nan_columns}")
        for col in nan_columns:
            print(f"     - {col}: {train_df[col].isna().sum()} NaN values")


    # ========== PREPARE FEATURES AND LABELS ==========
    print("\n3. Preparing features and labels...")
    
    # Identify non-feature columns to exclude
    non_feature_cols = ['label', 'filename']
    
    # Extract features (X) and labels (y) for each dataset
    X_train = train_df.drop([col for col in non_feature_cols if col in train_df.columns], axis=1)
    y_train = train_df['label']
    
    X_valid = valid_df.drop([col for col in non_feature_cols if col in valid_df.columns], axis=1)
    y_valid = valid_df['label']
    
    X_test = test_df.drop([col for col in non_feature_cols if col in test_df.columns], axis=1)
    y_test = test_df['label']
    
    # Normalize labels so that a banknote's front face is labled as the back
    print("\n4. Normalizing labels...")
    def normalize_label(label):
        label = str(label).lower().strip()

        # Remove side indicators only
        label = re.sub(r'\bfront\b', '', label)
        label = re.sub(r'\bback\b', '', label)

        # Clean extra spaces
        label = re.sub(r'\s+', ' ', label).strip()

        return label

    y_train = y_train.apply(normalize_label)
    y_valid = y_valid.apply(normalize_label)
    y_test  = y_test.apply(normalize_label)

        # ========== HANDLE MISSING VALUES ==========
    print("\n5. Handling missing values...")
    
    # Create imputer (use median for robustness)
    imputer = SimpleImputer(strategy='median')
    
    # Fit on training data only, then transform all sets
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)
    X_test_imputed = imputer.transform(X_test)
    
    print(f"✅ Missing values imputed using median strategy")

    # Encode string labels to numerical values
    print("\n6. Encoding labels...")
    label_encoder = LabelEncoder()
    
    # Fit encoder on training labels, then transform all sets
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"✅ Classes: {label_encoder.classes_}")
    print(f"✅ Number of features: {X_train.shape[1]}")
    
    # ========== SCALE FEATURES ==========
    print("\n7. Scaling features...")
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_valid_scaled = scaler.transform(X_valid_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print("✅ Features scaled using StandardScaler")
    
    return (X_train_scaled, y_train_encoded, 
            X_valid_scaled, y_valid_encoded, 
            X_test_scaled, y_test_encoded, 
            label_encoder, scaler, X_train.columns)


def train_and_evaluate_models(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names):
    """
    Train multiple traditional ML models and evaluate their performance.
    
    Args:
        X_train, y_train: Training features and labels
        X_valid, y_valid: Validation features and labels
        X_test, y_test: Test features and labels
        feature_names: Names of the features
        
    Returns:
        Dictionary containing trained models and their performance metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)
    
    # Define models to train (traditional ML as per project requirements)
    models = {
        # 'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
        # 'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            reg_alpha=0.5,
            reg_lambda=5,
            early_stopping_rounds=20
        )
        # 'Naive Bayes': GaussianNB(),
        # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Store results
    results = {
        'models': {},
        'train_acc': {},
        'valid_acc': {},
        'test_acc': {},
        'feature_importances': {},
        'test_probabilities': {}  # Added to store confidence scores
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Train the model
        if name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False)
            print(f"✅ XGBoost Training Complete. Best iteration: {model.get_booster().best_iteration}")
        else:
            model.fit(X_train, y_train,)
        results['models'][name] = model
        
        # Make predictions on all datasets
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results['train_acc'][name] = train_acc
        results['valid_acc'][name] = valid_acc
        results['test_acc'][name] = test_acc
        
        # Store confidence scores (probabilities) for test set
        try:
            # Try to get probability predictions
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)
                results['test_probabilities'][name] = y_test_proba
                print(f"   Confidence scores available ✓")
            else:
                print(f"   ⚠️ No probability predictions available for confidence scores")
        except Exception as e:
            print(f"   ⚠️ Could not get confidence scores: {e}")
        
        # Print results
        print(f"   Training Accuracy:   {train_acc:.4f}")
        print(f"   Validation Accuracy: {valid_acc:.4f}")
        print(f"   Test Accuracy:       {test_acc:.4f}")
        
        # Check for overfitting
        overfit_gap = train_acc - valid_acc
        if overfit_gap > 0.15:
            print(f"   ⚠️  Possible overfitting (gap: {overfit_gap:.3f})")
        
        # Extract feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importances'][name] = model.feature_importances_
    
    return results


def save_confidence_images(model, model_name, X_test, y_test, test_df, label_encoder, 
                          base_images_dir=BASE_DIR):
    """
    Save all test images with confidence scores annotated on them.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: True test labels (encoded)
        test_df: Original test dataframe with filenames
        label_encoder: Fitted label encoder
        base_images_dir: Base directory where original images are stored
    """
    print(f"\n📊 Generating confidence visualizations for {model_name}...")
    
    # Create output directory
    model_dir = os.path.join(CONFIDENCE_IMAGES_DIR, model_name.replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Get predictions and confidence scores
    try:
        y_test_pred = model.predict(X_test)
        
        # Try to get probability predictions
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)
            confidence_scores = np.max(y_test_proba, axis=1)
            all_probabilities = y_test_proba
        else:
            # For models without probability, use decision function if available
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                # Normalize decision scores to [0, 1] using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                all_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                confidence_scores = np.max(all_probabilities, axis=1)
            else:
                # Fallback: use 1.0 for correct predictions, 0.0 for incorrect
                confidence_scores = np.where(y_test_pred == y_test, 1.0, 0.0)
                all_probabilities = None
        
        # Decode labels
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
        
        # Store results for summary
        results_data = []
        
        # Process each test image
        for i in range(len(y_test)):
            if i < len(test_df):
                filename = test_df.iloc[i]['filename']
                original_label = str(test_df.iloc[i]['label'])
                
                # Get confidence score
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
                
                # Get top 3 predictions if probabilities are available
                top_predictions = []
                if all_probabilities is not None and i < len(all_probabilities):
                    probs = all_probabilities[i]
                    top_indices = np.argsort(probs)[-3:][::-1]
                    for idx in top_indices:
                        top_predictions.append({
                            'class': label_encoder.inverse_transform([idx])[0],
                            'probability': probs[idx]
                        })
                
                # Store for summary
                results_data.append({
                    'filename': filename,
                    'true_label': y_test_decoded[i],
                    'predicted_label': y_test_pred_decoded[i],
                    'confidence_score': confidence,
                    'is_correct': y_test_pred[i] == y_test[i],
                    'top_prediction_1': top_predictions[0]['class'] if top_predictions else '',
                    'top_probability_1': top_predictions[0]['probability'] if top_predictions else 0.0,
                    'top_prediction_2': top_predictions[1]['class'] if len(top_predictions) > 1 else '',
                    'top_probability_2': top_predictions[1]['probability'] if len(top_predictions) > 1 else 0.0,
                    'top_prediction_3': top_predictions[2]['class'] if len(top_predictions) > 2 else '',
                    'top_probability_3': top_predictions[2]['probability'] if len(top_predictions) > 2 else 0.0,
                })
                
                # Create annotated image
                annotated_filename = f"{i:04d}_True_{y_test_decoded[i]}_Pred_{y_test_pred_decoded[i]}_Conf_{confidence:.3f}.jpg"
                save_annotated_image_with_confidence(
                    filename, original_label, model_dir, annotated_filename,
                    y_test_decoded[i], y_test_pred_decoded[i], confidence,
                    top_predictions, base_images_dir
                )
        
        # Save summary CSV
        summary_df = pd.DataFrame(results_data)
        summary_path = os.path.join(model_dir, "confidence_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Calculate confidence statistics
        correct_confidences = summary_df[summary_df['is_correct']]['confidence_score']
        incorrect_confidences = summary_df[~summary_df['is_correct']]['confidence_score']
        
        print(f"✅ Confidence visualizations saved to: {model_dir}")
        print(f"   - Total images: {len(summary_df)}")
        print(f"   - Correct predictions: {len(correct_confidences)}")
        print(f"   - Incorrect predictions: {len(incorrect_confidences)}")
        if len(correct_confidences) > 0:
            print(f"   - Avg confidence (correct): {correct_confidences.mean():.3f}")
        if len(incorrect_confidences) > 0:
            print(f"   - Avg confidence (incorrect): {incorrect_confidences.mean():.3f}")
        
        # Create confidence distribution plot
        create_confidence_distribution_plot(summary_df, model_name, model_dir)
        
    except Exception as e:
        print(f"⚠️  Error generating confidence visualizations: {e}")
        import traceback
        traceback.print_exc()


def save_annotated_image_with_confidence(filename, original_label, output_dir, output_filename,
                                        true_label, pred_label, confidence, top_predictions,
                                        base_images_dir):
    """
    Save an image annotated with true label, predicted label, confidence score, and top predictions.
    
    Args:
        filename: Original image filename
        original_label: Label from the dataframe (for finding the image)
        output_dir: Directory to save annotated image
        output_filename: Name for the output file
        true_label: True label
        pred_label: Predicted label
        confidence: Confidence score (0-1)
        top_predictions: List of top predictions with probabilities
        base_images_dir: Base directory for original images
    """
    try:
        # Try to find the image
        source_path = None
        
        # 1. Check for cropped version
        cropped_dir = "cropped_banknotes"
        base_name_no_ext = os.path.splitext(filename)[0]
        cropped_fname = f"{original_label}_{base_name_no_ext}_cropped.jpg"
        cropped_path_check = os.path.join(cropped_dir, cropped_fname)
        
        if os.path.exists(cropped_path_check):
            source_path = cropped_path_check
        
        # 2. Search in dataset directories
        if not source_path:
            possible_paths = [
                os.path.join(base_images_dir, "test", original_label, filename),
                os.path.join(base_images_dir, "train", original_label, filename),
                os.path.join(base_images_dir, "valid", original_label, filename),
                os.path.join(base_images_dir, original_label, filename),
                filename  # Current directory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    source_path = path
                    break
        
        if not source_path or not os.path.exists(source_path):
            # print(f"  ⚠️  Image not found: {filename}")
            return
        
        # Load image
        img = cv2.imread(source_path)
        if img is None:
            return
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for consistent display
        max_height = 600
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
        
        # Add border for text
        border_top = 180  # More space for confidence info
        border_bottom = 20
        img_with_border = cv2.copyMakeBorder(img, border_top, border_bottom, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(245, 245, 245))
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(img_with_border, "BANKNOTE CLASSIFICATION CONFIDENCE", (20, 40),
                   font, 0.8, (0, 0, 0), 2)
        
        # True vs Predicted
        cv2.putText(img_with_border, f"True: {true_label}", (20, 80),
                   font, 0.7, (0, 100, 0), 2)
        cv2.putText(img_with_border, f"Predicted: {pred_label}", (20, 110),
                   font, 0.7, (0, 0, 200), 2)
        
        # Confidence score with color coding
        confidence_percent = confidence * 100
        if confidence >= 0.9:
            color = (0, 200, 0)  # Green for high confidence
        elif confidence >= 0.7:
            color = (200, 200, 0)  # Yellow for medium confidence
        else:
            color = (0, 0, 200)  # Red for low confidence
        
        cv2.putText(img_with_border, f"Confidence: {confidence_percent:.1f}%", (20, 140),
                   font, 0.7, color, 2)
        
        # Add confidence bar visualization
        bar_width = 200
        bar_height = 15
        bar_x = 20
        bar_y = 160
        
        # Background bar
        cv2.rectangle(img_with_border, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        cv2.rectangle(img_with_border, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Bar border
        cv2.rectangle(img_with_border, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
        
        # Top predictions (if available)
        if top_predictions:
            cv2.putText(img_with_border, "Top Predictions:", (250, 80),
                       font, 0.6, (0, 0, 0), 1)
            
            y_offset = 100
            for j, pred in enumerate(top_predictions[:3]):  # Show top 3
                pred_text = f"{j+1}. {pred['class']}: {pred['probability']*100:.1f}%"
                cv2.putText(img_with_border, pred_text, (250, y_offset),
                           font, 0.5, (50, 50, 50), 1)
                y_offset += 20
        
        # Correct/Wrong indicator
        is_correct = true_label == pred_label
        status_color = (0, 200, 0) if is_correct else (0, 0, 200)
        status_text = "✓ CORRECT" if is_correct else "✗ WRONG"
        
        cv2.putText(img_with_border, status_text, (img_with_border.shape[1] - 150, 80),
                   font, 0.7, status_color, 2)
        
        # Filename at bottom
        cv2.putText(img_with_border, f"File: {filename}", 
                   (10, img_with_border.shape[0] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Save the annotated image
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_with_border)
        
    except Exception as e:
        print(f"Error creating confidence image for {filename}: {e}")


def create_confidence_distribution_plot(summary_df, model_name, output_dir):
    """
    Create a plot showing confidence distribution for correct vs incorrect predictions.
    
    Args:
        summary_df: DataFrame with confidence scores
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Separate correct and incorrect predictions
        correct_confidences = summary_df[summary_df['is_correct']]['confidence_score']
        incorrect_confidences = summary_df[~summary_df['is_correct']]['confidence_score']
        
        # Create histogram
        bins = np.linspace(0, 1, 21)
        
        if len(correct_confidences) > 0:
            plt.hist(correct_confidences, bins=bins, alpha=0.7, label='Correct Predictions',
                    color='green', edgecolor='black')
        
        if len(incorrect_confidences) > 0:
            plt.hist(incorrect_confidences, bins=bins, alpha=0.7, label='Incorrect Predictions',
                    color='red', edgecolor='black')
        
        plt.title(f'Confidence Distribution - {model_name}', fontsize=14)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total: {len(summary_df)} images\n"
        stats_text += f"Correct: {len(correct_confidences)} ({len(correct_confidences)/len(summary_df)*100:.1f}%)\n"
        stats_text += f"Incorrect: {len(incorrect_confidences)} ({len(incorrect_confidences)/len(summary_df)*100:.1f}%)"
        
        if len(correct_confidences) > 0:
            stats_text += f"\nAvg confidence (correct): {correct_confidences.mean():.3f}"
        
        if len(incorrect_confidences) > 0:
            stats_text += f"\nAvg confidence (incorrect): {incorrect_confidences.mean():.3f}"
        
        plt.figtext(0.75, 0.75, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "confidence_distribution.png")
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"Error creating confidence distribution plot: {e}")


def analyze_results(results, label_encoder, X_test, y_test, feature_names):
    """
    Analyze and visualize the results of model training.
    
    Args:
        results: Dictionary containing models and their performance
        label_encoder: Fitted label encoder
        X_test, y_test: Test features and labels
        feature_names: Names of the features
    """
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    
    # ========== PERFORMANCE COMPARISON ==========
    print("\n1. MODEL PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Model':<25} {'Train Acc':<10} {'Valid Acc':<10} {'Test Acc':<10}")
    print("-" * 50)
    
    # Sort models by validation accuracy (descending)
    sorted_models = sorted(results['valid_acc'].items(), key=lambda x: x[1], reverse=True)
    
    for name, valid_acc in sorted_models:
        train_acc = results['train_acc'][name]
        test_acc = results['test_acc'][name]
        print(f"{name:<25} {train_acc:<10.4f} {valid_acc:<10.4f} {test_acc:<10.4f}")
    
    # Identify best model based on validation accuracy
    best_model_name = max(results['valid_acc'], key=results['valid_acc'].get)
    best_model = results['models'][best_model_name]
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Validation Accuracy: {results['valid_acc'][best_model_name]:.4f}")
    print(f"   Test Accuracy: {results['test_acc'][best_model_name]:.4f}")
    
    # ========== VISUALIZE MODEL PERFORMANCE ==========
    print("\n2. CREATING VISUALIZATIONS")
    
    # Prepare data for plotting
    model_names = list(results['test_acc'].keys())
    test_accuracies = list(results['test_acc'].values())
    
    # Sort by test accuracy for better visualization
    sorted_indices = np.argsort(test_accuracies)[::-1]
    model_names_sorted = [model_names[i] for i in sorted_indices]
    test_accuracies_sorted = [test_accuracies[i] for i in sorted_indices]
    
    # Create performance comparison plot
    plt.figure(figsize=(12, 6))
    
    # Bar chart of test accuracies
    colors = ['#2ca02c' if name == best_model_name else '#1f77b4' for name in model_names_sorted]
    bars = plt.bar(range(len(model_names_sorted)), test_accuracies_sorted, color=colors)
    
    plt.title('Model Performance Comparison on Test Set', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(len(model_names_sorted)), model_names_sorted, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, test_accuracies_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PLOT_FILE, dpi=100)
    print(f"✅ Model performance plot saved as {RESULTS_PLOT_FILE}")
    
    # ========== CONFUSION MATRIX FOR BEST MODEL ==========
    print(f"\n3. CONFUSION MATRIX FOR {best_model_name}")
    
    # Get predictions from best model
    y_test_pred = best_model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    
    # Create heatmap-like visualization without seaborn
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Set axis labels
    class_names = label_encoder.classes_
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=100)
    print(f"✅ Confusion matrix saved as {CONFUSION_MATRIX_FILE}")
    
    # ========== CLASSIFICATION REPORT ==========
    print("\n4. DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    
    report = classification_report(y_test, y_test_pred, 
                                   target_names=class_names, 
                                   digits=3)
    print(report)
    
    # ========== FEATURE IMPORTANCE ANALYSIS ==========
    if 'Random Forest' in results['feature_importances']:
        print("\n5. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        importances = results['feature_importances']['Random Forest']
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")
        
        # Create feature importance visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importances[indices], color='teal')
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14)
        plt.ylabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=100)
        print(f"✅ Feature importance plot saved as feature_importance.png")
    
    return best_model_name, best_model

def save_misclassified_images(results, best_model_name, X_test, y_test, test_df, 
                              label_encoder, base_images_dir=BASE_DIR):
    """
    Save misclassified images to a folder with their predicted labels.
    
    Args:
        results: Dictionary containing models and their performance
        best_model_name: Name of the best model
        X_test: Test features
        y_test: True test labels (encoded)
        test_df: Original test dataframe with filenames
        label_encoder: Fitted label encoder
        base_images_dir: Base directory where original images are stored
    """
    print("\n" + "=" * 70)
    print("SAVING MISCLASSIFIED IMAGES")
    print("=" * 70)
    
    # Get the best model and predictions
    best_model = results['models'][best_model_name]
    y_test_pred = best_model.predict(X_test)
    
    # Decode labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
    
    # Directories
    misclassified_dir = "misclassified_images"
    cropped_dir = "cropped_banknotes"  # Folder created by extract_features.py
    os.makedirs(misclassified_dir, exist_ok=True)
    
    print(f"Saving misclassified images to: {misclassified_dir}")
    
    misclassified_count = 0
    
    # Ensure indices match
    if len(test_df) != len(y_test):
        test_df = test_df.reset_index(drop=True)
    
    for i in range(len(y_test)):
        true_label = y_test_decoded[i]
        pred_label = y_test_pred_decoded[i]
        
        if true_label != pred_label:
            misclassified_count += 1
            
            if i < len(test_df):
                filename = test_df.iloc[i]['filename']
                original_label = str(test_df.iloc[i]['label']) # Ensure string for path joining
                
                # Setup basic file info
                base_name_no_ext = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                new_filename = f"{i:04d}_True_{true_label}_Pred_{pred_label}_{filename}"
                
                # 1. CHECK FOR CROPPED VERSION FIRST
                # extract_features.py saves as: {class}_{name}_cropped.jpg
                cropped_fname = f"{original_label}_{base_name_no_ext}_cropped.jpg"
                cropped_path_check = os.path.join(cropped_dir, cropped_fname)
                
                source_path = None
                is_cropped_source = False
                
                if os.path.exists(cropped_path_check):
                    source_path = cropped_path_check
                    is_cropped_source = True
                    # If using cropped source, ensure output extension matches (jpg)
                    new_filename = os.path.splitext(new_filename)[0] + ".jpg"
                    ext = ".jpg" 
                
                # 2. FALLBACK TO ORIGINAL SEARCH
                if not source_path:
                    # Search in dataset directory structure
                    possible_paths = [
                        os.path.join(base_images_dir, "test", original_label, filename),
                        os.path.join(base_images_dir, "train", original_label, filename),
                        os.path.join(base_images_dir, "valid", original_label, filename),
                        os.path.join(base_images_dir, original_label, filename),
                        filename # Current dir
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            source_path = path
                            break
                
                # 3. COPY AND ANNOTATE
                if source_path and os.path.exists(source_path):
                    try:
                        dest_path = os.path.join(misclassified_dir, new_filename)
                        
                        # Copy image
                        shutil.copy2(source_path, dest_path)
                        
                        # Annotate
                        annotated_path = dest_path.replace(ext, f"_annotated{ext}")
                        save_annotated_image(source_path, annotated_path, true_label, pred_label)
                        
                        origin_type = "Cropped" if is_cropped_source else "Original"
                        # print(f"  ✓ {filename} ({origin_type}): {true_label} → {pred_label}")
                        
                    except Exception as e:
                        print(f"  ✗ Error copying {filename}: {e}")
                # else:
                #     print(f"  ✗ Could not find image: {filename}")
    
    # print(f"\n✅ Saved {misclassified_count} misclassified images to '{misclassified_dir}'")
    create_misclassified_summary(misclassified_dir, y_test_decoded, y_test_pred_decoded, test_df)
    
    return misclassified_count


def save_annotated_image(source_path, dest_path, true_label, pred_label):
    """
    Save an image with annotation text showing true vs predicted labels.
    
    Args:
        source_path: Path to source image
        dest_path: Path to save annotated image
        true_label: True label
        pred_label: Predicted label
    """
    try:
        # Load image
        img = cv2.imread(source_path)
        if img is None:
            return
        
        # Resize if too large (for display purposes)
        max_height = 800
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
        
        # Add border for text
        border_size = 100
        img_with_border = cv2.copyMakeBorder(img, border_size, 0, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=(240, 240, 240))
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(img_with_border, "MISCLASSIFIED BANKNOTE", (20, 40), 
                   font, 1, (0, 0, 0), 2)
        
        # True label (in green if correct, but here it's always wrong)
        cv2.putText(img_with_border, f"True: {true_label}", (20, 80), 
                   font, 0.8, (0, 100, 0), 2)
        
        # Predicted label (in red since it's wrong)
        cv2.putText(img_with_border, f"Predicted: {pred_label}", (20, 120), 
                   font, 0.8, (0, 0, 200), 2)
        
        # Add a colored box based on correctness
        if true_label == pred_label:
            color = (0, 255, 0)  # Green for correct
            status = "CORRECT"
        else:
            color = (0, 0, 255)  # Red for incorrect
            status = "WRONG"
        
        cv2.putText(img_with_border, f"Status: {status}", (img_with_border.shape[1] - 200, 80), 
                   font, 0.8, color, 2)
        
        # Save the annotated image
        cv2.imwrite(dest_path, img_with_border)
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")


def create_misclassified_summary(output_dir, y_true, y_pred, test_df):
    """
    Create a summary CSV file of misclassified images.
    
    Args:
        output_dir: Directory to save summary
        y_true: True labels
        y_pred: Predicted labels
        test_df: Test dataframe with filenames
    """
    summary_data = []
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i] and i < len(test_df):
            summary_data.append({
                'index': i,
                'filename': test_df.iloc[i]['filename'],
                'true_label': y_true[i],
                'predicted_label': y_pred[i],
                'original_label_in_df': test_df.iloc[i]['label']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "misclassified_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Created misclassification summary: {summary_path}")
        
        # Print some statistics
        print(f"\n📊 Misclassification Statistics:")
        print(f"   Total misclassified: {len(summary_df)}")
        
        # Count by true label
        misclass_by_true = summary_df['true_label'].value_counts()
        print(f"\n   Misclassified by true label:")
        for label, count in misclass_by_true.items():
            print(f"     {label}: {count}")
        
        # Confusion matrix of misclassifications
        print(f"\n   Most common misclassifications:")
        confusion_counts = summary_df.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
        confusion_counts = confusion_counts.sort_values('count', ascending=False)
        for _, row in confusion_counts.head(10).iterrows():
            print(f"     {row['true_label']} → {row['predicted_label']}: {row['count']}")


def save_models(best_model, scaler, label_encoder):
    """
    Save the trained models, scaler, and label encoder for future use.
    
    Args:
        best_model: The best performing model
        scaler: Fitted feature scaler
        label_encoder: Fitted label encoder
    """
    print("\n" + "=" * 70)
    print("SAVING MODELS AND TRANSFORMERS")
    print("=" * 70)
    
    try:
        import joblib
        
        # Save the best model
        joblib.dump(best_model, BEST_MODEL_FILE)
        print(f"✅ Best model saved as {BEST_MODEL_FILE}")
        
        # Save the scaler
        joblib.dump(scaler, SCALER_FILE)
        print(f"✅ Feature scaler saved as {SCALER_FILE}")
        
        # Save the label encoder
        joblib.dump(label_encoder, LABEL_ENCODER_FILE)
        print(f"✅ Label encoder saved as {LABEL_ENCODER_FILE}")
        
    except ImportError:
        print("⚠️  Joblib not installed. Install with: pip install joblib")
        print("Models not saved. Please install joblib to save models.")


def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("=" * 70)
    print("EGYPTIAN BANKNOTE CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)
    
    # Create confidence visualization directory
    os.makedirs(CONFIDENCE_IMAGES_DIR, exist_ok=True)
    
    # Step 1: Load and prepare data
    data = load_and_prepare_data()
    if data is None:
        return
    
    X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder, scaler, feature_names = data
    
    # Step 2: Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_valid, y_valid, 
                                        X_test, y_test, feature_names)
    
    # Step 3: Analyze results
    best_model_name, best_model = analyze_results(results, label_encoder, 
                                                  X_test, y_test, feature_names)
    
    # Step 4: Generate confidence visualizations for ALL models
    print("\n" + "=" * 70)
    print("GENERATING CONFIDENCE VISUALIZATIONS")
    print("=" * 70)
    
    try:
        # Reload test data to get filenames
        test_df_original = pd.read_csv(TEST_FEATURES_FILE)
        
        # Generate confidence visualizations for each model
        for model_name, model in results['models'].items():
            save_confidence_images(
                model, model_name, X_test, y_test,
                test_df_original, label_encoder,
                BASE_DIR
            )
        
        print(f"\n✅ All confidence visualizations saved to '{CONFIDENCE_IMAGES_DIR}'")
        
    except Exception as e:
        print(f"⚠️  Could not generate confidence visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Save misclassified images
    print("\n" + "=" * 70)
    print("PREPARING MISCLASSIFIED IMAGES ANALYSIS")
    print("=" * 70)
    
    try:
        # Reload test data to get filenames
        test_df_original = pd.read_csv(TEST_FEATURES_FILE)
        
        # Save misclassified images
        misclassified_count = save_misclassified_images(
            results, best_model_name, X_test, y_test, 
            test_df_original, label_encoder, BASE_DIR
        )
        
        # Calculate misclassification rate
        total_test = len(y_test)
        misclassification_rate = misclassified_count / total_test if total_test > 0 else 0
        print(f"\n📈 Misclassification rate: {misclassification_rate:.2%} ({misclassified_count}/{total_test})")
        
    except Exception as e:
        print(f"⚠️  Could not save misclassified images: {e}")
        print("This feature requires the original test CSV and image files.")
    
    # Step 6: Save models
    save_models(best_model, scaler, label_encoder)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"  • Total training samples: {len(y_train)}")
    print(f"  • Total validation samples: {len(y_valid)}")
    print(f"  • Total test samples: {len(y_test)}")
    print(f"  • Number of features: {len(feature_names)}")
    print(f"  • Number of classes: {len(label_encoder.classes_)}")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Best model test accuracy: {results['test_acc'][best_model_name]:.4f}")
    
    if 'misclassified_count' in locals():
        print(f"  • Misclassified images: {misclassified_count}")
    
    print(f"\n📁 Files created:")
    print(f"  • {RESULTS_PLOT_FILE} - Model performance comparison")
    print(f"  • {CONFUSION_MATRIX_FILE} - Confusion matrix")
    print(f"  • feature_importance.png - Feature importance (if available)")
    print(f"  • {CONFIDENCE_IMAGES_DIR}/ - Folder with confidence visualizations")
    print(f"  • misclassified_images/ - Folder with misclassified images")
    print(f"  • misclassified_images/misclassified_summary.csv - Summary of errors")
    
    print(f"\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()