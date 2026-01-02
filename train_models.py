"""
Egyptian Banknote Classification - Model Training Script
This script trains traditional ML models on extracted features
using proper train/validation/test splits.
"""

import pandas as pd
import numpy as np
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

# ========== CONFIGURATION ==========
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
    
    # ========== PREPARE FEATURES AND LABELS ==========
    print("\n2. Preparing features and labels...")
    
    # Identify non-feature columns to exclude
    non_feature_cols = ['label', 'filename']
    
    # Extract features (X) and labels (y) for each dataset
    X_train = train_df.drop([col for col in non_feature_cols if col in train_df.columns], axis=1)
    y_train = train_df['label']
    
    X_valid = valid_df.drop([col for col in non_feature_cols if col in valid_df.columns], axis=1)
    y_valid = valid_df['label']
    
    X_test = test_df.drop([col for col in non_feature_cols if col in test_df.columns], axis=1)
    y_test = test_df['label']
    
    # Encode string labels to numerical values
    print("3. Encoding labels...")
    label_encoder = LabelEncoder()
    
    # Fit encoder on training labels, then transform all sets
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"✅ Classes: {label_encoder.classes_}")
    print(f"✅ Number of features: {X_train.shape[1]}")
    
    # ========== SCALE FEATURES ==========
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
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
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Store results
    results = {
        'models': {},
        'train_acc': {},
        'valid_acc': {},
        'test_acc': {},
        'feature_importances': {}
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
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
    
    # Step 4: Save models
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
    print(f"\n📁 Files created:")
    print(f"  • {RESULTS_PLOT_FILE} - Model performance comparison")
    print(f"  • {CONFUSION_MATRIX_FILE} - Confusion matrix")
    print(f"  • feature_importance.png - Feature importance (if available)")
    print(f"\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()