#!/usr/bin/env python3
"""
Train the ultimate AI antivirus model with comprehensive validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the dataset."""
    print("📊 Loading dataset...")
    df = pd.read_csv('malware_dataset.csv')
    
    # Prepare features
    feature_columns = [
        'file_size_kb', 'entropy_score', 'creation_randomness',
        'pattern_hack', 'pattern_steal', 'pattern_crack', 'pattern_keygen',
        'pattern_cheat', 'pattern_free', 'pattern_cracked', 'pattern_premium',
        'pattern_unlock', 'pattern_bypass', 'pattern_admin', 'pattern_root',
        'pattern_system', 'pattern_kernel', 'pattern_driver', 'pattern_service',
        'pattern_daemon', 'pattern_bot', 'pattern_miner', 'pattern_malware',
        'pattern_virus', 'pattern_infect', 'pattern_spread'
    ]
    
    # Add file category dummies
    category_dummies = pd.get_dummies(df['file_category'], prefix='category')
    extension_dummies = pd.get_dummies(df['extension'], prefix='ext')
    
    # Combine features
    X = df[feature_columns].copy()
    X = pd.concat([X, category_dummies, extension_dummies], axis=1)
    y = df['label']
    
    print(f"📈 Features: {X.shape[1]}")
    print(f"📊 Samples: {len(X)}")
    print(f"🎯 Labels: {y.value_counts().to_dict()}")
    
    return X, y, df

def train_model(X, y):
    """Train the RandomForest model with comprehensive validation."""
    print("🧠 Training model...")
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"📚 Train: {len(X_train)} samples")
    print(f"🔍 Validation: {len(X_val)} samples")
    print(f"🧪 Test: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on all sets
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    
    print(f"📊 Train Accuracy: {train_score:.3f}")
    print(f"📊 Validation Accuracy: {val_score:.3f}")
    print(f"📊 Test Accuracy: {test_score:.3f}")
    
    return model, X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X_test, y_test, X_val, y_val):
    """Evaluate model performance."""
    print("📈 Evaluating model...")
    
    # Test set predictions
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Validation set predictions
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val)
    val_recall = recall_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val)
    
    print(f"🧪 Test Metrics:")
    print(f"  Accuracy: {test_accuracy:.3f}")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall: {test_recall:.3f}")
    print(f"  F1-Score: {test_f1:.3f}")
    
    print(f"🔍 Validation Metrics:")
    print(f"  Accuracy: {val_accuracy:.3f}")
    print(f"  Precision: {val_precision:.3f}")
    print(f"  Recall: {val_recall:.3f}")
    print(f"  F1-Score: {val_f1:.3f}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    }, y_pred_test, y_pred_val

def save_confusion_matrix(y_true, y_pred, title, filename):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'logs/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_feature_importance(model, feature_names, filename):
    """Save feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(min(20, len(feature_names))), importances[indices[:20]])
    plt.xticks(range(min(20, len(feature_names))), [feature_names[i] for i in indices[:20]], rotation=45)
    plt.tight_layout()
    plt.savefig(f'logs/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model_and_metrics(model, metrics, feature_names, y_test, y_pred_test, y_val, y_pred_val):
    """Save model and metrics."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    
    # Save model
    model_path = 'model/model.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = f'logs/model_metrics_{timestamp}.txt'
    
    with open(metrics_file, 'w') as f:
        f.write("🧠 ULTIMATE AI ANTIVIRUS MODEL METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("📊 TEST METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics['test_accuracy']:.3f}\n")
        f.write(f"Precision: {metrics['test_precision']:.3f}\n")
        f.write(f"Recall: {metrics['test_recall']:.3f}\n")
        f.write(f"F1-Score: {metrics['test_f1']:.3f}\n\n")
        
        f.write("🔍 VALIDATION METRICS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Accuracy: {metrics['val_accuracy']:.3f}\n")
        f.write(f"Precision: {metrics['val_precision']:.3f}\n")
        f.write(f"Recall: {metrics['val_recall']:.3f}\n")
        f.write(f"F1-Score: {metrics['val_f1']:.3f}\n\n")
        
        f.write("📈 FEATURE IMPORTANCE (Top 20)\n")
        f.write("-" * 30 + "\n")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(20, len(feature_names))):
            f.write(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}\n")
    
    print(f"📄 Metrics saved to: {metrics_file}")
    
    # Save plots
    save_confusion_matrix(y_test, y_pred_test, "Test Set", f"confusion_matrix_test_{timestamp}")
    save_confusion_matrix(y_val, y_pred_val, "Validation Set", f"confusion_matrix_val_{timestamp}")
    save_feature_importance(model, feature_names, f"feature_importance_{timestamp}")
    
    print("🎯 Model training complete!")

def main():
    """Main training function."""
    print("🚀 ULTIMATE AI ANTIVIRUS MODEL TRAINING")
    print("=" * 50)
    
    # Load data
    X, y, df = load_and_prepare_data()
    
    # Train model
    model, X_train, X_val, X_test, y_train, y_val, y_test = train_model(X, y)
    
    # Evaluate model
    metrics, y_pred_test, y_pred_val = evaluate_model(model, X_test, y_test, X_val, y_val)
    
    # Save everything
    save_model_and_metrics(model, metrics, X.columns.tolist(), y_test, y_pred_test, y_val, y_pred_val)
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()