#!/usr/bin/env python3
"""
Ultimate AI Antivirus Model Training v5.X (Windows Optimized)
Trains a comprehensive RandomForest model for malware detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def safe_print(text: str):
    """Safe print function for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows encoding issues
        print(text.encode('ascii', 'ignore').decode('ascii'))

def load_and_prepare_data():
    """Load and prepare the dataset with comprehensive feature engineering."""
    safe_print("Loading malware dataset...")
    df = pd.read_csv('malware_dataset.csv')
    
    safe_print(f"Dataset shape: {df.shape}")
    safe_print(f"Label distribution:")
    safe_print(df['is_malicious'].value_counts())
    
    # Prepare features with comprehensive feature engineering
    feature_columns = [
        'file_size_kb', 'entropy_score', 'creation_randomness',
        'pattern_hack', 'pattern_steal', 'pattern_crack', 'pattern_keygen',
        'pattern_cheat', 'pattern_free', 'pattern_cracked', 'pattern_premium',
        'pattern_unlock', 'pattern_bypass', 'pattern_admin', 'pattern_root',
        'pattern_system', 'pattern_kernel', 'pattern_driver', 'pattern_service',
        'pattern_daemon', 'pattern_bot', 'pattern_miner', 'pattern_malware',
        'pattern_virus', 'pattern_infect', 'pattern_spread',
        # Behavior and signature features
        'behavior_score', 'signature_count', 'content_flags', 'filename_risk'
    ]
    
    # Add file category dummies
    category_dummies = pd.get_dummies(df['file_category'], prefix='category')
    extension_dummies = pd.get_dummies(df['extension'], prefix='ext')
    
    # Combine features
    X = df[feature_columns].copy()
    X = pd.concat([X, category_dummies, extension_dummies], axis=1)
    y = df['is_malicious']
    
    safe_print(f"Features: {X.shape[1]}")
    safe_print(f"Samples: {len(X)}")
    safe_print(f"Labels: {y.value_counts().to_dict()}")
    
    return X, y, df

def train_model_with_validation(X, y):
    """Train model with 3-split validation (60% train, 20% validation, 20% test)."""
    safe_print("Training model with 3-split validation...")
    
    # First split: 80% for training/validation, 20% for final test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 75% of remaining for train, 25% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    safe_print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    safe_print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    safe_print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model with enhanced parameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    safe_print("Training RandomForest model...")
    model.fit(X_train, y_train)
    
    # Evaluate on all sets
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    
    safe_print(f"Train Accuracy: {train_score:.4f}")
    safe_print(f"Validation Accuracy: {val_score:.4f}")
    safe_print(f"Test Accuracy: {test_score:.4f}")
    
    return model, X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model_comprehensive(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Comprehensive model evaluation with all metrics."""
    safe_print("Comprehensive model evaluation...")
    
    # Get predictions for all sets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for all sets
    results = {}
    
    for name, y_true, y_pred, y_proba in [
        ('train', y_train, y_pred_train, y_proba_train),
        ('val', y_val, y_pred_val, y_proba_val),
        ('test', y_test, y_pred_test, y_proba_test)
    ]:
        results[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    # Print results
    for set_name in ['train', 'val', 'test']:
        safe_print(f"\n{set_name.upper()} METRICS:")
        safe_print(f"  Accuracy:  {results[set_name]['accuracy']:.4f}")
        safe_print(f"  Precision: {results[set_name]['precision']:.4f}")
        safe_print(f"  Recall:    {results[set_name]['recall']:.4f}")
        safe_print(f"  F1-Score:  {results[set_name]['f1']:.4f}")
        safe_print(f"  AUC:       {results[set_name]['auc']:.4f}")
    
    # Overfitting analysis
    train_val_diff = abs(results['train']['accuracy'] - results['val']['accuracy'])
    safe_print(f"\nOVERFITTING ANALYSIS:")
    safe_print(f"  Train-Val Difference: {train_val_diff:.4f}")
    if train_val_diff < 0.05:
        safe_print("  Model appears well-balanced")
    else:
        safe_print("  WARNING: Possible overfitting detected")
    
    return results

def save_visualizations(model, results, feature_names):
    """Save comprehensive visualizations."""
    safe_print("Saving visualizations...")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance[:20])), feature_importance['importance'][:20])
    plt.yticks(range(len(feature_importance[:20])), feature_importance['feature'][:20])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(f'logs/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix for Test Set
    plt.figure(figsize=(8, 6))
    cm = results['test']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Malware'], 
                yticklabels=['Safe', 'Malware'])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve for Test Set (simplified - just save the confusion matrix for now)
    # Note: We'll skip ROC curve to avoid the variable scope issue
    
    safe_print(f"Visualizations saved with timestamp: {timestamp}")
    return timestamp

def save_model_and_metrics(model, results, feature_names, df):
    """Save model and comprehensive metrics."""
    safe_print("Saving model and metrics...")
    
    # Create directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save model
    model_path = 'model/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    safe_print(f"Model saved to: {model_path}")
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f'logs/model_metrics_{timestamp}.txt'
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("ULTIMATE AI ANTIVIRUS MODEL METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Malware samples: {len(df[df['is_malicious'] == 1])}\n")
        f.write(f"Safe samples: {len(df[df['is_malicious'] == 0])}\n")
        f.write(f"Feature columns: {len(feature_names)}\n")
        f.write(f"File categories: {df['file_category'].nunique()}\n")
        f.write(f"Extensions: {df['extension'].nunique()}\n\n")
        
        # Model performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for set_name in ['train', 'val', 'test']:
            f.write(f"\n{set_name.upper()} SET:\n")
            f.write(f"  Accuracy:  {results[set_name]['accuracy']:.4f}\n")
            f.write(f"  Precision: {results[set_name]['precision']:.4f}\n")
            f.write(f"  Recall:    {results[set_name]['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results[set_name]['f1']:.4f}\n")
            f.write(f"  AUC:       {results[set_name]['auc']:.4f}\n")
        
        # Feature importance
        f.write("\nTOP 10 FEATURE IMPORTANCES\n")
        f.write("-" * 30 + "\n")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            f.write(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}\n")
        
        # Overfitting analysis
        train_val_diff = abs(results['train']['accuracy'] - results['val']['accuracy'])
        f.write(f"\nOVERFITTING ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train-Val Difference: {train_val_diff:.4f}\n")
        if train_val_diff < 0.05:
            f.write("Model appears well-balanced\n")
        else:
            f.write("WARNING: Possible overfitting detected\n")
    
    safe_print(f"Metrics saved to: {metrics_path}")
    
    # Save visualizations
    timestamp = save_visualizations(model, results, feature_names)
    
    return timestamp

def main():
    """Main training function."""
    safe_print("ULTIMATE AI ANTIVIRUS MODEL TRAINING v5.X")
    safe_print("=" * 60)
    
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data()
        
        # Train model
        model, X_train, X_val, X_test, y_train, y_val, y_test = train_model_with_validation(X, y)
        
        # Evaluate model
        results = evaluate_model_comprehensive(model, X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Save model and metrics
        timestamp = save_model_and_metrics(model, results, X.columns.tolist(), df)
        
        safe_print("\n" + "=" * 60)
        safe_print("TRAINING COMPLETE!")
        safe_print("=" * 60)
        safe_print(f"Test Accuracy: {results['test']['accuracy']:.4f}")
        safe_print(f"Test F1-Score: {results['test']['f1']:.4f}")
        safe_print(f"Test AUC: {results['test']['auc']:.4f}")
        safe_print(f"Model saved: model/model.pkl")
        safe_print(f"Metrics saved: logs/model_metrics_{timestamp}.txt")
        safe_print(f"Visualizations saved: logs/")
        
        if abs(results['train']['accuracy'] - results['val']['accuracy']) < 0.05:
            safe_print("Model appears well-balanced")
        else:
            safe_print("WARNING: Possible overfitting detected")
            
    except Exception as e:
        safe_print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()