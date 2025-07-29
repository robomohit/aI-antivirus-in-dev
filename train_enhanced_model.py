#!/usr/bin/env python3
"""
Enhanced AI Model Training with 3-Split Validation
Trains the Ultimate AI Antivirus model with comprehensive validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset with comprehensive feature engineering."""
    print("📊 Loading malware dataset...")
    df = pd.read_csv('malware_dataset.csv')
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🎯 Label distribution:")
    print(df['is_malicious'].value_counts())
    
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
    
    print(f"📊 Features: {X.shape[1]}")
    print(f"📊 Samples: {len(X)}")
    print(f"🎯 Labels: {y.value_counts().to_dict()}")
    
    return X, y, df

def train_model_with_validation(X, y):
    """Train model with 3-split validation (60% train, 20% validation, 20% test)."""
    print("🧠 Training model with 3-split validation...")
    
    # First split: 80% for training/validation, 20% for final test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 75% of remaining for train, 25% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"📚 Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"🔍 Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"🧪 Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
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
    
    print("🔄 Training RandomForest model...")
    model.fit(X_train, y_train)
    
    # Evaluate on all sets
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    
    print(f"📊 Train Accuracy: {train_score:.4f}")
    print(f"📊 Validation Accuracy: {val_score:.4f}")
    print(f"📊 Test Accuracy: {test_score:.4f}")
    
    return model, X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model_comprehensive(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Comprehensive model evaluation with all metrics."""
    print("📈 Comprehensive model evaluation...")
    
    # Get predictions for all sets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for all sets
    results = {}
    
    for set_name, y_true, y_pred, y_proba in [
        ('train', y_train, y_pred_train, y_proba_train),
        ('validation', y_val, y_pred_val, y_proba_val),
        ('test', y_test, y_pred_test, y_proba_test)
    ]:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
        
        results[set_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"\n📊 {set_name.upper()} METRICS:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
    
    # Check for overfitting
    train_acc = results['train']['accuracy']
    val_acc = results['validation']['accuracy']
    test_acc = results['test']['accuracy']
    
    overfitting_score = train_acc - val_acc
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    print(f"  Train-Val Difference: {overfitting_score:.4f}")
    
    if overfitting_score > 0.05:
        print("  ⚠️  WARNING: Potential overfitting detected!")
    elif overfitting_score > 0.02:
        print("  ⚠️  CAUTION: Slight overfitting possible")
    else:
        print("  ✅ Model appears well-balanced")
    
    return results

def save_visualizations(model, results, feature_names):
    """Save comprehensive visualizations."""
    print("📊 Saving visualizations...")
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_names))
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.title('Feature Importance (Top 20)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'logs/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix for Test Set
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(results['test']['y_true'], results['test']['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    for set_name in ['train', 'validation', 'test']:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results[set_name]['y_true'], results[set_name]['y_proba'])
        auc = results[set_name]['auc']
        plt.plot(fpr, tpr, label=f'{set_name.capitalize()} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/roc_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved with timestamp: {timestamp}")

def save_model_and_metrics(model, results, feature_names, df):
    """Save model and comprehensive metrics."""
    """Save model and comprehensive metrics."""
    print("💾 Saving model and metrics...")
    
    # Create directories
    Path('logs').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    
    # Save model
    model_path = 'model/model.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save comprehensive metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = f'logs/model_metrics_{timestamp}.txt'
    
    with open(metrics_file, 'w') as f:
        f.write("🧠 ULTIMATE AI ANTIVIRUS MODEL METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Features: {len(feature_names)}\n\n")
        
        # Dataset summary
        f.write("📊 DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Malware samples: {len(df[df['is_malicious'] == 1])}\n")
        f.write(f"Safe samples: {len(df[df['is_malicious'] == 0])}\n")
        f.write(f"Feature columns: {len(feature_names)}\n")
        f.write(f"File categories: {df['file_category'].nunique()}\n")
        f.write(f"Extensions: {df['extension'].nunique()}\n\n")
        
        # Model performance
        f.write("📈 MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for set_name in ['train', 'validation', 'test']:
            f.write(f"\n{set_name.upper()} SET:\n")
            f.write(f"  Accuracy:  {results[set_name]['accuracy']:.4f}\n")
            f.write(f"  Precision: {results[set_name]['precision']:.4f}\n")
            f.write(f"  Recall:    {results[set_name]['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results[set_name]['f1']:.4f}\n")
            f.write(f"  AUC:       {results[set_name]['auc']:.4f}\n")
        
        # Overfitting analysis
        train_acc = results['train']['accuracy']
        val_acc = results['validation']['accuracy']
        overfitting_score = train_acc - val_acc
        
        f.write(f"\n🔍 OVERFITTING ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train-Val Difference: {overfitting_score:.4f}\n")
        if overfitting_score > 0.05:
            f.write("Status: ⚠️  POTENTIAL OVERFITTING\n")
        elif overfitting_score > 0.02:
            f.write("Status: ⚠️  SLIGHT OVERFITTING\n")
        else:
            f.write("Status: ✅ WELL-BALANCED\n")
        
        # Feature importance
        f.write(f"\n📊 FEATURE IMPORTANCE (Top 20)\n")
        f.write("-" * 30 + "\n")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(20, len(feature_names))):
            f.write(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}\n")
    
    print(f"📄 Metrics saved to: {metrics_file}")
    
    # Save visualizations
    save_visualizations(model, results, feature_names)
    
    return timestamp

def main():
    """Main training function."""
    print("🚀 ULTIMATE AI ANTIVIRUS MODEL TRAINING v5.X")
    print("=" * 60)
    
    # Load data
    X, y, df = load_and_prepare_data()
    
    # Train model
    model, X_train, X_val, X_test, y_train, y_val, y_test = train_model_with_validation(X, y)
    
    # Evaluate model
    results = evaluate_model_comprehensive(model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save everything
    timestamp = save_model_and_metrics(model, results, X.columns.tolist(), df)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Test Accuracy: {results['test']['accuracy']:.4f}")
    print(f"📊 Test F1-Score: {results['test']['f1']:.4f}")
    print(f"📊 Test AUC: {results['test']['auc']:.4f}")
    print(f"💾 Model saved: model/model.pkl")
    print(f"📄 Metrics saved: logs/model_metrics_{timestamp}.txt")
    print(f"📊 Visualizations saved: logs/")
    
    # Check for overfitting
    overfitting_score = results['train']['accuracy'] - results['validation']['accuracy']
    if overfitting_score > 0.05:
        print("⚠️  WARNING: Potential overfitting detected!")
    elif overfitting_score > 0.02:
        print("⚠️  CAUTION: Slight overfitting possible")
    else:
        print("✅ Model appears well-balanced")

if __name__ == "__main__":
    main()