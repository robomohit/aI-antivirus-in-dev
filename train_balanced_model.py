#!/usr/bin/env python3
"""
Train Balanced Model
Train a model with better anti-overfitting techniques
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from datetime import datetime
from colorama import init, Fore, Style
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama
init()

class BalancedModelTrainer:
    def __init__(self):
        self.features_dir = "malware_features"
        self.models_dir = "balanced_models"
        Path(self.models_dir).mkdir(exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
    
    def find_latest_dataset(self):
        """Find the latest balanced dataset."""
        dataset_dir = Path("malware_features")
        if not dataset_dir.exists():
            raise FileNotFoundError("malware_features directory not found")
        
        # Look for balanced dataset
        csv_files = list(dataset_dir.glob("balanced_realistic_dataset_*.csv"))
        if not csv_files:
            # Fallback to any dataset
            csv_files = list(dataset_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV datasets found in malware_features directory")
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest_file = csv_files[0]
        print(f"{Fore.GREEN}✅ Found dataset: {latest_file.name}")
        return latest_file
    
    def load_dataset(self, dataset_path):
        """Load and prepare the dataset."""
        print(f"{Fore.CYAN}🔄 Loading dataset: {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"{Fore.GREEN}✅ Dataset loaded successfully!")
            print(f"{Fore.CYAN}📊 Dataset shape: {df.shape}")
            
            # Check for required columns
            required_cols = ['file_size', 'entropy', 'strings_count', 'avg_string_length', 
                           'printable_ratio', 'histogram_regularity', 'entropy_consistency', 'label']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"{Fore.RED}❌ Missing required columns: {missing_cols}")
                return None
            
            # Show class distribution
            malware_count = len(df[df['label'] == 1])
            benign_count = len(df[df['label'] == 0])
            print(f"{Fore.CYAN}📊 Malware samples: {malware_count}")
            print(f"{Fore.CYAN}📊 Benign samples: {benign_count}")
            
            return df
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading dataset: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for training."""
        print(f"{Fore.CYAN}🔄 Preparing features...")
        
        # Select feature columns
        feature_cols = ['file_size', 'entropy', 'strings_count', 'avg_string_length', 
                       'printable_ratio', 'histogram_regularity', 'entropy_consistency']
        
        X = df[feature_cols]
        y = df['label']
        
        print(f"{Fore.GREEN}✅ Features prepared!")
        print(f"{Fore.CYAN}📊 Feature matrix shape: {X.shape}")
        print(f"{Fore.CYAN}📊 Target vector shape: {y.shape}")
        
        return X, y, feature_cols
    
    def cross_validate_model(self, X, y):
        """Perform cross-validation with anti-overfitting."""
        print(f"{Fore.CYAN}🔄 Performing cross-validation...")
        
        # Use stratified k-fold to maintain class balance
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Anti-overfitting parameters (without early stopping for CV)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Reduced from 31
            'learning_rate': 0.03,  # Reduced from 0.05
            'feature_fraction': 0.8,  # Reduced from 0.9
            'bagging_fraction': 0.7,  # Reduced from 0.8
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 30,  # Increased from 20
            'min_gain_to_split': 0.2,  # Increased from 0.1
            'lambda_l1': 0.2,  # Increased from 0.1
            'lambda_l2': 0.2,  # Increased from 0.1
            'max_depth': 4,  # Reduced from 6
            'min_child_samples': 20,  # Added
            'subsample': 0.8,  # Added
            'colsample_bytree': 0.8  # Added
        }
        
        cv_scores = cross_val_score(
            lgb.LGBMClassifier(**params), 
            X, y, 
            cv=skf, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"{Fore.GREEN}📊 Cross-Validation Results:")
        print(f"   Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Individual Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        return cv_scores
    
    def train_model(self, X, y, feature_cols):
        """Train the model with anti-overfitting techniques."""
        print(f"{Fore.CYAN}🔄 Training balanced model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"{Fore.CYAN}📊 Training samples: {len(X_train)}")
        print(f"{Fore.CYAN}📊 Test samples: {len(X_test)}")
        print(f"{Fore.CYAN}📊 Features: {len(feature_cols)}")
        
        # Anti-overfitting parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 30,
            'min_gain_to_split': 0.2,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'max_depth': 4,
            'early_stopping_rounds': 100,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        print(f"{Fore.GREEN}✅ Model training completed!")
        
        return model, X_test, y_test, feature_cols
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model performance."""
        print(f"{Fore.CYAN}🔄 Evaluating model...")
        
        # Make predictions
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Benign', 'Malware'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # AUC score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # False positive rate
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"{Fore.GREEN}📊 MODEL EVALUATION:")
        print(f"{Fore.GREEN}{'='*40}")
        print(f"Classification Report:")
        print(report)
        print(f"Confusion Matrix:")
        print(f"True Negatives: {tn} (Benign files correctly identified)")
        print(f"False Positives: {fp} (Benign files incorrectly flagged)")
        print(f"False Negatives: {fn} (Malware files missed)")
        print(f"True Positives: {tp} (Malware files correctly detected)")
        print()
        print(f"📊 Performance Metrics:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC Score: {auc_score:.3f}")
        print(f"   False Positive Rate: {false_positive_rate:.3f}")
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'false_positive_rate': false_positive_rate,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, model, feature_cols, metrics, dataset_info):
        """Save the trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"balanced_model_{timestamp}.pkl"
        model_path = Path(self.models_dir) / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_filename = f"balanced_metadata_{timestamp}.pkl"
        metadata_path = Path(self.models_dir) / metadata_filename
        
        metadata = {
            'feature_cols': feature_cols,
            'metrics': metrics,
            'dataset_info': dataset_info,
            'model_info': {
                'type': 'LightGBM',
                'parameters': {
                    'num_leaves': 15,
                    'learning_rate': 0.03,
                    'max_depth': 4,
                    'min_data_in_leaf': 30,
                    'lambda_l1': 0.2,
                    'lambda_l2': 0.2
                }
            },
            'timestamp': timestamp
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"{Fore.GREEN}✅ Model saved!")
        print(f"{Fore.CYAN}📁 Model: {model_path}")
        print(f"{Fore.CYAN}📁 Metadata: {metadata_path}")
        
        return model_path, metadata_path
    
    def run_training_process(self):
        """Run the complete training process."""
        print(f"{Fore.CYAN}🛡️  BALANCED MODEL TRAINING")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Step 1: Find and load dataset
        dataset_path = self.find_latest_dataset()
        df = self.load_dataset(dataset_path)
        
        if df is None:
            return
        
        # Step 2: Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Step 3: Cross-validation
        cv_scores = self.cross_validate_model(X, y)
        
        # Step 4: Train model
        model, X_test, y_test, feature_cols = self.train_model(X, y, feature_cols)
        
        # Step 5: Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Step 6: Save model
        dataset_info = {
            'total_samples': len(df),
            'malware_samples': len(df[df['label'] == 1]),
            'benign_samples': len(df[df['label'] == 0]),
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std()
        }
        
        model_path, metadata_path = self.save_model(model, feature_cols, metrics, dataset_info)
        
        print(f"\n{Fore.GREEN}🎉 TRAINING COMPLETE!")
        print(f"{Fore.CYAN}📊 Model trained with anti-overfitting techniques")
        print(f"{Fore.CYAN}📊 Cross-validation accuracy: {cv_scores.mean():.3f}")
        print(f"{Fore.CYAN}📊 Test accuracy: {metrics['accuracy']:.3f}")
        print(f"{Fore.CYAN}📊 False positive rate: {metrics['false_positive_rate']:.3f}")

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting Balanced Model Training...")
    
    trainer = BalancedModelTrainer()
    trainer.run_training_process()

if __name__ == "__main__":
    main()