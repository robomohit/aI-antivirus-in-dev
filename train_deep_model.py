#!/usr/bin/env python3
"""
Ultimate AI Antivirus - Deep Learning Model Training (PyTorch)
Professional-grade malware detection with explainability and automation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional
import time

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MalwareDataset(Dataset):
    """PyTorch Dataset for malware detection."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FeedForwardClassifier(nn.Module):
    """Deep Neural Network for malware classification."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], dropout_rate: float = 0.3):
        super(FeedForwardClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class DeepLearningTrainer:
    """Professional trainer for deep learning malware detection."""
    
    def __init__(self, model: nn.Module = None, device: str = 'auto'):
        self.model = model
        self.device = self._get_device(device)
        if self.model is not None:
            self.model.to(self.device)
        
        # Training components
        self.criterion = nn.BCELoss()
        self.optimizer = None
        self.scheduler = None
        
        # Logging
        self.setup_logging()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _get_device(self, device: str) -> torch.device:
        """Get the best available device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/deep_training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, dataset_path: str = 'malware_dataset.csv') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare and preprocess the dataset."""
        self.logger.info("Loading and preparing dataset...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Label distribution:\n{df['is_malicious'].value_counts()}")
        
        # Feature engineering
        feature_columns = [
            'file_size_kb', 'entropy_score', 'creation_randomness',
            'behavior_score', 'signature_count', 'content_flags', 'filename_risk'
        ]
        
        # Add pattern flags
        pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
        feature_columns.extend(pattern_columns)
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_features = ['file_size_kb', 'entropy_score', 'creation_randomness', 
                            'behavior_score', 'signature_count', 'content_flags', 'filename_risk']
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        
        # Encode categorical features
        categorical_features = ['extension', 'file_category']
        label_encoders = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].fillna('unknown'))
                X[f'{feature}_encoded'] = df[f'{feature}_encoded']
                label_encoders[feature] = le
        
        # Prepare labels
        y = df['is_malicious'].values
        
        # Convert all features to float
        X = X.astype(float)
        
        # Save preprocessing artifacts
        self.save_preprocessing_artifacts(scaler, label_encoders, list(X.columns))
        
        self.logger.info(f"Final feature count: {X.shape[1]}")
        self.logger.info(f"Feature names: {list(X.columns)}")
        
        return X.values, y, list(X.columns)
    
    def save_preprocessing_artifacts(self, scaler: MinMaxScaler, label_encoders: Dict, feature_names: List[str]):
        """Save preprocessing artifacts for inference."""
        os.makedirs('model', exist_ok=True)
        
        artifacts = {
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': feature_names
        }
        
        with open('model/preprocessing_artifacts.pkl', 'wb') as f:
            pickle.dump(artifacts, f)
        
        self.logger.info("Preprocessing artifacts saved to model/preprocessing_artifacts.pkl")
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = MalwareDataset(X_train, y_train)
        val_dataset = MalwareDataset(X_val, y_val)
        test_dataset = MalwareDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features).squeeze()
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features).squeeze()
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, patience: int = 15) -> Dict:
        """Train the model with early stopping."""
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Device: {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate the model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features).squeeze()
                
                probabilities = outputs.cpu().numpy()
                predictions = (outputs >= 0.5).float().cpu().numpy()
                labels = batch_labels.cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels
        }
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1-Score: {f1:.4f}")
        self.logger.info(f"  AUC: {auc:.4f}")
        
        return results
    
    def save_model(self, model_path: str = 'model/ai_model.pt'):
        """Save the trained model."""
        os.makedirs('model', exist_ok=True)
        
        torch.save(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def save_metrics(self, results: Dict, history: Dict):
        """Save comprehensive metrics and visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to file
        metrics_path = f'logs/deep_model_metrics_{timestamp}.txt'
        with open(metrics_path, 'w') as f:
            f.write("DEEP LEARNING MALWARE DETECTION MODEL METRICS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODEL ARCHITECTURE\n")
            f.write("-" * 20 + "\n")
            f.write(str(self.model) + "\n\n")
            
            f.write("TRAINING PARAMETERS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Optimizer: Adam\n")
            f.write(f"Learning Rate: 0.001\n")
            f.write(f"Loss Function: BCELoss\n")
            f.write(f"Batch Size: 32\n\n")
            
            f.write("TEST RESULTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1']:.4f}\n")
            f.write(f"AUC: {results['auc']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 20 + "\n")
            f.write(str(results['confusion_matrix']) + "\n\n")
        
        # Create visualizations
        self.create_visualizations(results, history, timestamp)
        
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def create_visualizations(self, results: Dict, history: Dict, timestamp: str):
        """Create comprehensive visualizations."""
        os.makedirs('logs', exist_ok=True)
        
        # 1. Training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_accuracies'], label='Train Accuracy')
        plt.plot(history['val_accuracies'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Loss Curves (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'logs/training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Safe', 'Malware'],
                    yticklabels=['Safe', 'Malware'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        auc_score = results['auc']
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'logs/roc_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved with timestamp: {timestamp}")
    
    def initialize_optimizer(self):
        """Initialize optimizer and scheduler after model is set."""
        if self.model is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Deep Learning Malware Detection Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-sizes", nargs='+', type=int, default=[128, 64, 32], help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default='auto', help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--retrain", action='store_true', help="Force retraining")
    parser.add_argument("--save", type=str, default='model/ai_model.pt', help="Model save path")
    
    args = parser.parse_args()
    
    print("🚀 ULTIMATE AI ANTIVIRUS - DEEP LEARNING TRAINING")
    print("=" * 60)
    
    # Check if model exists and retrain flag
    if os.path.exists(args.save) and not args.retrain:
        print(f"Model already exists at {args.save}")
        print("Use --retrain to force retraining")
        return
    
    # Prepare data first
    trainer = DeepLearningTrainer(device=args.device)
    X, y, feature_names = trainer.prepare_data()
    
    # Create model
    input_size = X.shape[1]
    model = FeedForwardClassifier(input_size, args.hidden_sizes, args.dropout)
    trainer.model = model
    trainer.model.to(trainer.device)
    trainer.initialize_optimizer()
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(X, y, args.batch_size)
    
    # Train model
    history = trainer.train(train_loader, val_loader, args.epochs, args.patience)
    
    # Evaluate model
    results = trainer.evaluate(test_loader)
    
    # Save model and metrics
    trainer.save_model(args.save)
    trainer.save_metrics(results, history)
    
    print("\n" + "=" * 60)
    print("🎯 DEEP LEARNING TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Test Accuracy: {results['accuracy']:.4f}")
    print(f"📊 Test F1-Score: {results['f1']:.4f}")
    print(f"📊 Test AUC: {results['auc']:.4f}")
    print(f"💾 Model saved: {args.save}")
    print(f"📈 Visualizations saved: logs/")

if __name__ == "__main__":
    main()