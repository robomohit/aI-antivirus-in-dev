#!/usr/bin/env python3
"""
Deep Learning AI Antivirus Model Training with PyTorch
Trains a neural network for malware detection using static features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class MalwareClassifier(nn.Module):
    """Deep learning model for malware classification."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(MalwareClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ============================================================================
# DATASET CLASS
# ============================================================================

class MalwareDataset(Dataset):
    """PyTorch Dataset for malware classification."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data():
    """Load and prepare the dataset for PyTorch training."""
    print("📊 Loading malware dataset...")
    df = pd.read_csv('malware_dataset.csv')
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🎯 Label distribution:")
    print(df['label'].value_counts())
    
    # Prepare numerical features
    numerical_features = [
        'file_size_kb', 'entropy_score', 'creation_randomness',
        'behavior_score', 'signature_count', 'content_flags', 'filename_risk'
    ]
    
    # Prepare binary features (pattern flags)
    binary_features = [
        'pattern_hack', 'pattern_steal', 'pattern_crack', 'pattern_keygen',
        'pattern_cheat', 'pattern_free', 'pattern_cracked', 'pattern_premium',
        'pattern_unlock', 'pattern_bypass', 'pattern_admin', 'pattern_root',
        'pattern_system', 'pattern_kernel', 'pattern_driver', 'pattern_service',
        'pattern_daemon', 'pattern_bot', 'pattern_miner', 'pattern_malware',
        'pattern_virus', 'pattern_infect', 'pattern_spread'
    ]
    
    # Prepare categorical features
    categorical_features = ['file_category', 'extension']
    
    # Combine all features
    feature_columns = numerical_features + binary_features + categorical_features
    
    # Extract features and labels
    X = df[feature_columns].copy()
    y = df['label'].values
    
    # Handle missing values
    X = X.fillna(0)
    
    # Encode categorical features
    label_encoders = {}
    for cat_col in categorical_features:
        le = LabelEncoder()
        X[cat_col] = le.fit_transform(X[cat_col].astype(str))
        label_encoders[cat_col] = le
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Convert to numpy array
    X_array = X.values.astype(np.float32)
    
    print(f"📊 Features: {X_array.shape[1]}")
    print(f"📊 Samples: {len(X_array)}")
    print(f"🎯 Labels: {np.bincount(y)}")
    
    return X_array, y, scaler, label_encoders

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into train/validation/test sets."""
    # First split: 80% for train/val, 20% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: 75% of remaining for train, 25% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    print(f"📚 Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"🔍 Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"🧪 Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    return total_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Store predictions for metrics
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return total_loss / len(dataloader), correct / total, all_predictions, all_labels

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, input_size):
    """Train the PyTorch model."""
    print("🧠 Training PyTorch model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Create datasets
    train_dataset = MalwareDataset(X_train, y_train)
    val_dataset = MalwareDataset(X_val, y_val)
    test_dataset = MalwareDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = MalwareClassifier(input_size=input_size).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"🔄 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'model/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('model/best_model.pt'))
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)
    
    print(f"📊 Final Test Accuracy: {test_acc:.4f}")
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'test_acc': test_acc
    }

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    labels = np.array(all_labels).flatten()
    
    # Compute metrics
    predictions_binary = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(labels, predictions_binary)
    precision = precision_score(labels, predictions_binary)
    recall = recall_score(labels, predictions_binary)
    f1 = f1_score(labels, predictions_binary)
    auc = roc_auc_score(labels, predictions)
    
    print(f"📊 MODEL METRICS:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': predictions,
        'labels': labels
    }

def save_visualizations(history, metrics, timestamp):
    """Save training visualizations."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # 1. Training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.plot(history['val_accs'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'logs/training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['predictions'])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/roc_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(metrics['labels'], (metrics['predictions'] > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Malware'], 
                yticklabels=['Safe', 'Malware'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved with timestamp: {timestamp}")

def save_model_and_metrics(model, metrics, history, timestamp):
    """Save model and metrics."""
    # Create model directory
    Path("model").mkdir(exist_ok=True)
    
    # Save model
    torch.save(model, f'model/ai_model_{timestamp}.pt')
    torch.save(model.state_dict(), f'model/ai_model_state_{timestamp}.pt')
    
    # Save metrics
    metrics_text = f"""
PYTORCH AI ANTIVIRUS MODEL METRICS
==================================
Timestamp: {timestamp}

MODEL PERFORMANCE:
=================
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
AUC:       {metrics['auc']:.4f}

TRAINING HISTORY:
================
Final Train Accuracy: {history['train_accs'][-1]:.4f}
Final Val Accuracy:   {history['val_accs'][-1]:.4f}
Final Test Accuracy:  {history['test_acc']:.4f}

MODEL ARCHITECTURE:
==================
- Input Size: {model.network[0].in_features}
- Hidden Layers: {len([m for m in model.modules() if isinstance(m, nn.Linear)]) - 1}
- Activation: ReLU
- Dropout: 0.3
- Output: Sigmoid (Binary Classification)

SAVED FILES:
============
- Model: model/ai_model_{timestamp}.pt
- State Dict: model/ai_model_state_{timestamp}.pt
- Visualizations: logs/
"""
    
    with open(f'logs/model_metrics_{timestamp}.txt', 'w') as f:
        f.write(metrics_text)
    
    print(f"💾 Model saved to: model/ai_model_{timestamp}.pt")
    print(f"📄 Metrics saved to: logs/model_metrics_{timestamp}.txt")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PyTorch malware classifier')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()
    
    print("🚀 PYTORCH AI ANTIVIRUS MODEL TRAINING")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path('malware_dataset.csv').exists():
        print("❌ Dataset not found. Please run create_dataset.py first.")
        return
    
    # Prepare data
    X, y, scaler, label_encoders = prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train model
    model, history = train_model(X_train, X_val, X_test, y_train, y_val, y_test, X.shape[1])
    
    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = MalwareDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    metrics = evaluate_model(model, test_loader, device)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_visualizations(history, metrics, timestamp)
    save_model_and_metrics(model, metrics, history, timestamp)
    
    print("=" * 50)
    print("🎯 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"📊 Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 Test F1-Score: {metrics['f1']:.4f}")
    print(f"📊 Test AUC: {metrics['auc']:.4f}")
    print(f"💾 Model saved: model/ai_model_{timestamp}.pt")
    print(f"📄 Metrics saved: logs/model_metrics_{timestamp}.txt")
    print(f"📊 Visualizations saved: logs/")

if __name__ == "__main__":
    main()