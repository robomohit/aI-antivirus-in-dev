#!/usr/bin/env python3
"""
Test script for PyTorch model integration
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

# Define the model class
class FeedForwardClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(FeedForwardClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def test_model_loading():
    """Test loading the trained model."""
    try:
        # Load model
        model = torch.load('model/ai_model.pt', map_location='cpu', weights_only=False)
        model.eval()
        print("✅ Model loaded successfully")
        
        # Load preprocessing artifacts
        with open('model/preprocessing_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        feature_names = artifacts['feature_names']
        scaler = artifacts['scaler']
        print(f"✅ Preprocessing artifacts loaded. Features: {len(feature_names)}")
        
        # Test prediction
        test_features = np.random.rand(1, 36)  # 36 features
        test_tensor = torch.FloatTensor(test_features)
        
        with torch.no_grad():
            prediction = model(test_tensor)
            probability = prediction.item()
        
        print(f"✅ Test prediction successful: {probability:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing PyTorch model integration...")
    success = test_model_loading()
    if success:
        print("🎉 All tests passed! Model integration is working.")
    else:
        print("💥 Tests failed. Model integration needs fixing.")