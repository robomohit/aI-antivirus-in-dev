#!/usr/bin/env python3
"""
Explainability for Ultimate AI Antivirus Deep Model (SHAP)
Saves feature importance to logs/
"""
import torch
import shap
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch.nn as nn

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

os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Load model and preprocessing artifacts
with open('model/preprocessing_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
feature_names = artifacts['feature_names']
scaler = artifacts['scaler']

# Load a sample of the dataset with the exact features used for training
X = pd.read_csv('malware_dataset.csv')
X = X[feature_names].astype(float)
X_sample = X.sample(n=500, random_state=42)
print(f"SHAP: Using {len(feature_names)} features: {feature_names}")

# Recreate model and load state dict
input_size = X_sample.shape[1]
model = FeedForwardClassifier(input_size)
model = torch.load('model/ai_model.pt', map_location='cpu', weights_only=False)
model.eval()

# SHAP explainability
explainer = shap.DeepExplainer(model, torch.FloatTensor(X_sample.values))
shap_values = explainer.shap_values(torch.FloatTensor(X_sample.values))

# Mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
feature_importance = list(zip(feature_names, mean_abs_shap))
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Save top-10 features to txt
with open(f'logs/feature_importance_shap_{timestamp}.txt', 'w') as f:
    f.write('Top 10 SHAP Feature Importances\n')
    f.write('='*40+'\n')
    for i, (feat, val) in enumerate(feature_importance[:10]):
        f.write(f'{i+1:2d}. {feat:<30} {val:.5f}\n')

# Bar plot
plt.figure(figsize=(10,6))
plt.bar([f[0] for f in feature_importance[:10]], [f[1] for f in feature_importance[:10]])
plt.title('Top 10 SHAP Feature Importances')
plt.ylabel('Mean |SHAP value|')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'logs/feature_importance_shap_{timestamp}.png')
plt.close()

print('SHAP feature importance saved to logs/.')