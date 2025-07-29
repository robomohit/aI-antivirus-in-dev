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
from sklearn.preprocessing import LabelEncoder

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
model = torch.load('model/ai_model.pt', map_location='cpu', weights_only=False)
model.eval()
with open('model/preprocessing_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

feature_names = artifacts['feature_names']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']

print(f"Model expects {len(feature_names)} features: {feature_names}")

# Load the original dataset and recreate encoded columns
df = pd.read_csv('malware_dataset.csv')
print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Recreate encoded columns
for feature in ['extension', 'file_category']:
    if feature in df.columns and feature in label_encoders:
        le = label_encoders[feature]
        df[f'{feature}_encoded'] = le.transform(df[feature].fillna('unknown'))

# Select only the features used by the model
X = df[feature_names].astype(float)
print(f"Final X shape: {X.shape}")
print(f"Feature count matches model: {X.shape[1] == len(feature_names)}")

# Sample for SHAP (use smaller sample for faster computation)
X_sample = X.sample(n=200, random_state=42)
print(f"SHAP sample shape: {X_sample.shape}")

# Create SHAP explainer
explainer = shap.DeepExplainer(model, torch.FloatTensor(X_sample.values))
shap_values = explainer.shap_values(torch.FloatTensor(X_sample.values))

# Handle SHAP values shape (could be list or array)
print(f"SHAP values type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"SHAP values list length: {len(shap_values)}")
    shap_values = shap_values[0]  # Take first element if it's a list

print(f"SHAP values shape: {shap_values.shape}")

# Get feature importance (mean absolute SHAP values)
feature_importance = np.mean(np.abs(shap_values), axis=0)
feature_importance = feature_importance.squeeze()  # Remove extra dimension
print(f"Feature importance shape: {feature_importance.shape}")
print(f"Feature names length: {len(feature_names)}")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Save feature importance
importance_path = f'logs/feature_importance_{timestamp}.txt'
with open(importance_path, 'w') as f:
    f.write("FEATURE IMPORTANCE ANALYSIS (SHAP)\n")
    f.write("=" * 50 + "\n\n")
    f.write("Top 20 Most Important Features:\n")
    f.write("-" * 30 + "\n")
    for idx, row in importance_df.head(20).iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    f.write(f"\nTotal features analyzed: {len(feature_names)}\n")
    f.write(f"SHAP sample size: {len(X_sample)}\n")

# Create feature importance plot
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('SHAP Importance')
plt.title('Top 15 Feature Importances (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'logs/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# Create SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(f'logs/shap_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFeature importance saved to: {importance_path}")
print(f"Feature importance plot saved to: logs/feature_importance_{timestamp}.png")
print(f"SHAP summary plot saved to: logs/shap_summary_{timestamp}.png")