#!/usr/bin/env python3
"""
EDA for Ultimate AI Antivirus Dataset
Saves plots to logs/ for review
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True)
df = pd.read_csv('malware_dataset.csv')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. Class balance
plt.figure(figsize=(5,4))
df['is_malicious'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Class Balance')
plt.xticks([0,1], ['Safe','Malware'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'logs/eda_class_balance_{timestamp}.png')
plt.close()

# 2. Feature histograms
features = ['file_size_kb','entropy_score','behavior_score','signature_count','content_flags','filename_risk']
for feat in features:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df, x=feat, bins=30, kde=True, hue='is_malicious', palette=['green','red'], element='step')
    plt.title(f'{feat} Distribution')
    plt.tight_layout()
    plt.savefig(f'logs/eda_{feat}_{timestamp}.png')
    plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(12,10))
corr = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'logs/eda_corr_heatmap_{timestamp}.png')
plt.close()

# 4. Pattern feature prevalence
pattern_cols = [c for c in df.columns if c.startswith('pattern_')]
pattern_sums = df[pattern_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
pattern_sums.plot(kind='bar')
plt.title('Pattern Feature Prevalence (Malware+Safe)')
plt.tight_layout()
plt.savefig(f'logs/eda_pattern_prevalence_{timestamp}.png')
plt.close()

print('EDA plots saved to logs/.')