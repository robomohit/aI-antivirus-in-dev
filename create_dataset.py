#!/usr/bin/env python3
"""
Create Enhanced Malware Dataset for AI Training
Generates a comprehensive dataset with realistic features for malware detection
"""

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from utils import scan_file_content, check_behavior_flags, get_file_hash

def create_malware_dataset():
    """Create a comprehensive malware dataset for AI training."""
    
    print("🚀 Creating Enhanced Malware Dataset v5.X")
    print("=" * 50)
    
    # Dataset parameters
    total_samples = 5000
    malware_samples = 2500
    safe_samples = 2500
    
    data = []
    
    # Malware file types and patterns
    malware_extensions = ['.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.ps1']
    malware_names = [
        'trojan', 'virus', 'malware', 'spyware', 'keylogger', 'backdoor', 'rootkit',
        'stealer', 'miner', 'ransomware', 'worm', 'bot', 'hack', 'crack', 'cheat',
        'free_', 'premium_', 'unlock_', 'bypass_', 'admin_', 'system_', 'kernel_',
        'driver_', 'service_', 'daemon_', 'payload', 'inject', 'infect', 'spread'
    ]
    
    # Safe file types
    safe_extensions = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                      '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi', '.zip',
                      '.rar', '.7z', '.py', '.java', '.cpp', '.c', '.html', '.css', '.js']
    safe_names = [
        'document', 'report', 'presentation', 'image', 'photo', 'video', 'music',
        'archive', 'backup', 'data', 'config', 'settings', 'log', 'temp', 'cache',
        'readme', 'license', 'manual', 'guide', 'help', 'info', 'note', 'memo'
    ]
    
    print(f"📊 Generating {malware_samples} malware samples...")
    
    # Generate malware samples
    for i in range(malware_samples):
        # Random malware characteristics
        ext = random.choice(malware_extensions)
        name_base = random.choice(malware_names)
        name = f"{name_base}_{random.randint(1000, 9999)}"
        filename = f"{name}{ext}"
        
        # File size (malware tends to be smaller)
        file_size_kb = random.randint(10, 2048)
        
        # Entropy (malware has higher entropy)
        entropy_score = random.uniform(6.0, 8.0)
        
        # Creation randomness (malware often created at odd times)
        creation_randomness = random.uniform(0.7, 1.0)
        
        # Behavior score (malware has high behavior flags)
        behavior_score = random.randint(6, 10)
        
        # Signature matches (malware has signature matches)
        signature_count = random.randint(1, 5)
        
        # Content flags (malware has suspicious content)
        content_flags = random.randint(1, 3)
        
        # Filename risk (malware names are risky)
        filename_risk = 1
        
        # Pattern flags
        pattern_flags = {
            'pattern_hack': random.choice([True, False]),
            'pattern_steal': random.choice([True, False]),
            'pattern_crack': random.choice([True, False]),
            'pattern_keygen': random.choice([True, False]),
            'pattern_cheat': random.choice([True, False]),
            'pattern_free': random.choice([True, False]),
            'pattern_cracked': random.choice([True, False]),
            'pattern_premium': random.choice([True, False]),
            'pattern_unlock': random.choice([True, False]),
            'pattern_bypass': random.choice([True, False]),
            'pattern_admin': random.choice([True, False]),
            'pattern_root': random.choice([True, False]),
            'pattern_system': random.choice([True, False]),
            'pattern_kernel': random.choice([True, False]),
            'pattern_driver': random.choice([True, False]),
            'pattern_service': random.choice([True, False]),
            'pattern_daemon': random.choice([True, False]),
            'pattern_bot': random.choice([True, False]),
            'pattern_miner': random.choice([True, False]),
            'pattern_malware': random.choice([True, False]),
            'pattern_virus': random.choice([True, False]),
            'pattern_infect': random.choice([True, False]),
            'pattern_spread': random.choice([True, False])
        }
        
        # File category
        if ext in ['.exe', '.com', '.scr', '.pif']:
            file_category = 'executable'
        elif ext in ['.bat', '.cmd', '.vbs', '.js', '.ps1']:
            file_category = 'script'
        else:
            file_category = 'other'
        
        # Create sample
        sample = {
            'filename': filename,
            'file_size_kb': file_size_kb,
            'entropy_score': entropy_score,
            'extension': ext,
            'creation_randomness': creation_randomness,
            'behavior_score': behavior_score,
            'signature_count': signature_count,
            'content_flags': content_flags,
            'filename_risk': filename_risk,
            'file_category': file_category,
            'is_malicious': 1
        }
        
        # Add pattern flags
        sample.update(pattern_flags)
        
        data.append(sample)
    
    print(f"📊 Generating {safe_samples} safe samples...")
    
    # Generate safe samples
    for i in range(safe_samples):
        # Random safe characteristics
        ext = random.choice(safe_extensions)
        name_base = random.choice(safe_names)
        name = f"{name_base}_{random.randint(1000, 9999)}"
        filename = f"{name}{ext}"
        
        # File size (safe files vary in size)
        file_size_kb = random.randint(1, 10240)
        
        # Entropy (safe files have lower entropy)
        entropy_score = random.uniform(2.0, 6.0)
        
        # Creation randomness (safe files created at normal times)
        creation_randomness = random.uniform(0.0, 0.5)
        
        # Behavior score (safe files have low behavior flags)
        behavior_score = random.randint(0, 3)
        
        # Signature matches (safe files have no signature matches)
        signature_count = 0
        
        # Content flags (safe files have no suspicious content)
        content_flags = 0
        
        # Filename risk (safe names are not risky)
        filename_risk = 0
        
        # Pattern flags (safe files have no suspicious patterns)
        pattern_flags = {
            'pattern_hack': False,
            'pattern_steal': False,
            'pattern_crack': False,
            'pattern_keygen': False,
            'pattern_cheat': False,
            'pattern_free': False,
            'pattern_cracked': False,
            'pattern_premium': False,
            'pattern_unlock': False,
            'pattern_bypass': False,
            'pattern_admin': False,
            'pattern_root': False,
            'pattern_system': False,
            'pattern_kernel': False,
            'pattern_driver': False,
            'pattern_service': False,
            'pattern_daemon': False,
            'pattern_bot': False,
            'pattern_miner': False,
            'pattern_malware': False,
            'pattern_virus': False,
            'pattern_infect': False,
            'pattern_spread': False
        }
        
        # File category
        if ext in ['.txt', '.pdf', '.doc', '.docx']:
            file_category = 'document'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi']:
            file_category = 'media'
        elif ext in ['.zip', '.rar', '.7z']:
            file_category = 'archive'
        elif ext in ['.py', '.java', '.cpp', '.c']:
            file_category = 'script'
        elif ext in ['.html', '.css', '.js']:
            file_category = 'web'
        else:
            file_category = 'other'
        
        # Create sample
        sample = {
            'filename': filename,
            'file_size_kb': file_size_kb,
            'entropy_score': entropy_score,
            'extension': ext,
            'creation_randomness': creation_randomness,
            'behavior_score': behavior_score,
            'signature_count': signature_count,
            'content_flags': content_flags,
            'filename_risk': filename_risk,
            'file_category': file_category,
            'is_malicious': 0
        }
        
        # Add pattern flags
        sample.update(pattern_flags)
        
        data.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save dataset
    df.to_csv('malware_dataset.csv', index=False)
    
    print(f"✅ Dataset created successfully!")
    print(f"📊 Total samples: {len(df)}")
    print(f"🦠 Malware samples: {len(df[df['is_malicious'] == 1])}")
    print(f"✅ Safe samples: {len(df[df['is_malicious'] == 0])}")
    print(f"📁 Saved to: malware_dataset.csv")
    
    # Print feature summary
    print("\n📈 Feature Summary:")
    print(f"   - File size range: {df['file_size_kb'].min():.1f} - {df['file_size_kb'].max():.1f} KB")
    print(f"   - Entropy range: {df['entropy_score'].min():.2f} - {df['entropy_score'].max():.2f}")
    print(f"   - Behavior score range: {df['behavior_score'].min()} - {df['behavior_score'].max()}")
    print(f"   - Extensions: {df['extension'].nunique()} unique types")
    print(f"   - Categories: {df['file_category'].nunique()} categories")
    
    return df

if __name__ == "__main__":
    create_malware_dataset()