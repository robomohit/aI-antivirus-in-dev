#!/usr/bin/env python3
"""
Enhanced Malware Dataset Generator
Creates a comprehensive dataset with realistic malware patterns and features
"""

import pandas as pd
import numpy as np
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import os

def create_enhanced_dataset(total_samples=10000):
    """Create an enhanced malware dataset with realistic features."""
    
    print("🚀 Creating Enhanced Malware Dataset v6.0")
    print("=" * 50)
    
    # Dataset parameters
    malware_samples = total_samples // 2
    safe_samples = total_samples // 2
    
    data = []
    
    # Enhanced malware patterns
    malware_extensions = ['.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.ps1', '.dll', '.sys']
    malware_names = [
        'trojan', 'virus', 'malware', 'spyware', 'keylogger', 'backdoor', 'rootkit',
        'stealer', 'miner', 'ransomware', 'worm', 'bot', 'hack', 'crack', 'cheat',
        'free_', 'premium_', 'unlock_', 'bypass_', 'admin_', 'system_', 'kernel_',
        'driver_', 'service_', 'daemon_', 'payload', 'inject', 'infect', 'spread',
        'steal_', 'hijack_', 'phish_', 'exploit_', 'vulnerability_', 'zero_day_',
        'apt_', 'nation_state_', 'cyber_', 'digital_', 'electronic_', 'computer_'
    ]
    
    # Safe file patterns
    safe_extensions = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                      '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi', '.zip',
                      '.rar', '.7z', '.py', '.java', '.cpp', '.c', '.html', '.css', '.js',
                      '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.conf']
    safe_names = [
        'document', 'report', 'presentation', 'image', 'photo', 'video', 'music',
        'archive', 'backup', 'data', 'config', 'settings', 'log', 'temp', 'cache',
        'readme', 'license', 'manual', 'guide', 'help', 'info', 'note', 'memo',
        'resume', 'invoice', 'receipt', 'contract', 'agreement', 'policy', 'terms'
    ]
    
    print(f"📊 Generating {malware_samples} malware samples...")
    
    # Generate malware samples with enhanced features
    for i in range(malware_samples):
        # Random malware characteristics
        ext = random.choice(malware_extensions)
        name_base = random.choice(malware_names)
        name = f"{name_base}_{random.randint(1000, 9999)}"
        filename = f"{name}{ext}"
        
        # Enhanced file size (malware varies more)
        file_size_kb = random.randint(5, 5120)
        
        # Enhanced entropy (malware has higher entropy)
        entropy_score = random.uniform(6.5, 8.0)
        
        # Enhanced creation randomness
        creation_randomness = random.uniform(0.8, 1.0)
        
        # Enhanced behavior score (malware has high behavior flags)
        behavior_score = random.randint(7, 10)
        
        # Enhanced signature matches
        signature_count = random.randint(2, 8)
        
        # Enhanced content flags
        content_flags = random.randint(2, 5)
        
        # Enhanced filename risk
        filename_risk = 1
        
        # Enhanced pattern flags with more realistic combinations
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
            'pattern_spread': random.choice([True, False]),
            'pattern_exploit': random.choice([True, False]),
            'pattern_vulnerability': random.choice([True, False]),
            'pattern_phish': random.choice([True, False]),
            'pattern_hijack': random.choice([True, False])
        }
        
        # Enhanced file category
        if ext in ['.exe', '.com', '.scr', '.pif']:
            file_category = 'executable'
        elif ext in ['.bat', '.cmd', '.vbs', '.js', '.ps1']:
            file_category = 'script'
        elif ext in ['.dll', '.sys']:
            file_category = 'driver'
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
    
    # Generate safe samples with enhanced features
    for i in range(safe_samples):
        # Random safe characteristics
        ext = random.choice(safe_extensions)
        name_base = random.choice(safe_names)
        name = f"{name_base}_{random.randint(1000, 9999)}"
        filename = f"{name}{ext}"
        
        # Enhanced file size (safe files vary more)
        file_size_kb = random.randint(1, 20480)
        
        # Enhanced entropy (safe files have lower entropy)
        entropy_score = random.uniform(1.5, 5.5)
        
        # Enhanced creation randomness (safe files created at normal times)
        creation_randomness = random.uniform(0.0, 0.3)
        
        # Enhanced behavior score (safe files have low behavior flags)
        behavior_score = random.randint(0, 2)
        
        # Enhanced signature matches (safe files have no signature matches)
        signature_count = 0
        
        # Enhanced content flags (safe files have no suspicious content)
        content_flags = 0
        
        # Enhanced filename risk (safe names are not risky)
        filename_risk = 0
        
        # Enhanced pattern flags (safe files have no suspicious patterns)
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
            'pattern_spread': False,
            'pattern_exploit': False,
            'pattern_vulnerability': False,
            'pattern_phish': False,
            'pattern_hijack': False
        }
        
        # Enhanced file category
        if ext in ['.txt', '.pdf', '.doc', '.docx']:
            file_category = 'document'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi']:
            file_category = 'media'
        elif ext in ['.zip', '.rar', '.7z']:
            file_category = 'archive'
        elif ext in ['.py', '.java', '.cpp', '.c']:
            file_category = 'script'
        elif ext in ['.html', '.css', '.js', '.json', '.xml']:
            file_category = 'web'
        elif ext in ['.log', '.ini', '.cfg', '.conf']:
            file_category = 'config'
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
    
    print(f"✅ Enhanced dataset created successfully!")
    print(f"📊 Total samples: {len(df)}")
    print(f"🦠 Malware samples: {len(df[df['is_malicious'] == 1])}")
    print(f"✅ Safe samples: {len(df[df['is_malicious'] == 0])}")
    print(f"📁 Saved to: malware_dataset.csv")
    
    # Print enhanced feature summary
    print("\n📈 Enhanced Feature Summary:")
    print(f"   - File size range: {df['file_size_kb'].min():.1f} - {df['file_size_kb'].max():.1f} KB")
    print(f"   - Entropy range: {df['entropy_score'].min():.2f} - {df['entropy_score'].max():.2f}")
    print(f"   - Behavior score range: {df['behavior_score'].min()} - {df['behavior_score'].max()}")
    print(f"   - Extensions: {df['extension'].nunique()} unique types")
    print(f"   - Categories: {df['file_category'].nunique()} categories")
    print(f"   - Pattern features: {len([col for col in df.columns if col.startswith('pattern_')])}")
    
    # Print pattern distribution
    pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
    print(f"\n🔍 Pattern Distribution (Malware vs Safe):")
    for pattern in pattern_cols[:10]:  # Show first 10 patterns
        malware_count = df[(df['is_malicious'] == 1) & (df[pattern] == True)].shape[0]
        safe_count = df[(df['is_malicious'] == 0) & (df[pattern] == True)].shape[0]
        print(f"   {pattern}: {malware_count} malware, {safe_count} safe")
    
    return df

if __name__ == "__main__":
    create_enhanced_dataset()