#!/usr/bin/env python3
"""
Create the complete malware dataset with feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
from utils import get_entropy, get_file_type, get_filename_pattern_flags

def extract_features(file_path: Path, label: int, source: str, method: str) -> dict:
    """Extract features from a file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Basic features
        file_size_kb = len(content) / 1024
        entropy_score = get_entropy(content)
        extension = file_path.suffix.lower()
        filename = file_path.name
        
        # Pattern flags
        pattern_flags = get_filename_pattern_flags(filename)
        
        # File type
        file_category = get_file_type(filename)
        
        # Hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Creation randomness (simulated)
        creation_randomness = np.random.random()
        
        # Threat level based on source
        if source == 'eicar':
            threat_level = 'HIGH_RISK'
        elif source == 'simulated':
            threat_level = 'SUSPICIOUS'
        else:
            threat_level = 'SAFE'
        
        return {
            'filename': filename,
            'file_path': str(file_path),
            'file_size_kb': file_size_kb,
            'entropy_score': entropy_score,
            'extension': extension,
            'file_category': file_category,
            'creation_randomness': creation_randomness,
            'pattern_hack': pattern_flags.get('hack', False),
            'pattern_steal': pattern_flags.get('steal', False),
            'pattern_crack': pattern_flags.get('crack', False),
            'pattern_keygen': pattern_flags.get('keygen', False),
            'pattern_cheat': pattern_flags.get('cheat', False),
            'pattern_free': pattern_flags.get('free', False),
            'pattern_cracked': pattern_flags.get('cracked', False),
            'pattern_premium': pattern_flags.get('premium', False),
            'pattern_unlock': pattern_flags.get('unlock', False),
            'pattern_bypass': pattern_flags.get('bypass', False),
            'pattern_admin': pattern_flags.get('admin', False),
            'pattern_root': pattern_flags.get('root', False),
            'pattern_system': pattern_flags.get('system', False),
            'pattern_kernel': pattern_flags.get('kernel', False),
            'pattern_driver': pattern_flags.get('driver', False),
            'pattern_service': pattern_flags.get('service', False),
            'pattern_daemon': pattern_flags.get('daemon', False),
            'pattern_bot': pattern_flags.get('bot', False),
            'pattern_miner': pattern_flags.get('miner', False),
            'pattern_malware': pattern_flags.get('malware', False),
            'pattern_virus': pattern_flags.get('virus', False),
            'pattern_infect': pattern_flags.get('infect', False),
            'pattern_spread': pattern_flags.get('spread', False),
            'sha256_hash': file_hash,
            'label': label,
            'source': source,
            'method': method,
            'threat_level': threat_level,
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_dataset():
    """Create the complete dataset."""
    print("🔍 Creating malware dataset...")
    
    dataset_rows = []
    
    # Process EICAR files
    print("📁 Processing EICAR files...")
    eicar_dir = Path("test_files/eicar")
    for file_path in eicar_dir.glob("*"):
        features = extract_features(file_path, label=1, source='eicar', method='SIGNATURE')
        if features:
            dataset_rows.append(features)
    
    # Process malware files
    print("🦠 Processing malware files...")
    malware_dir = Path("test_files/malware")
    for i, file_path in enumerate(malware_dir.glob("*")):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(list(malware_dir.glob('*')))}")
        features = extract_features(file_path, label=1, source='simulated', method='AI')
        if features:
            dataset_rows.append(features)
    
    # Process safe files
    print("✅ Processing safe files...")
    safe_dir = Path("test_files/safe")
    for i, file_path in enumerate(safe_dir.glob("*")):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(list(safe_dir.glob('*')))}")
        features = extract_features(file_path, label=0, source='safe', method='SAFE')
        if features:
            dataset_rows.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(dataset_rows)
    
    # Save dataset
    df.to_csv('malware_dataset.csv', index=False)
    
    print(f"🎯 Dataset created with {len(df)} samples")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    print(f"📁 Source distribution:")
    print(df['source'].value_counts())
    
    return df

if __name__ == "__main__":
    create_dataset()