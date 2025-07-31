#!/usr/bin/env python3
"""
Create Realistic Malware Dataset
Generate realistic malware and benign samples with proper features
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from colorama import init, Fore, Style
import hashlib
import random

# Initialize colorama
init()

class RealisticDatasetCreator:
    def __init__(self):
        self.features_dir = "malware_features"
        Path(self.features_dir).mkdir(exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
    
    def create_realistic_malware_features(self, count=500):
        """Create realistic malware features based on real patterns."""
        print(f"{Fore.CYAN}🔄 Creating {count} realistic malware samples...")
        
        malware_features = []
        
        # Real malware patterns (based on actual malware characteristics)
        malware_patterns = [
            # Ransomware patterns
            {"entropy_range": (7.0, 8.0), "printable_range": (0.1, 0.3), "size_range": (50000, 2000000)},
            # Trojan patterns
            {"entropy_range": (6.5, 7.5), "printable_range": (0.2, 0.4), "size_range": (100000, 500000)},
            # Cryptominer patterns
            {"entropy_range": (7.2, 8.0), "printable_range": (0.1, 0.25), "size_range": (200000, 1000000)},
            # Keylogger patterns
            {"entropy_range": (6.0, 7.0), "printable_range": (0.3, 0.5), "size_range": (50000, 300000)},
            # Backdoor patterns
            {"entropy_range": (6.8, 7.8), "printable_range": (0.15, 0.35), "size_range": (80000, 400000)}
        ]
        
        for i in range(count):
            # Select random pattern
            pattern = random.choice(malware_patterns)
            
            # Generate realistic features
            entropy = np.random.uniform(*pattern["entropy_range"])
            printable_ratio = np.random.uniform(*pattern["printable_range"])
            file_size = np.random.randint(*pattern["size_range"])
            
            # Generate other features
            strings_count = np.random.randint(1, 50)  # Malware has fewer strings
            avg_string_length = np.random.uniform(2.0, 8.0)  # Short strings
            histogram_regularity = np.random.uniform(0.6, 1.0)  # High irregularity
            entropy_consistency = np.random.uniform(0.8, 1.2)  # High consistency
            
            # Create feature dict
            features = {
                'file_size': file_size,
                'entropy': entropy,
                'strings_count': strings_count,
                'avg_string_length': avg_string_length,
                'printable_ratio': printable_ratio,
                'histogram_regularity': histogram_regularity,
                'entropy_consistency': entropy_consistency,
                'sha256_hash': f"malware_{hashlib.md5(str(i).encode()).hexdigest()[:16]}",
                'file_name': f"malware_sample_{i}.exe",
                'file_type': 'exe',
                'first_seen': datetime.now().isoformat(),
                'label': 1  # Malware
            }
            
            malware_features.append(features)
        
        print(f"{Fore.GREEN}✅ Created {len(malware_features)} malware samples")
        return malware_features
    
    def create_realistic_benign_features(self, count=500):
        """Create realistic benign features based on legitimate software."""
        print(f"{Fore.CYAN}🔄 Creating {count} realistic benign samples...")
        
        benign_features = []
        
        # Benign file patterns
        benign_patterns = [
            # Text files
            {"entropy_range": (3.0, 4.5), "printable_range": (0.8, 0.95), "size_range": (100, 100000)},
            # Configuration files
            {"entropy_range": (4.0, 5.5), "printable_range": (0.7, 0.9), "size_range": (500, 50000)},
            # Legitimate executables
            {"entropy_range": (5.0, 6.5), "printable_range": (0.4, 0.7), "size_range": (10000, 50000000)},
            # System files
            {"entropy_range": (5.5, 6.8), "printable_range": (0.3, 0.6), "size_range": (5000, 1000000)},
            # Document files
            {"entropy_range": (4.5, 6.0), "printable_range": (0.6, 0.8), "size_range": (1000, 100000)}
        ]
        
        for i in range(count):
            # Select random pattern
            pattern = random.choice(benign_patterns)
            
            # Generate realistic features
            entropy = np.random.uniform(*pattern["entropy_range"])
            printable_ratio = np.random.uniform(*pattern["printable_range"])
            file_size = np.random.randint(*pattern["size_range"])
            
            # Generate other features
            strings_count = np.random.randint(10, 1000)  # Benign has more strings
            avg_string_length = np.random.uniform(5.0, 20.0)  # Longer strings
            histogram_regularity = np.random.uniform(0.1, 0.4)  # Low irregularity
            entropy_consistency = np.random.uniform(0.1, 0.6)  # Low consistency
            
            # Create feature dict
            features = {
                'file_size': file_size,
                'entropy': entropy,
                'strings_count': strings_count,
                'avg_string_length': avg_string_length,
                'printable_ratio': printable_ratio,
                'histogram_regularity': histogram_regularity,
                'entropy_consistency': entropy_consistency,
                'sha256_hash': f"benign_{hashlib.md5(str(i).encode()).hexdigest()[:16]}",
                'file_name': f"benign_file_{i}.txt",
                'file_type': 'text',
                'first_seen': datetime.now().isoformat(),
                'label': 0  # Benign
            }
            
            benign_features.append(features)
        
        print(f"{Fore.GREEN}✅ Created {len(benign_features)} benign samples")
        return benign_features
    
    def save_dataset(self, malware_features, benign_features):
        """Save the complete dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realistic_malware_dataset_{timestamp}.csv"
        
        # Combine datasets
        all_features = malware_features + benign_features
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Save to CSV
        csv_path = Path(self.features_dir) / filename
        df.to_csv(csv_path, index=False)
        
        print(f"{Fore.GREEN}✅ Dataset saved: {csv_path}")
        print(f"{Fore.CYAN}📊 Total samples: {len(df)}")
        print(f"{Fore.CYAN}📊 Malware samples: {len(malware_features)}")
        print(f"{Fore.CYAN}📊 Benign samples: {len(benign_features)}")
        
        # Show feature statistics
        print(f"\n{Fore.YELLOW}📊 Feature Statistics:")
        print(f"   Malware Entropy: {df[df['label']==1]['entropy'].mean():.2f} ± {df[df['label']==1]['entropy'].std():.2f}")
        print(f"   Benign Entropy: {df[df['label']==0]['entropy'].mean():.2f} ± {df[df['label']==0]['entropy'].std():.2f}")
        print(f"   Malware Printable Ratio: {df[df['label']==1]['printable_ratio'].mean():.2f} ± {df[df['label']==1]['printable_ratio'].std():.2f}")
        print(f"   Benign Printable Ratio: {df[df['label']==0]['printable_ratio'].mean():.2f} ± {df[df['label']==0]['printable_ratio'].std():.2f}")
        
        return csv_path
    
    def run_creation_process(self, malware_count=500, benign_count=500):
        """Run the complete dataset creation process."""
        print(f"{Fore.CYAN}🛡️  REALISTIC DATASET CREATION")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Step 1: Create malware samples
        malware_features = self.create_realistic_malware_features(malware_count)
        
        # Step 2: Create benign samples
        benign_features = self.create_realistic_benign_features(benign_count)
        
        # Step 3: Save dataset
        dataset_path = self.save_dataset(malware_features, benign_features)
        
        print(f"\n{Fore.GREEN}🎉 DATASET CREATION COMPLETE!")
        print(f"{Fore.CYAN}📊 Realistic malware patterns created")
        print(f"{Fore.CYAN}📊 Realistic benign patterns created")
        print(f"{Fore.CYAN}📁 Dataset ready for training: {dataset_path}")
        
        return dataset_path

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting Realistic Dataset Creation...")
    
    creator = RealisticDatasetCreator()
    creator.run_creation_process(malware_count=500, benign_count=500)

if __name__ == "__main__":
    main()