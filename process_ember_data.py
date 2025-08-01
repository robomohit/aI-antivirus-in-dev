#!/usr/bin/env python3
"""
Process EMBER Dataset
Convert EMBER JSONL files to CSV format for training
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from colorama import init, Fore, Style
import logging

# Initialize colorama
init()

class EMBERDataProcessor:
    def __init__(self):
        self.ember_dir = "ember_dataset/ember2018"
        self.output_dir = "ember_dataset"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def process_jsonl_file(self, file_path):
        """Process a single JSONL file."""
        print(f"📊 Processing: {file_path.name}")
        
        data = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    # Parse JSON line
                    sample = json.loads(line.strip())
                    data.append(sample)
                    
                    # Progress indicator
                    if line_num % 10000 == 0:
                        print(f"   Processed {line_num:,} lines...")
                        
                except json.JSONDecodeError as e:
                    print(f"   Warning: Invalid JSON at line {line_num}: {e}")
                    continue
        
        print(f"   ✅ Processed {len(data):,} samples from {file_path.name}")
        return data
    
    def process_all_ember_data(self):
        """Process all EMBER data files."""
        print(f"{Fore.CYAN}🔄 Processing EMBER dataset...")
        
        # Find all JSONL files
        jsonl_files = list(Path(self.ember_dir).glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"{Fore.RED}❌ No JSONL files found in {self.ember_dir}")
            return False
        
        print(f"{Fore.GREEN}✅ Found {len(jsonl_files)} JSONL files")
        
        # Process each file
        all_data = []
        
        for file_path in jsonl_files:
            file_data = self.process_jsonl_file(file_path)
            all_data.extend(file_data)
        
        # Convert to DataFrame
        print(f"{Fore.CYAN}🔄 Converting to DataFrame...")
        df = pd.DataFrame(all_data)
        
        # Save processed data
        output_path = Path(self.output_dir) / "ember_processed.csv"
        df.to_csv(output_path, index=False)
        
        print(f"{Fore.GREEN}✅ Processed data saved to: {output_path}")
        print(f"📊 Total samples: {len(df):,}")
        
        # Show data distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"📊 Label distribution:")
            for label, count in label_counts.items():
                print(f"   Label {label}: {count:,} samples")
        
        # Show feature columns
        feature_cols = [col for col in df.columns if col not in ['label', 'sha256']]
        print(f"📊 Features: {len(feature_cols)} columns")
        print(f"   Feature columns: {feature_cols[:5]}...")
        
        return True
    
    def create_synthetic_data(self):
        """Create synthetic malware data to supplement EMBER."""
        print(f"{Fore.CYAN}🔄 Creating synthetic malware data...")
        
        # Load processed EMBER data
        ember_path = Path(self.output_dir) / "ember_processed.csv"
        if not ember_path.exists():
            print(f"{Fore.RED}❌ Processed EMBER data not found!")
            return None
        
        df = pd.read_csv(ember_path)
        
        # Create synthetic malware samples
        synthetic_samples = []
        
        # Generate synthetic malware with realistic features
        for i in range(1000):
            # High entropy, large size, low printable ratio = malware characteristics
            sample = {
                'file_size': 50000 + (i * 1000),  # Large files
                'entropy': 7.0 + (i % 10) * 0.1,  # High entropy
                'strings_count': 100 + (i % 50),
                'avg_string_length': 5 + (i % 10),
                'max_string_length': 20 + (i % 30),
                'printable_ratio': 0.3 + (i % 20) * 0.02,  # Low printable ratio
                'histogram_regularity': 0.2 + (i % 30) * 0.01,
                'entropy_consistency': 0.3 + (i % 40) * 0.01,
                'label': 1,  # Malware
                'malware_type': 'synthetic'
            }
            synthetic_samples.append(sample)
        
        # Create synthetic benign samples
        for i in range(1000):
            # Low entropy, smaller size, high printable ratio = benign characteristics
            sample = {
                'file_size': 1000 + (i * 100),  # Smaller files
                'entropy': 4.0 + (i % 10) * 0.1,  # Lower entropy
                'strings_count': 500 + (i % 100),
                'avg_string_length': 15 + (i % 10),
                'max_string_length': 50 + (i % 50),
                'printable_ratio': 0.8 + (i % 15) * 0.01,  # High printable ratio
                'histogram_regularity': 0.7 + (i % 20) * 0.01,
                'entropy_consistency': 0.8 + (i % 15) * 0.01,
                'label': 0,  # Benign
                'malware_type': 'synthetic_benign'
            }
            synthetic_samples.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        # Combine with EMBER data
        combined_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Save combined data
        combined_path = Path(self.output_dir) / "ember_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        
        print(f"{Fore.GREEN}✅ Combined data saved to: {combined_path}")
        print(f"📊 Total samples: {len(combined_df):,}")
        
        # Show final distribution
        label_counts = combined_df['label'].value_counts()
        print(f"📊 Final label distribution:")
        for label, count in label_counts.items():
            print(f"   Label {label}: {count:,} samples")
        
        return combined_df
    
    def run_processing(self):
        """Run the complete processing pipeline."""
        print(f"{Fore.CYAN}🛡️  EMBER DATA PROCESSING")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Step 1: Process EMBER data
        print(f"\n{Fore.CYAN}📦 Step 1: Processing EMBER data")
        if not self.process_all_ember_data():
            return False
        
        # Step 2: Create synthetic data
        print(f"\n{Fore.CYAN}📦 Step 2: Creating synthetic data")
        combined_df = self.create_synthetic_data()
        if combined_df is None:
            return False
        
        print(f"\n{Fore.GREEN}🎉 EMBER data processing complete!")
        return True

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting EMBER Data Processing...")
    
    processor = EMBERDataProcessor()
    success = processor.run_processing()
    
    if success:
        print(f"{Fore.GREEN}✅ EMBER data processing completed successfully!")
    else:
        print(f"{Fore.RED}❌ EMBER data processing failed!")

if __name__ == "__main__":
    main()