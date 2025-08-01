#!/usr/bin/env python3
"""
Download EMBER Dataset
Download and prepare the EMBER 2018 dataset for training
"""

import os
import sys
import requests
import tarfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from colorama import init, Fore, Style
import logging

# Initialize colorama
init()

class EMBERDownloader:
    def __init__(self):
        self.dataset_dir = "ember_dataset"
        self.ember_url = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
        self.dataset_path = "ember_dataset_2018_2.tar.bz2"
        
        # Create dataset directory
        Path(self.dataset_dir).mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def download_ember_dataset(self):
        """Download the EMBER dataset."""
        print(f"{Fore.CYAN}🔄 Downloading EMBER dataset...")
        
        if os.path.exists(self.dataset_path):
            print(f"{Fore.GREEN}✅ EMBER dataset already exists")
            return True
        
        try:
            print(f"{Fore.CYAN}📥 Downloading from: {self.ember_url}")
            print(f"{Fore.CYAN}📁 This is a large file (~1GB), please wait...")
            
            response = requests.get(self.ember_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(self.dataset_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 Download progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n{Fore.GREEN}✅ EMBER dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error downloading EMBER dataset: {e}")
            return False
    
    def extract_ember_dataset(self):
        """Extract the EMBER dataset."""
        print(f"{Fore.CYAN}🔄 Extracting EMBER dataset...")
        
        try:
            with tarfile.open(self.dataset_path, 'r:bz2') as tar:
                tar.extractall(self.dataset_dir)
            
            print(f"{Fore.GREEN}✅ EMBER dataset extracted successfully!")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error extracting EMBER dataset: {e}")
            return False
    
    def prepare_ember_data(self):
        """Prepare EMBER data for training."""
        print(f"{Fore.CYAN}🔄 Preparing EMBER data for training...")
        
        try:
            # Find extracted files
            ember_files = list(Path(self.dataset_dir).glob("*.json"))
            
            if not ember_files:
                print(f"{Fore.RED}❌ No EMBER files found!")
                return False
            
            print(f"{Fore.GREEN}✅ Found {len(ember_files)} EMBER files")
            
            # Process each file
            all_data = []
            
            for file_path in ember_files:
                print(f"📊 Processing: {file_path.name}")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                all_data.append(df)
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save processed data
            processed_path = Path(self.dataset_dir) / "ember_processed.csv"
            combined_df.to_csv(processed_path, index=False)
            
            print(f"{Fore.GREEN}✅ EMBER data prepared and saved to: {processed_path}")
            print(f"📊 Total samples: {len(combined_df)}")
            
            # Show data distribution
            if 'label' in combined_df.columns:
                label_counts = combined_df['label'].value_counts()
                print(f"📊 Label distribution:")
                for label, count in label_counts.items():
                    print(f"   Label {label}: {count} samples")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error preparing EMBER data: {e}")
            return False
    
    def run_download(self):
        """Run the complete download process."""
        print(f"{Fore.CYAN}🛡️  EMBER DATASET DOWNLOAD")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Step 1: Download dataset
        print(f"\n{Fore.CYAN}📦 Step 1: Downloading EMBER dataset")
        if not self.download_ember_dataset():
            return False
        
        # Step 2: Extract dataset
        print(f"\n{Fore.CYAN}📦 Step 2: Extracting EMBER dataset")
        if not self.extract_ember_dataset():
            return False
        
        # Step 3: Prepare data
        print(f"\n{Fore.CYAN}📦 Step 3: Preparing EMBER data")
        if not self.prepare_ember_data():
            return False
        
        print(f"\n{Fore.GREEN}🎉 EMBER dataset download complete!")
        return True

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting EMBER Dataset Download...")
    
    downloader = EMBERDownloader()
    success = downloader.run_download()
    
    if success:
        print(f"{Fore.GREEN}✅ EMBER dataset ready for training!")
    else:
        print(f"{Fore.RED}❌ EMBER dataset download failed!")

if __name__ == "__main__":
    main()