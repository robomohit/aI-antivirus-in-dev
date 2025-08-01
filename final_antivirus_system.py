#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE ANTIVIRUS SYSTEM
Complete antivirus system using the retrained model
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import hashlib
import random
import time
from datetime import datetime
from colorama import init, Fore, Style
import logging
import json
import subprocess
import threading
import queue

# Initialize colorama
init()

class FinalAntivirusSystem:
    def __init__(self):
        self.models_dir = "retrained_models"
        self.quarantine_dir = "quarantine"
        self.log_dir = "antivirus_logs"
        self.model = None
        self.metadata = None
        
        # Create directories
        Path(self.quarantine_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/antivirus.log"),
                logging.StreamHandler()
            ]
        )
        
        # Load the latest retrained model
        self._load_latest_model()
        
        # Scan statistics
        self.scan_stats = {
            'files_scanned': 0,
            'threats_found': 0,
            'quarantined_files': 0,
            'scan_start_time': None,
            'scan_end_time': None
        }
    
    def _load_latest_model(self):
        """Load the latest retrained model."""
        print(f"{Fore.CYAN}🔄 Loading latest retrained model...")
        
        try:
            # Find the latest model files
            model_files = list(Path(self.models_dir).glob("real_model_*.pkl"))
            metadata_files = list(Path(self.models_dir).glob("real_metadata_*.pkl"))
            
            if not model_files or not metadata_files:
                print(f"{Fore.RED}❌ No retrained models found!")
                return False
            
            # Get the latest model (by timestamp)
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            print(f"{Fore.GREEN}✅ Loading model: {latest_model.name}")
            print(f"{Fore.GREEN}✅ Loading metadata: {latest_metadata.name}")
            
            with open(latest_model, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(latest_metadata, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"{Fore.GREEN}✅ Final antivirus model loaded successfully!")
            
            # Show model info
            if 'evaluation_results' in self.metadata:
                results = self.metadata['evaluation_results']
                print(f"📊 Model Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading retrained model: {e}")
            return False
    
    def extract_features_fixed(self, file_path):
        """Extract features with FIXED numpy handling."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return None
            
            # Calculate features
            file_size = len(data)
            
            # FIXED: Proper numpy array handling
            data_array = np.frombuffer(data, dtype=np.uint8)
            byte_counts = np.bincount(data_array, minlength=256)
            byte_probs = byte_counts / len(data)
            entropy = -np.sum(byte_probs * np.log2(byte_probs + 1e-10))
            
            # Calculate printable ratio
            printable_chars = sum(1 for byte in data if 32 <= byte <= 126)
            printable_ratio = printable_chars / len(data)
            
            # Count strings
            strings_count = len([b for b in data if 32 <= b <= 126])
            
            # Calculate string features
            string_lengths = []
            current_string = 0
            for byte in data:
                if 32 <= byte <= 126:
                    current_string += 1
                else:
                    if current_string > 0:
                        string_lengths.append(current_string)
                        current_string = 0
            avg_string_length = np.mean(string_lengths) if string_lengths else 0
            max_string_length = max(string_lengths) if string_lengths else 0
            
            # Histogram features
            histogram = np.bincount(data_array, minlength=256)
            histogram_normalized = histogram / len(data)
            histogram_regularity = 1 - np.std(histogram_normalized)
            
            # Entropy consistency
            chunk_size = min(1024, len(data) // 10)
            if chunk_size > 0:
                entropies = []
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    if len(chunk) > 0:
                        chunk_array = np.frombuffer(chunk, dtype=np.uint8)
                        chunk_counts = np.bincount(chunk_array, minlength=256)
                        chunk_probs = chunk_counts / len(chunk)
                        chunk_entropy = -np.sum(chunk_probs * np.log2(chunk_probs + 1e-10))
                        entropies.append(chunk_entropy)
                entropy_consistency = 1 - np.std(entropies) if entropies else 0.5
            else:
                entropy_consistency = 0.5
            
            features = {
                'file_size': file_size,
                'entropy': entropy,
                'strings_count': strings_count,
                'avg_string_length': avg_string_length,
                'max_string_length': max_string_length,
                'printable_ratio': printable_ratio,
                'histogram_regularity': histogram_regularity,
                'entropy_consistency': entropy_consistency
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def predict_with_model(self, features):
        """Predict using the retrained model."""
        try:
            feature_cols = self.metadata.get('feature_cols', [])
            feature_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Get raw probability
            probability = self.model.predict(feature_array, num_iteration=self.model.best_iteration)[0]
            
            # Use standard threshold
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return 0, 0.5
    
    def scan_file(self, file_path):
        """Scan a single file with detailed analysis."""
        try:
            # Extract features
            features = self.extract_features_fixed(file_path)
            if features is None:
                return None
            
            # Predict using retrained model
            prediction, probability = self.predict_with_model(features)
            
            # Determine threat level
            if prediction == 1:
                if probability > 0.8:
                    threat_level = "HIGH"
                    color = Fore.RED
                    action = "QUARANTINE"
                elif probability > 0.6:
                    threat_level = "MEDIUM"
                    color = Fore.YELLOW
                    action = "WARN"
                else:
                    threat_level = "LOW"
                    color = Fore.CYAN
                    action = "MONITOR"
            else:
                threat_level = "CLEAN"
                color = Fore.GREEN
                action = "ALLOW"
            
            # Analyze file characteristics
            analysis = {
                'file_path': file_path,
                'prediction': prediction,
                'probability': probability,
                'threat_level': threat_level,
                'action': action,
                'color': color,
                'features': features
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error scanning {file_path}: {e}")
            return None
    
    def quarantine_file(self, file_path):
        """Quarantine a suspicious file."""
        try:
            # Create quarantine path
            file_name = Path(file_path).name
            quarantine_path = Path(self.quarantine_dir) / f"quarantined_{int(time.time())}_{file_name}"
            
            # Move file to quarantine
            os.rename(file_path, quarantine_path)
            
            # Log quarantine action
            logging.warning(f"File quarantined: {file_path} -> {quarantine_path}")
            
            return quarantine_path
            
        except Exception as e:
            logging.error(f"Error quarantining {file_path}: {e}")
            return None
    
    def scan_directory(self, directory_path, scan_mode="quick", quarantine_threats=True):
        """Scan a directory with comprehensive reporting."""
        print(f"{Fore.CYAN}🛡️  FINAL ANTIVIRUS SYSTEM")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"🔍 Scan mode: {scan_mode}")
        print(f"🔍 Target: {directory_path}")
        print(f"🔍 Quarantine threats: {quarantine_threats}")
        print(f"{Fore.CYAN}{'='*60}")
        
        # Initialize scan statistics
        self.scan_stats['scan_start_time'] = datetime.now()
        self.scan_stats['files_scanned'] = 0
        self.scan_stats['threats_found'] = 0
        self.scan_stats['quarantined_files'] = 0
        
        # Find files to scan
        files_to_scan = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip certain file types
                if file.endswith(('.log', '.tmp', '.cache', '.pyc')):
                    continue
                files_to_scan.append(file_path)
        
        print(f"📁 Found {len(files_to_scan)} files to scan")
        
        # Scan files
        threats_found = []
        
        for i, file_path in enumerate(files_to_scan):
            print(f"[{i+1}/{len(files_to_scan)}] 🔄 Scanning: {os.path.basename(file_path)}")
            
            result = self.scan_file(file_path)
            if result:
                self.scan_stats['files_scanned'] += 1
                
                if result['prediction'] == 1:
                    self.scan_stats['threats_found'] += 1
                    threats_found.append(result)
                    
                    print(f"{result['color']}🚨 THREAT DETECTED: {os.path.basename(file_path)}")
                    print(f"   Threat Level: {result['threat_level']}")
                    print(f"   Confidence: {result['probability']:.1%}")
                    print(f"   Action: {result['action']}")
                    print(f"   Entropy: {result['features']['entropy']:.2f}")
                    print()
                    
                    # Quarantine if enabled and threat is high/medium
                    if quarantine_threats and result['action'] in ['QUARANTINE', 'WARN']:
                        quarantined_path = self.quarantine_file(file_path)
                        if quarantined_path:
                            self.scan_stats['quarantined_files'] += 1
                            print(f"{Fore.YELLOW}📦 File quarantined: {quarantined_path}")
        
        # Final statistics
        self.scan_stats['scan_end_time'] = datetime.now()
        scan_duration = self.scan_stats['scan_end_time'] - self.scan_stats['scan_start_time']
        
        print(f"{Fore.CYAN}{'='*60}")
        print(f"📊 FINAL ANTIVIRUS SCAN RESULTS")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"✅ Files scanned: {self.scan_stats['files_scanned']}")
        print(f"🚨 Threats found: {self.scan_stats['threats_found']}")
        print(f"📦 Files quarantined: {self.scan_stats['quarantined_files']}")
        print(f"⏱️  Scan duration: {scan_duration}")
        
        if self.scan_stats['threats_found'] > 0:
            print(f"{Fore.RED}⚠️  System may be compromised!")
        else:
            print(f"{Fore.GREEN}✅ System appears clean!")
        
        # Save scan report
        self._save_scan_report(threats_found)
        
        return threats_found
    
    def _save_scan_report(self, threats_found):
        """Save detailed scan report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(self.log_dir) / f"scan_report_{timestamp}.json"
            
            report = {
                'scan_stats': self.scan_stats,
                'threats_found': [
                    {
                        'file_path': str(t['file_path']),
                        'threat_level': t['threat_level'],
                        'confidence': t['probability'],
                        'action': t['action'],
                        'entropy': t['features']['entropy'],
                        'file_size': t['features']['file_size']
                    }
                    for t in threats_found
                ]
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"{Fore.GREEN}📄 Scan report saved: {report_path}")
            
        except Exception as e:
            logging.error(f"Error saving scan report: {e}")
    
    def real_time_monitor(self, directory_path, check_interval=30):
        """Real-time monitoring of a directory."""
        print(f"{Fore.CYAN}🔄 Starting real-time monitoring...")
        print(f"📁 Monitoring: {directory_path}")
        print(f"⏱️  Check interval: {check_interval} seconds")
        print(f"{Fore.CYAN}{'='*60}")
        
        try:
            while True:
                # Scan for new files
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip certain file types
                        if file.endswith(('.log', '.tmp', '.cache', '.pyc')):
                            continue
                        
                        # Check if file is new (modified in last interval)
                        file_mtime = os.path.getmtime(file_path)
                        if time.time() - file_mtime < check_interval:
                            print(f"🔍 Checking new file: {file}")
                            
                            result = self.scan_file(file_path)
                            if result and result['prediction'] == 1:
                                print(f"{result['color']}🚨 THREAT DETECTED: {file}")
                                print(f"   Threat Level: {result['threat_level']}")
                                print(f"   Confidence: {result['probability']:.1%}")
                                
                                # Auto-quarantine high threats
                                if result['action'] == 'QUARANTINE':
                                    quarantined_path = self.quarantine_file(file_path)
                                    if quarantined_path:
                                        print(f"{Fore.YELLOW}📦 Auto-quarantined: {quarantined_path}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Real-time monitoring stopped")
    
    def restore_quarantined_file(self, quarantined_path, restore_path):
        """Restore a quarantined file."""
        try:
            os.rename(quarantined_path, restore_path)
            logging.info(f"File restored: {quarantined_path} -> {restore_path}")
            print(f"{Fore.GREEN}✅ File restored: {restore_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error restoring file: {e}")
            return False
    
    def list_quarantined_files(self):
        """List all quarantined files."""
        quarantined_files = list(Path(self.quarantine_dir).glob("quarantined_*"))
        
        if not quarantined_files:
            print(f"{Fore.GREEN}✅ No quarantined files found")
            return []
        
        print(f"{Fore.CYAN}📦 Quarantined Files:")
        print(f"{Fore.CYAN}{'='*50}")
        
        for i, file_path in enumerate(quarantined_files):
            file_size = file_path.stat().st_size
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"{i+1}. {file_path.name}")
            print(f"   Size: {file_size:,} bytes")
            print(f"   Quarantined: {mtime}")
            print()
        
        return quarantined_files

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting Final Antivirus System...")
    
    antivirus = FinalAntivirusSystem()
    
    if not antivirus.model:
        print(f"{Fore.RED}❌ Failed to load antivirus model!")
        return
    
    # Example usage
    print(f"\n{Fore.CYAN}🛡️  FINAL ANTIVIRUS SYSTEM READY")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"1. Quick scan current directory")
    print(f"2. Full system scan")
    print(f"3. Real-time monitoring")
    print(f"4. List quarantined files")
    print(f"5. Exit")
    print(f"{Fore.CYAN}{'='*60}")
    
    while True:
        try:
            choice = input(f"{Fore.CYAN}Select option (1-5): ").strip()
            
            if choice == "1":
                print(f"\n{Fore.CYAN}🔄 Starting quick scan...")
                antivirus.scan_directory(".", "quick", quarantine_threats=True)
                
            elif choice == "2":
                print(f"\n{Fore.CYAN}🔄 Starting full system scan...")
                antivirus.scan_directory("/", "full", quarantine_threats=True)
                
            elif choice == "3":
                print(f"\n{Fore.CYAN}🔄 Starting real-time monitoring...")
                antivirus.real_time_monitor(".", check_interval=10)
                
            elif choice == "4":
                print(f"\n{Fore.CYAN}📦 Listing quarantined files...")
                antivirus.list_quarantined_files()
                
            elif choice == "5":
                print(f"\n{Fore.GREEN}🎉 Final antivirus system shutdown")
                break
                
            else:
                print(f"{Fore.RED}❌ Invalid option!")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Final antivirus system stopped")
            break

if __name__ == "__main__":
    main()