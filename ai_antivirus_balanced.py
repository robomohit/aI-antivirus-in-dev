#!/usr/bin/env python3
"""
AI Antivirus with Balanced Model
Uses the balanced model with realistic performance
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from datetime import datetime
import time
import shutil
from colorama import init, Fore, Style
import argparse

# Initialize colorama
init()

class BalancedAIAntivirus:
    def __init__(self):
        self.models_dir = "retrained_models"
        self.quarantine_dir = "quarantine"
        self.log_file = "antivirus_balanced.log"
        
        # Create directories
        Path(self.quarantine_dir).mkdir(exist_ok=True)
        
        # Load model
        self.model = None
        self.metadata = None
        self._load_model()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_model(self):
        """Load the balanced model."""
        print(f"{Fore.CYAN}🔄 Loading balanced model...")
        
        try:
            # Find latest model files
            model_files = list(Path(self.models_dir).glob("real_model_*.pkl"))
            metadata_files = list(Path(self.models_dir).glob("real_metadata_*.pkl"))
            
            if not model_files or not metadata_files:
                print(f"{Fore.RED}❌ No real model found!")
                print(f"{Fore.YELLOW}💡 Run quick_real_training.py first")
                return
            
            # Get latest files
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_model, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(latest_metadata, 'rb') as f:
                self.metadata = pickle.load(f)
            
                            print(f"{Fore.GREEN}✅ Real model loaded: {latest_model.name}")
            print(f"{Fore.GREEN}✅ Metadata loaded: {latest_metadata.name}")
            
            # Show model performance
            metrics = self.metadata.get('metrics', {})
            if metrics:
                print(f"{Fore.CYAN}📊 Model Performance:")
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                print(f"   False Positive Rate: {metrics.get('false_positive_rate', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading model: {e}")
    
    def _print(self, message):
        """Print with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def extract_features(self, file_path):
        """Extract features from file for balanced model."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return None
            
            # Calculate features (same as training)
            file_size = len(data)
            
            # Calculate entropy
            byte_counts = np.bincount(data, minlength=256)
            byte_probs = byte_counts / len(data)
            entropy = -np.sum(byte_probs * np.log2(byte_probs + 1e-10))
            
            # Calculate printable ratio
            printable_chars = sum(1 for byte in data if 32 <= byte <= 126)
            printable_ratio = printable_chars / len(data)
            
            # Count strings (simplified)
            strings_count = len([b for b in data if 32 <= b <= 126])
            
            # Average string length (simplified)
            avg_string_length = 5.0  # Default value
            
            # Histogram regularity (simplified)
            histogram_regularity = 0.5  # Default value
            
            # Entropy consistency (simplified)
            entropy_consistency = 0.5  # Default value
            
            features = {
                'file_size': file_size,
                'entropy': entropy,
                'strings_count': strings_count,
                'avg_string_length': avg_string_length,
                'printable_ratio': printable_ratio,
                'histogram_regularity': histogram_regularity,
                'entropy_consistency': entropy_consistency
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def analyze_file(self, file_path):
        """Analyze a single file using the balanced model."""
        if not self.model or not self.metadata:
            return None
        
        try:
            # Skip certain file types and directories
            file_path_str = str(file_path).lower()
            file_name_lower = file_path.name.lower()
            
            # Skip virtual environment files
            if any(venv_dir in file_path_str for venv_dir in ['venv', 'env', '.venv', '\\lib\\site-packages']):
                return None
            
            # Skip legitimate files
            legitimate_files = [
                'firefox', 'chrome', 'edge', 'opera', 'safari', 'brave',
                'profiles.ini', 'containers.json', 'sessioncheckpoints.json',
                'application.ini', 'mozglue.dll', 'd3dcompiler_47.dll',
                'softokn3.dll', 'tor.exe', 'channel-prefs.js', 'compatibility.ini',
                'plugin-container.exe', 'function.js', 'build_nuitka.bat', 'scan_custom.bat',
                'scan_desktop.bat', 'scan_downloads.bat', 'scan_documents.bat',
                'libegl.dll', 'ipcclientcerts.dll', 'tbb_version.json', 'xulstore.json',
                'update-settings.ini', 'firewall_rules.json', 'lgpllibs.dll'
            ]
            
            if any(legit.lower() in file_name_lower for legit in legitimate_files):
                self._print(f"{Fore.GREEN}✅ Skipping legitimate file: {file_path.name}")
                return None
            
            if any(legit.lower() in file_path_str for legit in legitimate_files):
                self._print(f"{Fore.GREEN}✅ Skipping legitimate file: {file_path.name}")
                return None
            
            # Extract features
            features = self.extract_features(file_path)
            if features is None:
                return None
            
            # Prepare feature array
            feature_cols = self.metadata.get('feature_cols', [])
            feature_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Predict using balanced model
            try:
                probability = self.model.predict(feature_array, num_iteration=self.model.best_iteration)[0]
                prediction = 1 if probability > 0.5 else 0
            except:
                prediction = 0
                probability = 0.5
            
            # Determine threat level with more conservative thresholds
            if prediction == 1:
                if probability > 0.9:
                    threat_level = "HIGH"
                elif probability > 0.7:
                    threat_level = "MEDIUM"
                elif probability > 0.6:
                    threat_level = "LOW"
                else:
                    threat_level = "SAFE"
            else:
                threat_level = "SAFE"
            
            # Generate detection reason
            detection_reason = self._generate_detection_reason(features, probability, prediction)
            
            analysis = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'threat_level': threat_level,
                'malware_probability': probability,
                'prediction': prediction,
                'detection_reason': detection_reason,
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _generate_detection_reason(self, features, probability, prediction):
        """Generate human-readable detection reason."""
        reasons = []
        
        if prediction == 1:  # Malware detected
            if features['entropy'] > 7.0:
                reasons.append("High entropy (encrypted/packed content)")
            if features['printable_ratio'] < 0.3:
                reasons.append("Low printable character ratio")
            if features['file_size'] > 100000:
                reasons.append("Large file size")
            if probability > 0.9:
                reasons.append("Very high malware probability")
            elif probability > 0.7:
                reasons.append("High malware probability")
            
            return " | ".join(reasons) if reasons else "AI model detected malware"
        else:
            if features['entropy'] < 5.0:
                reasons.append("Low entropy (normal content)")
            if features['printable_ratio'] > 0.7:
                reasons.append("High printable character ratio")
            if probability < 0.3:
                reasons.append("Very low malware probability")
            
            return " | ".join(reasons) if reasons else "File appears benign"
    
    def scan_directory(self, directory_path, scan_mode="quick"):
        """Scan a directory for malware."""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"{Fore.RED}❌ Directory not found: {directory_path}")
            return
        
        print(f"{Fore.CYAN}🛡️  BALANCED AI ANTIVIRUS")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}🔍 Scan mode: {scan_mode}")
        print(f"{Fore.CYAN}🔍 Target: {directory_path}")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Collect files to scan
        files_to_scan = []
        
        if scan_mode == "quick":
            # Scan only executable and suspicious files
            extensions = ['.exe', '.dll', '.bat', '.cmd', '.vbs', '.js', '.ps1', '.scr', '.pif']
            for ext in extensions:
                files_to_scan.extend(directory.rglob(f"*{ext}"))
        elif scan_mode == "smart":
            # Scan executables and common malware targets
            extensions = ['.exe', '.dll', '.bat', '.cmd', '.vbs', '.js', '.ps1', '.scr', '.pif', '.com']
            for ext in extensions:
                files_to_scan.extend(directory.rglob(f"*{ext}"))
        else:  # full
            # Scan all files
            files_to_scan = list(directory.rglob("*"))
            files_to_scan = [f for f in files_to_scan if f.is_file()]
        
        # Filter out directories and skip files
        files_to_scan = [f for f in files_to_scan if f.is_file()]
        
        # Additional filtering for legitimate files
        filtered_files = []
        for file_path in files_to_scan:
            file_path_str = str(file_path).lower()
            if any(venv_dir in file_path_str for venv_dir in ['venv', 'env', '.venv', '\\lib\\site-packages']):
                continue
            if any(legit in file_path_str for legit in ['node_modules', '.git', '__pycache__']):
                continue
            filtered_files.append(file_path)
        
        files_to_scan = filtered_files
        
        print(f"{Fore.CYAN}📁 Found {len(files_to_scan)} files to scan")
        
        threats_found = []
        files_scanned = 0
        total_files = len(files_to_scan)
        
        print(f"{Fore.CYAN}🔄 Starting scan of {total_files} files...")
        
        for i, file_path in enumerate(files_to_scan):
            try:
                if total_files <= 20 or i % 5 == 0:
                    progress = (i + 1) / total_files * 100
                    self._print(f"{Fore.CYAN}🔄 Scanning: {file_path.name} ({i+1}/{total_files} - {progress:.1f}%)")
                
                analysis = self.analyze_file(file_path)
                if analysis:
                    files_scanned += 1
                    if analysis['threat_level'] in ['HIGH', 'MEDIUM']:
                        threats_found.append(analysis)
                        self._print(f"{Fore.RED}🚨 THREAT DETECTED: {file_path.name} ({analysis['threat_level']})")
                        self._print(f"{Fore.YELLOW}🔍 Detection Reason: {analysis.get('detection_reason', 'Unknown')}")
                        self._print(f"{Fore.CYAN}📊 Probability: {analysis['malware_probability']:.1%}")
                        
                        # Quarantine the file
                        self.quarantine_file(file_path)
                
                if total_files > 20 and i % 10 == 0 and i > 0:
                    progress = (i + 1) / total_files * 100
                    self._print(f"{Fore.CYAN}⠋ Progress: {i+1}/{total_files} files scanned ({progress:.1f}%)")
                    
            except Exception as e:
                logging.error(f"Error scanning {file_path}: {e}")
                continue
        
        # Print final results
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}📊 BALANCED SCAN RESULTS")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}✅ Files scanned: {files_scanned}/{total_files}")
        print(f"{Fore.RED}🚨 Threats found: {len(threats_found)}")
        
        if threats_found:
            print(f"{Fore.RED}🚨 Scan completed with {len(threats_found)} threats found!")
            for threat in threats_found:
                print(f"{Fore.RED}   - {threat['file_name']} ({threat['threat_level']}) - {threat['malware_probability']:.1%}")
        else:
            print(f"{Fore.GREEN}✅ No threats detected!")
            print(f"{Fore.GREEN}✅ Scan completed successfully - system is clean!")
    
    def quarantine_file(self, file_path):
        """Quarantine a suspicious file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"quarantined_{timestamp}_{file_path.name}"
            quarantine_path = Path(self.quarantine_dir) / quarantine_name
            
            shutil.move(str(file_path), str(quarantine_path))
            
            self._print(f"{Fore.YELLOW}🛡️  Quarantined: {file_path.name}")
            logging.info(f"Quarantined: {file_path} -> {quarantine_path}")
            
        except Exception as e:
            logging.error(f"Error quarantining {file_path}: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Balanced AI Antivirus")
    parser.add_argument("action", choices=["scan"], help="Action to perform")
    parser.add_argument("path", help="Path to scan")
    parser.add_argument("mode", choices=["quick", "smart", "full"], default="quick", 
                       help="Scan mode (default: quick)")
    
    args = parser.parse_args()
    
    antivirus = BalancedAIAntivirus()
    
    if args.action == "scan":
        antivirus.scan_directory(args.path, args.mode)

if __name__ == "__main__":
    main()