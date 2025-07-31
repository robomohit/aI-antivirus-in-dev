#!/usr/bin/env python3
"""
Hybrid AI Antivirus - Combines Old and New Models
Uses old model for proven threats, new model for modern patterns
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from colorama import init, Fore, Style
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Initialize colorama
init()

class HybridAIAntivirus:
    def __init__(self):
        self.old_model_path = "ember_real_models/ember_real_model_20250730_185819.pkl"
        self.old_metadata_path = "ember_real_models/ember_real_metadata_20250730_185819.pkl"
        self.new_model_path = "comprehensive_diverse_model_20250730_222728.pkl"
        self.new_metadata_path = "comprehensive_diverse_metadata_20250730_222728.pkl"
        
        self.old_model = None
        self.old_feature_cols = None
        self.new_model = None
        self.new_feature_cols = None
        
        self.quarantine_dir = "quarantine"
        self.scan_mode = "quick"
        
        # Create quarantine directory
        Path(self.quarantine_dir).mkdir(exist_ok=True)
        
        self.setup_logging()
        self.load_models()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hybrid_antivirus.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_models(self):
        """Load both old and new models."""
        try:
            # Load old model (2018 EMBER)
            with open(self.old_model_path, 'rb') as f:
                self.old_model = pickle.load(f)
            with open(self.old_metadata_path, 'rb') as f:
                old_metadata = pickle.load(f)
                self.old_feature_cols = old_metadata['feature_cols']
            
            # Load new model (2025 Comprehensive)
            with open(self.new_model_path, 'rb') as f:
                self.new_model = pickle.load(f)
            with open(self.new_metadata_path, 'rb') as f:
                new_metadata = pickle.load(f)
                self.new_feature_cols = new_metadata['feature_cols']
            
            print(f"{Fore.GREEN}✅ Hybrid system loaded successfully!")
            print(f"{Fore.CYAN}📊 Old Model (2018): {len(self.old_feature_cols)} features")
            print(f"{Fore.CYAN}📊 New Model (2025): {len(self.new_feature_cols)} features")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading models: {e}")
            sys.exit(1)
    
    def extract_old_features(self, file_path):
        """Extract features for old model (EMBER-style)."""
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # EMBER-style features
            features = {
                'file_size': file_size,
                'entropy': self.calculate_entropy(data),
                'strings_count': self.count_strings(data),
                'avg_string_length': self.calculate_avg_string_length(data),
                'printable_ratio': self.calculate_printable_ratio(data),
                'histogram_regularity': self.calculate_histogram_regularity(data),
                'entropy_consistency': self.calculate_entropy_consistency(data)
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting old features: {e}")
            return None
    
    def extract_new_features(self, file_path):
        """Extract features for new model (comprehensive)."""
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Comprehensive features
            features = {
                'file_size': file_size,
                'entropy': self.calculate_entropy(data),
                'strings_count': self.count_strings(data),
                'avg_string_length': self.calculate_avg_string_length(data),
                'printable_ratio': self.calculate_printable_ratio(data),
                'malware_suspicion': self.calculate_malware_suspicion(data),
                'benign_score': self.calculate_benign_score(data),
                'histogram_regularity': self.calculate_histogram_regularity(data),
                'entropy_consistency': self.calculate_entropy_consistency(data)
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting new features: {e}")
            return None
    
    def calculate_entropy(self, data):
        """Calculate Shannon entropy."""
        if not data:
            return 0
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        entropy = 0
        data_len = len(data)
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        return entropy
    
    def count_strings(self, data):
        """Count printable strings."""
        try:
            strings = data.decode('utf-8', errors='ignore').split('\x00')
            return len([s for s in strings if len(s) > 3])
        except:
            return 0
    
    def calculate_avg_string_length(self, data):
        """Calculate average string length."""
        try:
            strings = data.decode('utf-8', errors='ignore').split('\x00')
            valid_strings = [s for s in strings if len(s) > 3]
            return np.mean([len(s) for s in valid_strings]) if valid_strings else 0
        except:
            return 0
    
    def calculate_printable_ratio(self, data):
        """Calculate ratio of printable characters."""
        try:
            printable = sum(1 for b in data if 32 <= b <= 126)
            return printable / len(data) if data else 0
        except:
            return 0
    
    def calculate_histogram_regularity(self, data):
        """Calculate histogram regularity."""
        if not data:
            return 0
        histogram = np.histogram(data, bins=256, range=(0, 256))[0]
        return np.std(histogram) / np.mean(histogram) if np.mean(histogram) > 0 else 0
    
    def calculate_entropy_consistency(self, data):
        """Calculate entropy consistency."""
        if len(data) < 1024:
            return 0
        chunks = [data[i:i+1024] for i in range(0, len(data), 1024)]
        entropies = [self.calculate_entropy(chunk) for chunk in chunks]
        return np.std(entropies)
    
    def calculate_malware_suspicion(self, data):
        """Calculate malware suspicion score."""
        entropy = self.calculate_entropy(data)
        if entropy > 7.5:
            return 0.8
        elif entropy > 6.5:
            return 0.6
        else:
            return 0.2
    
    def calculate_benign_score(self, data):
        """Calculate benign score."""
        printable_ratio = self.calculate_printable_ratio(data)
        if printable_ratio > 0.8:
            return 0.8
        elif printable_ratio > 0.6:
            return 0.6
        else:
            return 0.2
    
    def predict_old_model(self, features):
        """Predict using old model."""
        try:
            feature_vector = []
            for col in self.old_feature_cols:
                feature_vector.append(features.get(col, 0.0))
            
            prediction = self.old_model.predict_proba([feature_vector])[0]
            return prediction[1]  # Malware probability
        except Exception as e:
            logging.error(f"Old model prediction error: {e}")
            return 0.5
    
    def predict_new_model(self, features):
        """Predict using new model."""
        try:
            feature_vector = []
            for col in self.new_feature_cols:
                feature_vector.append(features.get(col, 0.0))
            
            prediction = self.new_model.predict_proba([feature_vector])[0]
            return prediction[1]  # Malware probability
        except Exception as e:
            logging.error(f"New model prediction error: {e}")
            return 0.5
    
    def hybrid_decision(self, old_prob, new_prob, file_path):
        """Make hybrid decision based on both models."""
        file_ext = file_path.suffix.lower()
        
        # Modern file types (use new model more heavily)
        modern_extensions = ['.js', '.ps1', '.vbs', '.hta', '.wsf', '.jar', '.py']
        is_modern = file_ext in modern_extensions
        
        # Traditional file types (use old model more heavily)
        traditional_extensions = ['.exe', '.dll', '.sys', '.scr', '.com', '.pif']
        is_traditional = file_ext in traditional_extensions
        
        # Weight the models based on file type
        if is_modern:
            # New model gets more weight for modern threats
            hybrid_prob = (old_prob * 0.3) + (new_prob * 0.7)
            model_used = "NEW (Modern)"
        elif is_traditional:
            # Old model gets more weight for traditional threats
            hybrid_prob = (old_prob * 0.7) + (new_prob * 0.3)
            model_used = "OLD (Traditional)"
        else:
            # Balanced approach for other files
            hybrid_prob = (old_prob * 0.5) + (new_prob * 0.5)
            model_used = "HYBRID (Balanced)"
        
        # Determine threat level
        if hybrid_prob >= 0.8:
            threat_level = "HIGH"
        elif hybrid_prob >= 0.6:
            threat_level = "MEDIUM"
        elif hybrid_prob >= 0.4:
            threat_level = "LOW"
        else:
            threat_level = "SAFE"
        
        return hybrid_prob, threat_level, model_used
    
    def analyze_file(self, file_path):
        """Analyze a file using hybrid approach."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            # Skip virtual environment files
            file_path_str = str(file_path).lower()
            if any(venv_dir in file_path_str for venv_dir in ['venv', 'env', '.venv', '\\lib\\site-packages']):
                return None
            
            # Extract features for both models
            old_features = self.extract_old_features(file_path)
            new_features = self.extract_new_features(file_path)
            
            if old_features is None or new_features is None:
                return None
            
            # Get predictions from both models
            old_prob = self.predict_old_model(old_features)
            new_prob = self.predict_new_model(new_features)
            
            # Make hybrid decision
            hybrid_prob, threat_level, model_used = self.hybrid_decision(old_prob, new_prob, file_path)
            
            # Generate detection explanation
            detection_reason = self._generate_detection_explanation(
                old_prob, new_prob, hybrid_prob, threat_level, model_used, file_path
            )
            
            # Create analysis result
            analysis_result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': old_features.get('file_size', 0),
                'file_extension': file_path.suffix.lower(),
                'threat_level': threat_level,
                'hybrid_probability': hybrid_prob,
                'old_model_probability': old_prob,
                'new_model_probability': new_prob,
                'model_used': model_used,
                'detection_reason': detection_reason
            }
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _generate_detection_explanation(self, old_prob, new_prob, hybrid_prob, threat_level, model_used, file_path):
        """Generate detailed explanation for hybrid detection."""
        reasons = []
        
        # Model comparison
        if abs(old_prob - new_prob) > 0.3:
            reasons.append(f"Model disagreement: Old={old_prob:.1%}, New={new_prob:.1%}")
        
        # File type analysis
        file_ext = file_path.suffix.lower()
        if file_ext in ['.js', '.ps1', '.vbs', '.hta']:
            reasons.append(f"Modern script file ({file_ext})")
        elif file_ext in ['.exe', '.dll', '.sys']:
            reasons.append(f"Traditional executable ({file_ext})")
        
        # Probability analysis
        if hybrid_prob > 0.9:
            reasons.append(f"High hybrid confidence ({hybrid_prob:.1%})")
        elif hybrid_prob > 0.7:
            reasons.append(f"Moderate hybrid confidence ({hybrid_prob:.1%})")
        
        # Model used
        reasons.append(f"Decision: {model_used}")
        
        return " | ".join(reasons)
    
    def quarantine_file(self, file_path):
        """Quarantine a suspicious file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, None
            
            # Create quarantine filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"quarantined_{timestamp}_{file_path.name}"
            quarantine_path = Path(self.quarantine_dir) / quarantine_name
            
            # Move file to quarantine
            file_path.rename(quarantine_path)
            
            # Log quarantine action
            logging.info(f"Quarantined: {file_path} -> {quarantine_path}")
            
            return True, quarantine_path
            
        except Exception as e:
            logging.error(f"Error quarantining file {file_path}: {e}")
            return False, None
    
    def scan_directory(self, directory_path="."):
        """Scan a directory using hybrid approach."""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                print(f"{Fore.RED}❌ Directory not found: {directory_path}")
                return
            
            print(f"{Fore.CYAN}🔍 Hybrid scan mode: {self.scan_mode}")
            print(f"{Fore.CYAN}🔍 Starting hybrid scan of: {directory_path}")
            
            # Get files to scan
            files_to_scan = []
            
            if self.scan_mode == "quick":
                # Quick scan: only suspicious extensions
                suspicious_exts = ['.exe', '.dll', '.bat', '.cmd', '.vbs', '.js', '.ps1', '.scr', '.pif', '.hta', '.wsf']
                for ext in suspicious_exts:
                    files_to_scan.extend(directory_path.rglob(f"*{ext}"))
            else:  # smart or full
                # Scan all files
                files_to_scan = list(directory_path.rglob("*"))
                files_to_scan = [f for f in files_to_scan if f.is_file()]
            
            # Remove duplicates and filter
            files_to_scan = list(set(files_to_scan))
            files_to_scan = [f for f in files_to_scan if f.is_file()]
            
            print(f"{Fore.CYAN}📁 Found {len(files_to_scan)} files to scan")
            
            # Scan files
            threats_found = []
            files_scanned = 0
            total_files = len(files_to_scan)
            
            print(f"{Fore.CYAN}🔄 Starting hybrid scan of {total_files} files...")
            
            for i, file_path in enumerate(files_to_scan):
                try:
                    # Show progress
                    if total_files <= 20 or i % 5 == 0:
                        progress = (i + 1) / total_files * 100
                        print(f"{Fore.CYAN}🔄 Scanning: {file_path.name} ({i+1}/{total_files} - {progress:.1f}%)")
                    
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        files_scanned += 1
                        
                        if analysis['threat_level'] in ['HIGH', 'MEDIUM']:
                            threats_found.append(analysis)
                            
                            # Print threat details
                            print(f"{Fore.RED}🚨 THREAT DETECTED: {file_path.name} ({analysis['threat_level']})")
                            print(f"{Fore.YELLOW}🔍 Model: {analysis['model_used']}")
                            print(f"{Fore.YELLOW}🔍 Hybrid Probability: {analysis['hybrid_probability']:.1%}")
                            print(f"{Fore.YELLOW}🔍 Detection Reason: {analysis['detection_reason']}")
                            
                            # Quarantine high threats
                            if analysis['threat_level'] == 'HIGH':
                                success, quarantine_path = self.quarantine_file(file_path)
                                if success:
                                    print(f"{Fore.YELLOW}🛡️  Quarantined: {file_path.name}")
                    
                    # Progress indicator for larger scans
                    if total_files > 20 and i % 10 == 0 and i > 0:
                        progress = (i + 1) / total_files * 100
                        print(f"{Fore.CYAN}⠋ Progress: {i+1}/{total_files} files scanned ({progress:.1f}%)")
                
                except Exception as e:
                    logging.error(f"Error scanning {file_path}: {e}")
                    continue
            
            # Print results
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN}📊 HYBRID SCAN RESULTS")
            print(f"{Fore.CYAN}{'='*60}")
            print(f"{Fore.GREEN}✅ Files scanned: {files_scanned}/{total_files}")
            print(f"{Fore.RED}🚨 Threats found: {len(threats_found)}")
            
            if threats_found:
                print(f"{Fore.RED}🚨 Scan completed with {len(threats_found)} threats found!")
                for threat in threats_found:
                    print(f"{Fore.RED}   - {threat['file_name']} ({threat['threat_level']}) - {threat['model_used']}")
            else:
                print(f"{Fore.GREEN}✅ No threats detected!")
                print(f"{Fore.GREEN}✅ Hybrid scan completed successfully - system is clean!")
            
        except Exception as e:
            logging.error(f"Error in directory scan: {e}")
            print(f"{Fore.RED}❌ Error during scan: {e}")

def main():
    """Main function for hybrid antivirus."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid AI Antivirus')
    parser.add_argument('action', choices=['scan', 'monitor'], help='Action to perform')
    parser.add_argument('path', nargs='?', default='.', help='Path to scan/monitor')
    parser.add_argument('mode', nargs='?', choices=['quick', 'smart', 'full'], default='quick', help='Scan mode')
    
    args = parser.parse_args()
    
    # Initialize hybrid antivirus
    antivirus = HybridAIAntivirus()
    antivirus.scan_mode = args.mode
    
    # Perform action
    if args.action == 'scan':
        antivirus.scan_directory(args.path)
    elif args.action == 'monitor':
        print(f"{Fore.YELLOW}⚠️  Monitor mode not implemented yet")

if __name__ == "__main__":
    main()