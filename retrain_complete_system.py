#!/usr/bin/env python3
"""
COMPLETE RETRAINING SYSTEM
Retrain the entire malware detection system from scratch with proper data
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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize colorama
init()

class CompleteRetrainingSystem:
    def __init__(self):
        self.dataset_dir = "ember_dataset"
        self.models_dir = "retrained_models"
        self.test_dir = "retraining_test"
        
        # Create directories
        Path(self.models_dir).mkdir(exist_ok=True)
        Path(self.test_dir).mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Model parameters
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def load_ember_data(self):
        """Load EMBER dataset."""
        print(f"{Fore.CYAN}🔄 Loading EMBER dataset...")
        
        try:
            # Check for processed data
            processed_path = Path(self.dataset_dir) / "ember_processed.csv"
            
            if processed_path.exists():
                print(f"{Fore.GREEN}✅ Loading processed EMBER data...")
                df = pd.read_csv(processed_path)
            else:
                print(f"{Fore.YELLOW}⚠️  Processed data not found, checking raw data...")
                
                # Look for raw JSON files
                json_files = list(Path(self.dataset_dir).glob("*.json"))
                
                if not json_files:
                    print(f"{Fore.RED}❌ No EMBER data found!")
                    return None
                
                print(f"{Fore.GREEN}✅ Found {len(json_files)} EMBER files")
                
                # Load and combine data
                all_data = []
                for file_path in json_files:
                    print(f"📊 Loading: {file_path.name}")
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    df_chunk = pd.DataFrame(data)
                    all_data.append(df_chunk)
                
                df = pd.concat(all_data, ignore_index=True)
                
                # Save processed data
                df.to_csv(processed_path, index=False)
                print(f"{Fore.GREEN}✅ Saved processed data")
            
            print(f"{Fore.GREEN}✅ EMBER data loaded successfully!")
            print(f"📊 Dataset shape: {df.shape}")
            
            # Show data distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                print(f"📊 Label distribution:")
                for label, count in label_counts.items():
                    print(f"   Label {label}: {count} samples")
            
            return df
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading EMBER data: {e}")
            return None
    
    def create_synthetic_malware_data(self, num_samples=10000):
        """Create synthetic malware data to supplement EMBER."""
        print(f"{Fore.CYAN}🔄 Creating synthetic malware data...")
        
        malware_samples = []
        
        # Create different types of synthetic malware
        malware_types = [
            {
                "name": "ransomware",
                "patterns": [b"encrypt", b"decrypt", b"ransom", b"bitcoin", b"wallet"],
                "entropy_range": (7.0, 8.0),
                "size_range": (50000, 200000)
            },
            {
                "name": "trojan",
                "patterns": [b"backdoor", b"shell", b"execute", b"cmd", b"system"],
                "entropy_range": (6.5, 7.5),
                "size_range": (30000, 150000)
            },
            {
                "name": "keylogger",
                "patterns": [b"keyboard", b"hook", b"input", b"log", b"capture"],
                "entropy_range": (6.0, 7.0),
                "size_range": (20000, 100000)
            },
            {
                "name": "cryptominer",
                "patterns": [b"mining", b"hash", b"sha256", b"gpu", b"cpu"],
                "entropy_range": (7.5, 8.0),
                "size_range": (100000, 300000)
            }
        ]
        
        for i in range(num_samples):
            malware_type = random.choice(malware_types)
            
            # Create synthetic PE file
            data = self.create_synthetic_pe_file(
                patterns=malware_type["patterns"],
                entropy_range=malware_type["entropy_range"],
                size_range=malware_type["size_range"]
            )
            
            # Extract features
            features = self.extract_ember_features(data)
            
            if features:
                features['label'] = 1  # Malware
                features['malware_type'] = malware_type['name']
                malware_samples.append(features)
        
        print(f"{Fore.GREEN}✅ Created {len(malware_samples)} synthetic malware samples")
        return pd.DataFrame(malware_samples)
    
    def create_synthetic_pe_file(self, patterns, entropy_range, size_range):
        """Create a synthetic PE file with malware patterns."""
        data = bytearray()
        
        # DOS header
        data.extend(b'MZ')  # Magic number
        data.extend(b'\x90' * 58)  # DOS stub
        data.extend(b'\x00\x00')  # PE offset
        
        # PE header
        data.extend(b'PE\x00\x00')  # PE signature
        data.extend(b'\x4c\x01')  # Machine (x86)
        data.extend(b'\x01\x00')  # Number of sections
        data.extend(b'\x00' * 16)  # Time/date stamps, etc.
        data.extend(b'\xe0\x00')  # Size of optional header
        data.extend(b'\x0f\x01')  # Characteristics
        
        # Add malware patterns
        for pattern in patterns:
            data.extend(pattern)
            data.extend(b'\x00' * random.randint(10, 50))
        
        # Add realistic malware code
        malware_code = [
            b"\x55\x8B\xEC",  # push ebp; mov ebp, esp
            b"\x83\xEC\x20",  # sub esp, 32
            b"\x68\x00\x00\x00\x00",  # push 0
            b"\xFF\x15\x00\x00\x00\x00",  # call dword ptr [0]
            b"\x8B\x45\x08",  # mov eax, [ebp+8]
            b"\x89\x45\xFC",  # mov [ebp-4], eax
            b"\x8B\xE5",  # mov esp, ebp
            b"\x5D",  # pop ebp
            b"\xC3"  # ret
        ]
        
        for code in malware_code:
            data.extend(code)
            data.extend(b'\x90' * random.randint(1, 5))
        
        # Add data to reach target size and entropy
        target_size = random.randint(*size_range)
        target_entropy = random.uniform(*entropy_range)
        
        # Add high entropy data
        remaining_size = target_size - len(data)
        if remaining_size > 0:
            if target_entropy > 7.0:
                # High entropy - mostly random data
                data.extend(os.urandom(remaining_size))
            else:
                # Lower entropy - mix of patterns and random
                structured_data = b'This is a test file with some structure' * 100
                random_data = os.urandom(remaining_size - len(structured_data))
                data.extend(structured_data + random_data)
        
        return bytes(data)
    
    def extract_ember_features(self, data):
        """Extract EMBER-style features from data."""
        try:
            if len(data) == 0:
                return None
            
            # Calculate basic features
            file_size = len(data)
            
            # Calculate entropy
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
            
            # Create feature dictionary
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
            logging.error(f"Error extracting features: {e}")
            return None
    
    def prepare_training_data(self, ember_df, synthetic_df=None):
        """Prepare data for training."""
        print(f"{Fore.CYAN}🔄 Preparing training data...")
        
        # Combine EMBER and synthetic data
        if synthetic_df is not None:
            combined_df = pd.concat([ember_df, synthetic_df], ignore_index=True)
            print(f"📊 Combined dataset: {len(combined_df)} samples")
        else:
            combined_df = ember_df
        
        # Clean data
        combined_df = combined_df.dropna()
        
        # Separate features and labels
        feature_cols = [col for col in combined_df.columns if col not in ['label', 'malware_type']]
        X = combined_df[feature_cols]
        y = combined_df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Show class distribution
        print(f"📊 Training set distribution:")
        train_counts = y_train.value_counts()
        for label, count in train_counts.items():
            print(f"   Label {label}: {count} samples")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_model(self, X_train, y_train, feature_cols):
        """Train the malware detection model."""
        print(f"{Fore.CYAN}🔄 Training malware detection model...")
        
        try:
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Train model
            model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            print(f"{Fore.GREEN}✅ Model training completed!")
            return model
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error training model: {e}")
            return None
    
    def evaluate_model(self, model, X_test, y_test, feature_cols):
        """Evaluate the trained model."""
        print(f"{Fore.CYAN}🔄 Evaluating model...")
        
        try:
            # Make predictions
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{Fore.GREEN}📊 Model Evaluation Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            
            # Classification report
            print(f"\n📊 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n📊 Confusion Matrix:")
            print(cm)
            
            # Save evaluation results
            results = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': cm.tolist(),
                'feature_importance': dict(zip(feature_cols, model.feature_importance()))
            }
            
            return results
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error evaluating model: {e}")
            return None
    
    def save_model(self, model, feature_cols, results):
        """Save the trained model and metadata."""
        print(f"{Fore.CYAN}🔄 Saving model...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = Path(self.models_dir) / f"retrained_model_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'feature_cols': feature_cols,
                'model_params': self.model_params,
                'training_timestamp': timestamp,
                'evaluation_results': results
            }
            
            metadata_path = Path(self.models_dir) / f"retrained_metadata_{timestamp}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"{Fore.GREEN}✅ Model saved: {model_path}")
            print(f"{Fore.GREEN}✅ Metadata saved: {metadata_path}")
            
            return model_path, metadata_path
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error saving model: {e}")
            return None, None
    
    def run_complete_retraining(self):
        """Run the complete retraining process."""
        print(f"{Fore.CYAN}🛡️  COMPLETE RETRAINING SYSTEM")
        print(f"{Fore.CYAN}{'='*60}")
        
        # Step 1: Load EMBER data
        print(f"\n{Fore.CYAN}📦 Step 1: Loading EMBER dataset")
        ember_df = self.load_ember_data()
        if ember_df is None:
            print(f"{Fore.RED}❌ Failed to load EMBER data!")
            return False
        
        # Step 2: Create synthetic malware data
        print(f"\n{Fore.CYAN}📦 Step 2: Creating synthetic malware data")
        synthetic_df = self.create_synthetic_malware_data(num_samples=5000)
        
        # Step 3: Prepare training data
        print(f"\n{Fore.CYAN}📦 Step 3: Preparing training data")
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(
            ember_df, synthetic_df
        )
        
        # Step 4: Train model
        print(f"\n{Fore.CYAN}📦 Step 4: Training model")
        model = self.train_model(X_train, y_train, feature_cols)
        if model is None:
            print(f"{Fore.RED}❌ Model training failed!")
            return False
        
        # Step 5: Evaluate model
        print(f"\n{Fore.CYAN}📦 Step 5: Evaluating model")
        results = self.evaluate_model(model, X_test, y_test, feature_cols)
        if results is None:
            print(f"{Fore.RED}❌ Model evaluation failed!")
            return False
        
        # Step 6: Save model
        print(f"\n{Fore.CYAN}📦 Step 6: Saving model")
        model_path, metadata_path = self.save_model(model, feature_cols, results)
        if model_path is None:
            print(f"{Fore.RED}❌ Model saving failed!")
            return False
        
        print(f"\n{Fore.GREEN}🎉 Complete retraining finished!")
        print(f"{Fore.GREEN}✅ New model ready for deployment!")
        
        return True

def main():
    """Main function."""
    print(f"{Fore.CYAN}🛡️  Starting Complete Retraining System...")
    
    retrainer = CompleteRetrainingSystem()
    success = retrainer.run_complete_retraining()
    
    if success:
        print(f"{Fore.GREEN}✅ Retraining completed successfully!")
    else:
        print(f"{Fore.RED}❌ Retraining failed!")

if __name__ == "__main__":
    main()