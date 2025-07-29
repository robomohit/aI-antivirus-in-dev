#!/usr/bin/env python3
"""
AI-Enhanced Simple Antivirus in Python
Combines traditional extension-based detection with machine learning.
"""

import os
import sys
import time
import shutil
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import signal
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for colored output
colorama.init(autoreset=True)


class AIAntivirus:
    def __init__(self, monitor_path, quarantine_enabled=True, model_path="model/model.pkl"):
        """
        Initialize the AI-enhanced antivirus system.
        
        Args:
            monitor_path (str): Path to monitor for suspicious files
            quarantine_enabled (bool): Whether to move suspicious files to quarantine
            model_path (str): Path to the trained ML model
        """
        self.monitor_path = Path(monitor_path).resolve()
        self.quarantine_enabled = quarantine_enabled
        self.model_path = Path(model_path)
        
        # Create necessary directories
        self.logs_dir = Path("logs")
        self.quarantine_dir = Path("quarantine")
        self.model_dir = Path("model")
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Suspicious file extensions (traditional detection)
        self.suspicious_extensions = {
            '.exe', '.bat', '.vbs', '.scr', '.ps1', '.cmd', '.com', 
            '.pif', '.reg', '.js', '.jar', '.msi', '.dll', '.sys'
        }
        
        # Load or train the AI model
        self.model = self._load_or_train_model()
        
        # Initialize watchdog observer
        self.observer = Observer()
        self.event_handler = AIAntivirusEventHandler(self)
        
        self._print_startup_info()
    
    def _create_directories(self):
        """Create logs, quarantine, and model directories if they don't exist."""
        self.logs_dir.mkdir(exist_ok=True)
        if self.quarantine_enabled:
            self.quarantine_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / f"ai_antivirus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _print_startup_info(self):
        """Print colorful startup information."""
        print(f"{Fore.CYAN}🚀 AI-Enhanced Antivirus Started{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📁 Monitoring: {self.monitor_path}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🛡️ Quarantine enabled: {self.quarantine_enabled}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🧠 AI Model: {'Loaded' if self.model else 'Training...'}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}⚠️ Suspicious extensions: {', '.join(sorted(self.suspicious_extensions))}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}👁️ File monitoring started{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop monitoring{Style.RESET_ALL}\n")
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                self.logger.info("🧠 AI model loaded successfully")
                return model
            except Exception as e:
                self.logger.warning(f"❌ Failed to load model: {e}")
        
        # Train new model if loading fails
        self.logger.info("🧠 Training new AI model...")
        return self._train_model()
    
    def _create_training_data(self):
        """Create dummy training data for the AI model."""
        np.random.seed(42)  # For reproducible results
        
        # Generate dummy data
        data = []
        
        # Safe files (mostly small, common extensions)
        safe_extensions = ['.txt', '.pdf', '.jpg', '.png', '.mp3', '.mp4', '.doc', '.xls']
        for _ in range(100):
            ext = np.random.choice(safe_extensions)
            size = np.random.randint(1, 1000)  # 1-1000 KB
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 0
            })
        
        # Suspicious files (mostly large, dangerous extensions)
        suspicious_extensions = ['.exe', '.bat', '.vbs', '.scr', '.ps1', '.cmd', '.com']
        for _ in range(80):
            ext = np.random.choice(suspicious_extensions)
            size = np.random.randint(100, 5000)  # 100-5000 KB
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 1
            })
        
        # Some safe files with suspicious extensions (false positives)
        for _ in range(20):
            ext = np.random.choice(suspicious_extensions)
            size = np.random.randint(1, 100)  # Small size
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 0
            })
        
        # Some large safe files
        for _ in range(20):
            ext = np.random.choice(safe_extensions)
            size = np.random.randint(1000, 10000)  # Large size
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 0
            })
        
        df = pd.DataFrame(data)
        
        # Save training data
        training_data_path = self.model_dir / "training_data.csv"
        df.to_csv(training_data_path, index=False)
        self.logger.info(f"📊 Training data created: {training_data_path}")
        
        return df
    
    def _train_model(self):
        """Train the Random Forest model."""
        # Create training data
        df = self._create_training_data()
        
        # Prepare features
        # Convert extensions to numerical features (one-hot encoding)
        extension_dummies = pd.get_dummies(df['file_extension'], prefix='ext')
        
        # Combine features
        X = pd.concat([extension_dummies, df[['file_size_kb']]], axis=1)
        y = df['is_malicious']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"🧠 Model trained with accuracy: {accuracy:.3f}")
        self.logger.info(f"📊 Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.logger.info(f"💾 Model saved to: {self.model_path}")
        return model
    
    def _extract_features(self, file_path):
        """Extract features from a file for AI prediction."""
        try:
            # Get file stats
            stats = file_path.stat()
            file_size_kb = stats.st_size / 1024  # Convert to KB
            extension = file_path.suffix.lower()
            
            # Create feature vector
            features = {
                'file_size_kb': file_size_kb,
                'extension': extension
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting features from {file_path}: {e}")
            return None
    
    def _predict_with_ai(self, features):
        """Use AI model to predict if file is malicious."""
        try:
            # Prepare features for model
            extension = features['extension']
            file_size_kb = features['file_size_kb']
            
            # Create feature vector (matching training data format)
            feature_vector = np.zeros(len(self.model.feature_names_in_))
            
            # Set file size
            size_idx = np.where(self.model.feature_names_in_ == 'file_size_kb')[0]
            if len(size_idx) > 0:
                feature_vector[size_idx[0]] = file_size_kb
            
            # Set extension (one-hot encoding)
            ext_prefix = 'ext_'
            for i, feature_name in enumerate(self.model.feature_names_in_):
                if feature_name.startswith(ext_prefix):
                    if feature_name == f'ext_{extension}':
                        feature_vector[i] = 1
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            confidence = self.model.predict_proba([feature_vector])[0]
            
            return {
                'is_malicious': bool(prediction),
                'confidence': max(confidence),
                'prediction': prediction
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI prediction error: {e}")
            return {
                'is_malicious': False,
                'confidence': 0.0,
                'prediction': 0
            }
    
    def is_suspicious_by_extension(self, file_path):
        """Check if a file is suspicious based on its extension."""
        return file_path.suffix.lower() in self.suspicious_extensions
    
    def analyze_file(self, file_path):
        """Analyze a file using both traditional and AI methods."""
        if not file_path.exists():
            return None
        
        # Extract features
        features = self._extract_features(file_path)
        if not features:
            return None
        
        # Traditional extension-based detection
        extension_suspicious = self.is_suspicious_by_extension(file_path)
        
        # AI-based detection
        ai_result = self._predict_with_ai(features)
        
        # Combine both methods
        ai_suspicious = ai_result['is_malicious']
        confidence = ai_result['confidence']
        
        # Determine final result
        is_suspicious = extension_suspicious or ai_suspicious
        
        # Determine detection method
        if extension_suspicious and ai_suspicious:
            detection_method = "BOTH"
        elif extension_suspicious:
            detection_method = "EXTENSION"
        elif ai_suspicious:
            detection_method = "AI"
        else:
            detection_method = "SAFE"
        
        return {
            'file_path': file_path,
            'is_suspicious': is_suspicious,
            'extension_suspicious': extension_suspicious,
            'ai_suspicious': ai_suspicious,
            'ai_confidence': confidence,
            'detection_method': detection_method,
            'file_size_kb': features['file_size_kb'],
            'extension': features['extension']
        }
    
    def quarantine_file(self, file_path):
        """Move a suspicious file to the quarantine folder."""
        try:
            if not self.quarantine_enabled:
                return False
            
            # Create quarantine filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Move file to quarantine
            shutil.move(str(file_path), str(quarantine_path))
            return True, quarantine_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to quarantine {file_path}: {e}")
            return False, None
    
    def handle_suspicious_file(self, analysis_result):
        """Handle a detected suspicious file."""
        file_path = analysis_result['file_path']
        detection_method = analysis_result['detection_method']
        ai_confidence = analysis_result['ai_confidence']
        
        # Print colorful alert
        if detection_method == "BOTH":
            alert_color = Fore.RED
            alert_icon = "🚨"
        elif detection_method == "AI":
            alert_color = Fore.YELLOW
            alert_icon = "🤖"
        else:
            alert_color = Fore.ORANGE
            alert_icon = "⚠️"
        
        print(f"\n{alert_color}{alert_icon} SUSPICIOUS FILE DETECTED!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📁 File: {file_path}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🔍 Detection: {detection_method}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 Size: {analysis_result['file_size_kb']:.1f} KB{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🧠 AI Confidence: {ai_confidence:.2%}{Style.RESET_ALL}")
        
        # Log the detection
        self.logger.warning(f"🚨 SUSPICIOUS FILE DETECTED: {file_path}")
        self.logger.info(f"📊 File size: {analysis_result['file_size_kb']:.1f} KB")
        self.logger.info(f"🔍 Detection method: {detection_method}")
        self.logger.info(f"🧠 AI confidence: {ai_confidence:.2%}")
        self.logger.info(f"🕒 Last modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}")
        
        # Try to quarantine the file
        if self.quarantine_enabled:
            success, quarantine_path = self.quarantine_file(file_path)
            if success:
                print(f"{Fore.RED}🚫 Quarantined: {quarantine_path}{Style.RESET_ALL}")
                self.logger.info(f"🚫 Quarantined: {file_path} -> {quarantine_path}")
            else:
                print(f"{Fore.RED}❌ Failed to quarantine{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ File left in place (quarantine disabled){Style.RESET_ALL}")
            self.logger.info(f"⚠️ File left in place (quarantine disabled)")
    
    def scan_directory(self):
        """Scan the monitored directory for existing suspicious files."""
        self.logger.info(f"🔍 Scanning directory: {self.monitor_path}")
        
        suspicious_count = 0
        for file_path in self.monitor_path.rglob("*"):
            if file_path.is_file():
                analysis_result = self.analyze_file(file_path)
                if analysis_result and analysis_result['is_suspicious']:
                    suspicious_count += 1
                    self.handle_suspicious_file(analysis_result)
        
        self.logger.info(f"✅ Initial scan complete. Found {suspicious_count} suspicious files.")
    
    def start_monitoring(self):
        """Start monitoring the directory for new files."""
        try:
            # Schedule the observer
            self.observer.schedule(
                self.event_handler, 
                str(self.monitor_path), 
                recursive=True
            )
            self.observer.start()
            
            # Keep the script running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop_monitoring()
        except Exception as e:
            self.logger.error(f"❌ Error during monitoring: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the file monitoring."""
        self.observer.stop()
        self.observer.join()
        print(f"\n{Fore.YELLOW}🛑 AI Antivirus monitoring stopped{Style.RESET_ALL}")
        self.logger.info("🛑 AI Antivirus monitoring stopped")


class AIAntivirusEventHandler(FileSystemEventHandler):
    """Event handler for file system events."""
    
    def __init__(self, antivirus):
        self.antivirus = antivirus
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        # Small delay to ensure file is fully written
        time.sleep(0.5)
        if file_path.exists():
            analysis_result = self.antivirus.analyze_file(file_path)
            if analysis_result and analysis_result['is_suspicious']:
                self.antivirus.handle_suspicious_file(analysis_result)
    
    def on_moved(self, event):
        """Handle file move events."""
        if event.is_directory:
            return
        
        file_path = Path(event.dest_path)
        analysis_result = self.antivirus.analyze_file(file_path)
        if analysis_result and analysis_result['is_suspicious']:
            self.antivirus.handle_suspicious_file(analysis_result)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n{Fore.YELLOW}🛑 Received interrupt signal. Shutting down...{Style.RESET_ALL}")
    sys.exit(0)


def main():
    """Main function to run the AI antivirus."""
    parser = argparse.ArgumentParser(description="AI-Enhanced Simple Antivirus in Python")
    parser.add_argument(
        "--path", 
        default="Downloads", 
        help="Path to monitor (default: Downloads)"
    )
    parser.add_argument(
        "--no-quarantine", 
        action="store_true", 
        help="Disable quarantine functionality"
    )
    parser.add_argument(
        "--scan-only", 
        action="store_true", 
        help="Only scan existing files, don't monitor for new ones"
    )
    parser.add_argument(
        "--retrain", 
        action="store_true", 
        help="Retrain the AI model"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize antivirus
    antivirus = AIAntivirus(
        monitor_path=args.path,
        quarantine_enabled=not args.no_quarantine
    )
    
    # Retrain model if requested
    if args.retrain:
        print(f"{Fore.CYAN}🔄 Retraining AI model...{Style.RESET_ALL}")
        antivirus.model = antivirus._train_model()
    
    # Perform initial scan
    antivirus.scan_directory()
    
    # Start monitoring if not scan-only mode
    if not args.scan_only:
        antivirus.start_monitoring()
    else:
        print(f"{Fore.GREEN}✅ Scan complete. Exiting...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()