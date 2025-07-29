#!/usr/bin/env python3
"""
Ultimate AI Antivirus v5.X - Windows Optimized Version
Enhanced security agent with AI integration and Windows compatibility
"""

import os
import sys
import time
import signal
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import platform

# Data science imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Rich terminal output (Windows compatible)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    import colorama
    from colorama import Fore, Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic output")

import threading

# Import our modules
from config import (
    SUSPICIOUS_EXTENSIONS, THREAT_LEVELS, LOGS_DIR, QUARANTINE_DIR,
    MODEL_PATH, SCAN_THRESHOLDS, DETECTION_METHODS
)
from utils import (
    get_high_risk_paths, get_full_scan_paths, create_log_folders,
    print_colored, create_timestamp, get_file_metadata, format_size,
    validate_file_path, get_platform_info, get_file_hash, is_known_malware,
    add_to_known_malware, get_known_malware_count, get_entropy,
    get_file_type, get_filename_pattern_flags, simulate_file_creation_randomness
)

# Initialize colorama and rich with Windows compatibility
if RICH_AVAILABLE:
    colorama.init(autoreset=True)
    console = Console(force_terminal=True, color_system="auto")
else:
    console = None

# ============================================================================
# WINDOWS-SAFE PRINT FUNCTIONS
# ============================================================================

def safe_print(text: str, color: str = "white"):
    """Safely print text with fallback for Windows encoding issues."""
    if not RICH_AVAILABLE:
        print(text)
        return
        
    try:
        console.print(text, style=color)
    except UnicodeEncodeError:
        # Fallback for Windows encoding issues
        clean_text = text.replace("⚠️", "WARNING").replace("❌", "ERROR").replace("✅", "OK").replace("🧠", "AI").replace("🛡️", "SHIELD")
        print(clean_text)
    except Exception as e:
        # Ultimate fallback
        print(text)

def safe_log(logger, message: str, level: str = "info"):
    """Safely log messages without emojis for Windows compatibility."""
    clean_message = message.replace("🚀", "").replace("📁", "").replace("🛡️", "").replace("🔍", "").replace("🧠", "AI").replace("❌", "ERROR").replace("✅", "OK")
    if level == "info":
        logger.info(clean_message)
    elif level == "warning":
        logger.warning(clean_message)
    elif level == "error":
        logger.error(clean_message)

# ============================================================================
# THREAT LEVEL FUNCTIONS
# ============================================================================

def get_threat_level(score: float) -> Dict:
    """Get threat level based on AI confidence score."""
    for level, info in THREAT_LEVELS.items():
        if score >= info['min_score']:
            return {
                'level': level,
                'emoji': info['emoji'],
                'color': info['color'],
                'score': score
            }
    return THREAT_LEVELS['SAFE']

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def create_timestamp() -> str:
    """Create timestamp string for logging."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# ENHANCED AI ANTIVIRUS CLASS
# ============================================================================

class UltimateAIAntivirus:
    def __init__(self, monitor_path: str, quarantine_enabled: bool = True, 
                 model_path: str = "model/model.pkl", gui_mode: bool = False,
                 scan_mode: str = "normal"):
        """
        Initialize the Ultimate AI Antivirus system.
        
        Args:
            monitor_path: Path to monitor for suspicious files
            quarantine_enabled: Whether to move suspicious files to quarantine
            model_path: Path to the trained ML model
            gui_mode: Whether to run in GUI mode
            scan_mode: Scan mode ('normal', 'smart', 'full')
        """
        self.monitor_path = Path(monitor_path).resolve()
        self.quarantine_enabled = quarantine_enabled
        self.model_path = Path(model_path)
        self.gui_mode = gui_mode
        self.scan_mode = scan_mode
        
        # Statistics tracking
        self.stats = {
            'files_scanned': 0,
            'threats_found': 0,
            'quarantined': 0,
            'ai_detections': 0,
            'extension_detections': 0,
            'both_detections': 0,
            'start_time': datetime.now(),
            'last_scan_time': None
        }
        
        # Create necessary directories
        self.logs_dir = Path("logs")
        self.quarantine_dir = Path("quarantine")
        self.model_dir = Path("model")
        self._create_directories()
        
        # Setup logging first
        self._setup_logging()
        
        # Load or train the AI model
        self.model = self._load_or_train_model()
        
        # Initialize watchdog observer
        self.observer = Observer()
        self.event_handler = UltimateAIAntivirusEventHandler(self)
        
        # Dashboard components
        self.dashboard_live = None
        self.dashboard_running = False
        
        # Print startup info
        self._print_startup_info()
    
    def _create_directories(self):
        """Create necessary directories."""
        self.logs_dir.mkdir(exist_ok=True)
        self.quarantine_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logs directory
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = create_timestamp()
        log_file = self.logs_dir / f"antivirus_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Log startup
        safe_log(self.logger, "Ultimate AI Antivirus v4.X Started")
        safe_log(self.logger, f"Monitoring path: {self.monitor_path}")
        safe_log(self.logger, f"Quarantine enabled: {self.quarantine_enabled}")
        safe_log(self.logger, f"Scan mode: {self.scan_mode.upper()}")
    
    def _print_startup_info(self):
        """Print enhanced startup information."""
        if not RICH_AVAILABLE:
            print("=" * 60)
            print("ULTIMATE AI ANTIVIRUS v4.X")
            print("Enhanced Security Agent with AI Integration")
            print(f"Monitoring: {self.monitor_path}")
            print(f"Quarantine: {'Enabled' if self.quarantine_enabled else 'Disabled'}")
            print(f"AI Model: {'Loaded' if self.model else 'Training...'}")
            print(f"Scan Mode: {self.scan_mode.upper()}")
            print("=" * 60)
            return
            
        try:
            console.print(Panel(
                f"[bold cyan]ULTIMATE AI ANTIVIRUS v4.X[/bold cyan]\n"
                f"[green]Enhanced Security Agent with AI Integration[/green]\n"
                f"[yellow]Monitoring: {self.monitor_path}[/yellow]\n"
                f"[yellow]Quarantine: {'Enabled' if self.quarantine_enabled else 'Disabled'}[/yellow]\n"
                f"[yellow]AI Model: {'Loaded' if self.model else 'Training...'}[/yellow]\n"
                f"[yellow]Scan Mode: {self.scan_mode.upper()}[/yellow]\n"
                f"[yellow]Suspicious extensions: {', '.join(list(SUSPICIOUS_EXTENSIONS)[:10])}...[/yellow]",
                border_style="blue"
            ))
        except UnicodeEncodeError:
            # Fallback for Windows encoding issues
            print("=" * 60)
            print("ULTIMATE AI ANTIVIRUS v4.X")
            print("Enhanced Security Agent with AI Integration")
            print(f"Monitoring: {self.monitor_path}")
            print(f"Quarantine: {'Enabled' if self.quarantine_enabled else 'Disabled'}")
            print(f"AI Model: {'Loaded' if self.model else 'Training...'}")
            print(f"Scan Mode: {self.scan_mode.upper()}")
            print("=" * 60)
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                safe_log(self.logger, "AI model loaded successfully")
                return model
            else:
                safe_log(self.logger, "No existing model found. Training new model...", "warning")
                return self._train_model()
        except Exception as e:
            safe_log(self.logger, f"Error loading model: {e}", "error")
            return self._train_model()
    
    def _create_training_data(self):
        """Create comprehensive training data."""
        # This would be replaced with real training data generation
        # For now, we'll use a simple approach
        data = []
        
        # Safe files
        safe_extensions = ['.txt', '.pdf', '.jpg', '.png', '.mp3', '.mp4', '.doc', '.xls']
        for ext in safe_extensions:
            for i in range(20):
                data.append({
                    'file_extension': ext,
                    'file_size_kb': np.random.randint(1, 1000),
                    'is_malicious': 0
                })
        
        # Suspicious files
        for ext in SUSPICIOUS_EXTENSIONS:
            for i in range(15):
                data.append({
                    'file_extension': ext,
                    'file_size_kb': np.random.randint(10, 5000),
                    'is_malicious': 1
                })
        
        return pd.DataFrame(data)
    
    def _train_model(self):
        """Train the Random Forest model."""
        try:
            # Create training data
            df = self._create_training_data()
            
            # Prepare features
            extension_dummies = pd.get_dummies(df['file_extension'], prefix='ext')
            X = pd.concat([extension_dummies, df[['file_size_kb']]], axis=1)
            y = df['is_malicious']
            
            # Train model
            model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            model.fit(X, y)
            
            # Save model
            self.model_dir.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)
            
            safe_log(self.logger, "AI model trained and saved successfully")
            return model
            
        except Exception as e:
            safe_log(self.logger, f"Error training model: {e}", "error")
            return None
    
    def _extract_features(self, file_path: Path) -> Optional[Dict]:
        """Extract features from a file for AI analysis."""
        try:
            stat = file_path.stat()
            
            # Read file content for entropy calculation
            try:
                with open(file_path, 'rb') as f:
                    content = f.read(1024)  # Read first 1KB for analysis
            except:
                content = b''
            
            # Calculate entropy
            entropy_score = get_entropy(content) if content else 0.0
            
            # Get file category
            file_category = get_file_type(file_path.name)
            
            # Get filename patterns
            pattern_flags = get_filename_pattern_flags(file_path.name)
            
            # Simulate creation randomness
            creation_randomness = simulate_file_creation_randomness()
            
            # Base features
            features = {
                'file_size_kb': stat.st_size / 1024,
                'entropy_score': entropy_score,
                'creation_randomness': creation_randomness,
                'extension': file_path.suffix.lower(),
                'file_category': file_category,
                'last_modified': datetime.fromtimestamp(stat.st_mtime)
            }
            
            # Add pattern flags
            features.update(pattern_flags)
            
            return features
            
        except Exception as e:
            safe_log(self.logger, f"Error extracting features from {file_path}: {e}", "error")
            return None
    
    def _predict_with_ai(self, features: Dict) -> Dict:
        """Make AI prediction for file analysis."""
        try:
            if not self.model:
                return {'is_malicious': False, 'confidence': 0.0}
            
            # Create DataFrame with proper feature names
            feature_df = pd.DataFrame([features])
            
            # Add category dummies
            category_dummies = pd.get_dummies([features.get('file_category', 'other')], prefix='category')
            extension_dummies = pd.get_dummies([features.get('extension', '.unknown')], prefix='ext')
            
            # Combine features
            feature_df = pd.concat([feature_df, category_dummies, extension_dummies], axis=1)
            
            # Ensure all model features are present
            if hasattr(self.model, 'feature_names_in_'):
                model_features = self.model.feature_names_in_
                for feature in model_features:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0.0
                
                # Reorder columns to match model expectations
                feature_df = feature_df[model_features]
            
            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            confidence = self.model.predict_proba(feature_df)[0][1]
            
            return {
                'is_malicious': bool(prediction),
                'confidence': confidence
            }
            
        except Exception as e:
            safe_log(self.logger, f"Error in AI prediction: {e}", "error")
            return {'is_malicious': False, 'confidence': 0.0}
    
    def is_suspicious_by_extension(self, file_path: Path) -> bool:
        """Check if file is suspicious based on extension."""
        extension = file_path.suffix.lower()
        
        # If it's a suspicious extension, check if it's a protected project file
        if extension in SUSPICIOUS_EXTENSIONS:
            # Don't flag project batch files as suspicious
            if extension == '.bat':
                file_name = file_path.name.lower()
                protected_bat_files = {'setup_windows.bat', 'run_antivirus.bat'}
                if file_name in protected_bat_files:
                    return False
            
            return True
        
        return False
    
    def analyze_file(self, file_path: Path) -> Optional[Dict]:
        """Analyze a file using both traditional and AI methods."""
        if not file_path.exists():
            return None
        
        # Check if this is a protected project file
        file_name = file_path.name.lower()
        protected_files = {
            'setup_windows.bat', 'run_antivirus.bat', 'ai_antivirus.py', 
            'ai_antivirus_windows.py', 'config.py', 'utils.py', 'requirements.txt',
            'README.md', 'README_WINDOWS.md', 'WINDOWS_SETUP_GUIDE.md',
            'test_scan.py', 'gui.py', 'test_suite.py', 'run_final_test.py'
        }
        
        if file_name in protected_files:
            # Skip analysis for protected files
            self.stats['files_scanned'] += 1
            return None
        
        # Check if file is already known malware
        file_hash = get_file_hash(str(file_path))
        if file_hash and is_known_malware(file_hash):
            safe_log(self.logger, f"Known malware detected: {file_path.name}")
            self.stats['files_scanned'] += 1
            self.stats['last_scan_time'] = datetime.now()
            return {
                'file_path': file_path,
                'is_suspicious': True,
                'extension_suspicious': False,
                'ai_suspicious': False,
                'ai_confidence': 1.0,
                'detection_method': "KNOWN_MALWARE",
                'file_size_kb': file_path.stat().st_size / 1024,
                'extension': file_path.suffix.lower(),
                'threat_level': "CRITICAL",
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            }
        
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
        
        # Get threat level
        threat_info = get_threat_level(confidence)
        
        # Update statistics
        self.stats['files_scanned'] += 1
        self.stats['last_scan_time'] = datetime.now()
        
        if is_suspicious:
            self.stats['threats_found'] += 1
            if detection_method == "AI":
                self.stats['ai_detections'] += 1
            elif detection_method == "EXTENSION":
                self.stats['extension_detections'] += 1
            elif detection_method == "BOTH":
                self.stats['both_detections'] += 1
        
        return {
            'file_path': file_path,
            'is_suspicious': is_suspicious,
            'extension_suspicious': extension_suspicious,
            'ai_suspicious': ai_suspicious,
            'ai_confidence': confidence,
            'detection_method': detection_method,
            'file_size_kb': features['file_size_kb'],
            'extension': features['extension'],
            'threat_level': threat_info['level'],
            'last_modified': features['last_modified']
        }
    
    def log_threat(self, analysis_result: Dict):
        """Log threat detection with enhanced details."""
        file_path = analysis_result['file_path']
        threat_level = analysis_result['threat_level']
        detection_method = analysis_result['detection_method']
        confidence = analysis_result['ai_confidence']
        
        # Create detailed log message
        log_message = (
            f"THREAT DETECTED: {file_path.name} | "
            f"Level: {threat_level} | "
            f"Method: {detection_method} | "
            f"Confidence: {confidence:.2f} | "
            f"Size: {format_file_size(int(analysis_result['file_size_kb'] * 1024))}"
        )
        
        safe_log(self.logger, log_message, "warning")
    
    def quarantine_file(self, file_path: Path) -> Tuple[bool, Optional[Path]]:
        """Move suspicious file to quarantine with timestamp."""
        try:
            if not file_path.exists():
                return False, None
            
            # Create quarantine filename with timestamp
            timestamp = create_timestamp()
            quarantine_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Move file to quarantine
            import shutil
            shutil.move(str(file_path), str(quarantine_path))
            
            safe_log(self.logger, f"File quarantined: {file_path.name} -> {quarantine_name}")
            self.stats['quarantined'] += 1
            
            return True, quarantine_path
            
        except Exception as e:
            safe_log(self.logger, f"Error quarantining {file_path}: {e}", "error")
            return False, None
    
    def handle_suspicious_file(self, analysis_result: Dict):
        """Handle suspicious file detection."""
        file_path = analysis_result['file_path']
        
        # Log the threat
        self.log_threat(analysis_result)
        
        # Add to known malware database if AI detected
        if analysis_result['ai_suspicious'] or analysis_result['detection_method'] == "BOTH":
            file_hash = get_file_hash(str(file_path))
            if file_hash:
                add_to_known_malware(
                    str(file_path),
                    analysis_result,
                    analysis_result['ai_confidence'],
                    analysis_result['detection_method']
                )
        
        # Quarantine if enabled
        if self.quarantine_enabled:
            self.quarantine_file(file_path)
    
    def scan_directory(self, show_progress: bool = True):
        """Scan directory for suspicious files."""
        safe_log(self.logger, "Finding files to scan...")
        files_to_scan = self._get_files_to_scan()
        
        if not files_to_scan:
            safe_log(self.logger, "No files to scan")
            return
        
        safe_log(self.logger, f"Starting scan of {len(files_to_scan)} files...")
        
        start_time = time.time()
        
        for i, file_path in enumerate(files_to_scan):
            try:
                # Analyze file
                result = self.analyze_file(file_path)
                
                if result and result['is_suspicious']:
                    self.handle_suspicious_file(result)
                
                # Show progress more frequently
                if show_progress:
                    if (i + 1) % 50 == 0:  # Every 50 files
                        progress = (i + 1) / len(files_to_scan) * 100
                        safe_log(self.logger, f"Scan progress: {progress:.1f}% ({i + 1}/{len(files_to_scan)})")
                    elif (i + 1) % 10 == 0:  # Every 10 files for small scans
                        if len(files_to_scan) < 1000:
                            progress = (i + 1) / len(files_to_scan) * 100
                            safe_log(self.logger, f"Scan progress: {progress:.1f}% ({i + 1}/{len(files_to_scan)})")
                    
            except Exception as e:
                safe_log(self.logger, f"Error scanning {file_path}: {e}", "error")
        
        scan_time = time.time() - start_time
        files_per_second = len(files_to_scan) / scan_time if scan_time > 0 else 0
        
        safe_log(self.logger, f"Scan complete. Scanned {self.stats['files_scanned']} files in {scan_time:.2f}s")
        safe_log(self.logger, f"Performance: {files_per_second:.1f} files/second")
        safe_log(self.logger, f"Threats found: {self.stats['threats_found']}")
    
    def _get_files_to_scan(self) -> List[Path]:
        """Get list of files to scan based on scan mode."""
        files_to_scan = []
        
        if self.scan_mode == "smart":
            # Smart scan: only high-risk directories
            paths = get_high_risk_paths()
            safe_log(self.logger, "Smart scan: scanning high-risk directories only")
        elif self.scan_mode == "full":
            # Full scan: entire system
            paths = get_full_scan_paths()
            safe_log(self.logger, "Full scan: scanning entire system (this may take a while)")
        else:
            # Normal scan: specified path
            paths = [self.monitor_path]
            safe_log(self.logger, f"Normal scan: scanning path {self.monitor_path}")
        
        safe_log(self.logger, f"Searching {len(paths)} directories for files...")
        
        for i, path in enumerate(paths):
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.is_file():
                    files_to_scan.append(path_obj)
                else:
                    # Recursively find all files
                    file_count = 0
                    for file_path in path_obj.rglob('*'):
                        if file_path.is_file() and self._should_scan_file(file_path):
                            files_to_scan.append(file_path)
                            file_count += 1
                            # Show progress for large directories
                            if file_count % 1000 == 0:
                                safe_log(self.logger, f"Found {file_count} files in {path_obj}...")
        
        safe_log(self.logger, f"Found {len(files_to_scan)} files to scan")
        return files_to_scan
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned."""
        # Exclude certain directories and files
        excluded_patterns = {
            'quarantine', 'logs', 'model', 'test_files',
            'known_malware.csv', '.git', '.svn', '.hg',
            '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', '.env'
        }
        
        # Important project files that should never be quarantined
        protected_files = {
            'setup_windows.bat', 'run_antivirus.bat', 'ai_antivirus.py', 
            'ai_antivirus_windows.py', 'config.py', 'utils.py', 'requirements.txt',
            'README.md', 'README_WINDOWS.md', 'WINDOWS_SETUP_GUIDE.md',
            'test_scan.py', 'gui.py', 'test_suite.py', 'run_final_test.py'
        }
        
        # Check if file path contains any excluded patterns
        file_path_str = str(file_path).lower()
        for pattern in excluded_patterns:
            if pattern in file_path_str:
                return False
        
        # Check if file is a protected project file
        file_name = file_path.name.lower()
        if file_name in {f.lower() for f in protected_files}:
            return False
        
        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                return False
        except:
            return False
        
        return True
    
    def start_monitoring(self):
        """Start file system monitoring."""
        try:
            self.observer.schedule(self.event_handler, str(self.monitor_path), recursive=True)
            self.observer.start()
            safe_log(self.logger, f"Started monitoring: {self.monitor_path}")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_monitoring()
                
        except Exception as e:
            safe_log(self.logger, f"Error starting monitoring: {e}", "error")
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        try:
            self.observer.stop()
            self.observer.join()
            safe_log(self.logger, "Monitoring stopped")
        except Exception as e:
            safe_log(self.logger, f"Error stopping monitoring: {e}", "error")

class UltimateAIAntivirusEventHandler(FileSystemEventHandler):
    def __init__(self, antivirus):
        self.antivirus = antivirus
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self.antivirus._should_scan_file(file_path):
                result = self.antivirus.analyze_file(file_path)
                if result and result['is_suspicious']:
                    self.antivirus.handle_suspicious_file(result)
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if self.antivirus._should_scan_file(file_path):
                result = self.antivirus.analyze_file(file_path)
                if result and result['is_suspicious']:
                    self.antivirus.handle_suspicious_file(result)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    safe_log(logging.getLogger(), "Received interrupt signal, shutting down...")
    sys.exit(0)

def create_gui_placeholder():
    """Launch GUI mode."""
    try:
        import subprocess
        import sys
        
        # Launch GUI
        subprocess.run([sys.executable, "gui.py"])
        
    except Exception as e:
        safe_print(f"Error launching GUI: {e}", "red")
        safe_print("You can run the GUI manually with: python3 gui.py", "yellow")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function with enhanced CLI options."""
    parser = argparse.ArgumentParser(
        description="Ultimate AI Antivirus v4.X - Enhanced Security Agent",
        epilog="Example: python3 ai_antivirus.py --path /home/user --smart-scan"
    )
    
    parser.add_argument('--path', type=str, default=".",
                       help='Path to monitor for suspicious files')
    parser.add_argument('--scan-only', action='store_true',
                       help='Scan once and exit (no monitoring)')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with sample files')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain the AI model')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI mode (placeholder)')
    parser.add_argument('--smart-scan', action='store_true',
                       help='Smart scan: only high-risk directories')
    parser.add_argument('--full-scan', action='store_true',
                       help='Full scan: entire system (use with caution)')
    parser.add_argument('--upload-logs', action='store_true',
                       help='Upload logs to cloud (placeholder)')
    parser.add_argument('--model-info', action='store_true',
                       help='Print model information and exit')
    
    args = parser.parse_args()
    
    # Validate scan mode conflicts
    scan_modes = [args.smart_scan, args.full_scan]
    if sum(scan_modes) > 1:
        safe_print("ERROR: Only one scan mode allowed (--smart-scan OR --full-scan)", "red")
        return
    
    # Validate path conflicts with scan modes
    if args.path != "." and (args.smart_scan or args.full_scan):
        safe_print("ERROR: Cannot use --path with --smart-scan or --full-scan", "red")
        safe_print("Use --path for specific directory OR use scan modes for system-wide scanning", "red")
        return
    
    # Determine scan mode
    if args.smart_scan:
        scan_mode = "smart"
        safe_print("AI Smart Scan mode active", "yellow")
    elif args.full_scan:
        scan_mode = "full"
        safe_print("WARNING: Full system scan requested!", "red")
        safe_print("This will scan your entire system. Continue? (y/N):", "red")
        response = input().lower()
        if response != 'y':
            safe_print("Scan cancelled.", "yellow")
            return
    else:
        scan_mode = "normal"
    
    # Handle special modes
    if args.gui:
        create_gui_placeholder()
        return
    
    if args.model_info:
        # Print model information
        model_path = Path("model/model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"[bold cyan]Model Information[/bold cyan]\n"
                    f"Model Type: Random Forest Classifier\n"
                    f"Estimators: {model.n_estimators}\n"
                    f"Features: {model.n_features_in_}\n"
                    f"Classes: {list(model.classes_)}",
                    border_style="blue"
                ))
            else:
                print("=" * 40)
                print("Model Information")
                print(f"Model Type: Random Forest Classifier")
                print(f"Estimators: {model.n_estimators}")
                print(f"Features: {model.n_features_in_}")
                print(f"Classes: {list(model.classes_)}")
                print("=" * 40)
        else:
            safe_print("ERROR No trained model found", "red")
        return
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create antivirus instance
    antivirus = UltimateAIAntivirus(
        monitor_path=args.path,
        quarantine_enabled=True,
        gui_mode=args.gui,
        scan_mode=scan_mode
    )
    
    # Handle retrain mode
    if args.retrain:
        safe_print("Retraining AI model...", "yellow")
        antivirus.model = antivirus._train_model()
        safe_print("OK Model retraining complete!", "green")
        return
    
    # Handle demo mode
    if args.demo:
        safe_print("Demo mode: Creating sample files...", "cyan")
        # Create some sample files for demonstration
        demo_dir = Path("demo_files")
        demo_dir.mkdir(exist_ok=True)
        
        # Create safe file
        (demo_dir / "safe_document.txt").write_text("This is a safe file for demo purposes.")
        
        # Create suspicious file
        (demo_dir / "suspicious_script.bat").write_text("@echo off\necho 'This is a demo suspicious file'")
        
        antivirus.monitor_path = demo_dir
        safe_print(f"OK Demo files created in: {demo_dir}", "green")
    
    # Handle scan-only mode
    if args.scan_only:
        safe_print("Performing one-time scan...", "cyan")
        antivirus.scan_directory(show_progress=True)
        safe_print("OK Scan complete. Exiting...", "green")
        return
    
    # Handle upload logs (placeholder)
    if args.upload_logs:
        safe_print("Upload logs feature (placeholder)", "yellow")
        safe_print("Future versions will include cloud log upload functionality.", "yellow")
    
    # Start monitoring
    antivirus.start_monitoring()

if __name__ == "__main__":
    main()