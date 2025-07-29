#!/usr/bin/env python3
"""
🚀 ULTIMATE AI ANTIVIRUS v4.X
Enhanced AI-powered security agent with modular design, real-time monitoring, and comprehensive threat detection.
"""

import os
import sys
import time
import signal
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import colorama
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
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
    add_to_known_malware, get_known_malware_count
)

# Initialize colorama and rich
colorama.init(autoreset=True)
console = Console()

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
        self.dashboard_active = False
        self.dashboard_thread = None
        
        self._print_startup_info()
    
    def _create_directories(self):
        """Create logs, quarantine, and model directories if they don't exist."""
        create_log_folders()
    
    def _setup_logging(self):
        """Setup enhanced logging configuration."""
        timestamp = create_timestamp()
        log_file = self.logs_dir / f"ultimate_antivirus_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log startup
        self.logger.info("🚀 Ultimate AI Antivirus v4.X Started")
        self.logger.info(f"📁 Monitoring path: {self.monitor_path}")
        self.logger.info(f"🛡️ Quarantine enabled: {self.quarantine_enabled}")
        self.logger.info(f"🔍 Scan mode: {self.scan_mode.upper()}")
    
    def _print_startup_info(self):
        """Print enhanced startup information."""
        console.print(Panel(
            f"[bold cyan]🚀 ULTIMATE AI ANTIVIRUS v4.X[/bold cyan]\n"
            f"[green]Enhanced Security Agent with AI Integration[/green]\n"
            f"[yellow]📁 Monitoring: {self.monitor_path}[/yellow]\n"
            f"[yellow]🛡️ Quarantine: {'Enabled' if self.quarantine_enabled else 'Disabled'}[/yellow]\n"
            f"[yellow]🧠 AI Model: {'Loaded' if self.model else 'Training...'}[/yellow]\n"
            f"[yellow]🔍 Scan Mode: {self.scan_mode.upper()}[/yellow]\n"
            f"[yellow]⚠️ Suspicious extensions: {', '.join(list(SUSPICIOUS_EXTENSIONS)[:10])}...[/yellow]",
            border_style="blue"
        ))
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                self.logger.info("🧠 AI model loaded successfully")
                return model
            else:
                self.logger.warning("🧠 No existing model found. Training new model...")
                return self._train_model()
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
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
            
            self.logger.info("🧠 AI model trained and saved successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error training model: {e}")
            return None
    
    def _extract_features(self, file_path: Path) -> Optional[Dict]:
        """Extract features from a file for AI analysis."""
        try:
            stat = file_path.stat()
            return {
                'file_size_kb': stat.st_size / 1024,
                'extension': file_path.suffix.lower(),
                'last_modified': datetime.fromtimestamp(stat.st_mtime)
            }
        except Exception as e:
            self.logger.error(f"❌ Error extracting features from {file_path}: {e}")
            return None
    
    def _predict_with_ai(self, features: Dict) -> Dict:
        """Make AI prediction for file analysis."""
        try:
            if not self.model:
                return {'is_malicious': False, 'confidence': 0.0}
            
            # Create DataFrame with proper feature names
            feature_df = pd.DataFrame([features])
            
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
            self.logger.error(f"❌ Error in AI prediction: {e}")
            return {'is_malicious': False, 'confidence': 0.0}
    
    def is_suspicious_by_extension(self, file_path: Path) -> bool:
        """Check if file is suspicious based on extension."""
        return file_path.suffix.lower() in SUSPICIOUS_EXTENSIONS
    
    def analyze_file(self, file_path: Path) -> Optional[Dict]:
        """Analyze a file using both traditional and AI methods."""
        if not file_path.exists():
            return None
        
        # Check if file is already known malware
        file_hash = get_file_hash(str(file_path))
        if file_hash and is_known_malware(file_hash):
            self.logger.info(f"🧠 Known malware detected: {file_path.name}")
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
        if is_suspicious:
            self.stats['threats_found'] += 1
            if detection_method == "AI":
                self.stats['ai_detections'] += 1
            elif detection_method == "EXTENSION":
                self.stats['extension_detections'] += 1
            elif detection_method == "BOTH":
                self.stats['both_detections'] += 1
        
        self.stats['last_scan_time'] = datetime.now()
        
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
        """Log threat detection with enhanced information."""
        file_path = analysis_result['file_path']
        detection_method = analysis_result['detection_method']
        ai_confidence = analysis_result['ai_confidence']
        threat_level = analysis_result['threat_level']
        file_size = analysis_result['file_size_kb']
        
        # Log detailed information
        self.logger.warning(f"🚨 THREAT DETECTED: {file_path}")
        self.logger.info(f"📊 File size: {file_size:.1f} KB")
        self.logger.info(f"🔍 Detection method: {detection_method}")
        self.logger.info(f"🧠 AI confidence: {ai_confidence:.2%}")
        self.logger.info(f"⚠️ Threat level: {threat_level}")
        self.logger.info(f"🕒 Last modified: {analysis_result['last_modified']}")
        self.logger.info(f"📁 Extension: {analysis_result['extension']}")
    
    def quarantine_file(self, file_path: Path) -> Tuple[bool, Optional[Path]]:
        """Move a suspicious file to the quarantine folder."""
        try:
            if not self.quarantine_enabled:
                return False, None
            
            # Create quarantine filename with timestamp
            timestamp = create_timestamp()
            quarantine_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Move file to quarantine
            shutil.move(str(file_path), str(quarantine_path))
            self.stats['quarantined'] += 1
            return True, quarantine_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to quarantine {file_path}: {e}")
            return False, None
    
    def handle_suspicious_file(self, analysis_result: Dict):
        """Handle a suspicious file with enhanced logging."""
        file_path = analysis_result['file_path']
        detection_method = analysis_result['detection_method']
        ai_confidence = analysis_result['ai_confidence']
        threat_level = analysis_result['threat_level']
        
        # Log the threat
        self.log_threat(analysis_result)
        
        # Print colored alert
        threat_emoji = THREAT_LEVELS.get(threat_level, {}).get('emoji', '⚠️')
        threat_color = THREAT_LEVELS.get(threat_level, {}).get('color', Fore.YELLOW)
        
        console.print(f"{threat_color}{threat_emoji} THREAT DETECTED! {threat_emoji}{Style.RESET_ALL}")
        console.print(f"📁 File: {file_path}")
        console.print(f"🔍 Detection: {detection_method}")
        console.print(f"📊 Size: {analysis_result['file_size_kb']:.1f} KB")
        console.print(f"🧠 AI Confidence: {ai_confidence:.2%}")
        console.print(f"⚠️ Threat Level: {threat_level} {threat_emoji}")
        
        # Add to known malware database if it's a new AI or BOTH detection
        if detection_method in ["AI", "BOTH"]:
            features = {
                'file_size_kb': analysis_result['file_size_kb'],
                'extension': analysis_result['extension'],
                'entropy': 0.0  # Will be calculated in add_to_known_malware
            }
            add_to_known_malware(str(file_path), features, ai_confidence, detection_method)
        
        # Quarantine if enabled
        if self.quarantine_enabled:
            success, quarantine_path = self.quarantine_file(file_path)
            if success:
                console.print(f"🚫 Quarantined: {quarantine_path}")
            else:
                console.print("❌ Failed to quarantine file")
    
    def scan_directory(self, show_progress: bool = True):
        """Scan directory with enhanced progress tracking."""
        if not self.monitor_path.exists():
            self.logger.error(f"❌ Monitor path does not exist: {self.monitor_path}")
            return
        
        # Get files to scan based on scan mode
        files_to_scan = self._get_files_to_scan()
        
        if not files_to_scan:
            self.logger.info("✅ No files to scan")
            return
        
        self.logger.info(f"🔍 Scanning directory: {self.monitor_path}")
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Scanning files...", total=len(files_to_scan))
                
                for file_path in files_to_scan:
                    progress.update(task, description=f"Scanning: {file_path.name}")
                    
                    try:
                        analysis_result = self.analyze_file(file_path)
                        if analysis_result and analysis_result['is_suspicious']:
                            self.handle_suspicious_file(analysis_result)
                    except Exception as e:
                        self.logger.error(f"❌ Error scanning {file_path}: {e}")
                    
                    progress.advance(task)
        else:
            for file_path in files_to_scan:
                try:
                    analysis_result = self.analyze_file(file_path)
                    if analysis_result and analysis_result['is_suspicious']:
                        self.handle_suspicious_file(analysis_result)
                except Exception as e:
                    self.logger.error(f"❌ Error scanning {file_path}: {e}")
        
        self.logger.info(f"✅ Initial scan complete. Found {self.stats['threats_found']} suspicious files.")
    
    def _get_files_to_scan(self) -> List[Path]:
        """Get files to scan based on scan mode."""
        files = []
        
        if self.scan_mode == "smart":
            # Smart scan: only high-risk directories
            scan_paths = get_high_risk_paths()
            self.logger.info(f"🧠 Smart scan mode: scanning {len(scan_paths)} high-risk directories")
            
        elif self.scan_mode == "full":
            # Full scan: entire system
            scan_paths = get_full_scan_paths()
            self.logger.info(f"🔍 Full scan mode: scanning {len(scan_paths)} system paths")
            
        else:
            # Normal scan: just the monitor path
            scan_paths = [self.monitor_path]
            self.logger.info(f"📁 Normal scan mode: scanning monitor path")
        
        # Collect files from all scan paths
        for scan_path in scan_paths:
            scan_path = Path(scan_path)
            if scan_path.exists():
                for file_path in scan_path.rglob("*"):
                    if file_path.is_file() and self._should_scan_file(file_path):
                        files.append(file_path)
        
        return files
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned based on size and type."""
        try:
            # Check file size limits
            stat = file_path.stat()
            if stat.st_size < SCAN_THRESHOLDS['min_file_size']:
                return False
            if stat.st_size > SCAN_THRESHOLDS['max_file_size']:
                return False
            
            # Skip system directories and common exclusions
            excluded_patterns = [
                '/proc', '/dev', '/sys', '/tmp', '/var/cache', '/var/log',
                '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
                'node_modules', '.venv', 'venv', '.env', 'quarantine', 'logs', 'model',
                'known_malware.csv'
            ]
            
            file_path_str = str(file_path)
            if any(pattern in file_path_str for pattern in excluded_patterns):
                return False
            
            return True
            
        except Exception:
            return False
    
    def create_dashboard(self) -> Layout:
        """Create enhanced real-time dashboard."""
        layout = Layout()
        
        # Header
        header = Panel(
            f"[bold cyan]🚀 ULTIMATE AI ANTIVIRUS v4.X DASHBOARD[/bold cyan]\n"
            f"[green]Real-time Security Monitoring[/green]",
            border_style="blue"
        )
        
        # Statistics table
        table = Table(title="📊 Live Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="yellow")
        
        # Calculate uptime
        uptime = datetime.now() - self.stats['start_time']
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
        
        # Calculate files per second
        if self.stats['last_scan_time']:
            time_diff = (datetime.now() - self.stats['last_scan_time']).total_seconds()
            files_per_sec = self.stats['files_scanned'] / max(time_diff, 1)
        else:
            files_per_sec = 0
        
        table.add_row("Files Scanned", str(self.stats['files_scanned']), "Total files analyzed")
        table.add_row("Threats Found", str(self.stats['threats_found']), "Suspicious files detected")
        table.add_row("Quarantined", str(self.stats['quarantined']), "Files moved to quarantine")
        table.add_row("AI Detections", str(self.stats['ai_detections']), "ML-based detections")
        table.add_row("Extension Detections", str(self.stats['extension_detections']), "Rule-based detections")
        table.add_row("Both Detections", str(self.stats['both_detections']), "AI + Extension detections")
        table.add_row("Known Malware", str(get_known_malware_count()), "Hash-based detections")
        table.add_row("Uptime", uptime_str, "System running time")
        table.add_row("Files/Second", f"{files_per_sec:.1f}", "Processing speed")
        table.add_row("Scan Mode", self.scan_mode.upper(), "Current scan mode")
        
        layout.split_column(
            Layout(header, size=3),
            Layout(table, size=10)
        )
        
        return layout
    
    def start_dashboard(self):
        """Start the real-time dashboard."""
        if self.gui_mode:
            return
        
        self.dashboard_active = True
        
        def dashboard_loop():
            with Live(self.create_dashboard(), refresh_per_second=2) as live:
                while self.dashboard_active:
                    live.update(self.create_dashboard())
                    time.sleep(0.5)
        
        self.dashboard_thread = threading.Thread(target=dashboard_loop, daemon=True)
        self.dashboard_thread.start()
    
    def stop_dashboard(self):
        """Stop the real-time dashboard."""
        self.dashboard_active = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=1)
    
    def start_monitoring(self):
        """Start file system monitoring."""
        if self.gui_mode:
            return
        
        # Start dashboard if not in GUI mode
        self.start_dashboard()
        
        # Schedule initial scan
        self.scan_directory(show_progress=True)
        
        # Start file system monitoring
        self.observer.schedule(self.event_handler, str(self.monitor_path), recursive=True)
        self.observer.start()
        
        console.print("👁️ File monitoring started")
        console.print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        self.stop_dashboard()
        self.observer.stop()
        self.observer.join()
        console.print("\n🛑 Monitoring stopped")
    
    def get_system_tray_status(self) -> Dict:
        """Get system tray status (placeholder for future implementation)."""
        return {
            'active': True,
            'threats_detected': self.stats['threats_found'],
            'files_scanned': self.stats['files_scanned'],
            'uptime': str(datetime.now() - self.stats['start_time'])
        }

# ============================================================================
# FILE SYSTEM EVENT HANDLER
# ============================================================================

class UltimateAIAntivirusEventHandler(FileSystemEventHandler):
    def __init__(self, antivirus):
        self.antivirus = antivirus
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self.antivirus._should_scan_file(file_path):
            try:
                analysis_result = self.antivirus.analyze_file(file_path)
                if analysis_result and analysis_result['is_suspicious']:
                    self.antivirus.handle_suspicious_file(analysis_result)
            except Exception as e:
                self.antivirus.logger.error(f"❌ Error analyzing new file {file_path}: {e}")
    
    def on_moved(self, event):
        """Handle file move events."""
        if event.is_directory:
            return
        
        file_path = Path(event.dest_path)
        if self.antivirus._should_scan_file(file_path):
            try:
                analysis_result = self.antivirus.analyze_file(file_path)
                if analysis_result and analysis_result['is_suspicious']:
                    self.antivirus.handle_suspicious_file(analysis_result)
            except Exception as e:
                self.antivirus.logger.error(f"❌ Error analyzing moved file {file_path}: {e}")

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    console.print("\n🛑 Shutdown signal received. Stopping antivirus...")
    sys.exit(0)

# ============================================================================
# GUI PLACEHOLDER
# ============================================================================

def create_gui_placeholder():
    """Launch the GUI interface."""
    try:
        import subprocess
        import sys
        console.print("[cyan]🖥️ Launching GUI...[/cyan]")
        subprocess.run([sys.executable, "gui.py"])
    except Exception as e:
        console.print(f"[red]❌ Error launching GUI: {e}[/red]")
        console.print("[yellow]You can run the GUI manually with: python3 gui.py[/yellow]")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function with enhanced CLI options."""
    parser = argparse.ArgumentParser(
        description="🚀 Ultimate AI Antivirus v4.X - Enhanced Security Agent",
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
        console.print("[red]❌ Error: Only one scan mode allowed (--smart-scan OR --full-scan)[/red]")
        return
    
    # Validate path conflicts with scan modes
    if args.path != "." and (args.smart_scan or args.full_scan):
        console.print("[red]❌ Error: Cannot use --path with --smart-scan or --full-scan[/red]")
        console.print("[red]Use --path for specific directory OR use scan modes for system-wide scanning[/red]")
        return
    
    # Determine scan mode
    if args.smart_scan:
        scan_mode = "smart"
        console.print("[yellow]🧠 Smart Scan mode active[/yellow]")
    elif args.full_scan:
        scan_mode = "full"
        console.print("[red]⚠️  WARNING: Full system scan requested![/red]")
        console.print("[red]This will scan your entire system. Continue? (y/N):[/red]")
        response = input().lower()
        if response != 'y':
            console.print("[yellow]Scan cancelled.[/yellow]")
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
            console.print(Panel(
                f"[bold cyan]🧠 Model Information[/bold cyan]\n"
                f"Model Type: Random Forest Classifier\n"
                f"Estimators: {model.n_estimators}\n"
                f"Features: {model.n_features_in_}\n"
                f"Classes: {list(model.classes_)}",
                border_style="blue"
            ))
        else:
            console.print("[red]❌ No trained model found[/red]")
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
        console.print("[yellow]🔄 Retraining AI model...[/yellow]")
        antivirus.model = antivirus._train_model()
        console.print("[green]✅ Model retraining complete![/green]")
        return
    
    # Handle demo mode
    if args.demo:
        console.print("[cyan]🎮 Demo mode: Creating sample files...[/cyan]")
        # Create some sample files for demonstration
        demo_dir = Path("demo_files")
        demo_dir.mkdir(exist_ok=True)
        
        # Create safe file
        (demo_dir / "safe_document.txt").write_text("This is a safe file for demo purposes.")
        
        # Create suspicious file
        (demo_dir / "suspicious_script.bat").write_text("@echo off\necho 'This is a demo suspicious file'")
        
        antivirus.monitor_path = demo_dir
        console.print(f"[green]✅ Demo files created in: {demo_dir}[/green]")
    
    # Handle scan-only mode
    if args.scan_only:
        console.print("[cyan]🔍 Performing one-time scan...[/cyan]")
        antivirus.scan_directory(show_progress=True)
        console.print("✅ Scan complete. Exiting...")
        return
    
    # Handle upload logs (placeholder)
    if args.upload_logs:
        console.print("[yellow]☁️ Upload logs feature (placeholder)[/yellow]")
        console.print("[yellow]Future versions will include cloud log upload functionality.[/yellow]")
    
    # Start monitoring
    antivirus.start_monitoring()

if __name__ == "__main__":
    main()