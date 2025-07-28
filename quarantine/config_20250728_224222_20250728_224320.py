#!/usr/bin/env python3
"""
Configuration settings for AI Antivirus v4.X
Centralized configuration for extensions, thresholds, and paths.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ====================================
# 🛡️ SUSPICIOUS EXTENSIONS
# ====================================

SUSPICIOUS_EXTENSIONS = {
    '.exe', '.com', '.bat', '.cmd', '.vbs', '.scr', '.ps1', '.pif',
    '.reg', '.js', '.jar', '.msi', '.dll', '.sys', '.dmg', '.app',
    '.deb', '.rpm', '.pkg', '.sh', '.bash', '.zsh', '.fish'
}

# ====================================
# 🎯 THREAT LEVELS & EMOJIS
# ====================================

THREAT_LEVELS = {
    'CRITICAL': {
        'emoji': '🔥',
        'color': '\033[31m',  # Red
        'min_score': 0.80,
        'description': 'High confidence malicious file'
    },
    'HIGH_RISK': {
        'emoji': '⚠️',
        'color': '\033[35m',  # Magenta
        'min_score': 0.70,
        'description': 'Likely malicious file'
    },
    'SUSPICIOUS': {
        'emoji': '🟡',
        'color': '\033[33m',  # Yellow
        'min_score': 0.50,
        'description': 'Potentially suspicious file'
    },
    'SAFE': {
        'emoji': '✅',
        'color': '\033[32m',  # Green
        'min_score': 0.00,
        'description': 'Safe file'
    }
}

# ====================================
# 📁 PATHS & DIRECTORIES
# ====================================

# Base directories
BASE_DIR = Path.cwd()
LOGS_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "model"
QUARANTINE_DIR = BASE_DIR / "quarantine"
TEST_FILES_DIR = BASE_DIR / "test_files"

# Ensure directories exist
for directory in [LOGS_DIR, MODEL_DIR, QUARANTINE_DIR, TEST_FILES_DIR]:
    directory.mkdir(exist_ok=True)

# Model files
MODEL_PATH = MODEL_DIR / "model.pkl"
TRAINING_DATA_PATH = MODEL_DIR / "training_data.csv"

# Log files
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ====================================
# 🧠 ML MODEL CONFIGURATION
# ====================================

# Model parameters
MODEL_CONFIG = {
    'n_estimators': 150,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Feature configuration
FEATURE_CONFIG = {
    'use_entropy': True,
    'use_filename_patterns': True,
    'use_file_type': True,
    'use_creation_time': True,
    'use_complexity': True,
    'max_file_size_for_analysis': 1024 * 1024  # 1MB
}

# Training configuration
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'stratify': True,
    'shuffle': True
}

# ====================================
# 🔍 SCAN CONFIGURATION
# ====================================

# Scan thresholds
SCAN_THRESHOLDS = {
    'min_file_size': 1,  # bytes
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'max_files_per_scan': 10000,
    'scan_timeout_seconds': 300,  # 5 minutes
    'progress_update_interval': 50  # files
}

# File type categories for ML features
FILE_TYPE_CATEGORIES = {
    'executable': ['.exe', '.com', '.bat', '.cmd', '.msi', '.dll', '.sys'],
    'script': ['.ps1', '.vbs', '.js', '.py', '.sh', '.bash'],
    'document': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
    'media': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp3', '.mp4', '.avi', '.mov'],
    'archive': ['.zip', '.rar', '.7z', '.tar', '.gz'],
    'config': ['.json', '.xml', '.yaml', '.yml', '.ini', '.cfg'],
    'web': ['.html', '.htm', '.css', '.js'],
    'other': []
}

# ====================================
# 🚨 DETECTION CONFIGURATION
# ====================================

# Detection methods
DETECTION_METHODS = {
    'EXTENSION': 'Extension-based detection',
    'AI': 'AI model prediction',
    'BOTH': 'Both extension and AI detection',
    'SAFE': 'File marked as safe'
}

# Suspicious filename patterns
SUSPICIOUS_PATTERNS = {
    'cheat_keywords': ['cheat', 'hack', 'crack', 'keygen', 'serial'],
    'malware_keywords': ['virus', 'trojan', 'malware', 'spyware', 'worm'],
    'suspicious_keywords': ['free', 'download', 'cracked', 'nulled', 'warez'],
    'random_chars_threshold': 0.7,  # Files with <70% alphanumeric chars
    'multiple_extensions': True,  # Flag files with multiple dots
    'hidden_files': True  # Flag hidden files
}

# ====================================
# 📊 LOGGING CONFIGURATION
# ====================================

# Log levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Log file patterns
LOG_FILE_PATTERNS = {
    'antivirus_log': "ultimate_antivirus_{timestamp}.log",
    'model_metrics': "model_metrics_{timestamp}.txt",
    'test_results': "test_results_{timestamp}.txt",
    'performance_summary': "performance_summary_{timestamp}.txt"
}

# ====================================
# 🖥️ GUI CONFIGURATION
# ====================================

# GUI settings
GUI_CONFIG = {
    'window_title': "AI Antivirus v4.X",
    'window_size': "800x600",
    'theme': "default",
    'refresh_rate_ms': 1000,
    'max_log_lines': 100
}

# GUI colors
GUI_COLORS = {
    'background': '#2b2b2b',
    'foreground': '#ffffff',
    'button': '#4a4a4a',
    'button_active': '#6a6a6a',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'info': '#2196f3'
}

# ====================================
# ☁️ CLOUD INTEGRATION CONFIG
# ====================================

# Cloud upload settings (placeholder for future implementation)
CLOUD_CONFIG = {
    'enabled': False,
    'upload_endpoint': "https://api.antivirus-cloud.com/upload",
    'api_key': os.getenv('ANTIVIRUS_API_KEY', ''),
    'max_upload_size': 10 * 1024 * 1024,  # 10MB
    'upload_timeout': 30,  # seconds
    'retry_attempts': 3
}

# ====================================
# 🔧 SYSTEM CONFIGURATION
# ====================================

# Platform-specific settings
SYSTEM_CONFIG = {
    'max_workers': os.cpu_count() or 1,
    'memory_limit_mb': 512,
    'temp_dir': os.getenv('TEMP', '/tmp'),
    'user_home': os.path.expanduser('~')
}

# Performance settings
PERFORMANCE_CONFIG = {
    'batch_size': 100,
    'max_concurrent_scans': 4,
    'cache_size': 1000,
    'enable_progress_bars': True,
    'enable_real_time_logging': True
}

# ====================================
# 🧪 TESTING CONFIGURATION
# ====================================

# Test suite settings
TEST_CONFIG = {
    'eicar_string': r'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*',
    'fake_malware_count': 10,
    'safe_file_count': 10,
    'edge_case_count': 7,
    'test_timeout_seconds': 60
}

# Test file templates
TEST_FILE_TEMPLATES = {
    'fake_malware': {
        'free_cheats.exe': '@echo off\necho "HACKED"\npause',
        'ransomware.bat': '@echo off\necho "Your files are encrypted!"\npause',
        'trojan.ps1': 'Write-Host "Suspicious PowerShell script"',
        'spyware.js': 'console.log("Suspicious JavaScript");',
        'virus.scr': 'MsgBox "Suspicious screen saver"',
        'worm.reg': 'Windows Registry Editor Version 5.00\n[HKEY_LOCAL_MACHINE\\Test]',
        'keylogger.vbs': 'MsgBox "Suspicious VBScript"',
        'backdoor.pif': 'echo "Suspicious PIF file"',
        'rootkit.dll': '// Suspicious DLL content',
        'malware.com': 'echo "Suspicious COM file"'
    },
    'safe_files': {
        'document.pdf': '%PDF-1.4\n%Test PDF content',
        'image.jpg': 'JFIF\nTest image content',
        'text.txt': 'This is a safe text file.',
        'data.csv': 'name,age\nJohn,25\nJane,30',
        'config.json': '{"name": "test", "value": 123}',
        'script.py': 'print("Safe Python script")',
        'webpage.html': '<html><body>Safe webpage</body></html>',
        'stylesheet.css': 'body { color: black; }',
        'archive.zip': 'PK\nTest archive content',
        'readme.md': '# Safe README file\nThis is safe content.'
    }
}

# ====================================
# 🔍 VALIDATION FUNCTIONS
# ====================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check required directories
    for dir_name, dir_path in [
        ('LOGS_DIR', LOGS_DIR),
        ('MODEL_DIR', MODEL_DIR),
        ('QUARANTINE_DIR', QUARANTINE_DIR),
        ('TEST_FILES_DIR', TEST_FILES_DIR)
    ]:
        if not dir_path.exists():
            errors.append(f"Directory {dir_name} does not exist: {dir_path}")
    
    # Check model parameters
    if MODEL_CONFIG['n_estimators'] <= 0:
        errors.append("n_estimators must be positive")
    
    if MODEL_CONFIG['max_depth'] <= 0:
        errors.append("max_depth must be positive")
    
    # Check scan thresholds
    if SCAN_THRESHOLDS['min_file_size'] < 0:
        errors.append("min_file_size cannot be negative")
    
    if SCAN_THRESHOLDS['max_file_size'] <= SCAN_THRESHOLDS['min_file_size']:
        errors.append("max_file_size must be greater than min_file_size")
    
    return errors


def get_config_summary() -> Dict:
    """Get a summary of current configuration."""
    return {
        'suspicious_extensions_count': len(SUSPICIOUS_EXTENSIONS),
        'threat_levels': list(THREAT_LEVELS.keys()),
        'model_parameters': MODEL_CONFIG,
        'scan_thresholds': SCAN_THRESHOLDS,
        'directories': {
            'logs': str(LOGS_DIR),
            'model': str(MODEL_DIR),
            'quarantine': str(QUARANTINE_DIR),
            'test_files': str(TEST_FILES_DIR)
        },
        'feature_config': FEATURE_CONFIG,
        'detection_methods': list(DETECTION_METHODS.keys())
    }


if __name__ == "__main__":
    # Test configuration
    print("🔧 AI Antivirus v4.X Configuration Test")
    print("=" * 50)
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid")
    
    # Print configuration summary
    summary = get_config_summary()
    print(f"\n📊 Configuration Summary:")
    print(f"   Suspicious extensions: {summary['suspicious_extensions_count']}")
    print(f"   Threat levels: {', '.join(summary['threat_levels'])}")
    print(f"   Detection methods: {', '.join(summary['detection_methods'])}")
    print(f"   Model estimators: {summary['model_parameters']['n_estimators']}")
    
    print("\n📁 Directories:")
    for name, path in summary['directories'].items():
        print(f"   {name}: {path}")
    
    print("\n✅ Configuration test complete!")