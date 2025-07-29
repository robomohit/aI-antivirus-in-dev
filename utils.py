#!/usr/bin/env python3
"""
Utility functions for AI Antivirus v4.X
Helper functions for file analysis, path detection, and system utilities.
"""

import os
import platform
import math
import random
import hashlib
import csv
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)


def get_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of file data."""
    if not data:
        return 0.0
    
    # Count byte frequencies
    byte_counts = {}
    for byte in data:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    data_length = len(data)
    
    for count in byte_counts.values():
        probability = count / data_length
        entropy -= probability * math.log2(probability)
    
    return entropy


def get_file_type(filename: str) -> str:
    """Categorize file type based on extension."""
    ext = Path(filename).suffix.lower()
    
    # Executable types
    if ext in ['.exe', '.com', '.bat', '.cmd', '.msi', '.dll', '.sys']:
        return 'executable'
    
    # Script types
    elif ext in ['.ps1', '.vbs', '.js', '.py', '.sh', '.bash']:
        return 'script'
    
    # Document types
    elif ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt']:
        return 'document'
    
    # Media types
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp3', '.mp4', '.avi', '.mov']:
        return 'media'
    
    # Archive types
    elif ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
        return 'archive'
    
    # Configuration types
    elif ext in ['.json', '.xml', '.yaml', '.yml', '.ini', '.cfg']:
        return 'config'
    
    # Web types
    elif ext in ['.html', '.htm', '.css', '.js']:
        return 'web'
    
    else:
        return 'other'


def get_high_risk_paths() -> List[str]:
    """Get platform-specific high-risk directories."""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows high-risk paths
        return [
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/AppData/Local/Temp"),
            os.path.expanduser("~/AppData/Roaming")
        ]
    else:
        # Linux/macOS high-risk paths
        return [
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/tmp"),
            "/tmp"
        ]


def get_full_scan_paths() -> List[str]:
    """Get platform-specific full scan paths."""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows - scan all mounted drives
        drives = []
        for drive in range(ord('A'), ord('Z') + 1):
            drive_letter = chr(drive) + ":/"
            if os.path.exists(drive_letter):
                drives.append(drive_letter)
        return drives
    else:
        # Linux/macOS - start from root but skip system directories
        return ["/"]


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def create_log_folders():
    """Create necessary log folders if they don't exist."""
    folders = ["logs", "model", "quarantine", "test_files"]
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)


def log_threat(logger: logging.Logger, file_path: str, analysis_result: Dict):
    """Log threat detection with detailed information."""
    logger.warning(f"🚨 THREAT DETECTED: {file_path}")
    logger.info(f"📊 File size: {analysis_result.get('size_formatted', 'Unknown')}")
    logger.info(f"🔍 Detection method: {analysis_result.get('detection_method', 'Unknown')}")
    logger.info(f"🧠 AI confidence: {analysis_result.get('ai_confidence', 0):.2f}%")
    logger.info(f"⚠️ Threat level: {analysis_result.get('threat_level', 'Unknown')}")
    logger.info(f"🕒 Last modified: {analysis_result.get('last_modified', 'Unknown')}")
    logger.info(f"📁 Extension: {analysis_result.get('extension', 'Unknown')}")


def get_filename_pattern_flags(filename: str) -> Dict[str, bool]:
    """Extract suspicious patterns from filename."""
    filename_lower = filename.lower()
    
    suspicious_patterns = {
        'pattern_hack': any(word in filename_lower for word in ['hack', 'hacker', 'hacking']),
        'pattern_steal': any(word in filename_lower for word in ['steal', 'stealer', 'stealing']),
        'pattern_crack': any(word in filename_lower for word in ['crack', 'cracker', 'cracking']),
        'pattern_keygen': any(word in filename_lower for word in ['keygen', 'serial', 'key']),
        'pattern_cheat': any(word in filename_lower for word in ['cheat', 'cheater', 'cheating']),
        'pattern_free': any(word in filename_lower for word in ['free', 'freeware']),
        'pattern_cracked': any(word in filename_lower for word in ['cracked', 'nulled', 'warez']),
        'pattern_premium': any(word in filename_lower for word in ['premium', 'pro', 'professional']),
        'pattern_unlock': any(word in filename_lower for word in ['unlock', 'unlocker']),
        'pattern_bypass': any(word in filename_lower for word in ['bypass', 'bypasser']),
        'pattern_admin': any(word in filename_lower for word in ['admin', 'administrator']),
        'pattern_root': any(word in filename_lower for word in ['root', 'rootkit']),
        'pattern_system': any(word in filename_lower for word in ['system', 'sys']),
        'pattern_kernel': any(word in filename_lower for word in ['kernel', 'driver']),
        'pattern_driver': any(word in filename_lower for word in ['driver', 'drv']),
        'pattern_service': any(word in filename_lower for word in ['service', 'svc']),
        'pattern_daemon': any(word in filename_lower for word in ['daemon', 'demon']),
        'pattern_bot': any(word in filename_lower for word in ['bot', 'robot']),
        'pattern_miner': any(word in filename_lower for word in ['miner', 'mining']),
        'pattern_malware': any(word in filename_lower for word in ['malware', 'malicious']),
        'pattern_virus': any(word in filename_lower for word in ['virus', 'viral']),
        'pattern_infect': any(word in filename_lower for word in ['infect', 'infection']),
        'pattern_spread': any(word in filename_lower for word in ['spread', 'worm'])
    }
    
    return suspicious_patterns


def simulate_file_creation_randomness() -> float:
    """Simulate file creation randomness (for ML features)."""
    # Simulate various factors that might indicate suspicious files
    factors = [
        random.random(),  # Random factor
        random.choice([0.1, 0.9]),  # Binary-like behavior
        random.gauss(0.5, 0.2),  # Normal distribution
    ]
    return sum(factors) / len(factors)


def get_platform_info() -> Dict[str, str]:
    """Get detailed platform information."""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }


def print_colored(text: str, color: str = Fore.WHITE, style: str = Style.RESET_ALL):
    """Print colored text with proper formatting."""
    print(f"{color}{text}{style}")


def create_timestamp() -> str:
    """Create a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_file_path(file_path: str) -> bool:
    """Validate if file path is safe to scan."""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def get_file_metadata(file_path: str) -> Dict:
    """Get comprehensive file metadata."""
    try:
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'size_bytes': stat.st_size,
            'size_formatted': format_size(stat.st_size),
            'created_time': datetime.fromtimestamp(stat.st_ctime),
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'accessed_time': datetime.fromtimestamp(stat.st_atime),
            'extension': path.suffix.lower(),
            'filename': path.name,
            'parent_dir': str(path.parent),
            'is_hidden': path.name.startswith('.'),
            'permissions': oct(stat.st_mode)[-3:]
        }
    except Exception as e:
        return {
            'error': str(e),
            'size_bytes': 0,
            'size_formatted': '0 B',
            'extension': '',
            'filename': Path(file_path).name
        }


def calculate_file_complexity(file_path: str) -> float:
    """Calculate file complexity score for ML features."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read(1024)  # Read first 1KB for analysis
        
        # Calculate various complexity factors
        entropy = get_entropy(data)
        size = len(data)
        unique_bytes = len(set(data))
        
        # Normalize factors
        entropy_score = min(entropy / 8.0, 1.0)  # Max entropy is 8 bits
        size_score = min(size / 1024.0, 1.0)  # Normalize to 1KB
        diversity_score = unique_bytes / 256.0  # Normalize to byte range
        
        # Combine factors
        complexity = (entropy_score + size_score + diversity_score) / 3.0
        return complexity
        
    except Exception:
        return 0.0


def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents."""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        print_colored(f"❌ Error computing hash for {file_path}: {e}", Fore.RED)
        return ""


def is_known_malware(hash_value: str) -> bool:
    """Check if hash is in known malware database."""
    try:
        known_malware_path = Path("known_malware.csv")
        if not known_malware_path.exists():
            return False
        
        with open(known_malware_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('sha256_hash') == hash_value:
                    return True
        return False
    except Exception as e:
        print_colored(f"❌ Error checking known malware: {e}", Fore.RED)
        return False


def add_to_known_malware(file_path: str, features: Dict, score: float, method: str):
    """Add detection to known malware database."""
    try:
        known_malware_path = Path("known_malware.csv")
        
        # Create file with headers if it doesn't exist
        if not known_malware_path.exists():
            with open(known_malware_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'sha256_hash', 'filename', 'size_kb', 'entropy', 
                    'extension', 'ai_score', 'method', 'date_detected'
                ])
        
        # Get file hash and metadata
        file_hash = get_file_hash(file_path)
        if not file_hash:
            return
        
        file_size_kb = features.get('file_size_kb', 0)
        entropy = features.get('entropy', 0.0)
        extension = features.get('extension', '')
        filename = Path(file_path).name
        
        # Append to CSV
        with open(known_malware_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                file_hash, filename, file_size_kb, entropy,
                extension, score, method, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        print_colored(f"🧠 Added to known malware database: {filename}", Fore.CYAN)
        
    except Exception as e:
        print_colored(f"❌ Error adding to known malware: {e}", Fore.RED)


def get_known_malware_count() -> int:
    """Get count of known malware entries."""
    try:
        known_malware_path = Path("known_malware.csv")
        if not known_malware_path.exists():
            return 0
        
        with open(known_malware_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return sum(1 for _ in reader)
    except Exception:
        return 0


if __name__ == "__main__":
    # Test utility functions
    print_colored("🧪 Testing utility functions...", Fore.CYAN)
    
    # Test entropy calculation
    test_data = b"Hello, World! This is a test string."
    entropy = get_entropy(test_data)
    print(f"Entropy of test data: {entropy:.2f}")
    
    # Test file type detection
    test_files = ["document.pdf", "script.ps1", "media.jpg", "executable.exe"]
    for filename in test_files:
        file_type = get_file_type(filename)
        print(f"File type for {filename}: {file_type}")
    
    # Test high-risk paths
    high_risk_paths = get_high_risk_paths()
    print(f"High-risk paths: {high_risk_paths}")
    
    # Test filename patterns
    test_filenames = ["free_cheats.exe", "document.pdf", ".hidden_file", "keygen_crack.exe"]
    for filename in test_filenames:
        patterns = get_filename_pattern_flags(filename)
        print(f"Pattern flags for {filename}: {patterns}")
    
    # Test hash functions
    print_colored("🧪 Testing hash functions...", Fore.CYAN)
    test_file = "test_hash.txt"
    with open(test_file, 'w') as f:
        f.write("Test content for hash calculation")
    
    file_hash = get_file_hash(test_file)
    print(f"File hash: {file_hash}")
    
    # Test known malware functions
    print(f"Known malware count: {get_known_malware_count()}")
    print(f"Is known malware: {is_known_malware(file_hash)}")
    
    # Clean up test file
    Path(test_file).unlink()
    
    print_colored("✅ Utility functions test complete!", Fore.GREEN)