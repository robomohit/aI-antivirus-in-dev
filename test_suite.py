#!/usr/bin/env python3
"""
🧪 ULTIMATE AI ANTIVIRUS TEST SUITE v3.0
Comprehensive testing and validation system for the AI antivirus with EICAR and fake malware simulation.
"""

import os
import sys
import time
import shutil
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import colorama
from colorama import Fore, Back, Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
import threading

# Import the antivirus
from ai_antivirus import UltimateAIAntivirus, get_threat_level, create_timestamp

# Initialize colorama and rich
colorama.init(autoreset=True)
console = Console()

# ============================================================================
# CONSTANTS
# ============================================================================

# EICAR test string (official antivirus test file)
EICAR_STRING = r'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'

# Test directories
TEST_DIR = Path("test_files")
QUARANTINE_DIR = Path("quarantine")
LOGS_DIR = Path("logs")

# Fake malware templates
FAKE_MALWARE_TEMPLATES = {
    'free_cheats.exe': [
        '@echo off',
        'echo "FREE GAME CHEATS - DOWNLOADING..."',
        'echo "HACKED BY VIRUS"',
        'pause'
    ],
    'ransomware.bat': [
        '@echo off',
        'echo "YOUR FILES ARE ENCRYPTED"',
        'echo "PAY BITCOIN TO DECRYPT"',
        'echo "HACKED BY RANSOMWARE"',
        'pause'
    ],
    'keylogger.vbs': [
        'Set objShell = CreateObject("WScript.Shell")',
        'MsgBox "VIRUS KEYLOGGER ACTIVATED"',
        'WScript.Echo "HACKED BY KEYLOGGER"'
    ],
    'trojan.ps1': [
        'Write-Host "TROJAN HORSE ACTIVATED"',
        'Write-Host "HACKED BY TROJAN"',
        'Start-Sleep -Seconds 2'
    ],
    'spyware.js': [
        'console.log("SPYWARE ACTIVATED");',
        'alert("HACKED BY SPYWARE");',
        'document.write("VIRUS DETECTED");'
    ],
    'malware.com': [
        '@echo off',
        'echo "MALWARE PAYLOAD EXECUTED"',
        'echo "SYSTEM COMPROMISED"',
        'pause'
    ],
    'virus.scr': [
        '@echo off',
        'echo "VIRUS SCREENSAVER ACTIVATED"',
        'echo "HACKED BY VIRUS"',
        'pause'
    ],
    'backdoor.pif': [
        '@echo off',
        'echo "BACKDOOR ACTIVATED"',
        'echo "SYSTEM ACCESS GRANTED"',
        'pause'
    ],
    'worm.reg': [
        'Windows Registry Editor Version 5.00',
        '[HKEY_LOCAL_MACHINE\\SOFTWARE\\Worm]',
        '"Payload"="HACKED BY WORM"'
    ],
    'rootkit.dll': [
        '// Fake DLL content',
        '// HACKED BY ROOTKIT',
        '// VIRUS PAYLOAD'
    ]
}

# Safe file templates
SAFE_FILE_TEMPLATES = {
    'document.txt': 'This is a safe text document for testing.',
    'image.jpg': 'Fake JPEG image data for testing.',
    'video.mp4': 'Fake MP4 video data for testing.',
    'archive.zip': 'Fake ZIP archive data for testing.',
    'pdf_document.pdf': 'Fake PDF document data for testing.',
    'spreadsheet.xls': 'Fake Excel spreadsheet data for testing.',
    'presentation.ppt': 'Fake PowerPoint presentation data for testing.',
    'code.py': 'print("This is safe Python code")',
    'config.json': '{"safe": true, "test": "configuration"}',
    'readme.md': '# Safe README file\nThis is a safe markdown file.'
}

# ============================================================================
# TEST SUITE CLASS
# ============================================================================

class UltimateAntivirusTestSuite:
    def __init__(self):
        """Initialize the comprehensive test suite."""
        self.test_results = {
            'total_files': 0,
            'safe_files': 0,
            'threats_found': 0,
            'quarantined': 0,
            'detection_methods': {
                'SAFE': 0,
                'EXTENSION': 0,
                'AI': 0,
                'BOTH': 0
            },
            'threat_levels': {
                'CRITICAL': 0,
                'HIGH_RISK': 0,
                'SUSPICIOUS': 0,
                'SAFE': 0
            },
            'performance': {
                'start_time': None,
                'end_time': None,
                'total_scan_time': 0,
                'files_per_second': 0
            },
            'errors': []
        }
        
        # Create test directories
        self._create_test_directories()
        
        # Initialize antivirus
        self.antivirus = UltimateAIAntivirus(
            monitor_path=str(TEST_DIR),
            quarantine_enabled=True
        )
        
        console.print(Panel.fit(
            "[bold cyan]🧪 ULTIMATE AI ANTIVIRUS TEST SUITE v3.0[/bold cyan]\n"
            "[green]Comprehensive Testing and Validation System[/green]",
            border_style="cyan"
        ))
    
    def _create_test_directories(self):
        """Create necessary test directories."""
        TEST_DIR.mkdir(exist_ok=True)
        QUARANTINE_DIR.mkdir(exist_ok=True)
        LOGS_DIR.mkdir(exist_ok=True)
        
        console.print(f"[cyan]📁 Test directory created:[/cyan] {TEST_DIR}")
        console.print(f"[cyan]📁 Quarantine directory:[/cyan] {QUARANTINE_DIR}")
        console.print(f"[cyan]📁 Logs directory:[/cyan] {LOGS_DIR}")
    
    def generate_eicar_test_file(self, path: Path) -> bool:
        """Generate EICAR test file for antivirus validation."""
        try:
            with open(path, 'w') as f:
                f.write(EICAR_STRING)
            
            console.print(f"[green]✅ EICAR test file created:[/green] {path}")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to create EICAR file:[/red] {e}")
            return False
    
    def generate_fake_malware_files(self) -> int:
        """Generate fake malware files for testing."""
        console.print(f"\n[cyan]🦠 Generating fake malware files...[/cyan]")
        
        created_count = 0
        
        for filename, content in FAKE_MALWARE_TEMPLATES.items():
            file_path = TEST_DIR / filename
            
            try:
                with open(file_path, 'w') as f:
                    if isinstance(content, list):
                        f.write('\n'.join(content))
                    else:
                        f.write(content)
                
                created_count += 1
                console.print(f"[yellow]🦠 Created fake malware:[/yellow] {filename}")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to create {filename}:[/red] {e}")
                self.test_results['errors'].append(f"Failed to create {filename}: {e}")
        
        console.print(f"[green]✅ Created {created_count} fake malware files[/green]")
        return created_count
    
    def generate_safe_files(self) -> int:
        """Generate safe files for testing."""
        console.print(f"\n[cyan]📄 Generating safe files...[/cyan]")
        
        created_count = 0
        
        for filename, content in SAFE_FILE_TEMPLATES.items():
            file_path = TEST_DIR / filename
            
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                
                created_count += 1
                console.print(f"[green]📄 Created safe file:[/green] {filename}")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to create {filename}:[/red] {e}")
                self.test_results['errors'].append(f"Failed to create {filename}: {e}")
        
        console.print(f"[green]✅ Created {created_count} safe files[/green]")
        return created_count
    
    def generate_edge_case_files(self) -> int:
        """Generate edge case files for fault resilience testing."""
        console.print(f"\n[cyan]🔍 Generating edge case files...[/cyan]")
        
        edge_cases = [
            # File with no extension
            ('no_extension', 'This file has no extension'),
            
            # Extremely long filename
            ('a' * 200 + '.txt', 'File with extremely long name'),
            
            # Empty file
            ('empty_file.txt', ''),
            
            # File with special characters
            ('special_chars_!@#$%^&*().txt', 'File with special characters'),
            
            # Hidden file
            ('.hidden_file.txt', 'Hidden file content'),
            
            # File with spaces
            ('file with spaces.txt', 'File with spaces in name'),
            
            # Large file simulation
            ('large_file.txt', 'A' * 10000),  # 10KB file
        ]
        
        created_count = 0
        
        for filename, content in edge_cases:
            file_path = TEST_DIR / filename
            
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                
                created_count += 1
                console.print(f"[blue]🔍 Created edge case:[/blue] {filename}")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to create {filename}:[/red] {e}")
                self.test_results['errors'].append(f"Failed to create {filename}: {e}")
        
        console.print(f"[green]✅ Created {created_count} edge case files[/green]")
        return created_count
    
    def scan_test_files(self) -> Dict:
        """Scan all test files and collect results."""
        console.print(f"\n[cyan]🔍 Scanning test files...[/cyan]")
        
        # Get all files in test directory
        test_files = list(TEST_DIR.rglob("*"))
        test_files = [f for f in test_files if f.is_file()]
        
        self.test_results['total_files'] = len(test_files)
        self.test_results['performance']['start_time'] = datetime.now()
        
        scan_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning files...", total=len(test_files))
            
            for file_path in test_files:
                try:
                    # Analyze file
                    analysis_result = self.antivirus.analyze_file(file_path)
                    
                    if analysis_result:
                        scan_results.append(analysis_result)
                        
                        # Update statistics
                        if analysis_result['is_suspicious']:
                            self.test_results['threats_found'] += 1
                            
                            # Check if quarantined
                            if (QUARANTINE_DIR / file_path.name).exists():
                                self.test_results['quarantined'] += 1
                        else:
                            self.test_results['safe_files'] += 1
                        
                        # Update detection method statistics
                        detection_method = analysis_result['detection_method']
                        self.test_results['detection_methods'][detection_method] += 1
                        
                        # Update threat level statistics
                        threat_level = analysis_result['threat_level']['level']
                        self.test_results['threat_levels'][threat_level] += 1
                    
                except Exception as e:
                    error_msg = f"Error scanning {file_path}: {e}"
                    console.print(f"[red]❌ {error_msg}[/red]")
                    self.test_results['errors'].append(error_msg)
                
                progress.advance(task)
        
        self.test_results['performance']['end_time'] = datetime.now()
        self.test_results['performance']['total_scan_time'] = (
            self.test_results['performance']['end_time'] - 
            self.test_results['performance']['start_time']
        ).total_seconds()
        
        if self.test_results['performance']['total_scan_time'] > 0:
            self.test_results['performance']['files_per_second'] = (
                self.test_results['total_files'] / 
                self.test_results['performance']['total_scan_time']
            )
        
        return scan_results
    
    def calculate_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics for the test results."""
        total_files = self.test_results['total_files']
        threats_found = self.test_results['threats_found']
        safe_files = self.test_results['safe_files']
        
        # Expected threats (fake malware + EICAR)
        expected_threats = len(FAKE_MALWARE_TEMPLATES) + 1  # +1 for EICAR
        
        # Expected safe files
        expected_safe = len(SAFE_FILE_TEMPLATES) + len(FAKE_MALWARE_TEMPLATES) + 7  # +7 for edge cases
        
        # Calculate metrics
        true_positives = threats_found
        false_positives = threats_found - expected_threats
        true_negatives = safe_files
        false_negatives = expected_threats - threats_found
        
        # Ensure non-negative values
        true_positives = max(0, true_positives)
        false_positives = max(0, false_positives)
        true_negatives = max(0, true_negatives)
        false_negatives = max(0, false_negatives)
        
        # Calculate accuracy metrics
        total_predictions = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / max(total_predictions, 1)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'expected_threats': expected_threats,
            'expected_safe': expected_safe
        }
    
    def create_results_table(self, accuracy_metrics: Dict) -> Table:
        """Create a comprehensive results table."""
        table = Table(title="📊 ULTIMATE AI ANTIVIRUS TEST RESULTS")
        
        # Add columns
        table.add_column("Category", style="cyan", width=25)
        table.add_column("Metric", style="green", width=20)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Details", style="blue", width=30)
        
        # File Statistics
        table.add_row("📁 Files", "Total Scanned", str(self.test_results['total_files']), "All test files")
        table.add_row("", "Safe Files", str(self.test_results['safe_files']), "No threats detected")
        table.add_row("", "Threats Found", str(self.test_results['threats_found']), "Suspicious files detected")
        table.add_row("", "Quarantined", str(self.test_results['quarantined']), "Files moved to quarantine")
        
        # Detection Methods
        table.add_row("🔍 Detection", "Extension Only", str(self.test_results['detection_methods']['EXTENSION']), "Rule-based detection")
        table.add_row("", "AI Only", str(self.test_results['detection_methods']['AI']), "Machine learning detection")
        table.add_row("", "Both Methods", str(self.test_results['detection_methods']['BOTH']), "AI + Extension detection")
        table.add_row("", "Safe Files", str(self.test_results['detection_methods']['SAFE']), "No detection triggered")
        
        # Threat Levels
        table.add_row("⚠️ Threat Levels", "Critical", str(self.test_results['threat_levels']['CRITICAL']), "🔥 High confidence threats")
        table.add_row("", "High Risk", str(self.test_results['threat_levels']['HIGH_RISK']), "⚠️ Medium confidence threats")
        table.add_row("", "Suspicious", str(self.test_results['threat_levels']['SUSPICIOUS']), "🟡 Low confidence threats")
        table.add_row("", "Safe", str(self.test_results['threat_levels']['SAFE']), "✅ No threats detected")
        
        # Performance
        scan_time = self.test_results['performance']['total_scan_time']
        files_per_sec = self.test_results['performance']['files_per_second']
        table.add_row("⚡ Performance", "Scan Time", f"{scan_time:.2f}s", "Total scan duration")
        table.add_row("", "Files/Second", f"{files_per_sec:.1f}", "Processing speed")
        table.add_row("", "Avg Time/File", f"{scan_time/max(self.test_results['total_files'], 1):.3f}s", "Average per file")
        
        # Accuracy Metrics
        table.add_row("🎯 Accuracy", "Overall Accuracy", f"{accuracy_metrics['accuracy']:.2%}", "Correct predictions")
        table.add_row("", "Precision", f"{accuracy_metrics['precision']:.2%}", "True positives / All positives")
        table.add_row("", "Recall", f"{accuracy_metrics['recall']:.2%}", "True positives / All threats")
        table.add_row("", "F1 Score", f"{accuracy_metrics['f1_score']:.2%}", "Harmonic mean of precision/recall")
        
        return table
    
    def save_test_results(self, accuracy_metrics: Dict):
        """Save test results to log file."""
        timestamp = create_timestamp()
        log_file = LOGS_DIR / f"test_results_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write("🧪 ULTIMATE AI ANTIVIRUS TEST SUITE RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Scanned: {self.test_results['total_files']}\n")
            f.write(f"Threats Found: {self.test_results['threats_found']}\n")
            f.write(f"Safe Files: {self.test_results['safe_files']}\n")
            f.write(f"Quarantined: {self.test_results['quarantined']}\n\n")
            
            f.write("Detection Methods:\n")
            for method, count in self.test_results['detection_methods'].items():
                f.write(f"  {method}: {count}\n")
            f.write("\n")
            
            f.write("Threat Levels:\n")
            for level, count in self.test_results['threat_levels'].items():
                f.write(f"  {level}: {count}\n")
            f.write("\n")
            
            f.write("Performance:\n")
            f.write(f"  Scan Time: {self.test_results['performance']['total_scan_time']:.2f}s\n")
            f.write(f"  Files/Second: {self.test_results['performance']['files_per_second']:.1f}\n")
            f.write("\n")
            
            f.write("Accuracy Metrics:\n")
            f.write(f"  Overall Accuracy: {accuracy_metrics['accuracy']:.2%}\n")
            f.write(f"  Precision: {accuracy_metrics['precision']:.2%}\n")
            f.write(f"  Recall: {accuracy_metrics['recall']:.2%}\n")
            f.write(f"  F1 Score: {accuracy_metrics['f1_score']:.2%}\n")
            f.write("\n")
            
            if self.test_results['errors']:
                f.write("Errors:\n")
                for error in self.test_results['errors']:
                    f.write(f"  {error}\n")
        
        console.print(f"[green]💾 Test results saved to:[/green] {log_file}")
    
    def run_comprehensive_test(self):
        """Run the complete test suite."""
        console.print(f"\n[bold cyan]🚀 Starting Comprehensive Test Suite[/bold cyan]")
        
        # Step 1: Generate EICAR test file
        console.print(f"\n[cyan]📋 Step 1: Creating EICAR test file[/cyan]")
        eicar_path = TEST_DIR / "eicar_test.com"
        self.generate_eicar_test_file(eicar_path)
        
        # Step 2: Generate fake malware
        console.print(f"\n[cyan]📋 Step 2: Creating fake malware files[/cyan]")
        malware_count = self.generate_fake_malware_files()
        
        # Step 3: Generate safe files
        console.print(f"\n[cyan]📋 Step 3: Creating safe files[/cyan]")
        safe_count = self.generate_safe_files()
        
        # Step 4: Generate edge cases
        console.print(f"\n[cyan]📋 Step 4: Creating edge case files[/cyan]")
        edge_count = self.generate_edge_case_files()
        
        # Step 5: Scan all files
        console.print(f"\n[cyan]📋 Step 5: Scanning all test files[/cyan]")
        scan_results = self.scan_test_files()
        
        # Step 6: Calculate accuracy metrics
        console.print(f"\n[cyan]📋 Step 6: Calculating accuracy metrics[/cyan]")
        accuracy_metrics = self.calculate_accuracy_metrics()
        
        # Step 7: Display results
        console.print(f"\n[cyan]📋 Step 7: Displaying results[/cyan]")
        results_table = self.create_results_table(accuracy_metrics)
        console.print(results_table)
        
        # Step 8: Save results
        console.print(f"\n[cyan]📋 Step 8: Saving results[/cyan]")
        self.save_test_results(accuracy_metrics)
        
        # Final summary
        console.print(f"\n[bold green]✅ Comprehensive test suite completed![/bold green]")
        console.print(f"[green]📊 Total files tested:[/green] {self.test_results['total_files']}")
        console.print(f"[green]🦠 Threats detected:[/green] {self.test_results['threats_found']}")
        console.print(f"[green]🎯 Detection accuracy:[/green] {accuracy_metrics['accuracy']:.2%}")
        console.print(f"[green]⚡ Performance:[/green] {self.test_results['performance']['files_per_second']:.1f} files/second")


def main():
    """Main function to run the test suite."""
    console.print(Panel.fit(
        "[bold cyan]🧪 ULTIMATE AI ANTIVIRUS TEST SUITE v3.0[/bold cyan]\n"
        "[green]Professional-Grade Testing and Validation System[/green]\n"
        "[yellow]Safe simulation lab - No real malware used[/yellow]",
        border_style="cyan"
    ))
    
    # Create and run test suite
    test_suite = UltimateAntivirusTestSuite()
    test_suite.run_comprehensive_test()


if __name__ == "__main__":
    main()