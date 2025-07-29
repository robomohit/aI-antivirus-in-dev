#!/usr/bin/env python3
"""
🧪 ULTIMATE AI ANTIVIRUS TEST SUITE v4.X
Comprehensive testing framework for AI antivirus with EICAR and fake malware validation.
"""

import os
import time
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Tuple

# Import our modules
from config import (
    TEST_CONFIG, TEST_FILE_TEMPLATES, LOGS_DIR, TEST_FILES_DIR, 
    QUARANTINE_DIR, SUSPICIOUS_EXTENSIONS
)
from utils import (
    create_log_folders, print_colored, create_timestamp,
    get_file_metadata, format_size, get_file_hash, is_known_malware,
    add_to_known_malware, get_known_malware_count
)
from ai_antivirus import UltimateAIAntivirus, get_threat_level

# Rich imports for enhanced UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


class UltimateAntivirusTestSuite:
    """Enhanced test suite for AI antivirus validation."""
    
    def __init__(self, lite_mode=False):
        """Initialize the test suite."""
        self.lite_mode = lite_mode
        self.test_results = {
            'total_files': 0,
            'threats_found': 0,
            'safe_files': 0,
            'detection_methods': {
                'SAFE': 0,
                'EXTENSION': 0,
                'AI': 0,
                'BOTH': 0,
                'KNOWN_MALWARE': 0
            },
            'threat_levels': {
                'CRITICAL': 0,
                'HIGH_RISK': 0,
                'SUSPICIOUS': 0,
                'SAFE': 0
            },
            'false_positives': [],
            'false_negatives': [],
            'scan_duration': 0,
            'files_per_second': 0,
            'avg_scan_time': 0,
            'known_malware_detected': 0
        }
        
        # Create test directories
        self._create_test_directories()
        
        # Initialize antivirus
        self.antivirus = UltimateAIAntivirus(
            monitor_path=str(TEST_FILES_DIR),
            quarantine_enabled=True,
            gui_mode=False
        )
        
        # Print startup panel
        title = "🧪 ULTIMATE AI ANTIVIRUS TEST SUITE v4.X"
        if lite_mode:
            title += " (LITE MODE)"
        
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]\n"
            f"[green]Enhanced testing framework with real-time logging[/green]\n"
            f"[yellow]Test directory: {TEST_FILES_DIR}[/yellow]\n"
            f"[yellow]Logs directory: {LOGS_DIR}[/yellow]",
            border_style="blue"
        ))
    
    def _create_test_directories(self):
        """Create necessary test directories."""
        create_log_folders()
        TEST_FILES_DIR.mkdir(exist_ok=True)
        QUARANTINE_DIR.mkdir(exist_ok=True)
    
    def generate_eicar_test_file(self, path: Path):
        """Generate EICAR test file."""
        eicar_content = TEST_CONFIG['eicar_string']
        path.write_text(eicar_content)
        print_colored(f"🧪 Generated EICAR test file: {path}", "cyan")
    
    def generate_fake_malware_files(self):
        """Generate fake malware files for testing."""
        if self.lite_mode:
            # Generate only 1 fake malware file in lite mode
            malware_files = list(TEST_FILE_TEMPLATES['fake_malware'].items())[:1]
        else:
            # Generate all fake malware files
            malware_files = TEST_FILE_TEMPLATES['fake_malware'].items()
        
        print_colored(f"🦠 Generating {len(malware_files)} fake malware files...", "red")
        
        for filename, content in malware_files:
            file_path = TEST_FILES_DIR / filename
            file_path.write_text(content)
            print_colored(f"   Created: {filename}", "yellow")
    
    def generate_safe_files(self):
        """Generate safe files for testing."""
        if self.lite_mode:
            # Generate only 1 safe file in lite mode
            safe_files = list(TEST_FILE_TEMPLATES['safe_files'].items())[:1]
        else:
            # Generate all safe files
            safe_files = TEST_FILE_TEMPLATES['safe_files'].items()
        
        print_colored(f"✅ Generating {len(safe_files)} safe files...", "green")
        
        for filename, content in safe_files:
            file_path = TEST_FILES_DIR / filename
            file_path.write_text(content)
            print_colored(f"   Created: {filename}", "green")
    
    def generate_edge_case_files(self):
        """Generate edge case files for resilience testing."""
        if self.lite_mode:
            # Skip edge cases in lite mode
            return
        
        print_colored("🔍 Generating edge case files...", "yellow")
        
        edge_cases = [
            ("no_extension_file", "Content without extension"),
            ("very_long_filename_" + "x" * 100 + ".txt", "Long filename test"),
            ("empty_file.txt", ""),
            (".hidden_file.txt", "Hidden file test"),
            ("file_with_special_chars_!@#$%^&*().txt", "Special chars test"),
            ("file_with_spaces in name.txt", "Spaces in name test"),
            ("file.with.multiple.dots.txt", "Multiple dots test")
        ]
        
        for filename, content in edge_cases:
            file_path = TEST_FILES_DIR / filename
            file_path.write_text(content)
            print_colored(f"   Created: {filename}", "yellow")
    
    def scan_test_files(self):
        """Scan all test files with real-time progress."""
        test_files = list(TEST_FILES_DIR.glob("*"))
        self.test_results['total_files'] = len(test_files)
        
        print_colored(f"🔍 Scanning {len(test_files)} test files...", "cyan")
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Scanning files...", total=len(test_files))
            
            for i, file_path in enumerate(test_files):
                progress.update(task, description=f"Scanning file {i+1} of {len(test_files)}: {file_path.name}")
                
                try:
                    # Analyze file
                    analysis_result = self.antivirus.analyze_file(file_path)
                    
                    # Update statistics
                    if analysis_result['is_suspicious']:
                        self.test_results['threats_found'] += 1
                        
                        # Track detection method
                        detection_method = analysis_result['detection_method']
                        self.test_results['detection_methods'][detection_method] += 1
                        
                        # Track hash-based detections
                        if detection_method == 'KNOWN_MALWARE':
                            self.test_results['known_malware_detected'] += 1
                        
                        # Track threat level
                        threat_level = analysis_result['threat_level']
                        self.test_results['threat_levels'][threat_level] += 1
                        
                        # Check for false positives (safe files marked as suspicious)
                        if file_path.suffix.lower() not in SUSPICIOUS_EXTENSIONS:
                            self.test_results['false_positives'].append({
                                'file': str(file_path),
                                'reason': f"Marked as {detection_method}",
                                'confidence': analysis_result['ai_confidence']
                            })
                    else:
                        self.test_results['safe_files'] += 1
                        self.test_results['threat_levels']['SAFE'] += 1
                        
                        # Check for false negatives (suspicious files marked as safe)
                        if file_path.suffix.lower() in SUSPICIOUS_EXTENSIONS:
                            self.test_results['false_negatives'].append({
                                'file': str(file_path),
                                'reason': "Not detected as suspicious",
                                'confidence': analysis_result['ai_confidence']
                            })
                
                except Exception as e:
                    print_colored(f"⚠️ Error scanning {file_path.name}: {e}", "yellow")
                
                progress.advance(task)
        
        end_time = time.time()
        self.test_results['scan_duration'] = end_time - start_time
        self.test_results['files_per_second'] = len(test_files) / self.test_results['scan_duration']
        self.test_results['avg_scan_time'] = self.test_results['scan_duration'] / len(test_files)
    
    def calculate_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics including false positives/negatives."""
        total_files = self.test_results['total_files']
        threats_found = self.test_results['threats_found']
        safe_files = self.test_results['safe_files']
        
        # Calculate basic metrics
        true_positives = threats_found - len(self.test_results['false_positives'])
        true_negatives = safe_files - len(self.test_results['false_negatives'])
        false_positives = len(self.test_results['false_positives'])
        false_negatives = len(self.test_results['false_negatives'])
        
        # Calculate accuracy metrics
        accuracy = (true_positives + true_negatives) / total_files if total_files > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def create_enhanced_results_table(self, accuracy_metrics: Dict) -> Table:
        """Create enhanced results table with detailed metrics."""
        table = Table(title="🧪 AI Antivirus Test Results", show_header=True, header_style="bold magenta")
        
        # File statistics
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        table.add_row("Total Files", str(self.test_results['total_files']), "100%")
        table.add_row("Threats Found", str(self.test_results['threats_found']), 
                     f"{self.test_results['threats_found']/self.test_results['total_files']*100:.1f}%")
        table.add_row("Safe Files", str(self.test_results['safe_files']), 
                     f"{self.test_results['safe_files']/self.test_results['total_files']*100:.1f}%")
        
        # Detection methods
        table.add_row("", "", "")
        table.add_row("[bold]Detection Methods[/bold]", "", "")
        for method, count in self.test_results['detection_methods'].items():
            if count > 0:
                table.add_row(f"  {method}", str(count), 
                             f"{count/self.test_results['total_files']*100:.1f}%")
        
        # Hash-based detection
        if self.test_results['known_malware_detected'] > 0:
            table.add_row("  [bold cyan]Known Malware (Hash)[/bold cyan]", 
                         str(self.test_results['known_malware_detected']), 
                         f"{self.test_results['known_malware_detected']/self.test_results['total_files']*100:.1f}%")
        
        # Threat levels
        table.add_row("", "", "")
        table.add_row("[bold]Threat Levels[/bold]", "", "")
        for level, count in self.test_results['threat_levels'].items():
            if count > 0:
                table.add_row(f"  {level}", str(count), 
                             f"{count/self.test_results['total_files']*100:.1f}%")
        
        # Performance metrics
        table.add_row("", "", "")
        table.add_row("[bold]Performance[/bold]", "", "")
        table.add_row("  Scan Duration", f"{self.test_results['scan_duration']:.2f}s", "")
        table.add_row("  Files/Second", f"{self.test_results['files_per_second']:.1f}", "")
        table.add_row("  Avg Scan Time", f"{self.test_results['avg_scan_time']:.3f}s", "")
        
        # Accuracy metrics
        table.add_row("", "", "")
        table.add_row("[bold]Accuracy Metrics[/bold]", "", "")
        table.add_row("  True Positives", str(accuracy_metrics['true_positives']), "")
        table.add_row("  True Negatives", str(accuracy_metrics['true_negatives']), "")
        table.add_row("  False Positives", str(accuracy_metrics['false_positives']), "")
        table.add_row("  False Negatives", str(accuracy_metrics['false_negatives']), "")
        table.add_row("  Accuracy", f"{accuracy_metrics['accuracy']:.1%}", "")
        table.add_row("  Precision", f"{accuracy_metrics['precision']:.1%}", "")
        table.add_row("  Recall", f"{accuracy_metrics['recall']:.1%}", "")
        table.add_row("  F1-Score", f"{accuracy_metrics['f1_score']:.1%}", "")
        
        return table
    
    def save_enhanced_test_results(self, accuracy_metrics: Dict):
        """Save detailed test results to file."""
        timestamp = create_timestamp()
        results_file = LOGS_DIR / f"test_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("🧪 ULTIMATE AI ANTIVIRUS TEST SUITE RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Mode: {'LITE' if self.lite_mode else 'FULL'}\n\n")
            
            # File statistics
            f.write("📊 FILE STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Files: {self.test_results['total_files']}\n")
            f.write(f"Threats Found: {self.test_results['threats_found']}\n")
            f.write(f"Safe Files: {self.test_results['safe_files']}\n\n")
            
            # Detection methods
            f.write("🔍 DETECTION METHODS\n")
            f.write("-" * 20 + "\n")
            for method, count in self.test_results['detection_methods'].items():
                if count > 0:
                    f.write(f"{method}: {count}\n")
            f.write("\n")
            
            # Threat levels
            f.write("⚠️ THREAT LEVELS\n")
            f.write("-" * 15 + "\n")
            for level, count in self.test_results['threat_levels'].items():
                if count > 0:
                    f.write(f"{level}: {count}\n")
            f.write("\n")
            
            # Performance metrics
            f.write("⚡ PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Scan Duration: {self.test_results['scan_duration']:.2f}s\n")
            f.write(f"Files/Second: {self.test_results['files_per_second']:.1f}\n")
            f.write(f"Average Scan Time: {self.test_results['avg_scan_time']:.3f}s\n\n")
            
            # Accuracy metrics
            f.write("🎯 ACCURACY METRICS\n")
            f.write("-" * 18 + "\n")
            f.write(f"True Positives: {accuracy_metrics['true_positives']}\n")
            f.write(f"True Negatives: {accuracy_metrics['true_negatives']}\n")
            f.write(f"False Positives: {accuracy_metrics['false_positives']}\n")
            f.write(f"False Negatives: {accuracy_metrics['false_negatives']}\n")
            f.write(f"Accuracy: {accuracy_metrics['accuracy']:.1%}\n")
            f.write(f"Precision: {accuracy_metrics['precision']:.1%}\n")
            f.write(f"Recall: {accuracy_metrics['recall']:.1%}\n")
            f.write(f"F1-Score: {accuracy_metrics['f1_score']:.1%}\n\n")
            
            # False positives
            if self.test_results['false_positives']:
                f.write("❌ FALSE POSITIVES\n")
                f.write("-" * 17 + "\n")
                for fp in self.test_results['false_positives']:
                    f.write(f"File: {fp['file']}\n")
                    f.write(f"Reason: {fp['reason']}\n")
                    f.write(f"Confidence: {fp['confidence']:.1%}\n\n")
            
            # False negatives
            if self.test_results['false_negatives']:
                f.write("❌ FALSE NEGATIVES\n")
                f.write("-" * 17 + "\n")
                for fn in self.test_results['false_negatives']:
                    f.write(f"File: {fn['file']}\n")
                    f.write(f"Reason: {fn['reason']}\n")
                    f.write(f"Confidence: {fn['confidence']:.1%}\n\n")
        
        print_colored(f"📄 Test results saved to: {results_file}", "green")
        
        # Also save performance summary
        self.save_performance_summary(results_file)
    
    def save_performance_summary(self, results_file: Path):
        """Save performance summary to separate file."""
        timestamp = create_timestamp()
        performance_file = LOGS_DIR / f"performance_summary_{timestamp}.txt"
        
        with open(performance_file, 'w') as f:
            f.write("⚡ AI ANTIVIRUS PERFORMANCE SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Mode: {'LITE' if self.lite_mode else 'FULL'}\n\n")
            
            f.write(f"Total Files Scanned: {self.test_results['total_files']}\n")
            f.write(f"Scan Duration: {self.test_results['scan_duration']:.2f} seconds\n")
            f.write(f"Files Per Second: {self.test_results['files_per_second']:.1f}\n")
            f.write(f"Average Scan Time: {self.test_results['avg_scan_time']:.3f} seconds\n")
            f.write(f"Threats Detected: {self.test_results['threats_found']}\n")
            f.write(f"Detection Rate: {self.test_results['threats_found']/self.test_results['total_files']*100:.1f}%\n")
        
        print_colored(f"📊 Performance summary saved to: {performance_file}", "green")
    
    def run_comprehensive_test(self):
        """Run the complete test suite."""
        print_colored("🚀 Starting comprehensive test suite...", "cyan")
        
        # Generate test files
        print_colored("\n📁 Generating test files...", "cyan")
        self.generate_eicar_test_file(TEST_FILES_DIR / "eicar_test.com")
        self.generate_fake_malware_files()
        self.generate_safe_files()
        
        if not self.lite_mode:
            self.generate_edge_case_files()
        
        # First scan - should detect threats and add to known malware
        print_colored("\n🔍 First scan - detecting threats and building hash database...", "cyan")
        self.scan_test_files()
        
        # Second scan - should detect known malware via hashes
        print_colored("\n🧠 Second scan - testing hash-based detection...", "cyan")
        self.scan_test_files()
        
        # Calculate metrics
        print_colored("\n📊 Calculating metrics...", "cyan")
        accuracy_metrics = self.calculate_accuracy_metrics()
        
        # Display results
        print_colored("\n📋 Test Results:", "cyan")
        results_table = self.create_enhanced_results_table(accuracy_metrics)
        console.print(results_table)
        
        # Save results
        print_colored("\n💾 Saving results...", "cyan")
        self.save_enhanced_test_results(accuracy_metrics)
        
        # Print summary
        print_colored("\n🎉 Test suite completed!", "green")
        print_colored(f"📊 Final Accuracy: {accuracy_metrics['accuracy']:.1%}", "cyan")
        print_colored(f"⚡ Performance: {self.test_results['files_per_second']:.1f} files/second", "cyan")
        print_colored(f"🧠 Known malware detected: {self.test_results['known_malware_detected']}", "cyan")


def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description="AI Antivirus v4.X Test Suite")
    parser.add_argument('--lite', action='store_true',
                       help='Run lite test with minimal files (EICAR + 1 malware + 1 safe)')
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = UltimateAntivirusTestSuite(lite_mode=args.lite)
    test_suite.run_comprehensive_test()


if __name__ == "__main__":
    main()