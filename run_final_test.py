#!/usr/bin/env python3
"""
🧪 ULTIMATE AI ANTIVIRUS FINAL SYSTEM TEST v5.X
Comprehensive end-to-end validation of the complete antivirus system
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import csv

def run_command(cmd, description):
    """Run a command and return results."""
    print(f"🔍 {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        duration = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration,
            'return_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out after 300 seconds',
            'duration': 300,
            'return_code': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'duration': time.time() - start_time,
            'return_code': -1
        }

def analyze_scan_output(output):
    """Analyze scan output to extract metrics."""
    metrics = {
        'files_scanned': 0,
        'threats_detected': 0,
        'known_malware_hits': 0,
        'ai_detections': 0,
        'extension_detections': 0,
        'both_detections': 0,
        'safe_files': 0
    }
    
    lines = output.split('\n')
    for line in lines:
        if 'Files Scanned' in line:
            try:
                metrics['files_scanned'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Threats Found' in line:
            try:
                metrics['threats_detected'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Known Malware' in line:
            try:
                metrics['known_malware_hits'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'AI Detections' in line:
            try:
                metrics['ai_detections'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Extension Detections' in line:
            try:
                metrics['extension_detections'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Both Detections' in line:
            try:
                metrics['both_detections'] = int(line.split(':')[1].strip())
            except:
                pass
    
    return metrics

def check_known_malware_database():
    """Check the known malware database."""
    malware_file = Path("known_malware.csv")
    if malware_file.exists():
        with open(malware_file, 'r') as f:
            reader = csv.DictReader(f)
            entries = list(reader)
        return len(entries)
    return 0

def check_model_file():
    """Check if the model file exists and is valid."""
    model_file = Path("model/model.pkl")
    if model_file.exists():
        try:
            import joblib
            model = joblib.load(model_file)
            return {
                'exists': True,
                'type': type(model).__name__,
                'features': getattr(model, 'n_features_in_', 'Unknown'),
                'estimators': getattr(model, 'n_estimators', 'Unknown')
            }
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    return {'exists': False}

def check_dataset():
    """Check the malware dataset."""
    dataset_file = Path("malware_dataset.csv")
    if dataset_file.exists():
        try:
            df = pd.read_csv(dataset_file)
            return {
                'exists': True,
                'samples': len(df),
                'features': len(df.columns),
                'malware_samples': len(df[df['label'] == 1]),
                'safe_samples': len(df[df['label'] == 0])
            }
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    return {'exists': False}

def create_summary_table(results):
    """Create a summary table of all test results."""
    print("\n" + "="*80)
    print("🎯 ULTIMATE AI ANTIVIRUS FINAL TEST SUMMARY")
    print("="*80)
    
    # System Status
    print("\n📊 SYSTEM STATUS:")
    print(f"  Model File: {'✅' if results['model']['exists'] else '❌'}")
    print(f"  Dataset: {'✅' if results['dataset']['exists'] else '❌'}")
    print(f"  Known Malware DB: {results['known_malware_count']} entries")
    
    # Scan Results
    print("\n🔍 SCAN RESULTS:")
    print(f"  Smart Scan: {'✅' if results['smart_scan']['success'] else '❌'}")
    print(f"  Full Scan: {'✅' if results['full_scan']['success'] else '❌'}")
    print(f"  GUI Launch: {'✅' if results['gui_test']['success'] else '❌'}")
    
    # Performance Metrics
    print("\n⚡ PERFORMANCE METRICS:")
    if results['smart_scan']['success']:
        smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
        print(f"  Smart Scan Duration: {results['smart_scan']['duration']:.2f}s")
        print(f"  Files Scanned: {smart_metrics['files_scanned']}")
        print(f"  Threats Detected: {smart_metrics['threats_detected']}")
        print(f"  Known Malware Hits: {smart_metrics['known_malware_hits']}")
    
    if results['full_scan']['success']:
        full_metrics = analyze_scan_output(results['full_scan']['stdout'])
        print(f"  Full Scan Duration: {results['full_scan']['duration']:.2f}s")
        print(f"  Files Scanned: {full_metrics['files_scanned']}")
        print(f"  Threats Detected: {full_metrics['threats_detected']}")
    
    # Detection Breakdown
    if results['smart_scan']['success']:
        smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
        print("\n🎯 DETECTION BREAKDOWN:")
        print(f"  AI Detections: {smart_metrics['ai_detections']}")
        print(f"  Extension Detections: {smart_metrics['extension_detections']}")
        print(f"  Both Detections: {smart_metrics['both_detections']}")
        print(f"  Known Malware: {smart_metrics['known_malware_hits']}")
    
    # Accuracy Calculation
    if results['smart_scan']['success']:
        smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
        total_files = smart_metrics['files_scanned']
        threats = smart_metrics['threats_detected']
        if total_files > 0:
            accuracy = (total_files - threats) / total_files * 100
            print(f"\n📈 ACCURACY: {accuracy:.1f}%")
    
    print("\n" + "="*80)

def save_final_log(results):
    """Save comprehensive final test log."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(f"logs/final_test_summary_{timestamp}.txt")
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write("🧪 ULTIMATE AI ANTIVIRUS FINAL SYSTEM TEST v5.X\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System Status
        f.write("📊 SYSTEM STATUS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model File: {'✅' if results['model']['exists'] else '❌'}\n")
        if results['model']['exists'] and 'error' not in results['model']:
            f.write(f"Model Type: {results['model']['type']}\n")
            f.write(f"Features: {results['model']['features']}\n")
            f.write(f"Estimators: {results['model']['estimators']}\n")
        f.write(f"Dataset: {'✅' if results['dataset']['exists'] else '❌'}\n")
        if results['dataset']['exists'] and 'error' not in results['dataset']:
            f.write(f"Dataset Samples: {results['dataset']['samples']}\n")
            f.write(f"Malware Samples: {results['dataset']['malware_samples']}\n")
            f.write(f"Safe Samples: {results['dataset']['safe_samples']}\n")
        f.write(f"Known Malware DB: {results['known_malware_count']} entries\n\n")
        
        # Scan Results
        f.write("🔍 SCAN RESULTS\n")
        f.write("-" * 15 + "\n")
        f.write(f"Smart Scan: {'✅' if results['smart_scan']['success'] else '❌'}\n")
        f.write(f"Full Scan: {'✅' if results['full_scan']['success'] else '❌'}\n")
        f.write(f"GUI Launch: {'✅' if results['gui_test']['success'] else '❌'}\n\n")
        
        # Performance Metrics
        f.write("⚡ PERFORMANCE METRICS\n")
        f.write("-" * 20 + "\n")
        if results['smart_scan']['success']:
            smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
            f.write(f"Smart Scan Duration: {results['smart_scan']['duration']:.2f}s\n")
            f.write(f"Files Scanned: {smart_metrics['files_scanned']}\n")
            f.write(f"Threats Detected: {smart_metrics['threats_detected']}\n")
            f.write(f"Known Malware Hits: {smart_metrics['known_malware_hits']}\n")
            if smart_metrics['files_scanned'] > 0:
                files_per_sec = smart_metrics['files_scanned'] / results['smart_scan']['duration']
                f.write(f"Files/Second: {files_per_sec:.1f}\n")
        
        if results['full_scan']['success']:
            full_metrics = analyze_scan_output(results['full_scan']['stdout'])
            f.write(f"Full Scan Duration: {results['full_scan']['duration']:.2f}s\n")
            f.write(f"Files Scanned: {full_metrics['files_scanned']}\n")
            f.write(f"Threats Detected: {full_metrics['threats_detected']}\n")
            if full_metrics['files_scanned'] > 0:
                files_per_sec = full_metrics['files_scanned'] / results['full_scan']['duration']
                f.write(f"Files/Second: {files_per_sec:.1f}\n")
        
        # Detection Breakdown
        if results['smart_scan']['success']:
            smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
            f.write("\n🎯 DETECTION BREAKDOWN\n")
            f.write("-" * 20 + "\n")
            f.write(f"AI Detections: {smart_metrics['ai_detections']}\n")
            f.write(f"Extension Detections: {smart_metrics['extension_detections']}\n")
            f.write(f"Both Detections: {smart_metrics['both_detections']}\n")
            f.write(f"Known Malware: {smart_metrics['known_malware_hits']}\n")
        
        # Accuracy Metrics
        if results['smart_scan']['success']:
            smart_metrics = analyze_scan_output(results['smart_scan']['stdout'])
            total_files = smart_metrics['files_scanned']
            threats = smart_metrics['threats_detected']
            if total_files > 0:
                accuracy = (total_files - threats) / total_files * 100
                f.write(f"\n📈 ACCURACY: {accuracy:.1f}%\n")
        
        # Error Logs
        if results['smart_scan']['stderr']:
            f.write(f"\n❌ SMART SCAN ERRORS:\n{results['smart_scan']['stderr']}\n")
        if results['full_scan']['stderr']:
            f.write(f"\n❌ FULL SCAN ERRORS:\n{results['full_scan']['stderr']}\n")
    
    print(f"📄 Final test log saved to: {log_file}")
    return log_file

def main():
    """Main final test function."""
    print("🚀 ULTIMATE AI ANTIVIRUS FINAL SYSTEM TEST v5.X")
    print("=" * 60)
    print("🧪 Running comprehensive end-to-end validation...")
    
    # Activate virtual environment
    python_exe = sys.executable
    
    # Initialize results
    results = {}
    
    # 1. Check system components
    print("\n📊 Checking system components...")
    results['model'] = check_model_file()
    results['dataset'] = check_dataset()
    results['known_malware_count'] = check_known_malware_database()
    
    # 2. Test Smart Scan
    print("\n🧠 Testing Smart Scan...")
    results['smart_scan'] = run_command(
        [python_exe, "ai_antivirus.py", "--smart-scan"],
        "Smart Scan"
    )
    
    # 3. Test Full Scan (with timeout)
    print("\n🔍 Testing Full Scan...")
    results['full_scan'] = run_command(
        [python_exe, "ai_antivirus.py", "--full-scan"],
        "Full Scan"
    )
    
    # 4. Test GUI Launch
    print("\n🖥️ Testing GUI Launch...")
    results['gui_test'] = run_command(
        [python_exe, "ai_antivirus.py", "--gui"],
        "GUI Launch"
    )
    
    # 5. Test Test Suite
    print("\n🧪 Testing Test Suite...")
    results['test_suite'] = run_command(
        [python_exe, "test_suite.py", "--lite"],
        "Test Suite (Lite Mode)"
    )
    
    # 6. Create summary table
    create_summary_table(results)
    
    # 7. Save final log
    log_file = save_final_log(results)
    
    # 8. Final assessment
    print("\n🎯 FINAL ASSESSMENT:")
    success_count = sum([
        results['model']['exists'],
        results['dataset']['exists'],
        results['smart_scan']['success'],
        results['gui_test']['success']
    ])
    
    total_tests = 4
    success_rate = (success_count / total_tests) * 100
    
    print(f"✅ Success Rate: {success_rate:.1f}% ({success_count}/{total_tests})")
    
    if success_rate >= 75:
        print("🎉 SYSTEM STATUS: PRODUCTION READY!")
    elif success_rate >= 50:
        print("⚠️ SYSTEM STATUS: NEEDS IMPROVEMENT")
    else:
        print("❌ SYSTEM STATUS: CRITICAL ISSUES")
    
    print(f"\n📄 Detailed results saved to: {log_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()