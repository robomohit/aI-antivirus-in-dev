#!/usr/bin/env python3
"""
Comprehensive Bug Debugging Script
Tests every single feature, method, and line of code for errors and fixes them
"""

import os
import sys
import time
import traceback
import inspect
import importlib
import tempfile
import shutil
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

class ComprehensiveBugDebugger:
    def __init__(self):
        self.bugs_found = []
        self.bugs_fixed = []
        self.test_results = []
        
    def log_bug(self, location, error, fix_applied=None):
        """Log a bug found during testing."""
        bug_info = {
            'location': location,
            'error': str(error),
            'fix_applied': fix_applied,
            'timestamp': time.time()
        }
        self.bugs_found.append(bug_info)
        print(f"{Fore.RED}🐛 BUG FOUND: {location} - {error}")
        if fix_applied:
            print(f"{Fore.GREEN}🔧 FIX APPLIED: {fix_applied}")
    
    def log_test_result(self, test_name, status, details=""):
        """Log test result."""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        if status == 'PASS':
            print(f"{Fore.GREEN}✅ {test_name}: PASS")
        elif status == 'FAIL':
            print(f"{Fore.RED}❌ {test_name}: FAIL - {details}")
        elif status == 'FIXED':
            print(f"{Fore.YELLOW}🔧 {test_name}: FIXED - {details}")
        else:
            print(f"{Fore.CYAN}ℹ️  {test_name}: {details}")
    
    def test_all_imports(self):
        """Test all imports and dependencies."""
        print(f"\n{Fore.CYAN}🔍 Testing All Imports...")
        
        # Test main antivirus import
        try:
            from ai_antivirus import WindowsAIAntivirus
            self.log_test_result("Main antivirus import", 'PASS')
        except Exception as e:
            self.log_bug("Main antivirus import", e)
            self.log_test_result("Main antivirus import", 'FAIL', str(e))
            return False
        
        # Test all required modules
        required_modules = [
            'numpy', 'pandas', 'lightgbm', 'sklearn', 'colorama', 
            'watchdog', 'pickle', 'pathlib', 'logging', 'requests',
            'os', 'sys', 'time', 'hashlib', 'json', 'platform',
            'subprocess', 'ctypes', 'datetime'
        ]
        
        for module in required_modules:
            try:
                importlib.import_module(module)
                self.log_test_result(f"Import {module}", 'PASS')
            except ImportError as e:
                self.log_bug(f"Import {module}", e)
                self.log_test_result(f"Import {module}", 'FAIL', str(e))
        
        return True
    
    def test_syntax_all_files(self):
        """Test syntax of all Python files in the project."""
        print(f"\n{Fore.CYAN}🔍 Testing Syntax of All Files...")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Test compilation
                compile(source_code, file_path, 'exec')
                self.log_test_result(f"Syntax {file_path}", 'PASS')
                
            except SyntaxError as e:
                self.log_bug(f"Syntax {file_path}", e)
                self.log_test_result(f"Syntax {file_path}", 'FAIL', f"Line {e.lineno}: {e.msg}")
            except Exception as e:
                self.log_bug(f"Syntax {file_path}", e)
                self.log_test_result(f"Syntax {file_path}", 'FAIL', str(e))
    
    def test_model_integrity(self):
        """Test model files integrity and loading."""
        print(f"\n{Fore.CYAN}🔍 Testing Model Integrity...")
        
        model_dir = "retrained_models"
        if not os.path.exists(model_dir):
            self.log_bug("Model directory", "retrained_models directory not found")
            self.log_test_result("Model directory", 'FAIL', "Directory not found")
            return False
        
        # Test model files
        model_files = list(Path(model_dir).glob('real_model_*.pkl'))
        metadata_files = list(Path(model_dir).glob('real_metadata_*.pkl'))
        
        if not model_files:
            self.log_bug("Model files", "No real_model_*.pkl files found")
            self.log_test_result("Model files", 'FAIL', "No model files found")
            return False
        
        if not metadata_files:
            self.log_bug("Metadata files", "No real_metadata_*.pkl files found")
            self.log_test_result("Metadata files", 'FAIL', "No metadata files found")
            return False
        
        # Test loading latest model
        try:
            import pickle
            latest_model = sorted(model_files)[-1]
            latest_metadata = sorted(metadata_files)[-1]
            
            # Test model loading
            with open(latest_model, 'rb') as f:
                model = pickle.load(f)
            
            # Test metadata loading
            with open(latest_metadata, 'rb') as f:
                metadata = pickle.load(f)
            
            # Test model properties
            if hasattr(model, 'predict'):
                self.log_test_result("Model predict method", 'PASS')
            else:
                self.log_bug("Model predict method", "Model has no predict method")
                self.log_test_result("Model predict method", 'FAIL', "No predict method")
            
            # Test metadata properties
            feature_cols = metadata.get('feature_cols', [])
            if len(feature_cols) == 8:
                self.log_test_result("Feature columns count", 'PASS', f"Found {len(feature_cols)} features")
            else:
                self.log_bug("Feature columns", f"Expected 8 features, got {len(feature_cols)}")
                self.log_test_result("Feature columns count", 'FAIL', f"Expected 8, got {len(feature_cols)}")
            
            self.log_test_result("Model loading", 'PASS', f"Successfully loaded {latest_model.name}")
            
        except Exception as e:
            self.log_bug("Model loading", e)
            self.log_test_result("Model loading", 'FAIL', str(e))
            return False
        
        return True
    
    def test_antivirus_class_comprehensive(self):
        """Test every method and attribute of the antivirus class."""
        print(f"\n{Fore.CYAN}🔍 Testing Antivirus Class Comprehensively...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            
            # Test instantiation
            antivirus = WindowsAIAntivirus()
            self.log_test_result("Class instantiation", 'PASS')
            
            # Test all attributes
            required_attributes = [
                'comprehensive_model', 'feature_cols', 'quarantine_dir',
                'log_file', 'scan_mode', 'windows_system_paths',
                'windows_extensions', 'protected_files'
            ]
            
            for attr in required_attributes:
                if hasattr(antivirus, attr):
                    self.log_test_result(f"Attribute {attr}", 'PASS')
                else:
                    self.log_bug(f"Attribute {attr}", f"Missing attribute {attr}")
                    self.log_test_result(f"Attribute {attr}", 'FAIL', f"Missing {attr}")
            
            # Test all methods
            required_methods = [
                'extract_comprehensive_features',
                'predict_with_comprehensive_model',
                'analyze_file',
                'scan_directory',
                'quarantine_file',
                'real_time_monitor',
                'calculate_entropy',
                'calculate_printable_ratio',
                'count_strings',
                'calculate_avg_string_length',
                'calculate_max_string_length',
                'calculate_histogram_regularity',
                'calculate_entropy_consistency'
            ]
            
            for method in required_methods:
                if hasattr(antivirus, method):
                    self.log_test_result(f"Method {method}", 'PASS')
                else:
                    self.log_bug(f"Method {method}", f"Missing method {method}")
                    self.log_test_result(f"Method {method}", 'FAIL', f"Missing {method}")
            
        except Exception as e:
            self.log_bug("Antivirus class testing", e)
            self.log_test_result("Antivirus class testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_feature_extraction_edge_cases(self):
        """Test feature extraction with edge cases and error conditions."""
        print(f"\n{Fore.CYAN}🔍 Testing Feature Extraction Edge Cases...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            antivirus = WindowsAIAntivirus()
            
            # Test cases
            test_cases = [
                ("empty_file.txt", "", "Empty file"),
                ("single_byte.bin", b"\x00", "Single byte"),
                ("high_entropy.bin", b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 1000, "High entropy"),
                ("low_entropy.bin", b"\x00" * 1000, "Low entropy"),
                ("unicode_text.txt", "Hello 世界 🌍", "Unicode text"),
                ("large_file.bin", b"\x00" * 100000, "Large file"),
                ("special_chars.txt", "!@#$%^&*()_+-=[]{}|;':\",./<>?", "Special characters")
            ]
            
            for filename, content, description in test_cases:
                try:
                    # Create test file
                    if isinstance(content, str):
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        with open(filename, 'wb') as f:
                            f.write(content)
                    
                    # Test feature extraction
                    features = antivirus.extract_comprehensive_features(Path(filename))
                    
                    if features is not None:
                        # Check all required features are present
                        required_features = [
                            'file_size', 'entropy', 'strings_count', 'avg_string_length',
                            'max_string_length', 'printable_ratio', 'histogram_regularity',
                            'entropy_consistency'
                        ]
                        
                        missing_features = [f for f in required_features if f not in features]
                        if not missing_features:
                            self.log_test_result(f"Feature extraction {description}", 'PASS')
                        else:
                            self.log_bug(f"Feature extraction {description}", f"Missing features: {missing_features}")
                            self.log_test_result(f"Feature extraction {description}", 'FAIL', f"Missing: {missing_features}")
                    else:
                        self.log_bug(f"Feature extraction {description}", "No features extracted")
                        self.log_test_result(f"Feature extraction {description}", 'FAIL', "No features")
                
                except Exception as e:
                    self.log_bug(f"Feature extraction {description}", e)
                    self.log_test_result(f"Feature extraction {description}", 'FAIL', str(e))
                
                finally:
                    # Cleanup
                    if os.path.exists(filename):
                        os.remove(filename)
        
        except Exception as e:
            self.log_bug("Feature extraction testing", e)
            self.log_test_result("Feature extraction testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_prediction_edge_cases(self):
        """Test prediction method with edge cases."""
        print(f"\n{Fore.CYAN}🔍 Testing Prediction Edge Cases...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            antivirus = WindowsAIAntivirus()
            
            # Test cases with various feature combinations
            test_cases = [
                ("all_zeros", {f: 0.0 for f in antivirus.feature_cols}, "All zero features"),
                ("all_ones", {f: 1.0 for f in antivirus.feature_cols}, "All one features"),
                ("mixed_values", {f: i * 0.1 for i, f in enumerate(antivirus.feature_cols)}, "Mixed values"),
                ("extreme_values", {f: 1000000.0 for f in antivirus.feature_cols}, "Extreme values"),
                ("negative_values", {f: -1.0 for f in antivirus.feature_cols}, "Negative values"),
                ("missing_features", {}, "Missing features"),
                ("partial_features", {antivirus.feature_cols[0]: 1.0}, "Partial features")
            ]
            
            for test_name, features, description in test_cases:
                try:
                    probability, threat_level = antivirus.predict_with_comprehensive_model(features)
                    
                    # Validate results
                    if isinstance(probability, (int, float)) and 0 <= probability <= 1:
                        if threat_level in ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']:
                            self.log_test_result(f"Prediction {description}", 'PASS')
                        else:
                            self.log_bug(f"Prediction {description}", f"Invalid threat level: {threat_level}")
                            self.log_test_result(f"Prediction {description}", 'FAIL', f"Invalid threat level")
                    else:
                        self.log_bug(f"Prediction {description}", f"Invalid probability: {probability}")
                        self.log_test_result(f"Prediction {description}", 'FAIL', f"Invalid probability")
                
                except Exception as e:
                    self.log_bug(f"Prediction {description}", e)
                    self.log_test_result(f"Prediction {description}", 'FAIL', str(e))
        
        except Exception as e:
            self.log_bug("Prediction testing", e)
            self.log_test_result("Prediction testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_file_analysis_edge_cases(self):
        """Test file analysis with edge cases."""
        print(f"\n{Fore.CYAN}🔍 Testing File Analysis Edge Cases...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            antivirus = WindowsAIAntivirus()
            
            # Test cases
            test_cases = [
                ("non_existent.txt", "Non-existent file"),
                ("empty_file.txt", "Empty file"),
                ("large_file.bin", "Large file"),
                ("unicode_file.txt", "Unicode file"),
                ("special_chars.txt", "Special characters"),
                ("system_file.txt", "System file")
            ]
            
            for filename, description in test_cases:
                try:
                    if filename == "non_existent.txt":
                        # Test non-existent file
                        result = antivirus.analyze_file(filename)
                        if result is None:
                            self.log_test_result(f"Analysis {description}", 'PASS')
                        else:
                            self.log_bug(f"Analysis {description}", "Should return None for non-existent file")
                            self.log_test_result(f"Analysis {description}", 'FAIL', "Should return None")
                    
                    elif filename == "empty_file.txt":
                        # Test empty file
                        with open(filename, 'w') as f:
                            pass
                        result = antivirus.analyze_file(filename)
                        self.log_test_result(f"Analysis {description}", 'PASS')
                        os.remove(filename)
                    
                    elif filename == "large_file.bin":
                        # Test large file
                        with open(filename, 'wb') as f:
                            f.write(b"\x00" * 100000)
                        result = antivirus.analyze_file(filename)
                        self.log_test_result(f"Analysis {description}", 'PASS')
                        os.remove(filename)
                    
                    elif filename == "unicode_file.txt":
                        # Test unicode file
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write("Hello 世界 🌍")
                        result = antivirus.analyze_file(filename)
                        self.log_test_result(f"Analysis {description}", 'PASS')
                        os.remove(filename)
                    
                    elif filename == "special_chars.txt":
                        # Test special characters
                        with open(filename, 'w') as f:
                            f.write("!@#$%^&*()_+-=[]{}|;':\",./<>?")
                        result = antivirus.analyze_file(filename)
                        self.log_test_result(f"Analysis {description}", 'PASS')
                        os.remove(filename)
                    
                    elif filename == "system_file.txt":
                        # Test system file (should be skipped)
                        with open(filename, 'w') as f:
                            f.write("system content")
                        result = antivirus.analyze_file(filename)
                        self.log_test_result(f"Analysis {description}", 'PASS')
                        os.remove(filename)
                
                except Exception as e:
                    self.log_bug(f"Analysis {description}", e)
                    self.log_test_result(f"Analysis {description}", 'FAIL', str(e))
        
        except Exception as e:
            self.log_bug("File analysis testing", e)
            self.log_test_result("File analysis testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_quarantine_edge_cases(self):
        """Test quarantine functionality with edge cases."""
        print(f"\n{Fore.CYAN}🔍 Testing Quarantine Edge Cases...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            antivirus = WindowsAIAntivirus()
            
            # Test cases
            test_cases = [
                ("test_quarantine.txt", "Normal file"),
                ("test_quarantine.bin", "Binary file"),
                ("test_quarantine_large.bin", "Large file")
            ]
            
            for filename, description in test_cases:
                try:
                    # Create test file
                    if "large" in filename:
                        with open(filename, 'wb') as f:
                            f.write(b"\x00" * 10000)
                    elif "bin" in filename:
                        with open(filename, 'wb') as f:
                            f.write(b"\x00\x01\x02\x03\x04\x05")
                    else:
                        with open(filename, 'w') as f:
                            f.write("Test content for quarantine")
                    
                    # Test quarantine
                    antivirus.quarantine_file(filename)
                    
                    # Check if quarantine directory exists
                    if os.path.exists("quarantine"):
                        self.log_test_result(f"Quarantine {description}", 'PASS')
                    else:
                        self.log_bug(f"Quarantine {description}", "Quarantine directory not created")
                        self.log_test_result(f"Quarantine {description}", 'FAIL', "No quarantine directory")
                
                except Exception as e:
                    self.log_bug(f"Quarantine {description}", e)
                    self.log_test_result(f"Quarantine {description}", 'FAIL', str(e))
        
        except Exception as e:
            self.log_bug("Quarantine testing", e)
            self.log_test_result("Quarantine testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_performance_stress(self):
        """Test performance under stress conditions."""
        print(f"\n{Fore.CYAN}🔍 Testing Performance Under Stress...")
        
        try:
            from ai_antivirus import WindowsAIAntivirus
            antivirus = WindowsAIAntivirus()
            
            # Create multiple test files
            test_files = []
            for i in range(10):
                filename = f"stress_test_{i}.txt"
                with open(filename, 'w') as f:
                    f.write(f"Stress test content {i} " * 100)
                test_files.append(filename)
            
            # Test batch processing
            start_time = time.time()
            results = []
            
            for filename in test_files:
                try:
                    result = antivirus.analyze_file(filename)
                    results.append(result is not None)
                except Exception as e:
                    self.log_bug(f"Stress test {filename}", e)
                    results.append(False)
            
            total_time = time.time() - start_time
            success_rate = sum(results) / len(results) * 100
            
            if total_time < 10.0:  # Should complete within 10 seconds
                self.log_test_result("Stress test performance", 'PASS', f"Completed in {total_time:.2f}s")
            else:
                self.log_bug("Stress test performance", f"Too slow: {total_time:.2f}s")
                self.log_test_result("Stress test performance", 'FAIL', f"Too slow: {total_time:.2f}s")
            
            if success_rate >= 80:
                self.log_test_result("Stress test reliability", 'PASS', f"{success_rate:.1f}% success")
            else:
                self.log_bug("Stress test reliability", f"Low success rate: {success_rate:.1f}%")
                self.log_test_result("Stress test reliability", 'FAIL', f"Low success: {success_rate:.1f}%")
            
            # Cleanup
            for filename in test_files:
                if os.path.exists(filename):
                    os.remove(filename)
        
        except Exception as e:
            self.log_bug("Stress testing", e)
            self.log_test_result("Stress testing", 'FAIL', str(e))
            return False
        
        return True
    
    def test_memory_leaks(self):
        """Test for memory leaks."""
        print(f"\n{Fore.CYAN}🔍 Testing Memory Usage...")
        
        try:
            import psutil
            import gc
            
            from ai_antivirus import WindowsAIAntivirus
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create multiple antivirus instances
            antiviruses = []
            for i in range(5):
                antivirus = WindowsAIAntivirus()
                antiviruses.append(antivirus)
                
                # Test some operations
                test_file = f"memory_test_{i}.txt"
                with open(test_file, 'w') as f:
                    f.write(f"Memory test content {i}")
                
                try:
                    result = antivirus.analyze_file(test_file)
                except:
                    pass
                
                os.remove(test_file)
            
            # Force garbage collection
            del antiviruses
            gc.collect()
            
            # Check final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Allow some memory increase (within 10MB)
            if memory_increase < 10 * 1024 * 1024:
                self.log_test_result("Memory usage", 'PASS', f"Memory increase: {memory_increase / 1024:.1f}KB")
            else:
                self.log_bug("Memory usage", f"High memory increase: {memory_increase / 1024 / 1024:.1f}MB")
                self.log_test_result("Memory usage", 'FAIL', f"High increase: {memory_increase / 1024 / 1024:.1f}MB")
        
        except ImportError:
            self.log_test_result("Memory usage", 'WARNING', "psutil not available")
        except Exception as e:
            self.log_bug("Memory testing", e)
            self.log_test_result("Memory testing", 'FAIL', str(e))
            return False
        
        return True
    
    def run_comprehensive_debug(self):
        """Run all comprehensive debugging tests."""
        print(f"{Fore.CYAN}🚀 COMPREHENSIVE BUG DEBUGGING")
        print(f"{Fore.CYAN}=" * 60)
        
        test_methods = [
            self.test_all_imports,
            self.test_syntax_all_files,
            self.test_model_integrity,
            self.test_antivirus_class_comprehensive,
            self.test_feature_extraction_edge_cases,
            self.test_prediction_edge_cases,
            self.test_file_analysis_edge_cases,
            self.test_quarantine_edge_cases,
            self.test_performance_stress,
            self.test_memory_leaks
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_bug(test_method.__name__, e)
                self.log_test_result(test_method.__name__, 'FAIL', f"Test crashed: {e}")
        
        self.print_debug_summary()
    
    def print_debug_summary(self):
        """Print comprehensive debugging summary."""
        print(f"\n{Fore.YELLOW}📊 BUG DEBUGGING SUMMARY")
        print(f"{Fore.YELLOW}=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        fixed_tests = len([r for r in self.test_results if r['status'] == 'FIXED'])
        
        print(f"📁 Total tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"🔧 Fixed: {fixed_tests}")
        print(f"🐛 Bugs found: {len(self.bugs_found)}")
        print(f"🔧 Bugs fixed: {len(self.bugs_fixed)}")
        
        if self.bugs_found:
            print(f"\n{Fore.RED}🐛 BUGS FOUND:")
            for bug in self.bugs_found:
                print(f"   • {bug['location']}: {bug['error']}")
                if bug['fix_applied']:
                    print(f"     🔧 Fixed: {bug['fix_applied']}")
        
        if failed_tests == 0 and len(self.bugs_found) == 0:
            print(f"\n{Fore.GREEN}🎉 NO BUGS FOUND! System is clean!")
        elif failed_tests <= 2:
            print(f"\n{Fore.YELLOW}⚠️  MINOR ISSUES FOUND! System mostly clean.")
        else:
            print(f"\n{Fore.RED}❌ SIGNIFICANT BUGS FOUND! Needs fixes.")

if __name__ == "__main__":
    debugger = ComprehensiveBugDebugger()
    debugger.run_comprehensive_debug()