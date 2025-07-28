#!/usr/bin/env python3
"""
Training Data Generator and Model Trainer for AI Antivirus v4.X
Creates dummy training data and trains the Random Forest model with proper validation.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import colorama
from colorama import Fore, Style
import logging
from datetime import datetime
import argparse
import sys

# Import our modules
from config import (
    MODEL_CONFIG, TRAINING_CONFIG, FEATURE_CONFIG, 
    LOGS_DIR, MODEL_PATH, TRAINING_DATA_PATH,
    SUSPICIOUS_EXTENSIONS, FILE_TYPE_CATEGORIES
)
from utils import (
    get_entropy, get_file_type, get_filename_pattern_flags,
    simulate_file_creation_randomness, calculate_file_complexity,
    create_log_folders, print_colored
)

# Initialize colorama
colorama.init(autoreset=True)


def setup_logging():
    """Setup logging for model training metrics."""
    create_log_folders()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"model_metrics_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def create_enhanced_training_data(output_path=TRAINING_DATA_PATH):
    """Create comprehensive dummy training data with enhanced features."""
    print_colored("📊 Creating enhanced training data...", Fore.CYAN)
    
    np.random.seed(42)  # For reproducible results
    data = []
    
    # Safe file extensions and their typical characteristics
    safe_extensions = {
        '.txt': (1, 100),      # 1-100 KB
        '.pdf': (50, 2000),    # 50-2000 KB
        '.jpg': (10, 500),     # 10-500 KB
        '.png': (10, 800),     # 10-800 KB
        '.mp3': (100, 5000),   # 100-5000 KB
        '.mp4': (1000, 50000), # 1000-50000 KB
        '.doc': (50, 1000),    # 50-1000 KB
        '.xls': (20, 500),     # 20-500 KB
        '.csv': (1, 100),      # 1-100 KB
        '.json': (1, 50),      # 1-50 KB
        '.py': (1, 200),       # 1-200 KB
        '.html': (1, 100),     # 1-100 KB
        '.css': (1, 50),       # 1-50 KB
        '.js': (1, 200),       # 1-200 KB
    }
    
    # Suspicious file extensions and their characteristics
    suspicious_extensions = {
        '.exe': (100, 10000),   # 100-10000 KB
        '.bat': (1, 50),        # 1-50 KB
        '.vbs': (1, 100),       # 1-100 KB
        '.scr': (100, 5000),    # 100-5000 KB
        '.ps1': (1, 200),       # 1-200 KB
        '.cmd': (1, 50),        # 1-50 KB
        '.com': (50, 2000),     # 50-2000 KB
        '.pif': (100, 5000),    # 100-5000 KB
        '.reg': (1, 20),        # 1-20 KB
        '.js': (1, 200),        # 1-200 KB (can be malicious)
        '.jar': (100, 10000),   # 100-10000 KB
        '.msi': (1000, 50000),  # 1000-50000 KB
        '.dll': (50, 5000),     # 50-5000 KB
        '.sys': (100, 10000),   # 100-10000 KB
    }
    
    # Generate safe files with enhanced features
    print_colored("✅ Generating safe files with enhanced features...", Fore.GREEN)
    for ext, (min_size, max_size) in safe_extensions.items():
        for i in range(25):  # Increased samples
            size = np.random.randint(min_size, max_size)
            
            # Generate realistic filename
            filename = f"document_{i}_{ext[1:]}.{ext[1:]}"
            
            # Enhanced features
            file_type = get_file_type(filename)
            pattern_flags = get_filename_pattern_flags(filename)
            entropy = np.random.uniform(3.0, 6.0)  # Lower entropy for safe files
            complexity = np.random.uniform(0.2, 0.6)  # Lower complexity
            creation_randomness = simulate_file_creation_randomness()
            
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'filename': filename,
                'file_type': file_type,
                'entropy': entropy,
                'complexity': complexity,
                'creation_randomness': creation_randomness,
                'contains_cheat': pattern_flags['contains_cheat'],
                'contains_keygen': pattern_flags['contains_keygen'],
                'contains_virus': pattern_flags['contains_virus'],
                'contains_suspicious': pattern_flags['contains_suspicious'],
                'has_random_chars': pattern_flags['has_random_chars'],
                'is_hidden': pattern_flags['is_hidden'],
                'has_multiple_extensions': pattern_flags['has_multiple_extensions'],
                'is_malicious': 0
            })
    
    # Generate suspicious files with enhanced features
    print_colored("⚠️ Generating suspicious files with enhanced features...", Fore.RED)
    for ext, (min_size, max_size) in suspicious_extensions.items():
        for i in range(20):  # Increased samples
            size = np.random.randint(min_size, max_size)
            
            # Generate suspicious filename patterns
            suspicious_names = [
                f"free_cheats_{i}.{ext[1:]}",
                f"keygen_crack_{i}.{ext[1:]}",
                f"virus_trojan_{i}.{ext[1:]}",
                f"hack_tool_{i}.{ext[1:]}",
                f"malware_{i}.{ext[1:]}"
            ]
            filename = np.random.choice(suspicious_names)
            
            # Enhanced features
            file_type = get_file_type(filename)
            pattern_flags = get_filename_pattern_flags(filename)
            entropy = np.random.uniform(6.0, 8.0)  # Higher entropy for suspicious files
            complexity = np.random.uniform(0.7, 1.0)  # Higher complexity
            creation_randomness = simulate_file_creation_randomness()
            
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'filename': filename,
                'file_type': file_type,
                'entropy': entropy,
                'complexity': complexity,
                'creation_randomness': creation_randomness,
                'contains_cheat': pattern_flags['contains_cheat'],
                'contains_keygen': pattern_flags['contains_keygen'],
                'contains_virus': pattern_flags['contains_virus'],
                'contains_suspicious': pattern_flags['contains_suspicious'],
                'has_random_chars': pattern_flags['has_random_chars'],
                'is_hidden': pattern_flags['is_hidden'],
                'has_multiple_extensions': pattern_flags['has_multiple_extensions'],
                'is_malicious': 1
            })
    
    # Add edge cases
    print_colored("🔍 Adding edge cases with enhanced features...", Fore.YELLOW)
    
    # Small suspicious files (potential false negatives)
    for ext in ['.exe', '.bat', '.vbs', '.ps1']:
        for i in range(10):
            size = np.random.randint(1, 10)
            filename = f"small_suspicious_{i}.{ext[1:]}"
            
            file_type = get_file_type(filename)
            pattern_flags = get_filename_pattern_flags(filename)
            
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'filename': filename,
                'file_type': file_type,
                'entropy': np.random.uniform(4.0, 7.0),
                'complexity': np.random.uniform(0.5, 0.8),
                'creation_randomness': simulate_file_creation_randomness(),
                'contains_cheat': pattern_flags['contains_cheat'],
                'contains_keygen': pattern_flags['contains_keygen'],
                'contains_virus': pattern_flags['contains_virus'],
                'contains_suspicious': pattern_flags['contains_suspicious'],
                'has_random_chars': pattern_flags['has_random_chars'],
                'is_hidden': pattern_flags['is_hidden'],
                'has_multiple_extensions': pattern_flags['has_multiple_extensions'],
                'is_malicious': 1
            })
    
    # Large safe files (potential false positives)
    for ext in ['.pdf', '.mp4', '.doc']:
        for i in range(10):
            size = np.random.randint(10000, 50000)
            filename = f"large_safe_{i}.{ext[1:]}"
            
            file_type = get_file_type(filename)
            pattern_flags = get_filename_pattern_flags(filename)
            
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'filename': filename,
                'file_type': file_type,
                'entropy': np.random.uniform(2.0, 5.0),
                'complexity': np.random.uniform(0.1, 0.4),
                'creation_randomness': simulate_file_creation_randomness(),
                'contains_cheat': pattern_flags['contains_cheat'],
                'contains_keygen': pattern_flags['contains_keygen'],
                'contains_virus': pattern_flags['contains_virus'],
                'contains_suspicious': pattern_flags['contains_suspicious'],
                'has_random_chars': pattern_flags['has_random_chars'],
                'is_hidden': pattern_flags['is_hidden'],
                'has_multiple_extensions': pattern_flags['has_multiple_extensions'],
                'is_malicious': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save training data
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print_colored(f"📊 Enhanced training data created: {output_path}", Fore.GREEN)
    print_colored("📈 Enhanced dataset statistics:", Fore.CYAN)
    print(f"   Total samples: {len(df)}")
    print(f"   Safe files: {len(df[df['is_malicious'] == 0])}")
    print(f"   Suspicious files: {len(df[df['is_malicious'] == 1])}")
    print(f"   Unique extensions: {df['file_extension'].nunique()}")
    print(f"   Features included: {list(df.columns)}")
    
    return df


def prepare_enhanced_features(df):
    """Prepare enhanced features for model training."""
    # Convert extensions to numerical features (one-hot encoding)
    extension_dummies = pd.get_dummies(df['file_extension'], prefix='ext')
    
    # Convert file types to numerical features
    file_type_dummies = pd.get_dummies(df['file_type'], prefix='type')
    
    # Convert boolean features to numerical
    boolean_features = [
        'contains_cheat', 'contains_keygen', 'contains_virus', 
        'contains_suspicious', 'has_random_chars', 'is_hidden', 
        'has_multiple_extensions'
    ]
    
    # Combine all features
    feature_columns = [
        'file_size_kb', 'entropy', 'complexity', 'creation_randomness'
    ] + boolean_features
    
    X = pd.concat([
        extension_dummies,
        file_type_dummies,
        df[feature_columns]
    ], axis=1)
    
    y = df['is_malicious']
    
    return X, y


def evaluate_model(model, X_test, y_test, y_pred, split_name="Test"):
    """Evaluate model performance with comprehensive metrics."""
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'split_name': split_name
    }


def print_enhanced_metrics(metrics, split_name="Test"):
    """Print formatted enhanced metrics."""
    print_colored(f"\n📊 {split_name} Set Performance:", Fore.CYAN)
    print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']:.1%})")
    print(f"   Recall:    {metrics['recall']:.3f} ({metrics['recall']:.1%})")
    print(f"   F1-Score:  {metrics['f1_score']:.3f} ({metrics['f1_score']:.1%})")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    print_colored(f"\n📋 Confusion Matrix ({split_name}):", Fore.MAGENTA)
    print(f"                Predicted")
    print(f"                Safe  Suspicious")
    print(f"Actual Safe     {cm[0,0]:4d} {cm[0,1]:10d}")
    print(f"      Suspicious {cm[1,0]:4d} {cm[1,1]:10d}")


def save_confusion_matrix_plot(cm, split_name, log_file):
    """Save confusion matrix as PNG image."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe', 'Suspicious'],
                   yticklabels=['Safe', 'Suspicious'])
        plt.title(f'Confusion Matrix - {split_name} Set')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save to logs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = LOGS_DIR / f"confusion_matrix_{split_name.lower()}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print_colored(f"📊 Confusion matrix saved: {plot_path}", Fore.GREEN)
        logging.info(f"Confusion matrix plot saved: {plot_path}")
        
    except Exception as e:
        print_colored(f"⚠️ Could not save confusion matrix plot: {e}", Fore.YELLOW)


def log_enhanced_metrics(metrics, log_file):
    """Log enhanced metrics to file."""
    split_name = metrics['split_name']
    logging.info(f"=== {split_name.upper()} SET METRICS ===")
    logging.info(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    logging.info(f"Precision: {metrics['precision']:.3f} ({metrics['precision']:.1%})")
    logging.info(f"Recall: {metrics['recall']:.3f} ({metrics['recall']:.1%})")
    logging.info(f"F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']:.1%})")
    logging.info(f"Confusion Matrix:")
    logging.info(f"{metrics['confusion_matrix']}")


def check_overfitting(train_metrics, test_metrics):
    """Check for overfitting by comparing train and test metrics."""
    print_colored("\n🔍 Overfitting Analysis:", Fore.YELLOW)
    
    accuracy_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    f1_diff = train_metrics['f1_score'] - test_metrics['f1_score']
    
    print(f"   Train Accuracy: {train_metrics['accuracy']:.3f}")
    print(f"   Test Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"   Accuracy Difference: {accuracy_diff:.3f}")
    
    print(f"   Train F1-Score: {train_metrics['f1_score']:.3f}")
    print(f"   Test F1-Score:  {test_metrics['f1_score']:.3f}")
    print(f"   F1-Score Difference: {f1_diff:.3f}")
    
    # Overfitting assessment
    if accuracy_diff > 0.05 or f1_diff > 0.05:
        print_colored("⚠️  WARNING: Potential overfitting detected!", Fore.RED)
        print(f"   Consider reducing model complexity or adding more training data.")
    elif accuracy_diff > 0.02 or f1_diff > 0.02:
        print_colored("⚠️  Minor overfitting detected.", Fore.YELLOW)
    else:
        print_colored("✅ No significant overfitting detected.", Fore.GREEN)


def print_model_info(model, feature_names):
    """Print detailed model information."""
    print_colored("\n🧠 Model Information:", Fore.CYAN)
    print(f"   Model Type: Random Forest Classifier")
    print(f"   Estimators: {model.n_estimators}")
    print(f"   Max Depth: {model.max_depth}")
    print(f"   Min Samples Split: {model.min_samples_split}")
    print(f"   Min Samples Leaf: {model.min_samples_leaf}")
    print(f"   Total Features: {len(feature_names)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print_colored("\n🔍 Top 10 Most Important Features:", Fore.BLUE)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.3f}")
    
    # Model statistics
    print_colored("\n📊 Model Statistics:", Fore.MAGENTA)
    print(f"   Feature Names: {list(feature_names)}")
    print(f"   Classes: {model.classes_}")
    print(f"   N Features: {model.n_features_in_}")
    print(f"   N Outputs: {model.n_outputs_}")


def train_enhanced_model(training_data_path=TRAINING_DATA_PATH, model_path=MODEL_PATH):
    """Train the Random Forest model with enhanced features and proper validation."""
    print_colored("🧠 Training AI model with enhanced features and validation...", Fore.CYAN)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("=== AI ANTIVIRUS v4.X MODEL TRAINING ===")
    logging.info(f"Training started at: {datetime.now()}")
    
    # Load training data
    df = pd.read_csv(training_data_path)
    logging.info(f"Loaded {len(df)} training samples")
    
    # Prepare enhanced features
    X, y = prepare_enhanced_features(df)
    
    print_colored(f"📊 Enhanced feature matrix shape: {X.shape}", Fore.BLUE)
    print_colored(f"📊 Target vector shape: {y.shape}", Fore.BLUE)
    
    # Split data with proper validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TRAINING_CONFIG['test_size'], 
        random_state=TRAINING_CONFIG['random_state'], 
        shuffle=TRAINING_CONFIG['shuffle'], 
        stratify=y if TRAINING_CONFIG['stratify'] else None
    )
    
    print_colored(f"📈 Training set size: {len(X_train)} ({100-TRAINING_CONFIG['test_size']*100:.0f}%)", Fore.GREEN)
    print_colored(f"📊 Test set size: {len(X_test)} ({TRAINING_CONFIG['test_size']*100:.0f}%)", Fore.GREEN)
    
    logging.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train model with enhanced parameters
    model = RandomForestClassifier(**MODEL_CONFIG)
    
    print_colored("🔄 Training Random Forest model with enhanced features...", Fore.YELLOW)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_model(model, X_train, y_train, y_train_pred, "Training")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_model(model, X_test, y_test, y_test_pred, "Test")
    
    # Print results
    print_enhanced_metrics(train_metrics, "Training")
    print_enhanced_metrics(test_metrics, "Test")
    
    # Log metrics
    log_enhanced_metrics(train_metrics, log_file)
    log_enhanced_metrics(test_metrics, log_file)
    
    # Save confusion matrix plots
    save_confusion_matrix_plot(train_metrics['confusion_matrix'], "Training", log_file)
    save_confusion_matrix_plot(test_metrics['confusion_matrix'], "Test", log_file)
    
    # Check for overfitting
    check_overfitting(train_metrics, test_metrics)
    
    # Detailed classification report
    print_colored("\n📋 Detailed Classification Report (Test Set):", Fore.MAGENTA)
    print(classification_report(y_test, y_test_pred, target_names=['Safe', 'Suspicious']))
    
    # Print model information
    print_model_info(model, model.feature_names_in_)
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print_colored(f"💾 Enhanced model saved to: {model_path}", Fore.GREEN)
    logging.info(f"Enhanced model saved to: {model_path}")
    
    # Final summary
    print_colored("\n🎉 Enhanced Model Training Complete!", Fore.GREEN)
    print_colored(f"📊 Final Test Accuracy: {test_metrics['accuracy']:.1%}", Fore.CYAN)
    
    return model, test_metrics


def test_enhanced_model(model_path=MODEL_PATH):
    """Test the trained enhanced model with sample files."""
    print_colored("🧪 Testing enhanced model with sample files...", Fore.CYAN)
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Test cases with enhanced features
    test_cases = [
        {'extension': '.exe', 'size_kb': 500, 'filename': 'free_cheats.exe', 'expected': 'Suspicious'},
        {'extension': '.txt', 'size_kb': 10, 'filename': 'document.txt', 'expected': 'Safe'},
        {'extension': '.bat', 'size_kb': 5, 'filename': 'keygen_crack.bat', 'expected': 'Suspicious'},
        {'extension': '.pdf', 'size_kb': 2000, 'filename': 'document.pdf', 'expected': 'Safe'},
        {'extension': '.vbs', 'size_kb': 50, 'filename': 'virus_trojan.vbs', 'expected': 'Suspicious'},
        {'extension': '.jpg', 'size_kb': 100, 'filename': 'image.jpg', 'expected': 'Safe'},
        {'extension': '.ps1', 'size_kb': 150, 'filename': 'hack_tool.ps1', 'expected': 'Suspicious'},
        {'extension': '.dll', 'size_kb': 2000, 'filename': 'malware.dll', 'expected': 'Suspicious'},
    ]
    
    print_colored("📋 Enhanced Test Results:", Fore.YELLOW)
    print(f"{'File Type':<15} {'Size (KB)':<10} {'Filename':<20} {'Prediction':<12} {'Confidence':<12} {'Expected':<12} {'Status':<8}")
    print("-" * 95)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for test_case in test_cases:
        # Create enhanced feature vector
        feature_vector = np.zeros(len(model.feature_names_in_))
        
        # Set basic features
        size_idx = np.where(model.feature_names_in_ == 'file_size_kb')[0]
        if len(size_idx) > 0:
            feature_vector[size_idx[0]] = test_case['size_kb']
        
        # Set extension
        ext_prefix = 'ext_'
        for i, feature_name in enumerate(model.feature_names_in_):
            if feature_name.startswith(ext_prefix):
                if feature_name == f'ext_{test_case["extension"]}':
                    feature_vector[i] = 1
        
        # Set file type
        file_type = get_file_type(test_case['filename'])
        type_prefix = 'type_'
        for i, feature_name in enumerate(model.feature_names_in_):
            if feature_name.startswith(type_prefix):
                if feature_name == f'type_{file_type}':
                    feature_vector[i] = 1
        
        # Set pattern flags
        pattern_flags = get_filename_pattern_flags(test_case['filename'])
        for flag_name, flag_value in pattern_flags.items():
            if flag_name in model.feature_names_in_:
                idx = np.where(model.feature_names_in_ == flag_name)[0]
                if len(idx) > 0:
                    feature_vector[idx[0]] = 1 if flag_value else 0
        
        # Set simulated features
        if 'entropy' in model.feature_names_in_:
            entropy_idx = np.where(model.feature_names_in_ == 'entropy')[0]
            if len(entropy_idx) > 0:
                feature_vector[entropy_idx[0]] = np.random.uniform(4.0, 7.0)
        
        if 'complexity' in model.feature_names_in_:
            complexity_idx = np.where(model.feature_names_in_ == 'complexity')[0]
            if len(complexity_idx) > 0:
                feature_vector[complexity_idx[0]] = np.random.uniform(0.3, 0.8)
        
        if 'creation_randomness' in model.feature_names_in_:
            randomness_idx = np.where(model.feature_names_in_ == 'creation_randomness')[0]
            if len(randomness_idx) > 0:
                feature_vector[randomness_idx[0]] = simulate_file_creation_randomness()
        
        # Make prediction
        prediction = model.predict([feature_vector])[0]
        confidence = max(model.predict_proba([feature_vector])[0])
        
        prediction_text = "Suspicious" if prediction else "Safe"
        confidence_text = f"{confidence:.1%}"
        
        # Check if prediction matches expected
        is_correct = prediction_text == test_case['expected']
        if is_correct:
            correct_predictions += 1
            status = "✅"
            status_color = Fore.GREEN
        else:
            status = "❌"
            status_color = Fore.RED
        
        print(f"{status_color}{test_case['extension']:<15} {test_case['size_kb']:<10} "
              f"{test_case['filename']:<20} {prediction_text:<12} {confidence_text:<12} "
              f"{test_case['expected']:<12} {status:<8}{Style.RESET_ALL}")
    
    print_colored(f"\n📊 Enhanced Test Summary:", Fore.CYAN)
    print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"   Test accuracy: {correct_predictions/total_predictions:.1%}")


def main():
    """Main function to run the enhanced training process."""
    parser = argparse.ArgumentParser(description="AI Antivirus v4.X Model Training")
    parser.add_argument('--model-info', action='store_true', 
                       help='Print model information and exit')
    parser.add_argument('--create-data-only', action='store_true',
                       help='Create training data only, skip training')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only')
    
    args = parser.parse_args()
    
    if args.model_info:
        # Print model information
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print_model_info(model, model.feature_names_in_)
        else:
            print_colored("❌ No trained model found. Run training first.", Fore.RED)
        return
    
    if args.test_only:
        # Test existing model
        if MODEL_PATH.exists():
            test_enhanced_model()
        else:
            print_colored("❌ No trained model found. Run training first.", Fore.RED)
        return
    
    print_colored("🛡️ AI Antivirus v4.X Enhanced Model Training", Fore.CYAN)
    print("=" * 60)
    
    # Create enhanced training data
    training_data = create_enhanced_training_data()
    
    if args.create_data_only:
        print_colored("✅ Training data created. Skipping model training.", Fore.GREEN)
        return
    
    # Train enhanced model with validation
    model, test_metrics = train_enhanced_model()
    
    # Test enhanced model
    test_enhanced_model()
    
    print_colored("\n🎉 Enhanced training process complete!", Fore.GREEN)
    print_colored("📖 Next steps:", Fore.CYAN)
    print(f"   1. Run: python3 ai_antivirus.py --path /path/to/monitor")
    print(f"   2. Test with: python3 ai_antivirus.py --scan-only")
    print(f"   3. Retrain with: python3 ai_antivirus.py --retrain")
    print(f"   4. Check logs/model_metrics_*.txt for detailed metrics")
    print(f"   5. View confusion matrix plots in logs/ directory")


if __name__ == "__main__":
    main()