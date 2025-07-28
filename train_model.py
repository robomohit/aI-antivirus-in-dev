#!/usr/bin/env python3
"""
Training Data Generator and Model Trainer for AI Antivirus
Creates dummy training data and trains the Random Forest model with proper validation.
"""

import pandas as pd
import numpy as np
import pickle
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

# Initialize colorama
colorama.init(autoreset=True)


def setup_logging():
    """Setup logging for model training metrics."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"model_metrics_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def create_training_data(output_path="model/training_data.csv"):
    """Create comprehensive dummy training data for the AI model."""
    print(f"{Fore.CYAN}📊 Creating training data...{Style.RESET_ALL}")
    
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
    
    # Generate safe files
    print(f"{Fore.GREEN}✅ Generating safe files...{Style.RESET_ALL}")
    for ext, (min_size, max_size) in safe_extensions.items():
        for _ in range(20):  # Increased samples for better training
            size = np.random.randint(min_size, max_size)
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 0
            })
    
    # Generate suspicious files
    print(f"{Fore.RED}⚠️ Generating suspicious files...{Style.RESET_ALL}")
    for ext, (min_size, max_size) in suspicious_extensions.items():
        for _ in range(18):  # Increased samples for better training
            size = np.random.randint(min_size, max_size)
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 1
            })
    
    # Add some edge cases
    print(f"{Fore.YELLOW}🔍 Adding edge cases...{Style.RESET_ALL}")
    
    # Small suspicious files (potential false negatives)
    for ext in ['.exe', '.bat', '.vbs', '.ps1']:
        for _ in range(8):
            data.append({
                'file_extension': ext,
                'file_size_kb': np.random.randint(1, 10),
                'is_malicious': 1
            })
    
    # Large safe files (potential false positives)
    for ext in ['.pdf', '.mp4', '.doc']:
        for _ in range(8):
            data.append({
                'file_extension': ext,
                'file_size_kb': np.random.randint(10000, 50000),
                'is_malicious': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save training data
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"{Fore.GREEN}📊 Training data created: {output_path}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 Dataset statistics:{Style.RESET_ALL}")
    print(f"   Total samples: {len(df)}")
    print(f"   Safe files: {len(df[df['is_malicious'] == 0])}")
    print(f"   Suspicious files: {len(df[df['is_malicious'] == 1])}")
    print(f"   Unique extensions: {df['file_extension'].nunique()}")
    
    return df


def prepare_features(df):
    """Prepare features for model training."""
    # Convert extensions to numerical features (one-hot encoding)
    extension_dummies = pd.get_dummies(df['file_extension'], prefix='ext')
    
    # Combine features
    X = pd.concat([extension_dummies, df[['file_size_kb']]], axis=1)
    y = df['is_malicious']
    
    return X, y


def evaluate_model(model, X_test, y_test, y_pred):
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
        'confusion_matrix': cm
    }


def print_metrics(metrics, split_name="Test"):
    """Print formatted metrics."""
    print(f"\n{Fore.CYAN}📊 {split_name} Set Performance:{Style.RESET_ALL}")
    print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']:.1%})")
    print(f"   Recall:    {metrics['recall']:.3f} ({metrics['recall']:.1%})")
    print(f"   F1-Score:  {metrics['f1_score']:.3f} ({metrics['f1_score']:.1%})")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    print(f"\n{Fore.MAGENTA}📋 Confusion Matrix ({split_name}):{Style.RESET_ALL}")
    print(f"                Predicted")
    print(f"                Safe  Suspicious")
    print(f"Actual Safe     {cm[0,0]:4d} {cm[0,1]:10d}")
    print(f"      Suspicious {cm[1,0]:4d} {cm[1,1]:10d}")


def log_metrics(metrics, split_name, log_file):
    """Log metrics to file."""
    logging.info(f"=== {split_name} SET METRICS ===")
    logging.info(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    logging.info(f"Precision: {metrics['precision']:.3f} ({metrics['precision']:.1%})")
    logging.info(f"Recall: {metrics['recall']:.3f} ({metrics['recall']:.1%})")
    logging.info(f"F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']:.1%})")
    logging.info(f"Confusion Matrix:")
    logging.info(f"{metrics['confusion_matrix']}")


def check_overfitting(train_metrics, test_metrics):
    """Check for overfitting by comparing train and test metrics."""
    print(f"\n{Fore.YELLOW}🔍 Overfitting Analysis:{Style.RESET_ALL}")
    
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
        print(f"{Fore.RED}⚠️  WARNING: Potential overfitting detected!{Style.RESET_ALL}")
        print(f"   Consider reducing model complexity or adding more training data.")
    elif accuracy_diff > 0.02 or f1_diff > 0.02:
        print(f"{Fore.YELLOW}⚠️  Minor overfitting detected.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✅ No significant overfitting detected.{Style.RESET_ALL}")


def train_model(training_data_path="model/training_data.csv", model_path="model/model.pkl"):
    """Train the Random Forest model with proper validation."""
    print(f"{Fore.CYAN}🧠 Training AI model with validation...{Style.RESET_ALL}")
    
    # Setup logging
    log_file = setup_logging()
    logging.info("=== AI ANTIVIRUS MODEL TRAINING ===")
    logging.info(f"Training started at: {datetime.now()}")
    
    # Load training data
    df = pd.read_csv(training_data_path)
    logging.info(f"Loaded {len(df)} training samples")
    
    # Prepare features
    X, y = prepare_features(df)
    
    print(f"{Fore.BLUE}📊 Feature matrix shape: {X.shape}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📊 Target vector shape: {y.shape}{Style.RESET_ALL}")
    
    # Split data with 75% train / 25% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y
    )
    
    print(f"{Fore.GREEN}📈 Training set size: {len(X_train)} (75%){Style.RESET_ALL}")
    print(f"{Fore.GREEN}📊 Test set size: {len(X_test)} (25%){Style.RESET_ALL}")
    
    logging.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print(f"{Fore.YELLOW}🔄 Training Random Forest model...{Style.RESET_ALL}")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_model(model, X_train, y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_model(model, X_test, y_test, y_test_pred)
    
    # Print results
    print_metrics(train_metrics, "Training")
    print_metrics(test_metrics, "Test")
    
    # Log metrics
    log_metrics(train_metrics, "TRAINING", log_file)
    log_metrics(test_metrics, "TEST", log_file)
    
    # Check for overfitting
    check_overfitting(train_metrics, test_metrics)
    
    # Detailed classification report
    print(f"\n{Fore.MAGENTA}📋 Detailed Classification Report (Test Set):{Style.RESET_ALL}")
    print(classification_report(y_test, y_test_pred, target_names=['Safe', 'Suspicious']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{Fore.BLUE}🔍 Top 10 Most Important Features:{Style.RESET_ALL}")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<20} {row['importance']:.3f}")
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"{Fore.GREEN}💾 Model saved to: {model_path}{Style.RESET_ALL}")
    logging.info(f"Model saved to: {model_path}")
    
    # Final summary
    print(f"\n{Fore.GREEN}🎉 Model Training Complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Final Test Accuracy: {test_metrics['accuracy']:.1%}{Style.RESET_ALL}")
    
    return model, test_metrics


def test_model(model_path="model/model.pkl"):
    """Test the trained model with sample files."""
    print(f"{Fore.CYAN}🧪 Testing model with sample files...{Style.RESET_ALL}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Test cases
    test_cases = [
        {'extension': '.exe', 'size_kb': 500, 'expected': 'Suspicious'},
        {'extension': '.txt', 'size_kb': 10, 'expected': 'Safe'},
        {'extension': '.bat', 'size_kb': 5, 'expected': 'Suspicious'},
        {'extension': '.pdf', 'size_kb': 2000, 'expected': 'Safe'},
        {'extension': '.vbs', 'size_kb': 50, 'expected': 'Suspicious'},
        {'extension': '.jpg', 'size_kb': 100, 'expected': 'Safe'},
        {'extension': '.ps1', 'size_kb': 150, 'expected': 'Suspicious'},
        {'extension': '.dll', 'size_kb': 2000, 'expected': 'Suspicious'},
    ]
    
    print(f"{Fore.YELLOW}📋 Test Results:{Style.RESET_ALL}")
    print(f"{'File Type':<15} {'Size (KB)':<10} {'Prediction':<12} {'Confidence':<12} {'Expected':<12} {'Status':<8}")
    print("-" * 75)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for test_case in test_cases:
        # Create feature vector
        feature_vector = np.zeros(len(model.feature_names_in_))
        
        # Set file size
        size_idx = np.where(model.feature_names_in_ == 'file_size_kb')[0]
        if len(size_idx) > 0:
            feature_vector[size_idx[0]] = test_case['size_kb']
        
        # Set extension
        ext_prefix = 'ext_'
        for i, feature_name in enumerate(model.feature_names_in_):
            if feature_name.startswith(ext_prefix):
                if feature_name == f'ext_{test_case["extension"]}':
                    feature_vector[i] = 1
        
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
              f"{prediction_text:<12} {confidence_text:<12} {test_case['expected']:<12} {status:<8}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}📊 Test Summary:{Style.RESET_ALL}")
    print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"   Test accuracy: {correct_predictions/total_predictions:.1%}")


def main():
    """Main function to run the training process."""
    print(f"{Fore.CYAN}🛡️ AI Antivirus Model Training with Validation{Style.RESET_ALL}")
    print("=" * 60)
    
    # Create training data
    training_data = create_training_data()
    
    # Train model with validation
    model, test_metrics = train_model()
    
    # Test model
    test_model()
    
    print(f"\n{Fore.GREEN}🎉 Training process complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📖 Next steps:{Style.RESET_ALL}")
    print(f"   1. Run: python3 ai_antivirus.py --path /path/to/monitor")
    print(f"   2. Test with: python3 ai_antivirus.py --scan-only")
    print(f"   3. Retrain with: python3 ai_antivirus.py --retrain")
    print(f"   4. Check logs/model_metrics_*.txt for detailed metrics")


if __name__ == "__main__":
    main()