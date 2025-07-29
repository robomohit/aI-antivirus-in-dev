#!/usr/bin/env python3
"""
Training Data Generator and Model Trainer for AI Antivirus
Creates dummy training data and trains the Random Forest model.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)


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
        for _ in range(15):  # 15 samples per safe extension
            size = np.random.randint(min_size, max_size)
            data.append({
                'file_extension': ext,
                'file_size_kb': size,
                'is_malicious': 0
            })
    
    # Generate suspicious files
    print(f"{Fore.RED}⚠️ Generating suspicious files...{Style.RESET_ALL}")
    for ext, (min_size, max_size) in suspicious_extensions.items():
        for _ in range(12):  # 12 samples per suspicious extension
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
        for _ in range(5):
            data.append({
                'file_extension': ext,
                'file_size_kb': np.random.randint(1, 10),
                'is_malicious': 1
            })
    
    # Large safe files (potential false positives)
    for ext in ['.pdf', '.mp4', '.doc']:
        for _ in range(5):
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


def train_model(training_data_path="model/training_data.csv", model_path="model/model.pkl"):
    """Train the Random Forest model on the training data."""
    print(f"{Fore.CYAN}🧠 Training AI model...{Style.RESET_ALL}")
    
    # Load training data
    df = pd.read_csv(training_data_path)
    
    # Prepare features
    # Convert extensions to numerical features (one-hot encoding)
    extension_dummies = pd.get_dummies(df['file_extension'], prefix='ext')
    
    # Combine features
    X = pd.concat([extension_dummies, df[['file_size_kb']]], axis=1)
    y = df['is_malicious']
    
    print(f"{Fore.BLUE}📊 Feature matrix shape: {X.shape}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📊 Target vector shape: {y.shape}{Style.RESET_ALL}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"{Fore.GREEN}📈 Training set size: {len(X_train)}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}📊 Test set size: {len(X_test)}{Style.RESET_ALL}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print(f"{Fore.YELLOW}🔄 Training Random Forest model...{Style.RESET_ALL}")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{Fore.GREEN}✅ Model training complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Model Performance:{Style.RESET_ALL}")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    print(f"   Test samples: {len(y_test)}")
    
    # Detailed classification report
    print(f"\n{Fore.MAGENTA}📋 Classification Report:{Style.RESET_ALL}")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Suspicious']))
    
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
    
    return model, accuracy


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
    ]
    
    print(f"{Fore.YELLOW}📋 Test Results:{Style.RESET_ALL}")
    print(f"{'File Type':<15} {'Size (KB)':<10} {'Prediction':<12} {'Confidence':<12} {'Expected':<12}")
    print("-" * 70)
    
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
        
        # Color coding
        if prediction_text == test_case['expected']:
            status_color = Fore.GREEN
        else:
            status_color = Fore.RED
        
        print(f"{status_color}{test_case['extension']:<15} {test_case['size_kb']:<10} "
              f"{prediction_text:<12} {confidence_text:<12} {test_case['expected']:<12}{Style.RESET_ALL}")


def main():
    """Main function to run the training process."""
    print(f"{Fore.CYAN}🛡️ AI Antivirus Model Training{Style.RESET_ALL}")
    print("=" * 50)
    
    # Create training data
    training_data = create_training_data()
    
    # Train model
    model, accuracy = train_model()
    
    # Test model
    test_model()
    
    print(f"\n{Fore.GREEN}🎉 Training process complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📖 Next steps:{Style.RESET_ALL}")
    print(f"   1. Run: python3 ai_antivirus.py --path /path/to/monitor")
    print(f"   2. Test with: python3 ai_antivirus.py --scan-only")
    print(f"   3. Retrain with: python3 ai_antivirus.py --retrain")


if __name__ == "__main__":
    main()