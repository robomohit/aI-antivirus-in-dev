# 🧠 AI-Enhanced Antivirus in Python

A sophisticated file monitoring system that combines traditional extension-based detection with **machine learning** for intelligent threat detection.

## 🎯 **NEW AI FEATURES**

### 🤖 **Machine Learning Detection**
- **Random Forest Classifier** trained on 413 file samples
- **92.8% accuracy** on test data
- **Feature-based analysis**: file extension, file size, creation patterns
- **Confidence scoring** for each prediction

### 🔍 **Smart Detection Methods**
- **EXTENSION**: Traditional extension-based detection
- **AI**: Machine learning prediction
- **BOTH**: Detected by both methods (highest threat level)
- **SAFE**: Passed all checks

### 📊 **Enhanced Logging**
- **Detection method** (Extension/AI/Both)
- **AI confidence score** (0-100%)
- **File characteristics** (size, extension, timestamp)
- **Color-coded alerts** in terminal

## 🚀 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Train AI Model** (First time only)
```bash
python3 train_model.py
```

### 3. **Run AI Antivirus**
```bash
# Monitor Downloads folder
python3 ai_antivirus.py

# Monitor specific directory
python3 ai_antivirus.py --path /path/to/monitor

# Scan existing files only
python3 ai_antivirus.py --scan-only

# Retrain AI model
python3 ai_antivirus.py --retrain
```

## 📁 **File Structure**

```
.
├── ai_antivirus.py          # Main AI antivirus script
├── train_model.py           # AI model training script
├── ai_demo.py              # Interactive demo
├── requirements.txt         # Dependencies
├── README_AI.md           # This documentation
├── model/                 # AI model files
│   ├── model.pkl         # Trained Random Forest model
│   └── training_data.csv # Training dataset
├── logs/                 # Detection logs
│   └── ai_antivirus_YYYYMMDD_HHMMSS.log
└── quarantine/           # Suspicious files
    └── file_YYYYMMDD_HHMMSS.ext
```

## 🧠 **AI Model Details**

### **Training Data**
- **413 total samples**
- **225 safe files** (54.5%)
- **188 suspicious files** (45.5%)
- **27 unique file extensions**

### **Features Used**
1. **File Extension** (one-hot encoded)
2. **File Size** (in KB)

### **Model Performance**
- **Accuracy**: 92.8%
- **Precision**: 94% (Safe), 100% (Suspicious)
- **Recall**: 100% (Safe), 81% (Suspicious)

### **Top Features by Importance**
1. `file_size_kb` (0.199)
2. `ext_.bat` (0.093)
3. `ext_.ps1` (0.076)
4. `ext_.vbs` (0.073)
5. `ext_.exe` (0.064)

## 🚨 **Detection Examples**

### **High Threat (BOTH)**
```
🚨 SUSPICIOUS FILE DETECTED!
📁 File: /path/to/malware.exe
🔍 Detection: BOTH
📊 Size: 2500.0 KB
🧠 AI Confidence: 95.2%
```

### **AI-Only Detection**
```
🤖 SUSPICIOUS FILE DETECTED!
📁 File: /path/to/suspicious.js
🔍 Detection: AI
📊 Size: 150.0 KB
🧠 AI Confidence: 87.3%
```

### **Extension-Only Detection**
```
⚠️ SUSPICIOUS FILE DETECTED!
📁 File: /path/to/script.bat
🔍 Detection: EXTENSION
📊 Size: 5.0 KB
🧠 AI Confidence: 45.2%
```

## 📊 **Logging Format**

```log
2025-07-28 20:15:57,017 - WARNING - 🚨 SUSPICIOUS FILE DETECTED: /path/to/file.exe
2025-07-28 20:15:57,017 - INFO - 📊 File size: 2500.0 KB
2025-07-28 20:15:57,017 - INFO - 🔍 Detection method: BOTH
2025-07-28 20:15:57,017 - INFO - 🧠 AI confidence: 95.2%
2025-07-28 20:15:57,017 - INFO - 🕒 Last modified: 2025-07-28 20:15:53
2025-07-28 20:15:57,017 - INFO - 🚫 Quarantined: /path/to/file.exe -> quarantine/file_20250728_201557.exe
```

## 🎮 **Demo Scripts**

### **Interactive Demo**
```bash
python3 ai_demo.py
```

**Options:**
1. **Full AI antivirus demo** (training + monitoring)
2. **AI model analysis demo**
3. **Real-time monitoring demo**

### **Training Demo**
```bash
python3 train_model.py
```

**Features:**
- Creates comprehensive training data
- Trains Random Forest model
- Shows model performance metrics
- Tests model with sample files

## 🔧 **Advanced Usage**

### **Command Line Options**
```bash
# Basic monitoring
python3 ai_antivirus.py

# Monitor specific path
python3 ai_antivirus.py --path /custom/path

# Disable quarantine (log only)
python3 ai_antivirus.py --no-quarantine

# Scan existing files only
python3 ai_antivirus.py --scan-only

# Retrain AI model
python3 ai_antivirus.py --retrain
```

### **Custom Training Data**
Edit `train_model.py` to add new file types:

```python
# Add new safe extensions
safe_extensions = {
    '.txt': (1, 100),
    '.pdf': (50, 2000),
    '.your_ext': (min_size, max_size),  # Add your extension
}

# Add new suspicious extensions
suspicious_extensions = {
    '.exe': (100, 10000),
    '.bat': (1, 50),
    '.your_malicious_ext': (min_size, max_size),  # Add your extension
}
```

## 🎯 **Detection Logic**

### **Combined Detection Strategy**
1. **Extension Check**: Is file extension in suspicious list?
2. **AI Analysis**: Extract features and predict with ML model
3. **Combination**: Use both results for final decision

### **Detection Methods**
- **BOTH**: Extension + AI both flag as suspicious
- **EXTENSION**: Only extension-based detection
- **AI**: Only AI model flags as suspicious
- **SAFE**: Passes all checks

### **Confidence Scoring**
- **High (80-100%)**: Strong AI prediction
- **Medium (60-79%)**: Moderate confidence
- **Low (40-59%)**: Weak prediction
- **Very Low (<40%)**: Uncertain prediction

## 🛡️ **Security Features**

### **Quarantine System**
- **Timestamped filenames** prevent conflicts
- **Original metadata** preserved
- **Safe storage** in isolated folder
- **Manual review** capability

### **Logging System**
- **Comprehensive logs** with timestamps
- **Detection method** tracking
- **AI confidence** scores
- **File characteristics** recording

### **Real-time Monitoring**
- **File creation** detection
- **File movement** detection
- **Recursive directory** monitoring
- **Graceful shutdown** handling

## 🚀 **Performance**

### **Model Training**
- **Training time**: ~2-5 seconds
- **Model size**: ~50KB
- **Memory usage**: Minimal

### **Real-time Detection**
- **Response time**: <100ms per file
- **CPU usage**: Low
- **Memory footprint**: Small

## 🔮 **Future Enhancements**

### **Phase 3 Features**
- **Deep Learning** models (CNN, RNN)
- **File content analysis** (hex patterns, strings)
- **Network-based** threat intelligence
- **Behavioral analysis** (file operations)
- **Sandbox execution** for unknown files
- **Real-time model updates**

### **Advanced AI Features**
- **Anomaly detection** for unknown threats
- **Clustering** for threat families
- **Time-series analysis** for attack patterns
- **Multi-modal** analysis (extension + content + behavior)

## ⚠️ **Important Notes**

1. **Educational Purpose**: This is a learning tool, not production antivirus
2. **False Positives**: AI may flag legitimate files
3. **Training Data**: Uses synthetic data for demonstration
4. **Safe Operation**: No real malware used
5. **Model Limitations**: Accuracy depends on training data quality

## 📝 **Troubleshooting**

### **Common Issues**

**Model not loading:**
```bash
python3 train_model.py  # Retrain the model
```

**Dependencies missing:**
```bash
pip install -r requirements.txt --break-system-packages
```

**Permission errors:**
```bash
# Use a test directory instead of system folders
python3 ai_antivirus.py --path ./test_folder
```

### **Performance Tuning**

**For large directories:**
```bash
# Use scan-only mode for initial scan
python3 ai_antivirus.py --scan-only --path /large/directory
```

**For real-time monitoring:**
```bash
# Monitor specific subdirectories
python3 ai_antivirus.py --path /specific/folder
```

## 🤝 **Contributing**

Feel free to contribute:
- **New file types** for training data
- **Improved AI models**
- **Better feature extraction**
- **Enhanced logging**
- **Performance optimizations**

---

**🎉 Congratulations!** You now have a fully functional AI-enhanced antivirus system that combines traditional security with modern machine learning techniques.

**📚 Next Steps:**
1. Experiment with different file types
2. Customize the training data
3. Add new detection features
4. Explore advanced ML models
5. Build a GUI interface