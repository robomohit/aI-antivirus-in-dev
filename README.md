# 🛡️ Simple Antivirus in Python

A basic file monitoring system that detects suspicious files and logs them for security purposes.

## 🎯 Features

- **Background File Monitoring**: Uses `watchdog` to monitor directories in real-time
- **Suspicious File Detection**: Detects files with potentially dangerous extensions
- **Comprehensive Logging**: Logs all detections with timestamps to `logs/` folder
- **Quarantine System**: Optionally moves suspicious files to `quarantine/` folder
- **Graceful Shutdown**: Handles Ctrl+C interruption properly
- **Command Line Interface**: Flexible options for different use cases

## 📦 Installation

1. **Clone or download the files**:
   ```bash
   # Make sure you have the simple_antivirus.py and requirements.txt files
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage

Monitor the default `Downloads` folder:
```bash
python simple_antivirus.py
```

### Advanced Usage

Monitor a specific directory:
```bash
python simple_antivirus.py --path /path/to/monitor
```

Disable quarantine (only log detections):
```bash
python simple_antivirus.py --no-quarantine
```

Scan existing files only (no real-time monitoring):
```bash
python simple_antivirus.py --scan-only
```

Combine options:
```bash
python simple_antivirus.py --path ~/Downloads --no-quarantine --scan-only
```

## 📁 File Structure

The antivirus creates the following structure:
```
.
├── simple_antivirus.py    # Main antivirus script
├── requirements.txt       # Python dependencies
├── logs/                 # Log files (created automatically)
│   └── antivirus_YYYYMMDD_HHMMSS.log
└── quarantine/           # Quarantined files (if enabled)
    └── suspicious_file_YYYYMMDD_HHMMSS.ext
```

## 🚨 Suspicious File Extensions

The antivirus monitors for these potentially dangerous file types:
- `.exe` - Executable files
- `.bat` - Batch files
- `.vbs` - Visual Basic scripts
- `.scr` - Screen saver files (can be malicious)
- `.ps1` - PowerShell scripts
- `.cmd` - Command files
- `.com` - Command files
- `.pif` - Program information files
- `.reg` - Registry files
- `.js` - JavaScript files
- `.jar` - Java archive files
- `.msi` - Microsoft installer files
- `.dll` - Dynamic link libraries
- `.sys` - System files

## 📊 Logging

All detections are logged with:
- Timestamp
- File path
- File size
- Last modified date
- Action taken (quarantined or logged)

Example log entry:
```
2024-01-15 14:30:25,123 - WARNING - 🚨 SUSPICIOUS FILE DETECTED: /path/to/suspicious.exe
2024-01-15 14:30:25,124 - INFO - 📊 File size: 1024 bytes
2024-01-15 14:30:25,125 - INFO - 🕒 Last modified: 2024-01-15 14:30:20
2024-01-15 14:30:25,126 - INFO - 🚫 Quarantined: /path/to/suspicious.exe -> quarantine/suspicious_20240115_143025.exe
```

## 🛡️ Quarantine System

When quarantine is enabled:
- Suspicious files are moved to the `quarantine/` folder
- Original filename is preserved with timestamp suffix
- Files can be manually reviewed and restored if needed

## ⚠️ Important Notes

1. **Educational Purpose**: This is a learning tool, not a replacement for professional antivirus software
2. **False Positives**: Legitimate files with these extensions will be flagged
3. **No AI/ML**: This uses simple extension-based detection (Phase 2 will add AI)
4. **Safe Operation**: No actual malware is used or distributed

## 🔧 Customization

### Adding New Suspicious Extensions

Edit the `suspicious_extensions` set in the `SimpleAntivirus` class:

```python
self.suspicious_extensions = {
    '.exe', '.bat', '.vbs', '.scr', '.ps1', '.cmd', '.com', 
    '.pif', '.reg', '.js', '.jar', '.msi', '.dll', '.sys',
    '.your_extension'  # Add your custom extensions here
}
```

### Changing Log Format

Modify the logging configuration in the `_setup_logging` method.

## 🚀 Future Enhancements (Phase 2)

- AI/ML-based detection
- File hash checking
- Network-based threat intelligence
- Real-time virus signature updates
- GUI interface
- System tray integration

## 📝 License

This project is for educational purposes only.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**⚠️ Disclaimer**: This tool is for educational purposes only. It does not provide comprehensive protection against all types of malware and should not be used as a replacement for professional antivirus software.