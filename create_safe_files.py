#!/usr/bin/env python3
"""
Create 2,500 safe files for training dataset
"""

import random
import string
from pathlib import Path

# Safe file templates
SAFE_TEMPLATES = [
    'This is a safe document.',
    'Lorem ipsum dolor sit amet.',
    'Sample data for testing.',
    'Document content here.',
    'Safe file content.',
    'Test document.',
    'Sample text.',
    'Safe content.',
    'Document text.',
    'Test file content.'
]

SAFE_NAMES = [
    'document', 'report', 'resume', 'invoice', 'letter', 'memo',
    'presentation', 'spreadsheet', 'photo', 'image', 'video',
    'music', 'book', 'article', 'paper', 'thesis', 'manual',
    'guide', 'tutorial', 'notes', 'diary', 'journal', 'log',
    'data', 'backup', 'archive', 'config', 'settings', 'preferences'
]

SAFE_EXTENSIONS = ['.txt', '.doc', '.docx', '.pdf', '.csv', '.xls', '.xlsx',
                   '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi',
                   '.html', '.css', '.js', '.json', '.xml', '.zip', '.rar']

def create_safe_files(count=2500):
    """Create safe files quickly."""
    safe_dir = Path("test_files/safe")
    safe_dir.mkdir(exist_ok=True)
    
    print(f"✅ Creating {count} safe files...")
    
    for i in range(count):
        name = random.choice(SAFE_NAMES)
        ext = random.choice(SAFE_EXTENSIONS)
        suffix = ''.join(random.choices(string.digits, k=3))
        filename = f"{name}_{suffix}{ext}"
        
        # Generate content
        content_lines = []
        num_lines = random.randint(10, 50)
        
        for _ in range(num_lines):
            template = random.choice(SAFE_TEMPLATES)
            content_lines.append(template)
        
        content = '\n'.join(content_lines)
        
        file_path = safe_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        
        if (i + 1) % 500 == 0:
            print(f"  ✅ Created {i + 1}/{count}")
    
    print(f"🎯 Created {count} safe files")

if __name__ == "__main__":
    create_safe_files()