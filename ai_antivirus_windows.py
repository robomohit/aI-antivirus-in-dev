#!/usr/bin/env python3
"""
Ultimate AI Antivirus v5.X - Windows Optimized Version
Enhanced Security Agent with AI Integration and GUI Support
"""

import os
import sys
import argparse
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from pathlib import Path
import time

def safe_print(text: str):
    """Safe print function for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

def run_antivirus_command(command_args):
    """Run the antivirus with given arguments."""
    try:
        # Use the main antivirus script
        cmd = ['python', 'ai_antivirus.py'] + command_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

class AntivirusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate AI Antivirus v5.X")
        self.root.geometry("800x600")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.scan_running = False
        self.output_queue = queue.Queue()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the GUI interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Ultimate AI Antivirus v5.X", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Scan buttons frame
        button_frame = ttk.LabelFrame(main_frame, text="Scan Options", padding="10")
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Scan buttons
        ttk.Button(button_frame, text="Smart Scan", 
                  command=self.smart_scan).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="Full Scan", 
                  command=self.full_scan).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Custom Path", 
                  command=self.custom_scan).grid(row=0, column=2, padx=5, pady=5)
        
        # Utility buttons
        util_frame = ttk.LabelFrame(main_frame, text="Utilities", padding="10")
        util_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(util_frame, text="View Known Malware", 
                  command=self.view_known_malware).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(util_frame, text="Open Quarantine", 
                  command=self.open_quarantine).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(util_frame, text="Clear Logs", 
                  command=self.clear_logs).grid(row=0, column=2, padx=5, pady=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Scan Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Scan Output", padding="10")
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
    def smart_scan(self):
        """Run smart scan."""
        if self.scan_running:
            messagebox.showwarning("Scan Running", "A scan is already in progress!")
            return
        
        self.start_scan("Smart Scan", ["--smart-scan"])
        
    def full_scan(self):
        """Run full scan."""
        if self.scan_running:
            messagebox.showwarning("Scan Running", "A scan is already in progress!")
            return
        
        # Confirm full scan
        if not messagebox.askyesno("Full Scan Warning", 
                                 "Full scan will scan your entire system. This may take a long time. Continue?"):
            return
        
        self.start_scan("Full Scan", ["--full-scan"])
        
    def custom_scan(self):
        """Run custom path scan."""
        if self.scan_running:
            messagebox.showwarning("Scan Running", "A scan is already in progress!")
            return
        
        # Simple path input dialog
        path = tk.simpledialog.askstring("Custom Path", "Enter path to scan:")
        if path:
            self.start_scan("Custom Scan", ["--path", path])
        
    def start_scan(self, scan_type, args):
        """Start a scan in a separate thread."""
        self.scan_running = True
        self.progress_var.set(f"{scan_type} in progress...")
        self.progress_bar.start()
        self.status_var.set(f"Running {scan_type}")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Starting {scan_type}...\n")
        
        # Start scan thread
        scan_thread = threading.Thread(target=self.run_scan_thread, args=(args,))
        scan_thread.daemon = True
        scan_thread.start()
        
        # Start output monitoring
        self.root.after(100, self.check_output)
        
    def run_scan_thread(self, args):
        """Run scan in separate thread."""
        try:
            returncode, stdout, stderr = run_antivirus_command(args)
            self.output_queue.put(("complete", returncode, stdout, stderr))
        except Exception as e:
            self.output_queue.put(("error", 1, "", str(e)))
        
    def check_output(self):
        """Check for output from scan thread."""
        try:
            while True:
                msg_type, *data = self.output_queue.get_nowait()
                
                if msg_type == "complete":
                    returncode, stdout, stderr = data
                    self.scan_complete(returncode, stdout, stderr)
                    return
                elif msg_type == "error":
                    error_msg = data[0]
                    self.scan_error(error_msg)
                    return
                    
        except queue.Empty:
            # Continue monitoring
            self.root.after(100, self.check_output)
            
    def scan_complete(self, returncode, stdout, stderr):
        """Handle scan completion."""
        self.scan_running = False
        self.progress_bar.stop()
        
        if returncode == 0:
            self.progress_var.set("Scan completed successfully")
            self.status_var.set("Scan completed")
        else:
            self.progress_var.set("Scan completed with errors")
            self.status_var.set("Scan failed")
        
        # Display output
        if stdout:
            self.output_text.insert(tk.END, stdout)
        if stderr:
            self.output_text.insert(tk.END, f"\nErrors:\n{stderr}")
        
        self.output_text.see(tk.END)
        
    def scan_error(self, error_msg):
        """Handle scan error."""
        self.scan_running = False
        self.progress_bar.stop()
        self.progress_var.set("Scan failed")
        self.status_var.set("Error occurred")
        self.output_text.insert(tk.END, f"Error: {error_msg}\n")
        
    def view_known_malware(self):
        """Open known malware database."""
        try:
            if os.path.exists("known_malware.csv"):
                os.startfile("known_malware.csv")
            else:
                messagebox.showinfo("Info", "No known malware database found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open known malware database: {e}")
            
    def open_quarantine(self):
        """Open quarantine folder."""
        try:
            quarantine_path = Path("quarantine")
            if quarantine_path.exists():
                os.startfile(str(quarantine_path))
            else:
                messagebox.showinfo("Info", "No quarantine folder found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open quarantine folder: {e}")
            
    def clear_logs(self):
        """Clear log output."""
        if messagebox.askyesno("Clear Logs", "Clear the output log?"):
            self.output_text.delete(1.0, tk.END)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ultimate AI Antivirus v5.X")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--smart-scan", action="store_true", help="Run smart scan")
    parser.add_argument("--full-scan", action="store_true", help="Run full scan")
    parser.add_argument("--path", type=str, help="Scan specific path")
    parser.add_argument("--scan-only", action="store_true", help="Run scan and exit")
    
    args = parser.parse_args()
    
    if args.gui or not any([args.smart_scan, args.full_scan, args.path, args.scan_only]):
        # Launch GUI
        try:
            import tkinter.simpledialog
            root = tk.Tk()
            app = AntivirusGUI(root)
            root.mainloop()
        except ImportError:
            safe_print("GUI not available. Running in console mode...")
            # Fallback to console mode
            run_console_mode()
    else:
        # Run command line mode
        command_args = []
        if args.smart_scan:
            command_args.append("--smart-scan")
        elif args.full_scan:
            command_args.append("--full-scan")
        elif args.path:
            command_args.extend(["--path", args.path])
        elif args.scan_only:
            command_args.append("--scan-only")
            
        returncode, stdout, stderr = run_antivirus_command(command_args)
        
        if stdout:
            safe_print(stdout)
        if stderr:
            safe_print(f"Errors: {stderr}")
            
        return returncode

def run_console_mode():
    """Run console-based interface."""
    safe_print("Ultimate AI Antivirus v5.X - Console Mode")
    safe_print("=" * 50)
    
    while True:
        safe_print("\nChoose scan mode:")
        safe_print("1. Smart Scan")
        safe_print("2. Full Scan")
        safe_print("3. Custom Path")
        safe_print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                returncode, stdout, stderr = run_antivirus_command(["--smart-scan"])
            elif choice == "2":
                returncode, stdout, stderr = run_antivirus_command(["--full-scan"])
            elif choice == "3":
                path = input("Enter path to scan: ").strip()
                if path:
                    returncode, stdout, stderr = run_antivirus_command(["--path", path])
                else:
                    safe_print("Invalid path!")
                    continue
            elif choice == "4":
                safe_print("Goodbye!")
                break
            else:
                safe_print("Invalid choice!")
                continue
                
            if stdout:
                safe_print(stdout)
            if stderr:
                safe_print(f"Errors: {stderr}")
                
        except KeyboardInterrupt:
            safe_print("\nExiting...")
            break
        except Exception as e:
            safe_print(f"Error: {e}")

if __name__ == "__main__":
    sys.exit(main())