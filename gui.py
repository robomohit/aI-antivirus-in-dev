#!/usr/bin/env python3
"""
AI Antivirus Scanner GUI
Tkinter interface for the Ultimate AI Antivirus system
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import queue
import os
import sys
import csv
from pathlib import Path
from datetime import datetime
import re

class AIAntivirusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Antivirus Scanner v1.0")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Variables
        self.scan_process = None
        self.scan_thread = None
        self.log_queue = queue.Queue()
        self.threat_count = 0
        self.scan_running = False
        
        # Configure style
        self.setup_styles()
        
        # Create GUI components
        self.create_widgets()
        
        # Start log monitoring
        self.monitor_logs()
        
        # Center window
        self.center_window()
    
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Scan.TButton', 
                       font=('Arial', 10, 'bold'),
                       padding=10)
        style.configure('Action.TButton',
                       font=('Arial', 9),
                       padding=5)
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🛡️ AI Antivirus Scanner v1.0",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Scan buttons frame
        scan_frame = ttk.LabelFrame(main_frame, text="Scan Options", padding="10")
        scan_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        scan_frame.columnconfigure(0, weight=1)
        scan_frame.columnconfigure(1, weight=1)
        scan_frame.columnconfigure(2, weight=1)
        
        # Smart Scan button
        self.smart_scan_btn = ttk.Button(scan_frame,
                                        text="🧠 Smart Scan",
                                        command=self.start_smart_scan,
                                        style='Scan.TButton')
        self.smart_scan_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Full Scan button
        self.full_scan_btn = ttk.Button(scan_frame,
                                       text="🔍 Full Scan",
                                       command=self.start_full_scan,
                                       style='Scan.TButton')
        self.full_scan_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Stop Scan button
        self.stop_scan_btn = ttk.Button(scan_frame,
                                       text="⏹️ Stop Scan",
                                       command=self.stop_scan,
                                       style='Scan.TButton',
                                       state='disabled')
        self.stop_scan_btn.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Action buttons frame
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # View Known Malware button
        self.view_malware_btn = ttk.Button(action_frame,
                                          text="📋 View Known Malware",
                                          command=self.view_known_malware,
                                          style='Action.TButton')
        self.view_malware_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Open Quarantine button
        self.quarantine_btn = ttk.Button(action_frame,
                                        text="🗂️ Open Quarantine",
                                        command=self.open_quarantine,
                                        style='Action.TButton')
        self.quarantine_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Clear Logs button
        self.clear_logs_btn = ttk.Button(action_frame,
                                        text="🗑️ Clear Logs",
                                        command=self.clear_logs,
                                        style='Action.TButton')
        self.clear_logs_btn.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Log display frame
        log_frame = ttk.LabelFrame(main_frame, text="Scan Logs", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                 height=15,
                                                 font=('Consolas', 9),
                                                 wrap=tk.WORD,
                                                 bg='black',
                                                 fg='white',
                                                 insertbackground='white')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        status_frame.columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Status: Idle")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Threat count label
        self.threat_label = ttk.Label(status_frame, text="Threats: 0")
        self.threat_label.grid(row=0, column=2, sticky=tk.E)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame,
                                           variable=self.progress_var,
                                           mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
    
    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def start_smart_scan(self):
        """Start a smart scan."""
        self.start_scan("--smart-scan", "Smart Scan")
    
    def start_full_scan(self):
        """Start a full scan."""
        result = messagebox.askyesno("Full Scan Warning",
                                   "Full scan will scan your entire system.\n"
                                   "This may take a long time.\n\n"
                                   "Do you want to continue?")
        if result:
            self.start_scan("--full-scan", "Full Scan")
    
    def start_scan(self, scan_type, scan_name):
        """Start a scan with the specified type."""
        if self.scan_running:
            messagebox.showwarning("Scan Running", "A scan is already in progress.")
            return
        
        # Reset counters
        self.threat_count = 0
        self.threat_label.config(text="Threats: 0")
        
        # Update UI
        self.scan_running = True
        self.status_label.config(text=f"Status: {scan_name} in progress...")
        self.smart_scan_btn.config(state='disabled')
        self.full_scan_btn.config(state='disabled')
        self.stop_scan_btn.config(state='normal')
        self.progress_bar.start()
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"🚀 Starting {scan_name}...\n")
        self.log_text.insert(tk.END, f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_text.insert(tk.END, "=" * 50 + "\n\n")
        self.log_text.see(tk.END)
        
        # Start scan in separate thread
        self.scan_thread = threading.Thread(target=self.run_scan, args=(scan_type, scan_name))
        self.scan_thread.daemon = True
        self.scan_thread.start()
    
    def run_scan(self, scan_type, scan_name):
        """Run the scan in a separate thread."""
        try:
            # Determine Python executable
            python_exe = sys.executable
            
            # Build command
            cmd = [python_exe, "ai_antivirus.py", scan_type]
            
            # Start process
            self.scan_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output in real-time
            for line in iter(self.scan_process.stdout.readline, ''):
                if line:
                    # Put line in queue for GUI thread
                    self.log_queue.put(line.strip())
                    
                    # Check for threat detection
                    if self.detect_threat_in_line(line):
                        self.threat_count += 1
            
            # Wait for process to complete
            return_code = self.scan_process.wait()
            
            # Send completion message
            if return_code == 0:
                self.log_queue.put(f"\n✅ {scan_name} completed successfully!")
            else:
                self.log_queue.put(f"\n❌ {scan_name} completed with errors (code: {return_code})")
            
        except Exception as e:
            self.log_queue.put(f"\n❌ Error during scan: {str(e)}")
        finally:
            # Signal completion
            self.log_queue.put("SCAN_COMPLETE")
    
    def detect_threat_in_line(self, line):
        """Detect threat mentions in log line."""
        threat_indicators = [
            "🔥 CRITICAL",
            "⚠️ HIGH RISK", 
            "🟡 SUSPICIOUS",
            "🚨 THREAT DETECTED",
            "🦠 Malware detected",
            "🧠 Known malware detected",
            "🛡️ Quarantined"
        ]
        
        return any(indicator in line for indicator in threat_indicators)
    
    def stop_scan(self):
        """Stop the current scan."""
        if self.scan_process and self.scan_running:
            try:
                self.scan_process.terminate()
                self.log_queue.put("\n⏹️ Scan stopped by user")
            except:
                pass
            finally:
                self.scan_complete()
    
    def scan_complete(self):
        """Handle scan completion."""
        self.scan_running = False
        self.scan_process = None
        
        # Update UI
        self.status_label.config(text="Status: Scan Complete")
        self.smart_scan_btn.config(state='normal')
        self.full_scan_btn.config(state='normal')
        self.stop_scan_btn.config(state='disabled')
        self.progress_bar.stop()
        
        # Update threat count
        self.threat_label.config(text=f"Threats: {self.threat_count}")
        
        # Show completion message
        if self.threat_count > 0:
            messagebox.showinfo("Scan Complete", 
                              f"Scan completed!\n\n"
                              f"Threats detected: {self.threat_count}\n"
                              f"Check the quarantine folder for details.")
        else:
            messagebox.showinfo("Scan Complete", 
                              "Scan completed!\n\n"
                              "No threats detected.")
    
    def monitor_logs(self):
        """Monitor log queue and update GUI."""
        try:
            while True:
                try:
                    line = self.log_queue.get_nowait()
                    
                    if line == "SCAN_COMPLETE":
                        self.scan_complete()
                    else:
                        # Add line to log display
                        self.log_text.insert(tk.END, line + "\n")
                        self.log_text.see(tk.END)
                        
                        # Update threat count if needed
                        if self.detect_threat_in_line(line):
                            self.threat_label.config(text=f"Threats: {self.threat_count}")
                
                except queue.Empty:
                    break
        except:
            pass
        
        # Schedule next check
        self.root.after(100, self.monitor_logs)
    
    def view_known_malware(self):
        """Open known malware database."""
        malware_file = Path("known_malware.csv")
        if malware_file.exists():
            try:
                # Create a new window to display the data
                self.show_malware_window()
            except Exception as e:
                messagebox.showerror("Error", f"Could not open malware database: {str(e)}")
        else:
            messagebox.showinfo("No Data", "No known malware database found.")
    
    def show_malware_window(self):
        """Show known malware in a new window."""
        malware_window = tk.Toplevel(self.root)
        malware_window.title("Known Malware Database")
        malware_window.geometry("900x600")
        
        # Create treeview
        columns = ('Hash', 'Filename', 'Size (KB)', 'Extension', 'Method', 'Date')
        tree = ttk.Treeview(malware_window, columns=columns, show='headings')
        
        # Configure columns
        tree.heading('Hash', text='SHA-256 Hash')
        tree.heading('Filename', text='Filename')
        tree.heading('Size (KB)', text='Size (KB)')
        tree.heading('Extension', text='Extension')
        tree.heading('Method', text='Detection Method')
        tree.heading('Date', text='Date Detected')
        
        # Set column widths
        tree.column('Hash', width=200)
        tree.column('Filename', width=150)
        tree.column('Size (KB)', width=80)
        tree.column('Extension', width=80)
        tree.column('Method', width=100)
        tree.column('Date', width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(malware_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load data
        try:
            with open("known_malware.csv", 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    tree.insert('', tk.END, values=(
                        row.get('sha256_hash', '')[:16] + '...',
                        row.get('filename', ''),
                        row.get('size_kb', ''),
                        row.get('extension', ''),
                        row.get('method', ''),
                        row.get('date_detected', '')
                    ))
        except Exception as e:
            messagebox.showerror("Error", f"Could not load malware data: {str(e)}")
    
    def open_quarantine(self):
        """Open quarantine folder."""
        quarantine_path = Path("quarantine")
        if quarantine_path.exists():
            try:
                if sys.platform == "win32":
                    os.startfile(str(quarantine_path))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(quarantine_path)])
                else:
                    subprocess.run(["xdg-open", str(quarantine_path)])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open quarantine folder: {str(e)}")
        else:
            messagebox.showinfo("No Quarantine", "Quarantine folder does not exist.")
    
    def clear_logs(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.threat_count = 0
        self.threat_label.config(text="Threats: 0")
    
    def on_closing(self):
        """Handle window closing."""
        if self.scan_running:
            result = messagebox.askyesno("Scan Running", 
                                       "A scan is currently running.\n"
                                       "Do you want to stop the scan and exit?")
            if result:
                self.stop_scan()
            else:
                return
        
        self.root.destroy()

def main():
    """Main function to launch the GUI."""
    root = tk.Tk()
    app = AIAntivirusGUI(root)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()