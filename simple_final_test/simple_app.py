
import tkinter as tk
from tkinter import messagebox

class SimpleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Application")
        self.root.geometry("300x200")
        
        self.label = tk.Label(self.root, text="Hello, World!")
        self.label.pack(pady=20)
        
        self.button = tk.Button(self.root, text="Click Me!", command=self.show_message)
        self.button.pack(pady=10)
        
    def show_message(self):
        messagebox.showinfo("Info", "This is a simple application!")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleApp()
    app.run()
