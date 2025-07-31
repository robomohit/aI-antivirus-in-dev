
import tkinter as tk
from tkinter import filedialog, messagebox

class TextEditor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Simple Text Editor")
        self.text_area = tk.Text(self.window, width=80, height=25)
        self.text_area.pack()
        
    def new_file(self):
        self.text_area.delete(1.0, tk.END)
        
    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    self.text_area.delete(1.0, tk.END)
                    self.text_area.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
                
    def save_file(self):
        file_path = filedialog.asksaveasfilename()
        if file_path:
            try:
                content = self.text_area.get(1.0, tk.END)
                with open(file_path, 'w') as file:
                    file.write(content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
                
    def run(self):
        # Create menu
        menubar = tk.Menu(self.window)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        menubar.add_cascade(label="File", menu=file_menu)
        self.window.config(menu=menubar)
        
        self.window.mainloop()
        
if __name__ == "__main__":
    editor = TextEditor()
    editor.run()
