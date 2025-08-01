
import tkinter as tk
from tkinter import filedialog, messagebox

class SimpleTextEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Text Editor")
        self.root.geometry("600x400")
        
        self.text_area = tk.Text(self.root, wrap=tk.WORD)
        self.text_area.pack(expand=True, fill='both')
        
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.new_file)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        
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
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
                
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    editor = SimpleTextEditor()
    editor.run()
