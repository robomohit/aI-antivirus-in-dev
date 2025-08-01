
import tkinter as tk
from tkinter import ttk

class Calculator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Calculator")
        self.display = tk.Entry(self.window, width=30)
        self.display.grid(row=0, column=0, columnspan=4)
        
    def add_digit(self, digit):
        current = self.display.get()
        self.display.delete(0, tk.END)
        self.display.insert(0, current + str(digit))
        
    def calculate(self):
        try:
            result = eval(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(0, result)
        except:
            self.display.delete(0, tk.END)
            self.display.insert(0, "Error")
            
    def run(self):
        buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            '0', '.', '=', '+'
        ]
        
        row = 1
        col = 0
        for button in buttons:
            cmd = lambda x=button: self.add_digit(x) if x != '=' else self.calculate()
            tk.Button(self.window, text=button, command=cmd).grid(row=row, column=col)
            col += 1
            if col > 3:
                col = 0
                row += 1
                
        self.window.mainloop()
        
if __name__ == "__main__":
    calc = Calculator()
    calc.run()
