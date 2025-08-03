# This is a normal Python application
import os
import sys
import json
from datetime import datetime

class Calculator:
    def __init__(self):
        self.history = []
        
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
        
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
        
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
        
    def divide(self, a, b):
        if b == 0:
            return "Error: Division by zero"
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
        
    def get_history(self):
        return self.history
        
    def clear_history(self):
        self.history = []
        
def main():
    calc = Calculator()
    print("Simple Calculator")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Show History")
    print("6. Clear History")
    print("7. Exit")
    
    while True:
        choice = input("Enter choice (1-7): ")
        if choice == "7":
            break
        elif choice == "5":
            print("History:", calc.get_history())
        elif choice == "6":
            calc.clear_history()
            print("History cleared")
        else:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))
            if choice == "1":
                print("Result:", calc.add(a, b))
            elif choice == "2":
                print("Result:", calc.subtract(a, b))
            elif choice == "3":
                print("Result:", calc.multiply(a, b))
            elif choice == "4":
                print("Result:", calc.divide(a, b))
                
if __name__ == "__main__":
    main()
    