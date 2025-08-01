
import os
import shutil
from pathlib import Path

class FileManager:
    def __init__(self):
        self.current_dir = os.getcwd()
        
    def list_files(self, directory="."):
        try:
            files = os.listdir(directory)
            for file in files:
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"File: {file} ({size} bytes)")
                else:
                    print(f"Directory: {file}/")
        except Exception as e:
            print(f"Error listing files: {e}")
            
    def copy_file(self, source, destination):
        try:
            shutil.copy2(source, destination)
            print(f"File copied: {source} -> {destination}")
        except Exception as e:
            print(f"Error copying file: {e}")
            
    def delete_file(self, file_path):
        try:
            os.remove(file_path)
            print(f"File deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")
            
    def create_directory(self, dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory created: {dir_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")

def main():
    fm = FileManager()
    print("Simple File Manager")
    print("1. List files")
    print("2. Copy file")
    print("3. Delete file")
    print("4. Create directory")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        fm.list_files()
    elif choice == "2":
        source = input("Enter source file: ")
        dest = input("Enter destination: ")
        fm.copy_file(source, dest)
    elif choice == "3":
        file_path = input("Enter file to delete: ")
        fm.delete_file(file_path)
    elif choice == "4":
        dir_path = input("Enter directory name: ")
        fm.create_directory(dir_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
