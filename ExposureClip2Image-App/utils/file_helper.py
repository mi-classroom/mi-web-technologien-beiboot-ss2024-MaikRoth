import os

def ensure_directories_exist(directories):
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")