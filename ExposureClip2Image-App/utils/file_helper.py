import os
from typing import List

def ensure_directories_exist(directories: List[str]) -> None:
    # Iterate over the list of directory paths
    for folder in directories:
        # Check if the directory does not exist
        if not os.path.exists(folder):
            # Create the directory and any necessary parent directories
            os.makedirs(folder)
            # Print a message indicating the directory was created
            print(f"Created directory: {folder}")
