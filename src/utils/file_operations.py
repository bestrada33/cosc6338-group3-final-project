import os

def ensure_dir(path):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)