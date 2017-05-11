import os

def safe_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)