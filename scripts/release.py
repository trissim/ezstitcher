#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import re
import requests

def get_current_version():
    with open("ezstitcher/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return None

def get_pypi_version():
    try:
        response = requests.get("https://pypi.org/pypi/ezstitcher/json")
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except:
        pass
    return None

def main():
    # Get current version
    version = get_current_version()
    if not version:
        print("Error: Could not find version in __init__.py")
        sys.exit(1)
    
    # Get PyPI version
    pypi_version = get_pypi_version()
    print(f"Current package version: {version}")
    print(f"Current PyPI version: {pypi_version}")
    
    if pypi_version and version <= pypi_version:
        print(f"Error: Current version ({version}) must be greater than PyPI version ({pypi_version})")
        sys.exit(1)
    
    # Confirm with user
    response = input(f"Create release for v{version}? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    try:
        # Create and push tag
        subprocess.run(['git', 'tag', '-a', f'v{version}', '-m', f'Release version {version}'], check=True)
        subprocess.run(['git', 'push', 'origin', f'v{version}'], check=True)
        
        print(f"\nSuccessfully created and pushed tag v{version}")
        print("GitHub Actions workflow should start automatically.")
        print("Monitor progress at: https://github.com/YOUR_USERNAME/ezstitcher/actions")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during release process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()