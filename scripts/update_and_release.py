#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import re
from packaging import version

def run_command(command, check=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def get_modified_files():
    """Get list of modified files"""
    return run_command(['git', 'diff', '--name-only']).split('\n')

def check_unstaged_changes():
    """Check for unstaged changes"""
    result = subprocess.run(['git', 'diff', '--quiet'], capture_output=True)
    return result.returncode != 0

def commit_changes(files, message):
    """Commit specified files with a message"""
    try:
        # Check for unstaged changes first
        if check_unstaged_changes():
            print("You have unstaged changes. Please commit or stash them first.")
            print("\nYou can:")
            print("1. Stage and commit: git add . && git commit -m 'your message'")
            print("2. Or stash: git stash")
            sys.exit(1)
            
        # Pull latest changes
        subprocess.run(['git', 'pull', '--rebase', 'origin', 'main'], check=True)
        
        # Add files
        for file in files:
            subprocess.run(['git', 'add', file], check=True)
        
        # Commit
        subprocess.run(['git', 'commit', '-m', message], check=True)
        
        # Push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def update_version():
    """Run the version update script"""
    try:
        subprocess.run([sys.executable, 'scripts/update_version.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error updating version: {e}")
        sys.exit(1)

def create_release():
    """Run the release script"""
    try:
        subprocess.run([sys.executable, 'scripts/release.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating release: {e}")
        sys.exit(1)

def main():
    # 1. Get modified files
    modified_files = [f for f in get_modified_files() if f]
    if not modified_files:
        print("No modified files found.")
        sys.exit(0)
    
    # 2. Show modified files and confirm
    print("\nModified files:")
    for f in modified_files:
        print(f"- {f}")
    
    response = input("\nProceed with commit and release? [y/N] ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # 3. Commit changes
    commit_message = input("\nEnter commit message: ")
    if not commit_message:
        print("Commit message cannot be empty.")
        sys.exit(1)
    
    print("\nCommitting changes...")
    commit_changes(modified_files, commit_message)
    
    # 4. Update version
    print("\nUpdating version...")
    update_version()
    
    # 5. Create release
    print("\nCreating release...")
    create_release()
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
