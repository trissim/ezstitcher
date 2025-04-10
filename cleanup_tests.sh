#!/bin/bash

# This script removes redundant test files from the tests directory

# Make sure we're in the repository root
cd "$(git rev-parse --show-toplevel)" || exit 1

echo "Removing redundant test files..."

# Files to remove
files_to_remove=(
  "tests/test_synthetic_imagexpress_refactored.py"
  "tests/test_synthetic_opera_phenix_refactored.py"
  "tests/test_synthetic_opera_phenix_refactored_auto.py"
  "tests/test_process_plate_auto.py"
  "tests/test_auto_config.py"
  "tests/test_image_locator_integration.py"
  "tests/test_microscope_auto_detection.py"
  "tests/test_synthetic_imagexpress_refactored_auto.py"
  "tests/test_synthetic_opera_phenix_refactored_auto_new.py"
)

# Remove each file
for file in "${files_to_remove[@]}"; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    git rm "$file"
  else
    echo "File $file does not exist, skipping"
  fi
done

echo "Done!"
