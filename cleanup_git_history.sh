#!/bin/bash

# This script removes large files from the Git repository history

# Make sure we're in the repository root
cd "$(git rev-parse --show-toplevel)" || exit 1

echo "Creating backup branch..."
git checkout -b backup_before_cleanup

echo "Removing large files from repository history..."
git filter-branch --force --index-filter '
  git rm --cached --ignore-unmatch .coverage
  git rm --cached --ignore-unmatch class_based_test_output.txt
  git rm --cached --ignore-unmatch original_test_output.txt
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/yoga.wasm
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/cli.js
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/vendor/ripgrep/arm64-darwin/rg
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/vendor/ripgrep/arm64-linux/rg
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/vendor/ripgrep/x64-darwin/rg
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/vendor/ripgrep/x64-linux/rg
  git rm --cached --ignore-unmatch node_modules/@anthropic-ai/claude-code/vendor/ripgrep/x64-win32/rg.exe
  git rm -r --cached --ignore-unmatch node_modules/
' --prune-empty --tag-name-filter cat -- --all

echo "Cleaning up refs..."
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

echo "Running garbage collection..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Repository size before and after cleanup:"
du -sh .git

echo ""
echo "IMPORTANT: This has only cleaned up your local repository."
echo "To update the remote repository, you will need to force push:"
echo "  git push --force origin your-branch-name"
echo ""
echo "Make sure all collaborators are aware of this change, as they will need to re-clone the repository."
