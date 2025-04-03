# GitHub Setup Instructions

After creating your repository on GitHub, run these commands to connect your local repository:

```bash
# Add the remote repository
git remote add origin https://github.com/trissim/ezstitcher.git

# Push your code to GitHub
git push -u origin main
```

If you prefer to use SSH instead of HTTPS, use this command instead:

```bash
git remote add origin git@github.com:trissim/ezstitcher.git
```

## Next Steps

1. Set up GitHub Actions for CI/CD
2. Add documentation for new features
3. Create issues for planned features
4. Set up branch protection rules