#!/usr/bin/env python3
"""
Check Python version compatibility for packages on PyPI.
"""

import requests
import json
import sys

def get_package_info(package_name):
    """Get package information from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {package_name}: {response.status_code}")
        return None

def get_python_compatibility(package_name, version=None):
    """Get Python version compatibility for a package."""
    data = get_package_info(package_name)
    if not data:
        return None
    
    if version:
        # Check specific version
        if version in data["releases"]:
            releases = data["releases"][version]
            python_requires = set()
            for release in releases:
                if "requires_python" in release and release["requires_python"]:
                    python_requires.add(release["requires_python"])
                if "python_version" in release and release["python_version"]:
                    python_requires.add(f"=={release['python_version']}")
            return python_requires
        else:
            print(f"Version {version} not found for {package_name}")
            return None
    else:
        # Check latest version
        latest_version = data["info"]["version"]
        python_requires = data["info"].get("requires_python", "Not specified")
        return {
            "latest_version": latest_version,
            "python_requires": python_requires
        }

def main():
    """Main function."""
    packages = [
        "numpy",
        "scikit-image",
        "scipy",
        "pandas",
        "tifffile",
        "ashlar",
        "opencv-python",
        "matplotlib"
    ]
    
    print("Package Compatibility Information:")
    print("==================================")
    
    for package in packages:
        info = get_python_compatibility(package)
        if info:
            print(f"{package}: Latest version {info['latest_version']}, Python requirement: {info['python_requires']}")
        else:
            print(f"{package}: Unable to fetch information")
    
if __name__ == "__main__":
    main()
