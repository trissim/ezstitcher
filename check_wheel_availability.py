#!/usr/bin/env python3
"""
Check wheel availability for packages on PyPI for different Python versions.
"""

import requests
import json
import sys
from collections import defaultdict

def get_package_info(package_name):
    """Get package information from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {package_name}: {response.status_code}")
        return None

def check_wheel_availability(package_name, version=None):
    """Check wheel availability for different Python versions."""
    data = get_package_info(package_name)
    if not data:
        return None

    if not version:
        version = data["info"]["version"]  # Use latest version

    if version not in data["releases"]:
        print(f"Version {version} not found for {package_name}")
        return None

    releases = data["releases"][version]

    # Track which Python versions have wheels
    python_versions = defaultdict(list)

    for release in releases:
        if release["packagetype"] == "bdist_wheel":
            # Extract Python version from filename
            filename = release["filename"]
            if "cp" in filename:
                # Extract Python version from wheel filename
                parts = filename.split("-")
                for part in parts:
                    if part.startswith("cp") and not part.startswith("cp3"):
                        continue
                    if part.startswith("cp3"):
                        # Format: cp310 or cp39
                        py_version = part[2:]  # Remove 'cp'
                        if len(py_version) >= 2:
                            # Format: 310 -> 3.10 or 39 -> 3.9
                            py_version = f"3.{py_version[1:]}" if py_version.startswith("3") else py_version

                        # Check if it's platform specific
                        platform = "any"
                        if "manylinux" in filename:
                            platform = "manylinux"
                        elif "win" in filename:
                            platform = "windows"
                        elif "macosx" in filename:
                            platform = "macos"

                        python_versions[py_version].append(platform)

    # Deduplicate platforms
    for version in python_versions:
        python_versions[version] = list(set(python_versions[version]))

    return {
        "package": package_name,
        "version": version,
        "python_versions": dict(python_versions)
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

    print("Wheel Availability by Python Version:")
    print("====================================")

    for package in packages:
        info = check_wheel_availability(package)
        if info:
            print(f"\n{info['package']} {info['version']}:")
            if info['python_versions']:
                for py_version, platforms in sorted(info['python_versions'].items()):
                    print(f"  Python {py_version}: {', '.join(platforms)}")
            else:
                print("  No wheels found")
        else:
            print(f"\n{package}: Unable to fetch information")

if __name__ == "__main__":
    main()
