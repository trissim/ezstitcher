from setuptools import setup, find_packages
import re

# Read version from __init__.py
with open("ezstitcher/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read README and ensure absolute GitHub URL for PyPI
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    # Ensure logo URL is absolute and points to main branch
    github_base = 'https://raw.githubusercontent.com/trissim/ezstitcher/main'
    logo_path = 'docs/source/_static/ezstitcher_logo.png'

    # Replace any relative or malformed paths with the absolute GitHub URL
    long_description = re.sub(
        rf'{github_base}/{github_base}/{logo_path}|{github_base}/{logo_path}|{logo_path}',
        f'{github_base}/{logo_path}',
        long_description
    )

setup(
    name="ezstitcher",
    version=version,
    author="trissim",
    author_email="tristan.simas@mail.mcgill.ca",
    description="An easy-to-use microscopy image stitching and processing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trissim/ezstitcher",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.11.0",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "imageio>=2.9.0",
        "tifffile>=2021.1.1",
        "imagecodecs>=2021.1.1",
        "ashlar>=1.14.0",
        "opencv-python>=4.5.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.0",
            "pytest-cov==4.1.0",
            "coverage==7.3.2",
            "genbadge[coverage]",
            "sphinx",
            "sphinx-rtd-theme",
            "black",
            "flake8",
        ],
    },
)
