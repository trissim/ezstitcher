from setuptools import setup, find_packages

# Read version from __init__.py
with open("ezstitcher/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ezstitcher",
    version=version,  # Use version from __init__.py
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
    python_requires="==3.11",
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
)
