from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="axon_quant",
    version="0.1.0",
    author="trissim",
    author_email="your.email@example.com",
    description="A microscopy image stitching and processing tool for neuronal axon quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trissim/axon_quant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "imageio>=2.9.0",
        "tifffile>=2021.1.1",
        "ashlar>=1.14.0",
    ],
    entry_points={
        "console_scripts": [
            "axon_quant=axon_quant.__main__:main",
        ],
    },
)