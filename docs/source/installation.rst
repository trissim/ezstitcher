Installation
============

Requirements
-----------

- **Python 3.11.9** (recommended for best compatibility with all dependencies)
- Git

Linux/macOS
-----------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher

    # Create and activate a virtual environment with Python 3.11.9
    python3.11 -m venv .venv
    source .venv/bin/activate

    # Install the package in development mode
    pip install -e .

Windows
-------

.. code-block:: powershell

    # Clone the repository
    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher

    # Create and activate a virtual environment with Python 3.11.9
    py -3.11 -m venv .venv
    .venv\Scripts\activate

    # Install the package in development mode
    pip install -e .

Python Version Note
------------------

Python 3.11.9 is recommended because it provides the best compatibility with all required dependencies.

Dependencies
-----------

EZStitcher depends on the following packages:

- numpy
- scikit-image
- scipy
- pandas
- tifffile
- ashlar
- opencv-python
- pytest (for running tests)

These dependencies will be automatically installed when you install EZStitcher.
