Installation
============

This guide will help you install EZStitcher and its dependencies.

System Requirements
------------------

- **Python**: 3.8 or higher (but less than 3.12)
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum, 16GB or more recommended for large images
- **CPU**: Multi-core processor recommended for faster processing
- **Disk Space**: Depends on the size of your microscopy data

Installation via pip
-------------------

The easiest way to install EZStitcher is using pip:

.. code-block:: bash

    pip install ezstitcher

To verify the installation:

.. code-block:: bash

    python -c "import ezstitcher; print(ezstitcher.__version__)"

Installation from source
-----------------------

To install from source:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/trissim/ezstitcher.git
       cd ezstitcher

2. Install in development mode:

   .. code-block:: bash

       pip install -e .

Dependencies
-----------

EZStitcher depends on the following packages:

Core Dependencies
~~~~~~~~~~~~~~~~

- **numpy**: Numerical computing
- **scikit-image**: Image processing algorithms
- **scipy**: Scientific computing
- **pandas**: Data manipulation and analysis
- **imageio**: Image I/O
- **tifffile**: TIFF file handling
- **ashlar**: Image stitching backend
- **opencv-python**: Computer vision algorithms
- **pydantic**: Data validation
- **PyYAML**: YAML file handling

Optional Dependencies
~~~~~~~~~~~~~~~~~~~

- **matplotlib**: Visualization (recommended)
- **jupyter**: Interactive examples (optional)

These dependencies will be automatically installed when you install EZStitcher using pip.

Troubleshooting Installation Issues
----------------------------------

Common Issues
~~~~~~~~~~~~

1. **Missing Dependencies**:

   If you encounter errors about missing dependencies, try installing them manually:

   .. code-block:: bash

       pip install numpy scikit-image scipy pandas imageio tifffile ashlar opencv-python pydantic PyYAML

2. **Version Conflicts**:

   If you encounter version conflicts, try creating a new virtual environment:

   .. code-block:: bash

       python -m venv ezstitcher_env
       source ezstitcher_env/bin/activate  # On Windows: ezstitcher_env\Scripts\activate
       pip install ezstitcher

3. **Compilation Errors**:

   Some dependencies may require compilation. On Windows, you might need to install Visual C++ Build Tools. On Linux, you might need to install development packages:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get install python3-dev

       # Fedora/RHEL
       sudo dnf install python3-devel

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Windows**:

- If you encounter issues with OpenCV, try installing it separately:

  .. code-block:: bash

      pip install opencv-python

**macOS**:

- If you encounter issues with tifffile, try installing libtiff:

  .. code-block:: bash

      brew install libtiff

**Linux**:

- If you encounter issues with image I/O libraries, install the required system packages:

  .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install libtiff5-dev libopenjp2-7-dev

      # Fedora/RHEL
      sudo dnf install libtiff-devel openjpeg2-devel
