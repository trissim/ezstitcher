Installation Issues
=================

This page addresses common installation issues with EZStitcher.

Dependency Conflicts
-----------------

**Issue**: Conflicts between dependencies when installing EZStitcher.

**Solution**:

1. Create a clean virtual environment:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # Linux/macOS
       # or
       .venv\Scripts\activate     # Windows

2. Install dependencies with specific versions:

   .. code-block:: bash

       pip install -r requirements.txt

3. If you still encounter conflicts, try installing dependencies one by one:

   .. code-block:: bash

       pip install numpy==1.24.3
       pip install scikit-image==0.21.0
       pip install scipy==1.10.1
       pip install pandas==2.0.3
       pip install tifffile==2023.7.10
       pip install ashlar==1.15.0
       pip install opencv-python==4.8.0.76

NumPy/SciPy Issues
---------------

**Issue**: Errors related to NumPy or SciPy during installation.

**Solution**:

1. Uninstall and reinstall NumPy and SciPy:

   .. code-block:: bash

       pip uninstall numpy scipy
       pip install numpy==1.24.3 scipy==1.10.1

2. If you're on Windows, try installing pre-built wheels:

   .. code-block:: bash

       pip install --only-binary=numpy,scipy numpy==1.24.3 scipy==1.10.1

3. If you're on Linux, make sure you have the required build dependencies:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get install build-essential libopenblas-dev

       # CentOS/RHEL
       sudo yum install gcc-c++ openblas-devel

OpenCV Issues
----------

**Issue**: Errors related to OpenCV during installation.

**Solution**:

1. Try installing the headless version of OpenCV:

   .. code-block:: bash

       pip uninstall opencv-python
       pip install opencv-python-headless

2. If you're on Linux, make sure you have the required dependencies:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get install libgl1-mesa-glx

       # CentOS/RHEL
       sudo yum install mesa-libGL

3. If you're on macOS, try installing OpenCV via Homebrew:

   .. code-block:: bash

       brew install opencv
       pip install opencv-python

Ashlar Issues
----------

**Issue**: Errors related to Ashlar during installation.

**Solution**:

1. Install Ashlar from source:

   .. code-block:: bash

       git clone https://github.com/labsyspharm/ashlar.git
       cd ashlar
       pip install -e .

2. Make sure you have the required dependencies for Ashlar:

   .. code-block:: bash

       pip install numpy scipy scikit-image networkx matplotlib pyjnius

3. If you're on Windows, you might need to install Visual C++ Build Tools:

   - Download and install from https://visualstudio.microsoft.com/visual-cpp-build-tools/

Python Version Issues
-----------------

**Issue**: EZStitcher works best with Python 3.11, but you have a different version.

**Solution**:

1. We recommend using pyenv to manage Python versions:

   .. code-block:: bash

       # Install pyenv (Linux/WSL)
       curl https://pyenv.run | bash

       # Install pyenv (macOS)
       brew install pyenv

       # Install Python 3.11
       pyenv install 3.11
       pyenv global 3.11

2. Create a virtual environment with Python 3.11:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # Linux/macOS
       # or
       .venv\Scripts\activate     # Windows

3. If you can't use Python 3.11, EZStitcher should work with Python 3.8-3.10 as well.

Installation Steps
--------------------

Follow these steps to install EZStitcher:

.. code-block:: bash

    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher
    pip install -e .

This will install EZStitcher in development mode, which allows you to modify the code and see the changes immediately.

Verifying Installation
------------------

To verify that EZStitcher is installed correctly:

.. code-block:: bash

    python -c "import ezstitcher; print(ezstitcher.__version__)"

This should print the version number of EZStitcher. If you get an error, the installation was not successful.
