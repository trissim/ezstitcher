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

       pip install numpy scikit-image scipy pandas imageio tifffile ashlar opencv-python PyYAML

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

Recommended Installation with pyenv
----------------------------------

We recommend using `pyenv <https://github.com/pyenv/pyenv>`_ to manage Python versions and virtual environments. This approach provides several benefits:

- Install and manage multiple Python versions without affecting your system Python
- Create isolated environments for different projects
- Easily switch between Python versions
- Consistent environment across different operating systems

Installing with pyenv on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install pyenv dependencies:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get update
       sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
       libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
       liblzma-dev python-openssl git

       # Fedora/RHEL
       sudo dnf install -y make gcc zlib-devel bzip2 bzip2-devel readline-devel \
       sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel

2. Install pyenv:

   .. code-block:: bash

       curl https://pyenv.run | bash

3. Add pyenv to your shell configuration:

   .. code-block:: bash

       echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
       echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
       echo 'eval "$(pyenv init -)"' >> ~/.bashrc
       source ~/.bashrc

4. Install Python 3.11 and create a virtual environment for EZStitcher:

   .. code-block:: bash

       pyenv install 3.11
       pyenv global 3.11
       python -m venv ~/.venvs/ezstitcher
       source ~/.venvs/ezstitcher/bin/activate
       pip install ezstitcher

Installing with pyenv on Windows (WSL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Windows users, we strongly recommend using Windows Subsystem for Linux (WSL) for the best experience:

1. Install WSL by opening PowerShell as Administrator and running:

   .. code-block:: powershell

       wsl --install

2. After installation completes and you've set up your Linux user account, follow the Linux installation instructions above.

3. To access your Windows files from WSL, they are mounted at `/mnt/c/` (for the C: drive).

Installing with pyenv on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install Homebrew if you don't have it:

   .. code-block:: bash

       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install pyenv dependencies:

   .. code-block:: bash

       brew install openssl readline sqlite3 xz zlib tcl-tk

3. Install pyenv:

   .. code-block:: bash

       brew install pyenv

4. Add pyenv to your shell configuration:

   .. code-block:: bash

       echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
       echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
       echo 'eval "$(pyenv init -)"' >> ~/.zshrc
       source ~/.zshrc

   Note: If you're using bash instead of zsh, replace `.zshrc` with `.bash_profile` or `.bashrc`.

5. Install Python 3.11 and create a virtual environment for EZStitcher:

   .. code-block:: bash

       pyenv install 3.11
       pyenv global 3.11
       python -m venv ~/.venvs/ezstitcher
       source ~/.venvs/ezstitcher/bin/activate
       pip install ezstitcher

Using EZStitcher with the Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've set up your virtual environment, you can activate it whenever you want to use EZStitcher:

.. code-block:: bash

    # On Linux/macOS/WSL
    source ~/.venvs/ezstitcher/bin/activate
    
    # Run EZStitcher commands
    python -m ezstitcher --help

To deactivate the virtual environment when you're done:

.. code-block:: bash

    deactivate
