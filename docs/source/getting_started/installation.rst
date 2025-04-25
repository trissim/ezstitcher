Installation
============

This guide will help you install EZStitcher and its dependencies.

System Requirements
------------------

- **Python**: 3.8 or higher (3.11 recommended)
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum, 16GB or more recommended for large images
- **CPU**: Multi-core processor recommended for faster processing
- **Disk Space**: Depends on the size of your microscopy data

Recommended Installation with pyenv
----------------------------------

We recommend using `pyenv <https://github.com/pyenv/pyenv>`_ to manage Python versions. This approach provides isolated environments and consistent Python versions across platforms.

Quick Installation Steps
~~~~~~~~~~~~~~~~~~~~~~~

1. **Install pyenv** (platform-specific one-liners):

   .. code-block:: bash

       # macOS with Homebrew
       brew install pyenv

       # Windows with WSL
       curl https://pyenv.run | bash

       # Linux (Ubuntu/Debian)
       curl https://pyenv.run | bash

2. **Set up your environment**:

   .. code-block:: bash

       # Add pyenv to your shell (bash example)
       echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
       echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
       echo 'eval "$(pyenv init -)"' >> ~/.bashrc
       source ~/.bashrc

       # Install Python 3.11 and set up environment
       pyenv install 3.11
       pyenv global 3.11
       mkdir -p ~/projects/ezstitcher
       cd ~/projects/ezstitcher
       python -m venv .venv
       source .venv/bin/activate

3. **Install EZStitcher from source**:

   .. code-block:: bash

       git clone https://github.com/trissim/ezstitcher.git
       cd ezstitcher
       # Install in development mode
       python -m pip install -e .

Dependencies
-----------

EZStitcher's main dependencies will be installed automatically:

- **numpy**, **scikit-image**, **scipy**: Scientific computing
- **pandas**, **tifffile**, **ashlar**: Data handling and stitching
- **opencv-python**: Computer vision algorithms
- **PyYAML**: Configuration handling
- **imagecodecs**: Image compression/decompression

Optional dependencies for development and visualization include **matplotlib** and **jupyter**.

Basic Troubleshooting
-------------------

If you encounter issues during installation:

1. **Ensure you're using Python 3.8-3.11** (3.11 recommended)
2. **Check that your virtual environment is activated**
3. **Try installing in a fresh virtual environment**

For detailed troubleshooting, see the :doc:`../troubleshooting/installation` guide.

Verifying Installation
------------------

To verify that EZStitcher is installed correctly:

.. code-block:: bash

    python -c "import ezstitcher; print('EZStitcher installed successfully')"

This should print "EZStitcher installed successfully". If you get an error, the installation was not successful.

Platform-Specific Installation Details
----------------------------------

Linux (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install pyenv dependencies
    sudo apt-get update
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl \
    libncurses5-dev xz-utils tk-dev libffi-dev liblzma-dev

    # Install image processing dependencies
    sudo apt-get install -y libtiff5-dev libopenjp2-7-dev

macOS
~~~~

.. code-block:: bash

    # Install pyenv with Homebrew
    brew install pyenv

    # Install image processing dependencies
    brew install libtiff

    # Add to shell (for zsh)
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    source ~/.zshrc

Windows with WSL
~~~~~~~~~~~~

.. code-block:: bash

    # Install WSL from PowerShell (Admin)
    # wsl --install

    # Then in WSL:
    curl https://pyenv.run | bash

    # Add to shell
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    source ~/.bashrc

Getting Started
------------

After installation, you can start using EZStitcher through its Python API. For a quick introduction, see the :doc:`../user_guide/introduction` guide.
