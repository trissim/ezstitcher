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

3. **Install EZStitcher**:

   .. code-block:: bash

       git clone https://github.com/trissim/ezstitcher.git
       cd ezstitcher
       pip install -e .

Dependencies
-----------

EZStitcher's main dependencies will be installed automatically:

- **numpy**, **scikit-image**, **scipy**: Scientific computing
- **pandas**, **tifffile**, **ashlar**: Data handling and stitching
- **opencv-python**: Computer vision algorithms
- **PyYAML**: Configuration handling

Optional dependencies for development and visualization include **matplotlib** and **jupyter**.

Troubleshooting Installation Issues
----------------------------------

Common Issues
~~~~~~~~~~~~

1. **Missing Dependencies**:

   If you encounter errors about missing dependencies, try installing them manually:

   .. code-block:: bash

       pip install numpy scikit-image scipy pandas imageio tifffile ashlar opencv-python PyYAML

2. **Version Conflicts**:

   If you encounter version conflicts, try creating a fresh virtual environment:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate
       pip install -r requirements.txt

3. **Compilation Errors**:

   Some dependencies may require compilation tools:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get install python3-dev

       # macOS
       brew install libtiff

   On Windows, you might need to install Visual C++ Build Tools from the Microsoft website.

**Linux**:

- If you encounter issues with image I/O libraries, install the required system packages:

  .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install libtiff5-dev libopenjp2-7-dev

      # Fedora/RHEL
      sudo dnf install libtiff-devel openjpeg2-devel

Detailed Platform-Specific Installation
----------------------------------

**Linux (Ubuntu/Debian)**

.. code-block:: bash

    # Install pyenv dependencies
    sudo apt-get update
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl \
    libncurses5-dev xz-utils tk-dev libffi-dev liblzma-dev

    # Install pyenv
    curl https://pyenv.run | bash

    # Add to shell (add to ~/.bashrc)
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    source ~/.bashrc

**macOS**

.. code-block:: bash

    # Install pyenv with Homebrew
    brew install pyenv

    # Add to shell (for zsh)
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    source ~/.zshrc

**Windows with WSL**

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

Using EZStitcher
--------------

After installation, you can verify and use EZStitcher:

.. code-block:: bash

    # Verify installation
    python -c "import ezstitcher; print(ezstitcher.__version__)"

    # Run EZStitcher
    python -m ezstitcher --help
