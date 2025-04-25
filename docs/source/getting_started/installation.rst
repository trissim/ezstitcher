Installation
============

This guide will help you install EZStitcher using pyenv and pip.

System Requirements
------------------

- Python 3.11 required
- 8GB RAM minimum (16GB recommended for large images)
- Multi-core CPU recommended for faster processing with multithreaded support

Installation Steps
-----------------

1. Set up Python environment with pyenv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install pyenv (platform-specific)
    # macOS
    brew install pyenv

    # Linux/WSL
    curl https://pyenv.run | bash

    # Add pyenv to your shell (bash example)
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    source ~/.bashrc

    # Install Python 3.11
    pyenv install 3.11
    pyenv global 3.11

2. Create and activate a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create project directory
    mkdir -p ~/projects/ezstitcher
    cd ~/projects/ezstitcher

    # Create and activate virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install EZStitcher from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher

    # Install the package and dependencies
    pip install -e .

All dependencies will be automatically installed from the requirements.txt file.

Verifying Installation
--------------------

To verify that EZStitcher installed correctly:

.. code-block:: bash

    python -c "import ezstitcher; print('EZStitcher installed successfully')"

This should print "EZStitcher installed successfully".

Getting Started
---------------

After installation, you can start using EZStitcher through its Python API. For a quick introduction, see the :doc:`../user_guide/index` guide.