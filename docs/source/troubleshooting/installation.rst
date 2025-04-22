Installation Troubleshooting
=========================

This page addresses common installation issues with EZStitcher. For basic installation instructions, see the :doc:`../getting_started/installation` guide.

Common Issues
-----------

**Dependency Conflicts**

- Create a clean virtual environment and install dependencies with specific versions:

  .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # Linux/macOS
      # or
      .venv\Scripts\activate     # Windows
      pip install -r requirements.txt

**System Dependencies**

- Some packages require system libraries to be installed:

  .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install libtiff5-dev libopenjp2-7-dev libgl1-mesa-glx

      # macOS
      brew install libtiff openjpeg

**Python Version Issues**

- EZStitcher works best with Python 3.11
- If you can't use Python 3.11, versions 3.8-3.10 should work but may require additional configuration
- We recommend using pyenv to manage Python versions (see :doc:`../getting_started/installation` for details)
