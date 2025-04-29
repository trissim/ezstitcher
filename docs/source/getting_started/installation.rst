Installation
============

Installing EZStitcher is simple with pip.

System Requirements
------------------

- Python 3.11 (only supported version)
- 8GB RAM minimum (16GB recommended for large images)
- Multi-core CPU recommended for faster processing

Quick Install
------------

.. code-block:: bash

    pip install ezstitcher

Using pyenv (recommended)
------------------------

If you need to install Python 3.11, we recommend using `pyenv <https://github.com/pyenv/pyenv>`_:

.. code-block:: bash

    # Install Python 3.11 with pyenv
    pyenv install 3.11
    pyenv local 3.11

    # Create virtual environment and install ezstitcher
    python -m venv .venv
    source .venv/bin/activate
    pip install ezstitcher

Verifying Installation
--------------------

Verify that EZStitcher installed correctly:

.. code-block:: bash

    python -c "import ezstitcher; print('EZStitcher installed successfully')"

Next Steps
----------

After installation:

1. Follow the :doc:`quick_start` guide to run your first stitching pipeline
2. See :doc:`../user_guide/basic_usage` for basic usage examples
3. Explore :doc:`../concepts/architecture_overview` to learn about EZStitcher's architecture