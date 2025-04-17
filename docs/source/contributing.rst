Contributing
============

Thank you for your interest in contributing to EZStitcher! This document provides guidelines for contributing to the project.

Setting Up Development Environment
--------------------------------

1. **Clone the repository**:

.. code-block:: bash

    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher

2. **Create a virtual environment**:

.. code-block:: bash

    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install in development mode**:

.. code-block:: bash

    pip install -e .

4. **Install development dependencies**:

.. code-block:: bash

    pip install pytest pytest-cov sphinx sphinx-rtd-theme

Code Style
---------

EZStitcher follows these coding conventions:

- Use **snake_case** for variables, functions, and methods
- Use **CamelCase** for class names
- Use **UPPER_CASE** for constants
- Follow PEP 8 guidelines
- Include docstrings for all modules, classes, and functions
- Use type hints where appropriate

Testing
------

All code contributions should include tests:

1. **Run existing tests**:

.. code-block:: bash

    python -m unittest discover -s tests

2. **Add new tests** for new features or bug fixes:

.. code-block:: python

    # Example test
    import unittest
    from ezstitcher.core import some_function

    class TestSomeFunction(unittest.TestCase):
        def test_basic_functionality(self):
            result = some_function(input_data)
            self.assertEqual(result, expected_output)

3. **Check test coverage**:

.. code-block:: bash

    pytest --cov=ezstitcher tests/

Documentation
------------

All code contributions should include documentation:

1. **Add docstrings** to all modules, classes, and functions:

.. code-block:: python

    def some_function(param1, param2):
        """
        Brief description of the function.

        Args:
            param1 (type): Description of param1
            param2 (type): Description of param2

        Returns:
            type: Description of return value

        Raises:
            ExceptionType: When and why this exception is raised
        """
        # Function implementation

2. **Update the documentation** if you change existing functionality:

.. code-block:: bash

    cd docs
    make html

3. **Add examples** for new features:

.. code-block:: python

    # Example usage of new feature
    from ezstitcher.core import new_feature

    result = new_feature(input_data)
    print(result)

Pull Request Process
------------------

1. **Fork the repository** and create a new branch for your feature or bug fix
2. **Implement your changes** with appropriate tests and documentation
3. **Run the tests** to ensure they pass
4. **Submit a pull request** with a clear description of the changes
5. **Address any feedback** from the code review

Issue Reporting
-------------

If you find a bug or have a feature request:

1. **Check existing issues** to see if it has already been reported
2. **Create a new issue** with a clear description of the problem or request
3. **Include steps to reproduce** for bugs
4. **Include expected behavior** and actual behavior
5. **Include version information** (Python version, EZStitcher version, OS)
