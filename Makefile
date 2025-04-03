.PHONY: clean install test dev

# Install in development mode
dev:
	pip install -e .

# Install the package
install:
	pip install .

# Run tests
test:
	python -m unittest discover -s tests

# Clean up build files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Distribute package
dist: clean
	python setup.py sdist bdist_wheel