#!/usr/bin/env python3
"""
EZStitcher package entry point.

Note: Command-line interface is not yet supported.
EZStitcher is currently only available through its Python API.
"""

import sys


def main():
    """
    Main entry point for the package when run as a script.
    Currently only displays a message that CLI is not supported.

    Returns:
        int: Exit code (1 for error)
    """
    print("EZStitcher command-line interface is not yet supported.")
    print("Please use the Python API instead.")
    print("For documentation, visit: https://ezstitcher.readthedocs.io/")
    return 1


if __name__ == "__main__":
    sys.exit(main())