Image Locator
=============

.. module:: ezstitcher.core.image_locator

This module provides a class for locating images in various directory structures.

ImageLocator
----------

.. py:class:: ImageLocator

   Locates images in various directory structures.

   .. py:attribute:: DEFAULT_EXTENSIONS
      :type: list
      :value: ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

      Default file extensions for image files.

   .. py:staticmethod:: find_images_in_directory(directory, extensions=None, recursive=True, filename_parser=None)

      Find all images in a directory.

      :param directory: Directory to search
      :type directory: str or Path
      :param extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
      :type extensions: list, optional
      :param recursive: Whether to search recursively in subdirectories
      :type recursive: bool
      :param filename_parser: Optional filename parser (currently unused in this method)
      :type filename_parser: FilenameParser, optional
      :return: List of Path objects for image files
      :rtype: list

   .. py:staticmethod:: find_images_by_pattern(directory, pattern, extensions=None)

      Find images matching a pattern in a directory.

      :param directory: Directory to search
      :type directory: str or Path
      :param pattern: Regex pattern to match
      :type pattern: str or Pattern
      :param extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
      :type extensions: list, optional
      :return: List of Path objects for matching image files
      :rtype: list

   .. py:staticmethod:: find_z_stack_dirs(root_dir, pattern="ZStep_\\d+", recursive=True)

      Find directories matching a pattern (default: Z-step_#) recursively.

      :param root_dir: Root directory to start the search
      :type root_dir: str or Path
      :param pattern: Regex pattern to match directory names (default: pattern for Z-step folders)
      :type pattern: str
      :param recursive: Whether to search recursively in subdirectories
      :type recursive: bool
      :return: List of (z_index, directory) tuples where z_index is extracted from the pattern
      :rtype: list

   .. py:staticmethod:: find_image_locations(plate_folder, extensions=None)

      Find all image files recursively within plate_folder.

      :param plate_folder: Path to the plate folder
      :type plate_folder: str or Path
      :param extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
      :type extensions: list, optional
      :return: Dictionary with all images found in the plate folder
      :rtype: dict

   .. py:staticmethod:: find_image_directory(plate_folder, extensions=None)

      Find the directory where images are actually located.

      Handles both cases:
      1. Images directly in a folder (returns that folder)
      2. Images split across Z-step folders (returns parent of Z-step folders)

      :param plate_folder: Base directory to search
      :type plate_folder: str or Path
      :param extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
      :type extensions: list, optional
      :return: Path to the directory containing images
      :rtype: Path
