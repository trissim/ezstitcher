"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
import sphinx_rtd_theme

# Check if we're on Read the Docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# If we're on Read the Docs, use mock modules for dependencies that are hard to install
if on_rtd:
    try:
        from unittest.mock import MagicMock

        class Mock(MagicMock):
            @classmethod
            def __getattr__(cls, name):
                return MagicMock()

        # Systems modules that might cause issues during doc building
        MOCK_MODULES = [
            'cv2',
            'ashlar',
            'imageio',
            'tifffile',
            'skimage',
            'skimage.feature',
            'skimage.filters',
            'skimage.transform',
            'skimage.util',
            'skimage.measure',
            'skimage.morphology',
            'skimage.segmentation',
            'skimage.exposure',
        ]

        sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
    except ImportError:
        pass

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'EZStitcher'
copyright = '2024, EZStitcher Team'
author = 'EZStitcher Team'
version = '0.1.0'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'
master_doc = 'index'
language = 'en'

# HTML output configuration
html_theme = 'sphinx_rtd_theme'

# Only set html_theme_path if not on Read the Docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Autodoc configuration
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}
