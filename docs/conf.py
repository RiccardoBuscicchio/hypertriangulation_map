# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'hypertransform'
copyright = '2025'
author = 'Riccardo Buscicchio, Elinore Roebber, Janna Goldstein, Christopher J. Moore'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy docstrings
    'sphinx_rtd_theme',  # only for this theme
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
