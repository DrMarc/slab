# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../slab/'))

# -- Project information -----------------------------------------------------
project = 'slab'
copyright = '2020, Marc Schoenwiesner, Ole Bialas'
author = 'Marc Schoenwiesner, Ole Bialas'
with open('../VERSION.txt') as f:
    version = f.read().strip()
release = version

# -- General configuration ---------------------------------------------------
needs_sphinx = '1.8'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.napoleon']
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
#autodoc_default_options = {'member-order': 'bysource'}
intersphinx_mapping = {'python': ('https://docs.python.org/', None),
                       'matplotlib': ('http://matplotlib.org/', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None)}
html_theme = 'alabaster'
