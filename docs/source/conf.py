# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
from recommonmark.parser import CommonMarkParser
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cwd, '../../')))
sys.path.append(os.path.abspath(os.path.join(cwd, '../../../')))
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'lasr'
copyright = '2021, Soohwan Kim'
author = 'Soohwan Kim'

# The full version, including alpha/beta/rc tags
release = 'latest'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'recommonmark',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'Sphinx-Tutorialdoc'
latex_elements = {
}
latex_documents = [
    (master_doc, 'lasr.tex', 'LASR Documentation',
     'sooftware', 'manual'),
]
man_pages = [
    (master_doc, 'lasr', 'LASR Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'lasr', 'LASR Documentation',
     author, 'LASR', 'One line description of project.',
     'Miscellaneous'),
]
epub_title = project
epub_exclude_files = ['search.html']
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'PyTorch': ('http://pytorch.org/docs/master/', None),
}
todo_include_todos = True
