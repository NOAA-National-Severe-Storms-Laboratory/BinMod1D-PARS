# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BinMod1D'
copyright = '2026, Edwin Lee Dunnavan'
author = 'Edwin Lee Dunnavan'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx_copybutton'
    ]

myst_enable_extensions = [
    "dollarmath",           # Enables $...$ and $$...$$
    "amsmath",              # Enables LaTeX environments like \begin{align}
    "colon_fence",  # <--- Add this line!
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints']

nbsphinx_allow_errors = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_theme_options = {'logo_only':False,'display_version':True}

html_context = {
    "display_github": True,
    "github_user": "NOAA-National-Severe-Storms-Laboratory",
    "github_repo": "BinMod1D-PARS",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

def setup(app):
   app.add_css_file('custom.css')
