# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Variational Linear Attention'
copyright = '2026, DeepBrain Labs'
author = 'DeepBrain Labs'

version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "html_image",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4A90E2",
        "color-brand-content": "#4A90E2",
    },
    "dark_css_variables": {
        "color-brand-primary": "#50E3C2",
        "color-brand-content": "#50E3C2",
    },
}
