# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XuanCe'
copyright = '2023, Wenzhang Liu, etc.'
author = 'Wenzhang Liu.'
release = 'v1.2'

# The master toctree document.
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "notfound.extension",
    "sphinx_design",
    "sphinx_inline_tabs",
    "sphinx_favicon",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # before that is renku, sphinx_rtd_theme
html_static_path = ['figures']
html_theme_options = {
    "light_logo": "logo_2.png",
    "dark_logo": "logo_2.png",
}

favicons = [
    {"href": "favicon/favicon.svg"},  # => use `_static/icon.svg`
    {"href": "https://xuance.readthedocs.io/en/latest/favicon/favicon-96x96.png"},
    {
        "rel": "apple-touch-icon",
        "href": "https://xuance.readthedocs.io/en/latest/favicon/apple-touch-icon.png",
    },
]

