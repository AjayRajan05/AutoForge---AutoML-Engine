# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AutoForge'
copyright = '2024, AutoForge Team'
author = 'AutoForge Team'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom sidebar templates, must be a dictionary that maps template names to
# their filenames.
html_css_files = [
    'css/custom.css',
]

# -- Options for autodoc extension -----------------------------------------

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# -- Options for Napoleon extension -----------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for autodoc extension -----------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '_weakref_'
}

# -- Options for intersphinx extension --------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable/', None),
    'xgboost': ('https://xgboost.readthedocs.io/en/latest/', None),
    'lightgbm': ('https://lightgbm.readthedocs.io/en/latest/', None),
    'shap': ('https://shap.readthedocs.io/en/latest/', None),
}

# -- Options for coverage extension ----------------------------------------

coverage_show_missing_items = True

# -- Options for doctest extension ----------------------------------------

doctest_global_setup = '''
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
'''

# -- Custom setup ---------------------------------------------------------

def setup(app):
    """Custom setup for Sphinx."""
    app.add_css_file('css/custom.css')
    
    # Add custom roles if needed
    app.add_role('gh', textroles.XRefRole())

# -- Project-specific customizations ------------------------------------

# Add any custom Python modules that should be documented
autodoc_mock_imports = [
    'shap',
    'lightgbm',
    'xgboost',
    'optuna',
    'rich',
    'seaborn',
]

# -- LaTeX options (for PDF output) --------------------------------------

latex_engine = 'xelatex'
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{hyperref}
\usepackage{enumitem}
\setlistdepth{9}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}

# -- Man page options -----------------------------------------------------

man_pages = [
    (master_doc, 'autoforge', 'AutoForge Documentation',
     [author], 1)
]

# -- Texinfo options ------------------------------------------------------

texinfo_documents = [
    (master_doc, 'AutoForge', 'AutoForge Documentation',
     author, 'AutoForge', 'One line description of project.',
     'Miscellaneous'),
]

# -- EPUB options --------------------------------------------------------

epub_title = project
epub_exclude_files = ['search.html']
