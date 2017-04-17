#!/usr/bin/env python3
# -*- coding: utf-8 -*-

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode']

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'rBCM'
copyright = '2017, Lucas Kolstad'
author = 'Lucas Kolstad'
version = '0.2'
release = '0'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'default'
# html_theme_options = {}
# html_static_path = ['_static']
htmlhelp_basename = 'rBCMdoc'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_documents = [
    (master_doc, 'rBCM.tex', 'rBCM Documentation',
     'Lucas Kolstad', 'manual'),
]

man_pages = [
    (master_doc, 'rbcm', 'rBCM Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'rBCM', 'rBCM Documentation',
     author, 'rBCM', 'One line description of project.',
     'Miscellaneous'),
]
