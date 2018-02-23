# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python (SPOTPY).
:author: Philipp Kraft

A collection of helper functions to describe spotpy setups

Usage:

>>> spotpy.describe.sampler(sampler)
>>> spotpy.describe.setup(model)
"""
from __future__ import division, absolute_import, unicode_literals
import sys
from .parameter import get_parameters_from_setup
from .algorithms._algorithm import _algorithm
if sys.version_info[0] >= 3:
    from inspect import getdoc as _getdoc
    unicode = str
else:
    def _getdoc(obj):
        u = obj.__doc__
        try:
            return u'\n'.join(l.strip() for l in u.split(u'\n') if l.strip())
        except UnicodeDecodeError:
            raise AssertionError(
                '{}: Docstring uses unicode but {} misses the line ``from __future__ import unicode_literals``'
                .format(obj, type(obj).__module__)
                )

try:
    from docutils.core import publish_string
except ImportError:
    publish_string = None


def describe(obj):
    """
    Returns a long string description of a sampler with its model
    :param obj: A sampler
    :return: str
    """
    return 'Sampler:\n--------\n{}\n\nModel:\n------\n{}'.format(sampler(obj), setup(obj.setup))


def sampler(obj):
    """
    Returns a string representation of the sampler.
    By design, it is rather verbose and returns a
    large multiline description
    :return:
    """
    cname = unicode(type(obj).__name__)
    s = [cname, '=' * len(cname), _getdoc(obj),
         '    db format: ' + obj.dbformat,
         '    db name: ' + obj.dbname,
         '    save simulation: ' + str(obj.save_sim),
         '    parallel: ' + type(obj.repeat).__module__.split('.')[-1]]
    return '\n'.join(s)


def setup(obj):
    """
    Describes a spotpy setup using its class name, docstring and parameters
    :param obj: A spotpy compatible model setup
    :return: A describing string
    """
    # Get class name
    cname = unicode(type(obj).__name__)
    # Add doc string
    mdoc = _getdoc(obj).strip('\n').replace('\r', '\n')
    # Get parameters from class
    params = '\n'.join(' - {p}'.format(p=unicode(p)) for p in get_parameters_from_setup(obj))
    parts = [cname, '=' * len(cname), mdoc, 'Parameters:', '-' * 11, params]
    return '\n'.join(parts)


if sys.version_info > (3, 5):

    from pathlib import Path
    import webbrowser


    def _as_rst_caption(s, caption_sign):
        s = unicode(s)
        return s + '\n' + caption_sign * len(s) + '\n\n'

    class rst:
        """
        Creates a reStructuredText description of a sampler or a setup
        """
        def __init__(self, setup_or_sampler):
            if isinstance(setup_or_sampler, _algorithm):
                self.setup = setup_or_sampler.setup
                self.sampler = setup_or_sampler
                self.rst_text = self._sampler_text()
            else:
                self.setup = setup_or_sampler
                self.sampler = None
                self.rst_text = ''

            if self.setup:
                self.rst_text += self._setup_text()

        def add_rst(self, extra):
            """

            :param extra:
            :return:
            """
            self.rst_text += extra

        def __str__(self):
            if sys.version_info.major < 3:
                return self.rst_text.encode('utf-8', errors='ignore')
            else:
                return self.rst_text

        def __unicode__(self):
            return self.rst_text

        css = """
            body, table, div, p, dl {
                font-family: Lucida Grande, Verdana, Geneva, Arial, sans-serif;
                font-size: 16px;
            }
            li>p {
                margin: 0px;
            }
            /* @group Heading Levels */
            h1.title {
                background-color: #fff;
                color: #0040A0;
                text-align: left;
                font-size: 200%;
                border: solid 2px #1f6992;
            }
            h1 {
             background-color: #1f6992;
             color: #fff;
             padding: .2em .5em;
             font-size: 150%;
            }
            h2 {
             background-color: #cde;
             color: #000;
             padding: .2em .5em;
             border-bottom: solid 2px #1f6992;
             font-size: 120%;
            }
            
            h3 {
                font-size: 100%;
                border-bottom: solid 2px #0040A0;
            }
            div.line {
                font-family: "Lucida Console", "Lucida Sans Typewriter","DejaVu Sans Mono",monospace;
                font-size: 100%;
            }
            
            img {
                max-width: 720px;
            }
    
        """

        def as_html(self, css=None):
            """
            Converts the generated reStructuredText as html5

            :css: A string containing a cascading style sheet. If None, the default css is used
            :return: The html document as string
            """
            if publish_string is None:
                raise NotImplementedError('The docutils package needs to be installed')
            args = {'input_encoding': 'unicode',
                    'output_encoding' : 'unicode'}
            res = publish_string(source=self.rst_text,
                                 writer_name='html5',
                                 settings_overrides=args)
            style_idx = res.index('</style>')
            css = css or self.css
            # Include css
            res = res[:style_idx] + css + res[style_idx:]
            return res

        def show_in_browser(self, filename=None, css=None):
            """
            Writes the content as html to disk
            :param filename: The html filename, if None use <setup class name>.html
            :param css: A style string, if None the default style is used
            """
            html = self.html(css)
            fn = filename or type(self.setup).__name__ + '.html'
            path = Path(fn).absolute()
            path.write_text(html)
            webbrowser.open_new_tab(path.as_uri())

        def _sampler_text(self):
            obj = self.sampler
            cname = _as_rst_caption(type(obj).__name__, '=')
            s = [
                 '- **db format:** ' + obj.dbformat,
                 '- **db name:** ' + obj.dbname,
                 '- **save simulation:** ' + str(obj.save_sim),
                 '- **parallel:** ' + type(obj.repeat).__module__.split('.')[-1],
                 '', ''
                 ]
            return cname + _getdoc(obj).strip('\n') + '\n\n' + '\n'.join(s)

        def _setup_text(self):
            # Get class name
            obj = self.setup
            cname = _as_rst_caption(type(obj).__name__, '=')
            # Add doc string
            mdoc = _getdoc(obj).strip('\n').replace('\r', '\n') + '\n\n'
            # Get parameters from class
            param_caption = _as_rst_caption('Parameters', '-')
            params = '\n'.join('#. **{p.name}:** {p}'.format(p=p) for p in get_parameters_from_setup(obj))
            return cname + mdoc + param_caption + params

