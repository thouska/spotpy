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



    class rst:
        """
        Creates a reStructuredText description of a sampler or a setup

        Usage:
        >>>description = spotpy.describe.rst(sampler)
        >>>print(description) # Prints the rst source text
        >>># Add additional text section
        >>>description.append('#. One idea' + '\n' + '#. Another one.', title='Ideas', titlelevel=2)
        >>>description.append_image('media/image.png')
        >>>print(description.as_html()) # Create html
        >>>description.show_in_browser()
        """

        caption_characters = '=-#~*+^'

        def __init__(self, setup_or_sampler):
            """
            Creates a reStructuredText description of a sampler or a setup
            :param setup_or_sampler: Either a spotpy.algorithm sampler or a spotpy setup
            """
            if isinstance(setup_or_sampler, _algorithm):
                self.setup = setup_or_sampler.setup
                self.sampler = setup_or_sampler
                self.rst_text = [self._sampler_text()]
            else:
                self.setup = setup_or_sampler
                self.sampler = None
                self.rst_text = []

            if self.setup:
                self.rst_text.append(self._setup_text())

        def append(self, text='', title=None, titlelevel=1):
            """
            Appends additional descriptions in reStructured text to the generated text
            :param text: The rst text to add
            :param title: A title for the text
            :param titlelevel: The level of the section (0->h1.title, 1->h1, 2->h2, etc.)
            :return:
            """
            res = '\n'
            if title:
                res += rst._as_rst_caption(title, titlelevel)
            self.rst_text.append(res + text)

        def append_image(self, imgpath, **kwargs):
            """
            Links an image to the output
            :param imgpath: Path to the image (must be found from the http server)
            :param kwargs: Any keyword with value is translated in rst as `:keyword: value`
                            and added to the image description

            >>>description.append_image('https://img.shields.io/travis/thouska/spotpy/master.svg',
            ...                         target='https://github.com/thouska',
            ...                         width='200px')
            """
            rst = '.. image:: {}'.format(imgpath)
            for k, v in kwargs.items():
                rst += '\n  :{}: {}'.format(k, v)
            rst += '\n'
            self.append(rst)

        def append_math(self, latex):
            """
            Appends a block equation to the output
            :param latex: Latex formula
            """
            rst =  '.. math::\n'
            rst += '  ' + latex + '\n'
            self.append(rst)

        def __str__(self):
            return '\n'.join(self.rst_text)

        @classmethod
        def _as_rst_caption(cls, s, level=1):
            """
            Marks text as a section caption
            :param s: String to be marked as caption
            :param level: Caption level 0-6, translates to 0=h1.title, 1=h1, 2=h2, etc.
            :return: The string as rst caption
            """
            return s + '\n' + cls.caption_characters[level] * len(s) + '\n\n'

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
                    'output_encoding': 'unicode'}
            res = publish_string(source=str(self),
                                 writer_name='html5',
                                 settings_overrides=args)
            style_idx = res.index('</style>')
            css = css or self.css
            # Include css
            res = res[:style_idx] + css + res[style_idx:]
            return res

        def show_in_browser(self, filename=None, css=None):
            """
            Writes the content as html to disk and opens a browser showing the result

            :param filename: The html filename, if None use <setup class name>.html
            :param css: A style string, if None the default style is used
            """
            html = self.as_html(css).replace('unicode', 'utf-8')
            fn = filename or type(self.setup).__name__ + '.html'
            path = Path(fn).absolute()
            path.write_text(html, encoding='utf-8')
            webbrowser.open_new_tab(path.as_uri())

        def _sampler_text(self):
            """
            Generates the rst for the sampler
            :return:
            """
            obj = self.sampler
            cname = rst._as_rst_caption(type(obj).__name__, 0)
            s = [
                 '- **db format:** ' + obj.dbformat,
                 '- **db name:** ' + obj.dbname,
                 '- **save simulation:** ' + str(obj.save_sim),
                 '- **parallel:** ' + type(obj.repeat).__module__.split('.')[-1],
                 '', ''
                 ]
            return cname + _getdoc(obj).strip('\n') + '\n\n' + '\n'.join(s)

        def _setup_text(self):
            """
            Generates the rst for the setup
            :return:
            """
            # Get class name
            obj = self.setup
            cname = rst._as_rst_caption(type(obj).__name__, 0)
            # Add doc string
            mdoc = _getdoc(obj).strip('\n').replace('\r', '\n') + '\n\n'
            # Get parameters from class
            param_caption = rst._as_rst_caption('Parameters', 1)
            params = '\n'.join('#. **{p.name}:** {p}'.format(p=p) for p in get_parameters_from_setup(obj))
            return cname + mdoc + param_caption + params

