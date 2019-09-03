# -*- coding: utf-8 -*-
"""
Tests the describe module

:author: philippkraft
"""
from __future__ import unicode_literals, absolute_import, division, print_function
import sys
import unittest
sys.path.insert(0, '.')

import spotpy
from spotpy.parameter import Uniform
from io import StringIO

import inspect

class SpotSetup(object):
    """
    Just a fun setup using non ASCIÎ letters to chällenge describe in Python 2
    """
    a = Uniform(-1, 1, doc='α parameter')
    beta = Uniform(-1, 1, doc='β parameter')

    @staticmethod
    def simulation(par):
        return [par.a + par.beta]

    @staticmethod
    def evaluation():
        return [0]

    @staticmethod
    def objectivefunction(simulation, evaluation):
        return abs(simulation[0] - evaluation[0])


class TestDescribe(unittest.TestCase):

    def test_setup(self):
        t = spotpy.describe.setup(SpotSetup())
        io = StringIO()
        io.write(t)
        self.assertGreaterEqual(io.getvalue().count('\n'), 6)

    def sampler_test(self, sampler=None):
        if not sampler:
            return
        t = spotpy.describe.describe(sampler)
        io = StringIO()
        io.write(t)
        line_count = io.getvalue().count('\n')
        self.assertGreaterEqual(line_count, 14, '<{}> description to short ({} lines, but >14 expected)'
                                .format(sampler, line_count))

    def test_mc_sampler(self):
        sampler = spotpy.algorithms.mc(spot_setup=SpotSetup(), dbformat='ram', dbname='äöü')
        self.sampler_test(sampler)

    def test_rope_sampler(self):
        sampler = spotpy.algorithms.rope(spot_setup=SpotSetup(), dbformat='ram', dbname='äöü')
        self.sampler_test(sampler)


    def test_all_algorithms(self):
        for sname, scls in inspect.getmembers(spotpy.algorithms, inspect.isclass):
            if not sname.startswith('_'):
                model = SpotSetup()
                sampler = scls(spot_setup=model, dbformat='ram', dbname='äöü')
                self.sampler_test(sampler)

if sys.version_info > (3, 5):

    class TestDescribeRst(unittest.TestCase):

        def test_setup_rst(self):
            setup = SpotSetup()
            rst = spotpy.describe.rst(setup)

        def test_sampler_rst(self):
            for sname, scls in inspect.getmembers(spotpy.algorithms, inspect.isclass):
                if not sname.startswith('_'):
                    model = SpotSetup()
                    sampler = scls(spot_setup=model, dbformat='ram', dbname='äöü')
                    rst = spotpy.describe.rst(sampler)

                    rst.append('This is a test for ``rst.append().``\n' * 10, 'Appending', 1)
                    rst.append_math(r'c = \sqrt{a^2 + b^2}')
                    rst.append(title='Image', titlelevel=2)
                    rst.append_image('https://img.shields.io/travis/thouska/spotpy/master.svg',
                                     target='https://travis-ci.org/thouska/spotpy',
                                     width='200px')

                    rst.as_html()


if __name__ == '__main__':
    unittest.main()
