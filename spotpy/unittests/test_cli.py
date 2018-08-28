from __future__ import division, print_function, unicode_literals

import click
from click.testing import CliRunner
from spotpy.cli import main, cli
from spotpy.examples.spot_setup_rosenbrock import spot_setup
import unittest
import sys

class TestCLI(unittest.TestCase):
    def test_cli(self):

        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--config'])
        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()