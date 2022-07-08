# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Philipp Kraft
"""

import io
import os
import unittest

from click.testing import CliRunner

from spotpy.algorithms import _algorithm
from spotpy.cli import cli, get_config_from_file, get_sampler_from_string, run
from spotpy.examples.spot_setup_rosenbrock import spot_setup


class TestCLI(unittest.TestCase):
    def test_cli_config(self):

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config"])
        if result.exception:
            raise result.exception
        self.assertEqual(result.exit_code, 0)

    def test_cli_run(self):
        setup = spot_setup()
        runner = CliRunner()
        result = runner.invoke(run, ["-n 10", "-s", "lhs", "-p", "seq"], obj=setup)
        self.assertEqual(result.exit_code, 0)

    def test_config_file(self):
        config = dict(n="10", sampler="mc", parallel="seq")

        # 1. Create config file
        with io.open("spotpy.conf", "w", encoding="utf-8") as f:
            f.write("# A comment to be ignored\n")
            f.write("Some nonsense to fail\n")
            for k, v in config.items():
                f.write("{} = {}\n".format(k, v))

        try:
            # 2. Read config file
            config_from_file = get_config_from_file()
            self.assertDictEqual(
                config,
                config_from_file,
                "Configuration from file is not the same as before",
            )
        finally:
            os.unlink("spotpy.conf")

    def test_sampler_from_string(self):
        sampler_names = (
            "abc|demcz|dream|fast|fscabc|lhs|mc|mcmc|mle|rope|sa|sceua".split("|")
        )
        samplers = [
            get_sampler_from_string(sampler_name) for sampler_name in sampler_names
        ]
        wrong_samplers = [
            n for n, c in zip(sampler_names, samplers) if not issubclass(c, _algorithm)
        ]
        self.assertFalse(
            wrong_samplers, "Samplers not found from name: " + ", ".join(wrong_samplers)
        )


if __name__ == "__main__":
    unittest.main()
