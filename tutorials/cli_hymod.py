"""
Shows the usage of the command line interface CLI

"""

from __future__ import division, print_function, unicode_literals

from spotpy.cli import main
from spotpy.examples.spot_setup_hymod_python import spot_setup


if __name__ == '__main__':
    setup = spot_setup()
    main(setup)
