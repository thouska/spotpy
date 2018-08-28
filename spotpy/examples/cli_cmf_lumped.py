"""
Shows the usage of the matplotlib GUI

Needs at least Python 3.5
"""

from __future__ import division, print_function, unicode_literals


from spotpy.cli import main
#from spotpy.examples.spot_setup_cmf_lumped import SingleStorage as spot_setup
from spotpy.examples.spot_setup_rosenbrock import spot_setup

if __name__ == '__main__':
    setup = spot_setup()
    main(setup)