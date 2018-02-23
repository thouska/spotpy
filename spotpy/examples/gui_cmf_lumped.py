"""
Shows the usage of the matplotlib GUI

Needs at least Python 3.5
"""

from __future__ import division, print_function, unicode_literals


import datetime
import spotpy
from spotpy.gui.mpl import GUI
from spotpy.examples.spot_setup_cmf_lumped import SingleStorage

if __name__ == '__main__':
    # Create the model
    model = SingleStorage(datetime.datetime(1980, 1, 1),
                          datetime.datetime(1985, 12, 31))
    spotpy.describe.setup(model)
    gui = GUI(model)
    gui.show()
