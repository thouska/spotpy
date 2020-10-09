"""
Shows the usage of the matplotlib GUI

Needs at least Python 3.5
"""

from __future__ import division, print_function, unicode_literals


import spotpy
from spotpy.gui.mpl import GUI
from spotpy.examples.spot_setup_hymod_python import spot_setup
from  spotpy.objectivefunctions import rmse 

if __name__ == '__main__':
    setup_class=spot_setup(rmse)
    
    #Select number of maximum allowed repetitions
    rep=10000
        
    # Create the SCE-UA sampler of spotpy, alt_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.rmse) 
    sampler=spotpy.algorithms.sceua(setup_class, dbname='SCEUA_hymod', dbformat='csv', alt_objfun=None)
    
    #Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    #sampler.sample(rep,ngs=10, kstop=50, peps=0.1, pcento=0.1)  

    gui = GUI(spot_setup())
    gui.show()
