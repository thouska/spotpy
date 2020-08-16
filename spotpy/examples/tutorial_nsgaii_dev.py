from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import spotpy


from spotpy.examples.spot_setup_dtlz1 import spot_setup
from spotpy.algorithms.nsgaii_dev import TournamentSelection,Crossover,PolynomialMutation


if __name__ == "__main__":
    #Create samplers for every algorithm:
    results=[]
    spot_setup=spot_setup(n_var=5,n_obj=3)
    generations=10
    n_pop = 30
    skip_duplicates = True

    sampler=spotpy.algorithms.NSGAII_DEV(selection = TournamentSelection(pressure=2) ,
                                 crossover = Crossover(crossProb=0.9),
                                 mutation = PolynomialMutation(prob_mut=0.25,eta_mut=30),
                                 spot_setup=spot_setup,
                                 dbname='NSGA2',
                                 dbformat='csv',
                                 save_sim=True,
                                 parallel='mpi')
    sampler.sample(generations,n_pop=n_pop,skip_duplicates=skip_duplicates) 
