# -*- coding: utf-8 -*-
'''
Copyright 2018 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Benjamin Manns

This class holds example code how to use the SQL database class with a huge set of simulation data
'''

import numpy as np
import spotpy
import sqlite3, time, pandas as pd

class spot_setup(object):
    slow = 1000
    def __init__(self):
        x = spotpy.parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10)
        y = spotpy.parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10)

    def simulation(self, vector):
        return [np.random.uniform(0, 1, 2000)] * 50

    def evaluation(self):
        return [np.random.uniform(0, 1, 2000)] * 50

    def objectivefunction(self, simulation, evaluation):
        like = -spotpy.objectivefunctions.rmse(evaluation=evaluation, simulation=simulation)
        return like

spot_setup = spot_setup()

dbname = "TUT_SQL"

# sampler = spotpy.algorithms.mc(spot_setup,dbname=dbname, dbformat="csv")
# sampler.sample(1000)




start_sql = time.time()
sqlite3.connect("TUT_SQL.db")
db = sqlite3.connect("TUT_SQL.db")

# get all column names of table `dbname` of sqlite
columns = [h[1] for h in db.execute("PRAGMA table_info("+dbname+")")]

# simulation are concatenated in one string because of length limiting for column names in sqlite
sim_24 = [list(map(float,h[0].split(","))) for h in db.execute("select simulation_24 from " + dbname)]
# therefor we split up this string simply by using:
print("Size of %s x %s" % (len(sim_24), len(sim_24[0])))
print(sim_24)


# getting all likelihoods with their position:
likeli = [(i, l[0]) for i, l in
          enumerate(db.execute("select "+ ",".join([c for c in columns if c.startswith("like")]) +" from "+dbname+";"))]
print(likeli)


end_sql = time.time()




start_csv = time.time()
ofile = open(dbname + '.csv')
line = ofile.readline()
names = line.split(',')
ofile.close()
discharge_fields = [word for word in names if word.startswith('simulation24')]
discharge_data = pd.read_csv(dbname + '.csv', delimiter=',', usecols=discharge_fields, encoding='latin1')
print(discharge_data)

# also likelihood

like_fields = [word for word in names if word.startswith('like')]
like_data = pd.read_csv(dbname + '.csv', delimiter=',', usecols=like_fields, encoding='latin1')
print(like_data)



end_csv = time.time()

# now benchmarking and comparing
print("SQL loading a set of simulations needs a time of %s seconds" % (end_sql-start_sql))
print("CSV loading a set of simulations needs a time of %s seconds" % (end_csv-start_csv))