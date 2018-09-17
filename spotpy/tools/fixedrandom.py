'''
Copyright (c) 2018 by Benjamin Manns
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Benjamin Manns
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import os

# replaces numpy.random module in a way

class FixedRandomizerEndOfDataException(Exception):
    pass


class FixedRandomizer():
    def __init__(self):
        self.uniform_counter = 0
        self.normal_counter = 0
        self.uniform_list=list(np.loadtxt(os.path.dirname(__file__)+"/uniform_list.txt"))

        self.uniform_list*=3
        self.max_normal_counter = 10000
        self.max_uniform_counter = 30000

        self.normal_list = list(np.loadtxt(os.path.dirname(__file__)+"/normal_list.txt"))

    def rand(self,dim_x=1,dim_y=1):
        #x = np.array(dim_y * [dim_x * [0]])
        x = dim_x * [0]
        for i in range(dim_x):
            #for j in range(dim_y):
            if self.uniform_counter < self.max_uniform_counter:
                x[i] = self.uniform_list[self.uniform_counter]
                self.uniform_counter = self.uniform_counter + 1
            else:
                raise FixedRandomizerEndOfDataException("No more data left. Counter is: "+str(self.uniform_counter))
        if len(x) == 1:
            return x[0]
        else:
            return x

    def randint(self,x_from,x_to):
        vals = [j for j in range(x_from,x_to)]
        vals_size = len(vals)
        if vals_size == 0:
            raise ValueError("x_to >= x_from")
        fraq = 1 / vals_size
        if self.uniform_counter < self.max_uniform_counter:
            q_uni = self.uniform_list[self.uniform_counter]
            pos = np.int(np.floor(q_uni / fraq))
            self.uniform_counter += 1
            return vals[pos]
        else:
            raise FixedRandomizerEndOfDataException("No more data left.")

    def normal(self,loc,scale,size=1):
        x = []
        for j in range(size):
            if self.normal_counter < self.max_normal_counter:
                x.append(self.normal_list[self.normal_counter]*scale + loc)
                self.normal_counter += 1

            else:
                raise FixedRandomizerEndOfDataException("No more data left.")
        if len(x) == 1:
            return x[0]
        else:
            return x


# TODO UNITEST irgendwie

    #print(f_rand.normal(12,1,12))
    #print(np.random.normal(12,1,12))

    #f_rand.normal(12,1,12)-
    # print(np.var(np.random.normal(12,1,12)-np.random.normal(12,1,12)))
    # print("-------------------------")




# TODO Convert this to unittest
# f_rand = FixedRandomizer()
# print(f_rand.my_rand(10))
# print(np.random.rand(10))

# for k in range(100):
#     print(f_rand.my_randint(1,101010))
#     print(np.random.randint(1,101010))
#     print("----------------------")

# print(np.random.normal(0, 1))
# print(f_rand.my_randn(0,1))

