# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Optimization Tool (SPOTPY).

:author: Tobias Houska

This is the parent class of all algorithms, which can handle the database 
structure during the sample.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import io 

class database(object):
    """
    Parent class for database. It can handle the basic functionalities of all
    databases.
    """
    def __init__(self,dbname,parnames,like,randompar,simulations,chains=1,save_sim=True):
        self.save_sim = save_sim        
        #Just needed for the first line in the database        
        self.header=self.make_header(like,parnames,simulations)

    
    
    def make_header(self,objectivefunction,parnames,simulations):
        if type(objectivefunction)==type([]):
            header=[]
            for i in range(len(objectivefunction)):
                header.append('like'+str(i+1))
        else:
            header   = ['like']
        for name in parnames:
            header.append('par'+str(name))
        if self.save_sim==True:
            if type(simulations[0])==type([]):
                for i in range(len(simulations)):
                    for j in range(len(simulations[i])):
                        header.append('simulation'+str(i)+'_'+str(j))
            else:
                for i in range(len(simulations)):
                    header.append('simulation'+str(i))
        header.append('chain')  
        return header        
        
    def _get_list(self,objectivefunction,parameterlist,simulations=None):
        data=[]
        if type(objectivefunction)==type([]):
            for like in objectivefunction:
                data.append(like)
        else:
            data.append(objectivefunction)
        for par in parameterlist:
            data.append(float(par))
        if self.save_sim==True:
            if type(simulations[0])==type([]):
                for i in range(len(simulations)):
                    for sim in simulations[i]:
                        data.append(float(sim))
            else:
                for sim in simulations:
                    data.append(float(sim))      
        return data
    
      
class ram(database):
    """
    This class saves the process in the working storage. It can be used if
    time matters.
    """
    def __init__(self,dbname,parnames,like,randompar,simulations,chains=1,save_sim=True):
        database.__init__(self,dbname,parnames,like,randompar,simulations,chains=chains,save_sim=save_sim)        
        
        self.ram_par  = [list(randompar)]
        if type(like)==type([]):
            self.ram_like = [like]
        else:
            self.ram_like = [[like]]
        self.ram_sim  = [list(simulations)]
        self.chains   = [[chains]]  
        
    def save(self,objectivefunction,parameterlist,simulations=None,chains=1):
        self.ram_par.append(parameterlist)
        if type(objectivefunction)==type([]):
            self.ram_like.append(objectivefunction)
        else:
            self.ram_like.append([objectivefunction])
        if self.save_sim==True:
            self.ram_sim.append(simulations)
        self.chains.append([chains])
    
    def finalize(self):
        dt = np.dtype({'names': self.header,
                'formats': ['<f8']*len(self.header)})
        
        ram=[]
        for i in range(len(self.ram_like)):
            if self.save_sim==True:
                wrapped=tuple(self.ram_like[i]+self.ram_par[i]+self.ram_sim[i]+self.chains[i])
            else:
                wrapped=tuple(self.ram_like[i]+self.ram_par[i]+self.chains[i])                
            ram.append(wrapped)
        #ignore the first initialization run to reduce the risk of different 
        #objectivefunction mixing
        self.data=np.array(ram,dtype=dt)[1:]

    
    def getdata(self):
        #Expects a finalized database
        return self.data
    
    def get_last_par(self):
        if len(self.ram_par[-1])==1:
            return self.ram_par
        else:
            return self.ram_par[-1]
            


class csv(database):
    """
    This class saves the process in the working storage. It can be used if
    safety matters.
    """
    def __init__(self,dbname,parnames,like,randompar,simulations=None,chains=1,save_sim=True):
        database.__init__(self,dbname,parnames,like,randompar,simulations,chains=chains,save_sim=save_sim)  
        self.chains=chains
        self.dbname=dbname
        #Create a open file, which needs to be closed after the sampling
        self.db=io.open(self.dbname+'.csv', 'wb')
        
        header_as_str=''
        for name in self.header:
            header_as_str=header_as_str+str(name)+','
        header_as_str=header_as_str[:-1]+'\n'  
        self.db.write(bytes(header_as_str.encode('utf-8'))) 
        #self.save(like,randompar,simulations=simulations,save_sim=save_sim)
    
    def save(self,objectivefunction,parameterlist,simulations=None,chains=1):
        self.ram_par=parameterlist
        if self.save_sim==True:
            data=self._get_list(objectivefunction,parameterlist,simulations=simulations)
        else:
            data=self._get_list(objectivefunction,parameterlist,simulations=None)
        for value in data:
            try:
                self.db.write(bytes((str(value)+',').encode('utf-8')))
            except IOError:
                input("Please close the file "+self.dbname+" When done press Enter to continue...")
                self.db.write(bytes((str(value)+',').encode('utf-8')))
        self.db.write(bytes((str(chains)+'\n').encode('utf-8')))

    def finalize(self):
        self.db.close()
        
    def getdata(self,dbname=None):        
        data=np.genfromtxt(self.dbname+'.csv',delimiter=',',names=True)[1:]
        return data

    def get_last_par(self):
        return self.ram_par
    

    

        
