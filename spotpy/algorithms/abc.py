# -*- coding: utf-8 -*-
'''

:author: Patrick Lauer

This class holds the Artificial Bee Colony(ABC) algorithm, based on Karaboga (2007):

D. Karaboga, AN IDEA BASED ON HONEY BEE SWARM FOR NUMERICAL OPTIMIZATION,TECHNICAL REPORT-TR06, Erciyes University, Engineering Faculty, Computer Engineering Department 2005.

D. Karaboga, B. Basturk, A powerful and Efficient Algorithm for Numerical Function Optimization: Artificial Bee Colony (ABC) Algorithm, Journal of Global Optimization, Volume:39, Issue:3,pp:459-171, November 2007,ISSN:0925-5001 , doi: 10.1007/s10898-007-9149-x

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import spotpy
import numpy as np
import time 
import random
import itertools

class abc(_algorithm):
    '''
    Implements the ABC algorithm from Karaboga (2007).
    
    Input
    ----------
    spot_setup: class
        model: function 
            Should be callable with a parameter combination of the parameter-function 
            and return an list of simulation results (as long as evaluation list)
        parameter: function
            When called, it should return a random parameter combination. Which can 
            be e.g. uniform or Gaussian
        objectivefunction: function 
            Should return the objectivefunction for a given list of a model simulation and 
            observation.
        evaluation: function
            Should return the true values as return by the model.
            
    dbname: str
        * Name of the database where parameter, objectivefunction value and simulation results will be saved.
    
    dbformat: str
        * ram: fast suited for short sampling time. no file will be created and results are saved in an array.
        * csv: A csv file will be created, which you can import afterwards.        

    parallel: str
        * seq: Sequentiel sampling (default): Normal iterations on one core of your cpu.
        * mpc: Multi processing: Iterations on all available cores on your cpu (recommended for windows os).
        * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).
    
    save_sim: boolean
        *True:  Simulation results will be saved
        *False: Simulationt results will not be saved
     '''
    def __init__(self, spot_setup, dbname=None, dbformat=None, parallel='seq',save_sim=True):

        _algorithm.__init__(self,spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,save_sim=save_sim)
    
    def find_min_max(self):
        randompar=self.parameter()['random']        
        for i in range(1000):
            randompar=np.column_stack((randompar,self.parameter()['random']))
        return np.amin(randompar,axis=1),np.amax(randompar,axis=1)
    
    def simulate(self,id_params_tuple):
        id,params = id_params_tuple
        simulations=self.model(params)
        return id,params,simulations

    def mutate(self):
        random.seed()
        x=0.25
        while x==0.25 or x==0.5 or x== 0.75:
            x=random.random()
        step=random.randint(100, 10000000)
        call=0
        while step > call:
            x=4*x*(1-x)
            call+=1
        return x

    def sample(self,repetitions,lb,ub,eb=48,a=(1/10),kstop=100,pcento=0.0000001,peps=0.0001):
        """

        
        Parameters
        ----------
        repetitions: int
            maximum number of function evaluations allowed during optimization
        eb: int
            number of employed bees (half of population size)
        kstop: int
            maximum number of evolution loops before convergency
        pcento: int 
            the percentage change allowed in kstop loops before convergency
        peps: float
            Convergence criterium        
        """
        #Initialize the Progress bar
        starttime    = time.time()
        intervaltime = starttime
        #Initialize ABC parameters:
        randompar=self.parameter()['random']
        self.nopt=randompar.size
        #print(params[0][1])
        print(self.nopt)
        self.limit=eb/2
        random.seed()
        print(self.parameter()['name'])
        #Initialization
        work=[]
        param_generator = ((rep,list(self.parameter()['random'])) for rep in range(eb))
        for rep,randompar,simulations in self.repeat(param_generator):
            print(time.localtime())
            #Calculate the objective function
            like = self.objectivefunction(evaluation = self.evaluation, simulation = simulations)
            self.status(rep,like,randompar)
            #Save everything in the database
            self.datawriter.save(like,randompar,simulations=simulations)
            c=0
            p=0
            work.append([like,randompar,like,randompar,c,p])
            #Progress bar
            acttime=time.time()
            
            #get str showing approximate timeleft to end of simulation in H, M, S 
        
            timestr = time.strftime("%H:%M:%S", time.gmtime(round(((acttime-starttime)/
                                   (rep + 1))*(repetitions-(rep + 1 )))))
            #Refresh progressbar every second
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g) est. time remaining: %s' % (rep,repetitions,
                     self.status.objectivefunction,timestr)
                print(text)
                intervaltime=time.time()

        nloop=0
        icall=0
        gnrng=1e100
        while icall<repetitions and gnrng>peps: #and criter_change>pcento:
            psum=0
        #Employed bee phase
            #generieren der Nachbarparameter aus work        
            for i,val in enumerate(work):
                k=i
                while k==i: k=random.randint(0,(eb-1))
                j=random.randint(0,(self.nopt-1))
                work[i][3][j]=work[i][1][j]+random.uniform(-a,a)*(work[i][1][j]-work[k][1][j])
                if work[i][3][j]<lb[j]: work[i][3][j]=lb[j]
                if work[i][3][j]>ub[j]: work[i][3][j]=ub[j]
                '''
                #Scout bee phase
                if work[i][4] >= self.limit:
                    work[i][3]=self.parameter()['random']
                    work[i][4]=0
                '''
                #Simulationen durchführen

            param_generator = ((rep,work[rep][3]) for rep in range(eb)) 
            for rep,randompar,simulations in self.repeat(param_generator):
                #print(rep)
                clike = self.objectivefunction(evaluation = self.evaluation, simulation = simulations)
                #Fitting vergleichen/Counter erhöhen
                if clike > work[rep][0]:
                    work[rep][1]=work[rep][3]
                    work[rep][0]=clike
                    work[rep][4]=0
                else:
                    work[rep][4]=work[rep][4]+1
                self.status(rep,work[rep][0],work[rep][1])
                self.datawriter.save(clike,work[rep][3],simulations=simulations,chains=icall)
                icall += 1
                print(time.localtime(),icall)
            #Wahrscheinlichkeitsverteilung erstellen
            bn=[]
            for i,val in enumerate(work):
                psum=psum+(1/work[i][0])
            for i,val in enumerate(work):
                work[i][5]=((1/work[i][0])/psum)
                bn.append(work[i][5])
                #print(i,work[i][0])
            bounds = np.cumsum(bn)
                #print(work[i][5])
            #print(work)
            #Onlooker bee phase
            for i,val in enumerate(work):
                pn=random.uniform(0,1)
                k=i
                while k==i:
                    k=random.randint(0,eb-1)
                for t,vol in enumerate(bounds):
                    if bounds[t]-pn>=0:
                        z=t
                        break
                j=random.randint(0,(self.nopt-1))
                work[i][3][j]=work[z][1][j]+random.uniform(-a,a)*(work[z][1][j]-work[k][1][j])
                if work[i][3][j]<lb[j]: work[i][3][j]=lb[j]
                if work[i][3][j]>ub[j]: work[i][3][j]=ub[j]                
            param_generator = ((rep,work[rep][3]) for rep in range(eb))             
            for rep,randompar,simulations in self.repeat(param_generator):
                clike = self.objectivefunction(evaluation = self.evaluation, simulation = simulations)
            #Fitting vergleichen/Counter erhöhen
                if clike > work[rep][0]:
                    work[rep][1]=work[rep][3]
                    work[rep][0]=clike
                    work[rep][4]=0
                else:
                    work[rep][4]=work[rep][4]+1
                self.status(rep,work[rep][0],work[rep][1])
                self.datawriter.save(clike,work[rep][3],simulations=simulations,chains=icall)

                icall += 1
                print(time.localtime(),icall)
            #print(-self.status.objectivefunction)
            for i,val in enumerate(work):
                if work[i][4] >= self.limit:
                    work[i][1]=self.parameter()['random']
                    work[i][4]=0
                    t,work[i][0],simulations=self.simulate((icall,work[i][1]))
                    clike = self.objectivefunction(evaluation = self.evaluation, simulation = simulations)
                    self.datawriter.save(clike,work[rep][3],simulations=simulations,chains=icall)
                    work[i][0]=clike
                    icall += 1
                    print(time.localtime(),icall)
                    #print('!',icall)
            gnrng=-self.status.objectivefunction
            text='%i of %i (best like=%g) est. time remaining: %s' % (icall,repetitions,self.status.objectivefunction,timestr)
            print(text)
            if icall >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            if gnrng < peps:
                print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
        print('Best parameter set:')
        print(self.status.params)
        text='Duration:'+str(round((acttime-starttime),2))+' s'
        print(-self.status.objectivefunction)
        print(icall)
        try:
            self.datawriter.finalize()
        except AttributeError: #Happens if no database was assigned
            pass