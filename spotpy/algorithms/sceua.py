# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska and Stijn Van Hoey

This class holds the Shuffled Complex Evolution Algortithm (SCE-UA) algorithm, based on Duan (1994):

Duan, Q., Sorooshian, S. and Gupta, V. K.: Optimal use of the SCE-UA global optimization method for calibrating watershed models, J. Hydrol., 158(3), 265–284, 1994.

Based on Optimization_SCE
Copyright (c) 2011 Stijn Van Hoey.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import spotpy
import numpy as np
import time 

class sceua(_algorithm):
    '''
    Implements the SCE-UA algorithm from Duan (1994).
    
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
    """
    def simulate(self,params):
        if self.repeat.phase=='burnin':
            id,params = params
            simulations =
    """ 
    def simulate(self,id_params_tuple):
        """This overwrites the simple wrapper function of _algorithms.py
        and makes a two phase mpi parallelization possbile: 
        1) burn-in
        2) complex evolution
        """
        if not self.repeat.phase: #burn-in
            id,params = id_params_tuple
            return id,params,self.model(params)
        
        else:#complex-evolution
            igs,x,xf,icall,cx,cf,sce_vars= id_params_tuple
            self.npg,self.nopt,self.ngs,self.nspl,self.nps,self.bl,self.bu, self.status = sce_vars
            # Partition the population into complexes (sub-populations);
#            cx=np.zeros((self.npg,self.nopt))
#            cf=np.zeros((self.npg))
            #print(igs)
            k1=np.arange(self.npg,dtype=int)
            k2=k1*self.ngs+igs
            cx[k1,:] = x[k2,:]
            cf[k1] = xf[k2]
            # Evolve sub-population igs for self.self.nspl steps:
            likes=[]
            sims=[]
            pars=[]
            for loop in xrange(self.nspl):
                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs=np.array([0]*self.nps)
                lcs[0] = 1
                for k3 in range(1,self.nps):
                    for i in range(1000):
                        #lpos = 1 + int(np.floor(self.npg+0.5-np.sqrt((self.npg+0.5)**2 - self.npg*(self.npg+1)*np.random.random())))
                        lpos = int(np.floor(self.npg+0.5-np.sqrt((self.npg+0.5)**2 - self.npg*(self.npg+1)*np.random.random())))
                        #idx=find(lcs(1:k3-1)==lpos)
                        idx=(lcs[0:k3]==lpos).nonzero()  #check of element al eens gekozen
                        if idx[0].size == 0:
                            break
                    lcs[k3] = lpos
                lcs.sort()
                
                # Construct the simplex:
                s = np.zeros((self.nps,self.nopt))
                s=cx[lcs,:]
                sf = cf[lcs]
                
                snew,fnew,icall,simulation = self._cceua(s,sf,icall)
                likes.append(fnew)                
                pars.append(list(snew))
                self.status(igs,-fnew,snew)                
                sims.append(list(simulation))
                #self.datawriter.save(-fnew,list(snew), simulations = list(simulation),chains = igs)   
                # Replace the worst point in Simplex with the new point:
                s[-1,:] = snew
                sf[-1] = fnew
                
                # Replace the simplex into the complex;
                cx[lcs,:] = s
                cf[lcs] = sf
                
                # Sort the complex;
                idx = np.argsort(cf)
                cf = np.sort(cf)
                cx=cx[idx,:]
                
            # Replace the complex back into the population;


            return igs,likes,pars,sims,cx,cf,k1,k2

    def sample(self,repetitions,ngs=20,kstop=100,pcento=0.0000001,peps=0.0000001):
        """
        Samples from parameter distributions using SCE-UA (Duan, 2004), 
        converted to python by Van Hoey (2011).
        
        Parameters
        ----------
        repetitions: int
            maximum number of function evaluations allowed during optimization
        ngs: int
            number of complexes (sub-populations), take more then the number of
            analysed parameters
        kstop: int
            maximum number of evolution loops before convergency
        pcento: int 
            the percentage change allowed in kstop loops before convergency
        peps: float
            Convergence criterium        
        """
        # Initialize the Progress bar
        starttime    = time.time()
        intervaltime = starttime
        # Initialize SCE parameters:
        self.ngs=ngs
        randompar=self.parameter()['random']        
        self.nopt=randompar.size
        self.npg=2*self.nopt+1
        self.nps=self.nopt+1
        self.nspl=self.npg
        npt=self.npg*self.ngs
        self.iseed=1        
        self.bl,self.bu = self.find_min_max()
        bound = self.bu-self.bl  #np.array


        # Create an initial population to fill array x(npt,self.self.nopt):
        x=self._sampleinputmatrix(npt,self.nopt)
        
        #Set Ininitial parameter position         
        #iniflg=1        

        nloop=0
        icall=0
        xf=np.zeros(npt)
        
        #Burn in
        param_generator = ((rep,list(x[rep])) for rep in xrange(int(npt)))        
        for rep,randompar,simulations in self.repeat(param_generator):        
            #Calculate the objective function
            like = spotpy.objectivefunctions.rmse(self.evaluation,simulations)
            #Save everything in the database
            self.status(rep,-like,randompar)
            xf[rep] = like                        
            self.datawriter.save(-like,randompar,simulations=simulations)            
            icall += 1
            #Progress bar
            acttime=time.time()
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (rep,repetitions,self.status.objectivefunction)
                print(text)
                intervaltime=time.time()
   
        # Sort the population in order of increasing function values;
        idx = np.argsort(xf)
        xf = np.sort(xf)
        x=x[idx,:]

        # Record the best and worst points;
        bestx=x[0,:]
        bestf=xf[0]
        #worstx=x[-1,:]
        #worstf=xf[-1]

        BESTF=bestf
        BESTX=bestx
        ICALL=icall

        # Compute the standard deviation for each parameter
        #xnstd=np.std(x,axis=0)

        # Computes the normalized geometric range of the parameters
        gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

        # Check for convergency;
        if icall >= repetitions:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ')
            print(repetitions)
            print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
            print(icall)
            print('OF THE INITIAL LOOP!')

        if gnrng < peps:
            print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

        # Begin evolution loops:
        nloop = 0
        criter=[]
        criter_change=1e+5
        
        starttime=time.time()
        intervaltime=starttime
        acttime=time.time()
        self.repeat.setphase('ComplexEvo')
      
        while icall<repetitions and gnrng>peps and criter_change>pcento:
            nloop+=1
            #print nloop
            #print 'Start MPI'
            # Loop on complexes (sub-populations);
            cx=np.zeros((self.npg,self.nopt))
            cf=np.zeros((self.npg))
            
            sce_vars=[self.npg,self.nopt,self.ngs,self.nspl,self.nps,self.bl,self.bu, self.status]
            param_generator = ((rep,x,xf,icall,cx,cf,sce_vars) for rep in xrange(int(self.ngs))) 
            for igs,likes,pars,sims,cx,cf,k1,k2 in self.repeat(param_generator):
                icall+=len(likes)
                x[k2,:] = cx[k1,:]
                xf[k2] = cf[k1]
            
                for i in range(len(likes)):
                    self.status(icall,-likes[i],pars[i])
                    self.datawriter.save(-likes[i],list(pars[i]), simulations = list(sims[i]),chains = igs)   
 
            #Progress bar
            acttime=time.time()
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (icall,repetitions,self.status.objectivefunction)
                print(text)
                intervaltime=time.time()
            # End of Loop on Complex Evolution;
    
            # Shuffled the complexes;
            idx = np.argsort(xf)
            xf = np.sort(xf)
            x=x[idx,:]

            # Record the best and worst points;
            bestx=x[0,:]
            bestf=xf[0]
            #worstx=x[-1,:]
            #worstf=xf[-1]

            BESTX = np.append(BESTX,bestx, axis=0) #appenden en op einde reshapen!!
            BESTF = np.append(BESTF,bestf)
            ICALL = np.append(ICALL,icall)

            # Computes the normalized geometric range of the parameters
            gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

            # Check for convergency;
            if icall >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            if gnrng < peps:
                print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

            criter=np.append(criter,bestf)

            if nloop >= kstop: #nodig zodat minimum zoveel doorlopen worden
                criter_change= np.abs(criter[nloop-1]-criter[nloop-kstop])*100
                criter_change= criter_change/np.mean(np.abs(criter[nloop-kstop:nloop]))
                if criter_change < pcento:
                    text='THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f' %(kstop,pcento)
                    print(text)
                    print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')

        # End of the Outer Loops
        text='SEARCH WAS STOPPED AT TRIAL NUMBER: %d' %icall
        print(text)
        text='NORMALIZED GEOMETRIC RANGE = %f'  %gnrng
        print(text)
        text='THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f' %(kstop,criter_change)
        print(text)

        #reshape BESTX
        BESTX=BESTX.reshape(BESTX.size/self.nopt,self.nopt)
        self.repeat.terminate()
        try:
            self.datawriter.finalize()
        except AttributeError: #Happens if no database was assigned
            pass
        print('Best parameter set')
        print(self.status.params)
        text='Duration:'+str(round((acttime-starttime),2))+' s'
        print(text)
 
    
 
    def _cceua(self,s,sf,icall):
            #  This is the subroutine for generating a new point in a simplex
            #
            #   s(.,.) = the sorted simplex in order of increasing function values
            #   s(.) = function values in increasing order
            #
            # LIST OF LOCAL VARIABLES
            #   sb(.) = the best point of the simplex
            #   sw(.) = the worst point of the simplex
            #   w2(.) = the second worst point of the simplex
            #   fw = function value of the worst point
            #   ce(.) = the centroid of the simplex excluding wo
            #   snew(.) = new point generated from the simplex
            #   iviol = flag indicating if constraints are violated
            #         = 1 , yes
            #         = 0 , no

        self.nps,self.nopt=s.shape
        alpha = 1.0
        beta = 0.5

        # Assign the best and worst points:
        sw=s[-1,:]
        fw=sf[-1]

        # Compute the centroid of the simplex excluding the worst point:
        ce= np.mean(s[:-1,:],axis=0)

        # Attempt a reflection point
        snew = ce + alpha*(ce-sw)

        # Check if is outside the bounds:
        ibound=0
        s1=snew-self.bl
        idx=(s1<0).nonzero()
        if idx[0].size <> 0:
            ibound=1

        s1=self.bu-snew
        idx=(s1<0).nonzero()
        if idx[0].size <> 0:
            ibound=2

        if ibound >= 1:
            snew = self._sampleinputmatrix(1,self.nopt)[0]  #checken!!

        ##    fnew = functn(self.nopt,snew);
        simulation=self.model(snew)
        like = spotpy.objectivefunctions.rmse(simulation,self.evaluation)
        #like=-self.objectivefunction(simulation,self.evaluation)
        fnew = like#bcf.algorithms._makeSCEUAformat(self.model,self.observations,snew)
        #fnew = self.model(snew)
        icall += 1

        # Reflection failed; now attempt a contraction point:
        if fnew > fw:
            snew = sw + beta*(ce-sw)
            simulation=self.model(snew)
            like = spotpy.objectivefunctions.rmse(simulation,self.evaluation) 
            #like=-self.objectivefunction(simulation,self.evaluation)
            fnew = like
            icall += 1

        # Both reflection and contraction have failed, attempt a random point;
            if fnew > fw:
                snew = self._sampleinputmatrix(1,self.nopt)[0]  #checken!!
                simulation=self.model(snew)
                like = spotpy.objectivefunctions.rmse(simulation,self.evaluation)
                #like=-self.objectivefunction(simulation,self.evaluation)  
                fnew = like#bcf.algorithms._makeSCEUAformat(self.model,self.observations,snew)
                #print 'NSE = '+str((fnew-1)*-1)                    
                #fnew = self.model(snew)
                icall += 1

        # END OF CCE
        return snew,fnew,icall,simulation
                                
    def _sampleinputmatrix(self,nrows,npars):
        '''
        Create inputparameter matrix for nrows simualtions,
        for npars with bounds ub and lb (np.array from same size)
        distname gives the initial sampling ditribution (currently one for all parameters)

        returns np.array
        '''   
        x=np.zeros((nrows,npars))
        for i in range(nrows):
            x[i,:]= self.parameter()['random']
        return x        
        # Matrix=np.empty((nrows,npars))
        # for i in range(nrows):
            # Matrix[i]= self.parameter()['random']
        # return Matrix
  
  