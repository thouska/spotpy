

import numpy as np
import math
from spotpy.algorithms import _algorithm
import copy



class TournamentSelection:

    def __init__(self,pressure = 2):
        self.pressure = pressure

    def calc(self,pop_rank):

        n_select = len(pop_rank)
        n_random = n_select * self.pressure #n_select * n_parents * pressure

        n_perms = math.ceil(n_random / len(pop_rank))

        P = random_permuations(n_perms, len(pop_rank))[:n_random]

        P = np.reshape(P, (n_select, self.pressure))

        n_tournament,n_competitors = P.shape

        S = np.full(n_tournament,-1,dtype=np.int)

        for i in range(n_tournament):
            a,b = P[i]

            if pop_rank[a] < pop_rank[b]:
                S[i] = a
            else:
                S[i] = b


        return S


def random_permuations(n, l):
    perms = []
    for i in range(n):
        perms.append(np.random.permutation(l))
    P = np.concatenate(perms)
    return P


class Crossover:

    def __init__(self,crossProb=0.9): 

        self.crossProbThreshold = crossProb

    def calc(self,pop,n_var):

        n_pop = pop.shape[0]
        crossProbability = np.random.random((n_pop))
        do_cross = crossProbability <  self.crossProbThreshold
        R = np.random.randint(0,n_pop,(n_pop,2))
        parents = R[do_cross]
        crossPoint = np.random.randint(1,n_var,parents.shape[0])
        d = pop[parents,:]
        child = []
        for i in range(parents.shape[0]):
            child.append(np.concatenate([d[i,0,:crossPoint[i]],d[i,1,crossPoint[i]:]]))
        child = np.vstack(child)
        pop[do_cross,:] = child
        return pop




class PolynomialMutation:

    def __init__(self,prob_mut,eta_mut):

        self.prob_mut = prob_mut
        self.eta_mut = eta_mut

    def calc(self,x,xl,xu):

        X = copy.deepcopy(x)
        Y = np.full(X.shape,np.inf)

        do_mutation = np.random.random(X.shape) < self.prob_mut

        m = np.sum(np.sum(do_mutation))
        #print(f"mutants locations: {m}")

        Y[:,:] = X

        xl = np.repeat(xl[None,:],X.shape[0],axis=0)[do_mutation] #selecting who is mutating
        xu = np.repeat(xu[None,:],X.shape[0],axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu -xl)

        mut_pow = 1.0/(self.eta_mut + 1.0)


        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)
        #import pdb; pdb.set_trace()

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta_mut + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta_mut + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]
        #import pdb; pdb.set_trace()

        _Y = X + deltaq * (xu - xl)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        Y[do_mutation] = _Y

        return Y






class NSGAII_DEV(_algorithm):
    """
        Implements the "Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
        by Kalyanmoy Deb, Associate Member, IEEE, Amrit Pratap, Sameer Agarwal, and T. Meyarivan

    """

    def __init__(self,selection,crossover,mutation, *args, **kwargs):
        """
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
            * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).

        save_sim: boolean
            * True:  Simulation results will be saved
            * False: Simulation results will not be saved
        """

        super(NSGAII_DEV, self).__init__(*args, **kwargs)


        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation


    def fastSort(self,x):
        n = x.shape[0]
        S = np.zeros((n,n),dtype=bool)
        Np = np.zeros(n)


        for i in range(n):
            for j in range(n):
                S[i,j] = self.dominates(x[i,:],x[j,:])

        nDom = np.sum(S,axis=0) # the n solutions that dominates i
        Np[nDom == 0] = 1 # if i ==  0, i is non-dominated, set i rank to 1, i belongs to first non-dominated front
        k = 1
        # loop over pareto fronts
        while np.sum(Np == 0) > 0:
            l = np.arange(n)[Np==k] # first non-dominated front
            for i in l: # loop over the non-dominated front
                nDom[S[i,:]] = nDom[S[i,:]] -1 # reduce by 1 the rank of the solutions that i dominates
            k += 1
            # now nDom has been reduced by 1, so the next non-dominated front will be nDom ==  0
            # and Np == 0 ensure that we don't pass over the first ranked non-dom solutions
            Np[(nDom == 0) & (Np == 0) ] = k
    
        return Np.astype(int)


   
    def dominates(self,a,b):
        if len(a.shape) >1:
            ret = (np.sum(a <= b,axis =1) == a.shape[1]) & (np.sum(a < b,axis=1) >0)
        else:
            ret = (np.sum(a <= b) == len(a)) & (np.sum(a < b) >0)
        return ret


    def crowdDist(self,x):
        n = x.shape[0]

        nobj = x.shape[1]

        dist = np.zeros(n)


        ord = np.argsort(x,axis=0)
        #import pdb; pdb.set_trace()

        X = x[ord,range(nobj)]

        #import pdb; pdb.set_trace()
    
        dist = np.vstack([X,np.full(nobj,np.inf)]) - np.vstack([np.full(nobj,-np.inf),X])

    
        norm = np.max(X,axis=0) - np.min(X,axis=0)
        dist_to_last,dist_to_next = dist, np.copy(dist)
        dist_to_last,dist_to_next = dist_to_last[:-1]/norm ,dist_to_next[1:]/norm
        J = np.argsort(ord,axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(nobj)] + dist_to_next[J, np.arange(nobj)], axis=1) / nobj


        return _cd

    def crowdDist2(self,x):
        n = x.shape[0]
        
        dist = np.zeros(n)

        for obj in range(x.shape[1]):
            ord = np.argsort(x[:,obj])
            dist[ord[[0,-1]]] = np.inf
            #import pdb; pdb.set_trace()

            norm = np.max(x[:,obj]) - np.min(x[:,obj])

            for i in range(1,n-1):
                dist[i] = dist[ord[i]] + (x[ord[i+1],obj] - x[ord[i-1],obj])/norm

        return dist



    def sample(self, generations,n_pop=None, n_obj=None):
        self.n_pop = n_pop
        self.generation= generations
        self.n_obj = n_obj
        self.set_repetiton(self.generation*self.n_pop)

        Pt = np.vstack([self.setup.parameters()['random'] for i in range(self.n_pop)])
        
        #Burn-in
        #TODO: I would suggest to make the burin-in sample indiviudual for each cpu-core in case of parallel usage, compare dream.py, but not sure if this is defined in the publication
        # evaluate population
        param_generator = ((i,Pt[i,:]) for i in range(self.n_pop))
        Of = list(self.repeat(param_generator))

        Of = np.vstack([i[2] for i in Of]).reshape(self.n_pop,self.n_obj)
        nonDomRank = self.fastSort(Of)

        crDist = np.empty(self.n_pop)
        for rk in range(1,np.max(nonDomRank)+1):
            crDist[nonDomRank == rk] = self.crowdDist(Of[nonDomRank ==rk,:])


        # sorting

        rank = np.lexsort((-crDist,nonDomRank))
        Psort = Pt[rank]
        Ofsort = Of[rank]
        #import pdb; pdb.set_trace()
        for p in range(self.n_pop):
            self.postprocessing(0, Psort[p,:], Ofsort[p,:], p)

        # selection

        offsprings = self.selection.calc(pop_rank = rank)

        Qt = Psort[offsprings,:]

        #import pdb; pdb.set_trace()

        # crossover
        Qt = self.crossover.calc(pop =Qt,n_var = self.setup.n_var)

        # mutation
        self.varminbound = np.array([])
        self.varmaxbound = np.array([])
        for i in self.setup.params:
            self.varminbound = np.append(self.varminbound,i.minbound)
            self.varmaxbound = np.append(self.varmaxbound,i.maxbound)

        Qt = self.mutation.calc(x = Qt,xl = self.varminbound,xu = self.varmaxbound)
        
        for igen in range(1,self.generations - 1):

                Rt = np.vstack([Pt,Qt])

                #import pdb; pdb.set_trace()
                # evaluate population
                param_generator = ((i,Rt[i,:]) for i in range(self.n_pop*2))
                Of = list(self.repeat(param_generator))
                Of = np.vstack([i[2] for i in Of]).reshape(self.n_pop*2,self.setup.n_obj)
                nonDomRank = self.fastSort(Of)

                crDist = np.empty(self.n_pop*2)
                for rk in range(1,np.max(nonDomRank)+1):
                    crDist[nonDomRank == rk] = self.crowdDist(Of[nonDomRank ==rk,:])

                # sorting
                rank = np.lexsort((-crDist,nonDomRank))[:self.n_pop]
                Psort = Rt[rank]
                Ofsort = Of[rank]

                Pt = Psort[:,:]

                #import pdb; pdb.set_trace()
                for p in range(self.n_pop):
                    self.postprocessing(igen, Psort[p,:], Ofsort[p,:], p)

                # selection
                offsprings = self.selection.calc(pop_rank = rank)
                Qt = Psort[offsprings,:]
                # crossover
                Qt = self.crossover.calc(pop =Qt,n_var = self.setup.n_var)
                # mutation
                Qt = self.mutation.calc(x = Qt,xl = self.varminbound,xu =self.varmaxbound)



        self.final_call()
