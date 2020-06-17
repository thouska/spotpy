#NSGAII
import copy
import numpy as np
from numpy.random import random
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n_var = 5
n_obj = 3
n_pop = 100


def polynomial_mutation(x,xl,xu,prob_mut,eta_mut):

    X = copy.deepcopy(x)
    Y = np.full(X.shape,np.inf)

    do_mutation = np.random.random(X.shape) < prob_mut

    m = np.sum(np.sum(do_mutation))
    print(f"mutants locations: {m}")

    Y[:,:] = X
    #import pdb; pdb.set_trace()

    xl = np.repeat(xl[None,:],X.shape[0],axis=0)[do_mutation] #selecting who is mutating
    xu = np.repeat(xu[None,:],X.shape[0],axis=0)[do_mutation]

    X = X[do_mutation]

    delta1 = (X - xl) / (xu - xl)
    delta2 = (xu - X) / (xu -xl)

    mut_pow = 1.0/(eta_mut + 1.0)


    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)
    #import pdb; pdb.set_trace()

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta_mut + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta_mut + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]
    #import pdb; pdb.set_trace()

    _Y = X + deltaq * (xu - xl)

    _Y[_Y < xl] = xl[_Y < xl]
    _Y[_Y > xu] = xu[_Y > xu]

    Y[do_mutation] = _Y

    return Y


def tournament_selection_v1(n_pop,elitism,pressure=2):
    #import pdb; pdb.set_trace()
    #n_perms = elitism * pressure

    perms = []
    for p in range(pressure):
        perms.append(np.random.permutation(range(elitism)))
    n_tournaments = np.stack(perms,axis=1)# not n_perms

    #for ch in range(elitism,n_pop+1):
    #    parentid = np.random.randint(1,elitism,pressure)

    n_tournaments = perms.reshape((elitism,pressure))
    child = np.min(n_tournaments,axis=1) # competition: get lower index winning

    # after competition put children into new population
    
    return child

def tournament_selection_v2(n_pop,elitism,pressure=2):
    child = np.array([])
    if elitism:
        for ch in range(elitism,n_pop):
            parentid = np.random.randint(1,elitism,pressure)
            child =np.append(child,np.min(parentid))
    else:
        for ch in range(n_pop):
            parentid = np.random.randint(1,n_pop,pressure)
            child =np.append(child,np.min(parentid))
 
    return child.astype(int)

def dominates(a,b):
    #assert len(a.shape) > 1 and len(b.shape) > 1
    #pdb.set_trace()
    if len(a.shape) >1:
        ret = (np.sum(a <= b,axis =1) == a.shape[1]) & (np.sum(a < b,axis=1) >0)
    else:
        ret = (np.sum(a <= b) == len(a)) & (np.sum(a < b) >0)
    return ret

def crowdDist(x):
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

def fastSort(x):
    n = x.shape[0]
    S = np.zeros((n,n),dtype=bool)
    Np = np.zeros(n)

    # DEBUG: insert some fake obj function outcome close to the minimum

    for i in range(n):
        for j in range(n):
            #import pdb; pdb.set_trace()

            S[i,j] = dominates(x[i,:],x[j,:])

    #import pdb; pdb.set_trace()
    nDom = np.sum(S,axis=0) # the n solutions that dominates i
    #import pdb; pdb.set_trace()
    Np[nDom == 0] = 1 # if i ==  0, i is non-dominated, set i rank to 1, i belongs to first non-dominated front
    k = 1
    # loop over pareto fronts
    while np.sum(Np == 0) > 0:
        #import pdb; pdb.set_trace()
        l = np.arange(n)[Np==k] # first non-dominated front
        for i in l: # loop over the non-dominated front
            nDom[S[i,:]] = nDom[S[i,:]] -1 # reduce by 1 the rank of the solutions that i dominates
        #import pdb; pdb.set_trace()
        k += 1
        # now nDom has been reduced by 1, so the next non-dominated front will be nDom ==  0
        # and Np == 0 ensure that we don't pass over the first ranked non-dom solutions
        Np[(nDom == 0) & (Np == 0) ] = k
    #import pdb; pdb.set_trace()
    
    return Np.astype(int)


def g1(x,k):
    return 100*( k + np.sum(np.square(x - 0.5) - np.cos(20*np.pi*(x -0.5)), axis=1))


def dtlz1(x,n_var,n_obj):


    k = n_var - n_obj + 1

    X, X_M = x[:, :n_obj - 1], x[:, n_obj - 1:]
    g = g1(X_M,k)

    f = []
    for i in range(0,n_obj):
        _f = 0.5 * (1 + g)
        _f *= np.prod(X[:, :X.shape[1] -i],axis=1)
        if i> 0:
            _f *= 1 - X[:,X.shape[1] -i]
        f.append(_f)
    #import pdb; pdb.set_trace()
    
    return np.stack(f,axis=1)


objFunc = dtlz1





def main(elitism=None,xl=None,xu=None,generations=None):


    if not elitism:
        elitism = 0
    # loop for each generation
    store = []
    for igen in range(1,generations):
        print(f"Generation: {igen}")
        # initialisation
        if igen == 1:

            #import pdb; pdb.set_trace()

            Pt = random((n_pop,n_var))

            #P2t = transfPop(Pt) # why the transfer pop??

            # calculate obj func
            Ot = objFunc(Pt,n_var=n_var,n_obj=n_obj) # (n_pop,n_obj)
            store.append(copy.deepcopy(Ot))
            
        
        else:
            Ot = objFunc(Pt,n_var=n_var,n_obj=n_obj)
            #import pdb; pdb.set_trace()

            store.append(copy.deepcopy(Ot))
            # Np = domination count = number of solutions that dominates the solution p
            # Sp = set of solution that the solution p dominates
            nonDomRank = fastSort(Ot)
            crDist = np.empty(n_pop)
            #import pdb; pdb.set_trace()

            for rk in range(1,np.max(nonDomRank)+1):
                crDist[nonDomRank == rk] = crowdDist(Ot[nonDomRank==rk,:])


                # sort
                # by nonDomRand increasing
                # by crDist decreasing
            sortOt =  np.lexsort((-crDist,nonDomRank)) 
            Psort = Pt[sortOt]
            Osort = Ot[sortOt]
            #import pdb; pdb.set_trace()
            #plt.figure(figsize=(20,10))
            #for i in range(n_obj):
            #    y = Osort[:,i]
            #    plt.plot(y,"-")
            #plt.show() 

            # tournament selection [NEEDS REFACTOR, not happy about this, check pymoo]
            #n_select = 1
            #n_parents = 50
            pressure = 2

            #elitism = n_select * n_parents

            # create new population
            Qt = np.full(Psort.shape,np.nan)
            # apply elitism
            Qt[:elitism,:] = Psort[:elitism,:]
            # get children index
            # [TODO] child can be replicates with this method
            child = tournament_selection_v2(n_pop=n_pop,elitism=elitism,pressure=2)
            Qt[elitism:] = Psort[child]#[child[np.random.randint(0,len(child),n_pop -len(child))]]
            crossoverChance = 0.6

            # combine populations, elitism, Deb papers says it is 2*n_pop
            # crossover
            crossoverCount = 0
            #import pdb; pdb.set_trace()
            Pt[:elitism] = Qt[:elitism]

            for child in range(elitism,n_pop):
                if np.random.uniform(0,1) < crossoverChance:
                    parentID = []
                    #for p in range(pressure):
                    #    parentID.append(np.random.permutation(range(elitism,n_pop +1)))
                    parentID = np.random.randint(elitism,n_pop,2)
                    #parentID = np.stack(parentID,axis=1)
                    parents = Qt[parentID,:]
                    crossoverPoint = np.random.randint(0,n_var)
                    Pt[child,:] =  np.concatenate([parents[0,:crossoverPoint],parents[1,crossoverPoint:]])
                    crossoverCount += 1
            print(f"crossover applied to: {crossoverCount} individuals")
            # mutation (Polynomial mutation)

            prob_mut = 0.1
            eta_mut = 350
            Pt[elitism:,:] = polynomial_mutation(Pt[elitism:,:],xl,xu,prob_mut,eta_mut)
            
    return store,Ot



if __name__ == "__main__":


    generations = 200
    store,Ot = main(elitism=None,xl = np.array([0,0,0,0,0]),xu=np.array([1,1,1,1,1]),generations=generations)
    #import pdb; pdb.set_trace()
    
    O1 = np.vstack(store)
    
    fig,ax = plt.subplots(3,1,sharex = True,figsize=(20,10))
    for i in range(n_obj):
        ax[i].plot(O1[:,i],linewidth=0.2,alpha=0.4)
    x1,y1,z1 = O1[:,0],O1[:,1],O1[:,2]

    x,y,z = Ot[:100,0],Ot[:100,1],Ot[:100,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,marker="o")
    #ax.scatter(x1,y1,z1,color="red")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")
    plt.show()
    
