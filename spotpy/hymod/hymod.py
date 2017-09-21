import numpy as np

def hymod(cmax,bexp,alpha,Rs,Rq):

    # Define the rainfall
    PET,Precip,MaxT = [],[],[]
    # For more details to that headless file see: bound_units.xlsx
    for line in open('bound.txt', 'r'):
        fn, sn, av, na, nb, nc, nd, ne, nr = line.strip().split('  ')
        #print([fn, sn, av, na, nb, nc, nd, ne, nr])
        PET.append(float(nd))
        Precip.append(float(nc))
        MaxT.append(float(nb))

    m = MaxT.__len__()
    MaxT = max(MaxT)
    # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
    x_loss = 0.0
    # Initialize slow tank state
    x_slow = 2.3503 / (Rs * 22.5)
    # Initialize state(s) of quick tank(s)
    x_quick = [0,0,0]
    t = 1
    outflow = []
    output = np.array([])
    # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS

    while t < m:
        Pval = Precip[t]
        PETval = PET[t]
        # Compute excess precipitation and evaporation
        UT1, UT2, x_loss = excess(x_loss, cmax, bexp, Pval, PETval)
        UQ = alpha * UT2 + UT1
        US = (1 - alpha) * UT2
        # Route slow flow component with single linear reservoir
        inflow = US
        x_slow,outflow = linres(x_slow,inflow,outflow,Rs)
        QS = outflow
        # Route quick flow component with linear reservoirs
        inflow = UQ

        for i in range(3):
            x_quick[i], outflow = linres(x_quick[i], inflow, outflow, Rq)
            inflow = outflow

        # Compute total flow for timestep


        output = np.append(output,(QS + outflow) * 22.5)
        t = t+1


    return output[64:m]

def power(X,Y):
    return X**Y

def linres(x_slow,inflow,outflow,Rs):
    # Linear reservoir
    x_slow = (1 - Rs) * x_slow + (1 - Rs) * inflow
    outflow = (Rs / (1 - Rs)) * x_slow
    return x_slow,outflow


def excess(x_loss,cmax,bexp,Pval,PETval):
    # this function calculates excess precipitation and evaporation
    xn_prev = x_loss
    ct_prev = cmax * (1 - power((1 - ((bexp + 1) * (xn_prev) / cmax)), (1 / (bexp + 1))))
    UT1 = max((Pval - cmax + ct_prev), 0.0)
    Pval = Pval - UT1
    dummy = min(((ct_prev + Pval) / cmax), 1)
    xn = (cmax / (bexp + 1)) * (1 - power((1 - dummy), (bexp + 1)))
    UT2 = max(Pval - (xn - xn_prev), 0)
    evap = min(xn, PETval)
    xn = xn - evap
    return UT1,UT2,xn


