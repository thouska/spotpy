from numba import jit

@jit
def hymod(Precip, PET, cmax,bexp,alpha,Rs,Rq):
    """
    See https://www.proc-iahs.net/368/180/2015/piahs-368-180-2015.pdf for a scientific paper.

    :param cmax:
    :param bexp:
    :param alpha:
    :param Rs:
    :param Rq:
    :return: Dataset of water in hymod (has to be calculated in litres)
    :rtype: list
    """

    # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
    x_loss = 0.0
    # Initialize slow tank state
    x_slow = 2.3503 / (Rs * 22.5)
    x_slow = 0  # --> works ok if calibration data starts with low discharge
    # Initialize state(s) of quick tank(s)
    x_quick = [0,0,0]
    t = 0
    outflow = []
    output = []
    # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS

    while t <= len(Precip)-1:
        Pval = Precip[t]
        PETval = PET[t]
        # Compute excess precipitation and evaporation
        ER1, ER2, x_loss = excess(x_loss, cmax, bexp, Pval, PETval)
        # Calculate total effective rainfall
        ET = ER1 + ER2
        #  Now partition ER between quick and slow flow reservoirs
        UQ = alpha * ET
        US = (1 - alpha) * ET
        # Route slow flow component with single linear reservoir
        x_slow, QS = linres(x_slow, US, Rs)
        # Route quick flow component with linear reservoirs
        inflow = UQ

        for i in range(3):
            # Linear reservoir
            x_quick[i], outflow = linres(x_quick[i], inflow, Rq)
            inflow = outflow

        # Compute total flow for timestep
        output.append(QS + outflow)
        t = t+1


    return output

@jit
def power(X,Y):
    X=abs(X) # Needed to capture invalid overflow with netgative values
    return X**Y

@jit
def linres(x_slow,inflow,Rs):
    # Linear reservoir
    x_slow = (1 - Rs) * x_slow + (1 - Rs) * inflow
    outflow = (Rs / (1 - Rs)) * x_slow
    return x_slow,outflow

@jit
def excess(x_loss,cmax,bexp,Pval,PETval):
    # this function calculates excess precipitation and evaporation
    xn_prev = x_loss
    ct_prev = cmax * (1 - power((1 - ((bexp + 1) * (xn_prev) / cmax)), (1 / (bexp + 1))))
    # Calculate Effective rainfall 1
    ER1 = max((Pval - cmax + ct_prev), 0.0)
    Pval = Pval - ER1
    dummy = min(((ct_prev + Pval) / cmax), 1)
    xn = (cmax / (bexp + 1)) * (1 - power((1 - dummy), (bexp + 1)))

    # Calculate Effective rainfall 2
    ER2 = max(Pval - (xn - xn_prev), 0)

    # Alternative approach
    evap = (1 - (((cmax / (bexp + 1)) - xn) / (cmax / (bexp + 1)))) * PETval  # actual ET is linearly related to the soil moisture state
    xn = max(xn - evap, 0)  # update state

    return ER1,ER2,xn


if __name__ == '__main__':
    import sys, os, pandas as pd, numpy as np, re
    if len(sys.argv) != 6:
        print("Hymod reads in from file called 'Param.in'")
        with open("Param.in", "r") as param_in:
            param_line = param_in.readline()
            x = [np.float(i) for i in re.split("\s+", param_line) if len(i) > 0]
    else:
        print("Hymod reads in from stdin")
        x = sys.argv
        x.pop(0)

    owd = os.path.dirname(os.path.realpath(__file__))

    #  pyinstaller --onefile -c --add-data hymod_input.csv:hymod_input.csv  hymod.py
    hymod_path = owd + os.sep + 'hymod_input.csv'
    #print(hymod_path)

    hymod_data = pd.read_csv(hymod_path, delimiter=r";")
    Precip = hymod_data.values[:,1]
    PET = hymod_data.values[:,2]
    x = [np.float(i) for i in x]
    result = hymod(Precip,PET, x[0], x[1], x[2], x[3], x[4])
    np.savetxt("Q.out", result, delimiter=";")