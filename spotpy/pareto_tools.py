import numpy as np

def nd_check(nd_set, y, x):
    """
    It is the Non Dominated Solution Check (ND Check)
    :param nd_set: Pareto Fron
    :param y: ojective values
    :param x: parameter set
    :return: a new pareto front and a value if it was dominated or not (0,1,-1)
    """
    # Algorithm from PADDS Matlab Code

    dominance_flag = 0

    # These are simply reshaping problems if we want to loop over arrays but we have a single float given
    try:
        num_objs = y.shape[0]
    except IndexError:
        y = y.reshape((1, ))
        num_objs = y.shape[0]
    try:
        pareto_high = nd_set.shape[1]
    except IndexError:
        nd_set = nd_set.reshape(1,nd_set.shape[0])
        pareto_high = nd_set.shape[1]


    i = -1  # solution counter
    while i < nd_set.shape[0]-1:
        i += 1
        num_eql = np.sum(y == nd_set[i, :num_objs])
        num_imp = np.sum(y < nd_set[i, :num_objs])
        num_deg = np.sum(y > nd_set[i, :num_objs])

        if num_imp == 0 and num_deg > 0:  # x is dominated
            dominance_flag = -1
            return (nd_set, dominance_flag)
        elif num_eql == num_objs:
            # Objective functions are the same for x and archived solution i
            nd_set[0] = np.append(y, x)  # Replace solution i in ND_set with X
            dominance_flag = 0  # X is non - dominated
            return nd_set, dominance_flag
        elif num_imp > 0 and num_deg == 0:  # X dominates ith solution in the ND_set
            nd_set = np.delete(nd_set, i, 0)
            i = i - 1
            dominance_flag = 1

    if nd_set.size == 0:  # that means the array is completely empty
        nd_set = np.append(y, x)  # Set solution i in ND_set with X
    else:  # If X dominated a portion of solutions in ND_set
        nd_set = np.vstack(
            [nd_set, np.append(y, x)])  # Add the new solution to the end of ND_set (for later use and comparing!

    return nd_set, dominance_flag


def crowd_dist(points):
    """
    This function calculates the Normalized Crowding Distance for each member
    or "points". Deb book p236
     The structure of PF_set is as follows:
     PF_set = [obj_1, obj_2, ... obj_m, DV_1, DV_2, ..., DV_n]

     e.g. v = np.array([[1,10], [2,9.8], [3,5], [4,4], [8,2], [10,1]]); CDInd = crowd_dist(v)

    :param points: mainly is this a pareto front, but could be any set of data which a crowd distance should be calculated from
    :return: the crowd distance distance
    """

    # Normalize Objective Function Space
    try: # Python / Numpy interprets arrays sometimes with wrong shape, this is a fix
        length_x = points.shape[1]
    except IndexError:
        points = points.reshape((1, points.shape[0]))
        length_x = points.shape[1]


    max_f = np.max(points,0)
    min_f = np.min(points,0)
    levels = (max_f == min_f)
    length_y = points.shape[0]


    indicies = np.array(range(length_x))[levels]
    max_f[indicies] += 1


    MAX_f = np.transpose(max_f.repeat(length_y).reshape((length_x,length_y)))
    MIN_f = np.transpose(min_f.repeat(length_y).reshape((length_x,length_y)))


    points = np.divide(points - MIN_f,MAX_f - MIN_f)

    # resave Length
    length_x = points.shape[1]
    length_y = points.shape[0]

    # Initialization
    zero_column = np.array([[0] * length_y]).reshape((length_y, 1))
    index_column = np.array(range(length_y)).reshape((length_y,1))
    temp = np.concatenate((points, zero_column, index_column), 1)
    ij = temp.shape[1] - 2
    endpointIndx = np.array([0]*2*length_x)

    # Main Calculation
    if length_y <= length_x + 1:  # Less than or equal # obj + 1 solutions are non-dominated
        temp[:, ij] = 1   # The crowding distance is 1 for all archived solutions
        return temp[:, ij]
    else: # More than 2 solutions are non - dominated
        for i in range(length_x):
            #  https://stackoverflow.com/a/22699957/5885054
            temp = temp[temp[:,i].argsort()]
            temp[0, ij] = temp[0, ij] + 2 * (temp[1, i] - temp[0, i])
            temp[length_y-1, ij] = temp[length_y-1,ij] + 2*(temp[length_y-1,i] - temp[length_y-2,i])

            for j in range(1, length_y-1):
                temp[j, ij] = temp[j, ij] + (temp[j + 1, i] - temp[j - 1, i])

            endpointIndx[2 * (i - 1) + 0] = temp[0, -1]
            endpointIndx[2 * (i - 1) + 1] = temp[-1, -1]

    #  Endpoints of Pareto Front
    temp = temp[temp[:,temp.shape[1]-1].argsort()]   # Sort points based on the last column to restore the original order of points in the archive
    endpointIndx = np.unique(endpointIndx)


    non_endpointIndx = np.array(range(length_y)).reshape((length_y,1))
    non_endpointIndx=np.delete(non_endpointIndx, endpointIndx, 0)

    non_endpointIndx = non_endpointIndx.reshape((non_endpointIndx.shape[0]))

    Y = points[endpointIndx, :]
    X = points[non_endpointIndx, :]
    IDX = dsearchn(X,Y)   # Identify the closest point in the objective space to each endpoint (dsearchn in Matlab)
    if IDX.size > 0:
        for i in range(endpointIndx.shape[0]):
            temp[endpointIndx[i], ij] = np.max([temp[endpointIndx[i], ij],temp[non_endpointIndx[IDX[i]], ij]])   # IF the closest point to the endpoint has a higher CD value, assign that to the endpoint; otherwise leave the CD value of the endpoint unchanged

    return temp[:, ij]

def dsearchn(x,y):
    """
    Implement Octave / Matlab dsearchn without triangulation
    :param x: Search Points in
    :param y: Were points are stored
    :return: indices of points of x which have minimal distance to points of y
    """
    IDX = []
    for line in range(y.shape[0]):
        distances = np.sqrt(np.sum(np.power(x - y[line, :], 2), axis=1))
        found_min_dist_ind = (np.min(distances, axis=0) == distances)
        length = found_min_dist_ind.shape[0]
        IDX.append(np.array(range(length))[found_min_dist_ind][0])
    return np.array(IDX)



