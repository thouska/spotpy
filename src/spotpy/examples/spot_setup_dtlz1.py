import time

import numpy as np

import spotpy


def g1(x, k):
    return 100 * (
        k + np.sum(np.square(x - 0.5) - np.cos(20 * np.pi * (x - 0.5)), axis=1)
    )


def dtlz1(x, n_var, n_obj):

    k = n_var - n_obj + 1

    X, X_M = x[:, : n_obj - 1], x[:, n_obj - 1 :]
    g = g1(X_M, k)

    f = []
    for i in range(0, n_obj):
        _f = 0.5 * (1 + g)
        _f *= np.prod(X[:, : X.shape[1] - i], axis=1)
        if i > 0:
            _f *= 1 - X[:, X.shape[1] - i]
        f.append(_f)

    return f


class spot_setup(object):
    def __init__(self, n_var=5, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj

        self.params = []
        for i in range(self.n_var):
            self.params.append(spotpy.parameter.Uniform(str(i), 0, 1))

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        vars = np.array(vector)[None]
        sim = dtlz1(vars, n_var=self.n_var, n_obj=self.n_obj)
        # time.sleep(0.1)
        return sim

    def evaluation(self):
        observations = [0] * self.n_obj
        return observations

    def objectivefunction(self, simulation, evaluation):
        obj = []
        for i, f in enumerate(simulation):
            obj.append(
                spotpy.objectivefunctions.mae(
                    evaluation=[evaluation[i]], simulation=[simulation[i]]
                )
            )
        return obj
