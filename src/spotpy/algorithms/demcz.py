# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
"""

import numpy as np

from . import _algorithm
from spotpy import spotpylogging


class demcz(_algorithm):
    """
    Implements the DE-MC_Z algorithm from ter Braak and Vrugt (2008).

    Implements a variant of DE-MC_Z. The sampler is a multi-chain sampler that
    proposal states based on the differences between random past states.
    The sampler does not use the snooker updater but does use the crossover
    probability, probability distribution. Convergence assessment is based on a
    naive implementation of the Gelman-Rubin convergence statistics.

    The basis for this algorithm are the following papers:

        Provides the basis for the DE-MC_Z extension (also see second paper).
        C.J.F. ter Braak, and J.A. Vrugt, Differential evolution Markov chain with
        snooker updater and fewer chains, Statistics and Computing, 18(4),
        435-446, doi:10.1007/s11222-008-9104-9, 2008.

        Introduces the origional DREAM idea:
        J.A. Vrugt, C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and
        J.M. Hyman, Accelerating Markov chain Monte Carlo simulation by
        differential evolution with self-adaptive randomized subspace sampling,
        International Journal of Nonlinear Sciences and Numerical
        Simulation, 10(3), 273-290, 2009.

        This paper uses DREAM in an application
        J.A. Vrugt, C.J.F. ter Braak, M.P. Clark, J.M. Hyman, and B.A. Robinson,
        Treatment of input uncertainty in hydrologic modeling: Doing hydrology
        backward with Markov chain Monte Carlo simulation, Water Resources
        Research, 44, W00B09, doi:10.1029/2007WR006720, 2008.

    Based on multichain_mcmc 0.3
    Copyright (c) 2010 John Salvatier.
    All rights reserved.

    Redistribution and use in source and binary forms are permitted
    provided that the above copyright notice and this paragraph are
    duplicated in all such forms and that any documentation,
    advertising materials, and other materials related to such
    distribution and use acknowledge that the software was developed
    by the <organization>. The name of the
    <organization> may not be used to endorse or promote products derived
    from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
    """

    def __init__(self, *args, **kwargs):
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
            * mpc: Multi processing: Iterations on all available cores on your cpu (recommended for windows os).
            * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).

        save_sim: boolean
            * True:  Simulation results will be saved
            * False: Simulationt results will not be saved
        """
        kwargs["optimization_direction"] = "maximize"
        kwargs[
            "algorithm_name"
        ] = "Differential Evolution Markov Chain (DE-MC) algorithm"
        super(demcz, self).__init__(*args, **kwargs)

    def check_par_validity(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            self.logger.error('ERROR Bounds have not the same lenghts as Parameterarray')
        return par

    def sample(
        self,
        repetitions,
        nChains=3,
        burnIn=100,
        thin=1,
        convergenceCriteria=0.8,
        variables_of_interest=None,
        DEpairs=2,
        adaptationRate="auto",
        eps=5e-2,
        mConvergence=True,
        mAccept=True,
    ):
        """
        Samples from a posterior distribution using DREAM.

        Parameters
        ----------
        repetitions : int
            number of draws from the sample distribution to be returned
        nChains : int
            number of different chains to employ
        burnInSize : int
            number of iterations (meaning draws / nChains) to do before doing actual sampling.
        DEpairs : int
            number of pairs of chains to base movements off of
        eps : float
            used in jittering the chains

        Returns
        -------
            None : None
        """

        self.set_repetiton(repetitions)
        self.logger.info(
            "Starting the DEMCz algotrithm with %s repetitions...", 
            repetitions
        )

        self.min_bound, self.max_bound = (
            self.parameter()["minbound"],
            self.parameter()["maxbound"],
        )
        repetitions = int(repetitions / nChains)
        ndraw_max = repetitions * nChains
        maxChainDraws = int(ndraw_max / nChains)

        dimensions = len(self.parameter()["random"])

        # minbound,maxbound=self.find_min_max()
        # select variables if necessary
        if variables_of_interest is not None:
            slices = []
            for var in variables_of_interest:
                slices.append(self.slices[var])
        else:
            slices = [slice(None, None)]

        # make a list of starting chains that at least span the dimension space
        # in this case it will be of size 2*dim
        nSeedIterations = max(int(np.ceil(dimensions * 2 / nChains)), 2)

        # init a simulationhistory instance
        history = _SimulationHistory(
            maxChainDraws + nSeedIterations, nChains, dimensions
        )
        history.add_group("interest", slices)

        ### BURN_IN
        burnInpar = [np.zeros((nChains, dimensions))] * nSeedIterations
        for i in range(nSeedIterations):
            self._logPs = []
            simulationlist = []
            old_like = np.empty(nChains)
            param_generator = (
                (rep, self.parameter()["random"]) for rep in range(int(nChains))
            )

            for rep, vector, simulations in self.repeat(param_generator):
                burnInpar[i][rep] = vector
                likelist = self.postprocessing(i, vector, simulations, chains=rep)
                simulationlist.append(simulations)
                self._logPs.append(likelist)
                old_like[rep] = likelist
                burnInpar[i][rep] = vector
                if self.status.stop:
                    break
        if not self.status.stop:
            history.record(burnInpar[i], self._logPs, 1)

            gamma = None
            self.accepts_ratio = 0.000001

            # initilize the convergence diagnostic object
            grConvergence = _GRConvergence()
            covConvergence = _CovarianceConvergence()

            # get the starting log objectivefunction and position for each of the
            # chains
            currentVectors = burnInpar[-1]
            currentLogPs = self._logPs[-1]

            # 2)now loop through and sample
            cur_iter = 0
            accepts_ratio_weighting = 1 - np.exp(-1.0 / 30)
            lastRecalculation = 0
            # continue sampling if:
            # 1) we have not drawn enough samples to satisfy the minimum number of iterations
            # 2) or any of the dimensions have not converged
            # 3) and we have not done more than the maximum number of iterations

            while cur_iter < maxChainDraws:
                self.logger.debug("%s, %s", cur_iter, burnIn)
                if cur_iter == burnIn:
                    self.logger.debug('starting')
                    history.start_sampling()

                # every5th iteration allow a big jump
                if np.random.randint(5) == 0.0:
                    gamma = np.array([1.0])
                else:
                    gamma = np.array([2.38 / np.sqrt(2 * DEpairs * dimensions)])

                if cur_iter >= burnIn:
                    proposalVectors = _dream_proposals(
                        currentVectors,
                        history,
                        dimensions,
                        nChains,
                        DEpairs,
                        gamma,
                        0.05,
                        eps,
                    )
                    for i in range(len(proposalVectors)):
                        proposalVectors[i] = self.check_par_validity(proposalVectors[i])
                    # print proposalVectors
                else:
                    proposalVectors = []
                    for i in range(nChains):
                        proposalVectors.append(self.parameter()["random"])
                        proposalVectors[i] = self.check_par_validity(proposalVectors[i])

                # if self.bounds_ok(minbound,maxbound,proposalVectors,nChains):
                proposalLogPs = []
                old_simulationlist = simulationlist
                old_likelist = self._logPs[-1]
                new_simulationlist = []
                new_likelist = []

                param_generator = (
                    (rep, list(proposalVectors[rep])) for rep in range(int(nChains))
                )
                for rep, vector, simulations in self.repeat(param_generator):
                    new_simulationlist.append(simulations)
                    like = self.postprocessing(
                        cur_iter + nSeedIterations,
                        list(vector),
                        simulations,
                        chains=rep,
                    )
                    self._logPs.append(like)
                    new_likelist.append(like)
                    proposalLogPs.append(like)
                    if self.status.stop:
                        cur_iter = maxChainDraws
                        break

                if not self.status.stop:
                    # apply the metrop decision to decide whether to accept or reject
                    # each chain proposal
                    decisions, acceptance = self._metropolis_hastings(
                        currentLogPs, proposalLogPs, nChains
                    )
                    self._update_accepts_ratio(accepts_ratio_weighting, acceptance)
                    # choose from list of possible choices if 1d_decision is True at
                    # specific index, else use default choice
                    # np.choose(1d_decision[:,None], (list of possible choices, default
                    # choice)
                    save_likes = []
                    save_pars = []
                    save_sims = []

                    for curchain in range(nChains):
                        if decisions[curchain]:
                            save_likes.append(float(new_likelist[curchain]))
                            old_like[curchain] = float(new_likelist[curchain])
                            save_pars.append(proposalVectors[curchain])
                            save_sims.append(new_simulationlist[curchain])
                        else:
                            save_likes.append(old_like[curchain])
                            save_pars.append(currentVectors[curchain])
                            save_sims.append(old_simulationlist[curchain])

                    currentVectors = np.choose(
                        decisions[:, np.newaxis], (currentVectors, proposalVectors)
                    )
                    currentLogPs = np.choose(decisions, (currentLogPs, proposalLogPs))

                    simulationlist = [
                        [new_simulationlist, old_simulationlist][int(x)][ix]
                        for ix, x in enumerate(decisions)
                    ]

                    likelist = list(
                        np.choose(
                            decisions[:, np.newaxis], (new_likelist, old_likelist)
                        )
                    )

                    # we only want to recalculate convergence criteria when we are past
                    # the burn in period

                    if cur_iter % thin == 0:

                        historyStartMovementRate = adaptationRate
                        # try to adapt more when the acceptance rate is low and less
                        # when it is high
                        if adaptationRate == "auto":
                            historyStartMovementRate = min(
                                (0.234 / self.accepts_ratio) * 0.5, 0.95
                            )

                        history.record(
                            currentVectors,
                            currentLogPs,
                            historyStartMovementRate,
                            grConvergence=grConvergence.R,
                        )

                if (
                    history.nsamples > 0
                    and cur_iter > lastRecalculation * 1.1
                    and history.nsequence_histories > dimensions
                ):
                    lastRecalculation = cur_iter
                    grConvergence.update(history)
                    covConvergence.update(history, "all")
                    covConvergence.update(history, "interest")
                    if all(grConvergence.R < convergenceCriteria):
                        cur_iter = maxChainDraws
                        self.logger.info(
                            "All chains fullfil the convergence criteria. Sampling stopped."
                        )
                cur_iter += 1

        # 3) finalize
        # only make the second half of draws available because that's the only
        # part used by the convergence diagnostic
        self.history = history.samples
        self.histo = history
        self.iter = cur_iter
        self.burnIn = burnIn
        self.R = grConvergence.R
        
        self.logger.info('Gelman Rubin R=%s', self.R)
        self.status.rep = self.status.repetitions
        self.final_call()


    def _update_accepts_ratio(self, weighting, acceptances):
        self.accepts_ratio = (
            weighting * np.mean(acceptances) + (1 - weighting) * self.accepts_ratio
        )

    def _metropolis_hastings(
        self, currentLogPs, proposalLogPs, nChains, jumpLogP=0, reverseJumpLogP=0
    ):
        """
        makes a decision about whether the proposed vector should be accepted
        """
        logMetropHastRatio = np.array(proposalLogPs) - np.array(
            currentLogPs
        )  # + (reverseJumpLogP - jumpLogP)
        decision = np.log(np.random.uniform(size=nChains)) < logMetropHastRatio

        # replace values which ecoures an overflow for e^x with 100
        isATooBigValue = logMetropHastRatio >= 1e3
        logMetropHastRatio[isATooBigValue] = 1e2

        return decision, np.minimum(1, np.exp(logMetropHastRatio))


class _SimulationHistory(object):

    group_indicies = {"all": slice(None, None)}

    def __init__(self, maxChainDraws, nChains, dimensions):
        self._combined_history = np.zeros((nChains * maxChainDraws, dimensions))
        self._sequence_histories = np.zeros((nChains, dimensions, maxChainDraws))
        self._logPSequences = np.zeros((nChains, maxChainDraws))
        self._logPHistory = np.zeros(nChains * maxChainDraws)
        self.r_hat = [] * dimensions
        self._sampling_start = np.Inf

        self._nChains = nChains
        self._dimensions = dimensions
        self.relevantHistoryStart = 0
        self.relevantHistoryEnd = 0

    def add_group(self, name, slices):
        indexes = list(range(self._dimensions))
        indicies = []
        for s in slices:
            indicies.extend(indexes[s])

        self.group_indicies[name] = np.array(indicies)

    def record(self, vectors, logPs, increment, grConvergence=None):
        if len(vectors.shape) < 3:
            self._record(vectors, logPs, increment, grConvergence)
        else:
            for i in range(vectors.shape[2]):
                self._record(vectors[:, :, i], logPs[:, i], increment, grConvergence)

    def _record(self, vectors, logPs, increment, grConvergence):

        self._sequence_histories[:, :, self.relevantHistoryEnd] = vectors
        self._combined_history[
            (self.relevantHistoryEnd * self._nChains) : (
                self.relevantHistoryEnd * self._nChains + self._nChains
            ),
            :,
        ] = vectors
        self._logPSequences[:, self.relevantHistoryEnd] = logPs
        self._logPHistory[
            (self.relevantHistoryEnd * self._nChains) : (
                self.relevantHistoryEnd * self._nChains + self._nChains
            )
        ] = logPs
        self.relevantHistoryEnd += 1
        if np.isnan(increment):
            self.relevantHistoryStart += 0
        else:
            self.relevantHistoryStart += increment
        self.r_hat.append(grConvergence)

    def start_sampling(self):
        self._sampling_start = self.relevantHistoryEnd

    @property
    def sequence_histories(self):
        return self.group_sequence_histories("all")

    def group_sequence_histories(self, name):
        return self._sequence_histories[
            :,
            self.group_indicies[name],
            int(np.ceil(self.relevantHistoryStart)) : self.relevantHistoryEnd,
        ]

    @property
    def nsequence_histories(self):
        return self.sequence_histories.shape[2]

    @property
    def combined_history(self):
        return self.group_combined_history("all")

    def group_combined_history(self, name):
        return self._combined_history[
            (int(np.ceil(self.relevantHistoryStart)) * self._nChains) : (
                self.relevantHistoryEnd * self._nChains
            ),
            self.group_indicies[name],
        ]

    @property
    def ncombined_history(self):
        return self.combined_history.shape[0]

    @property
    def samples(self):
        return self.group_samples("all")

    def group_samples(self, name):
        if self._sampling_start < np.Inf:
            start = int(
                max(np.ceil(self.relevantHistoryStart), self._sampling_start)
                * self._nChains
            )
            end = self.relevantHistoryEnd * self._nChains
        else:
            start = 0
            end = 0
        return self._combined_history[start:end, self.group_indicies[name]]

    @property
    def nsamples(self):
        return self.samples.shape[0]

    @property
    def combined_history_logps(self):
        return self._logPHistory[
            (np.ceil(self.relevantHistoryStart) * self._nChains) : (
                self.relevantHistoryEnd * self._nChains
            )
        ]


def _random_no_replace(sampleSize, populationSize, numSamples):

    samples = np.zeros((numSamples, sampleSize), dtype=int)
    # Use Knuth's variable names
    n = sampleSize
    N = populationSize
    i = 0
    t = 0  # total input records dealt with
    m = 0  # number of items selected so far

    while i < numSamples:
        t = 0
        m = 0
        while m < n:
            # call a uniform(0,1) random number generator
            u = np.random.uniform()
            if (N - t) * u >= n - m:
                t += 1
            else:
                samples[i, m] = t
                t += 1
                m += 1
        i += 1
    return samples


class _CovarianceConvergence():
    
    relativeVariances = {}

    def update(self, history, group):

        relevantHistory = history.group_combined_history(group)

        self.relativeVariances[group] = self.rv(relevantHistory)

    @staticmethod
    def rv(relevantHistory):
        end = relevantHistory.shape[0]
        midpoint = int(np.floor(end / 2))

        covariance1 = np.cov(relevantHistory[0:midpoint, :].transpose())
        covariance2 = np.cov(relevantHistory[midpoint:end, :].transpose())

        _eigenvalues1, _eigenvectors1 = _eigen(covariance1)
        basis1 = np.sqrt(_eigenvalues1)[np.newaxis, :] * _eigenvectors1

        _eigenvalues2, _eigenvectors2 = _eigen(covariance2)
        basis2 = np.sqrt(_eigenvalues2)[np.newaxis, :] * _eigenvectors2

        # project the second basis onto the first basis

        try:
            projection = np.dot(np.linalg.inv(basis1), basis2)
        except np.linalg.linalg.LinAlgError as e:
            projection = np.array(basis1) * np.nan
            spotpylogging.get_logger("_CovarianceConvergence()").logger.debug("Exception happend!\nExcpetion:%s", e)

        # find the releative size in each of the basis1 directions
        return np.log(np.sum(projection**2, axis=0) ** 0.5)


def _eigen(a, n=-1):

    # if we got a 0-dimensional array we have to turn it back into a 2
    # dimensional one
    if len(a.shape) == 0:
        a = a[np.newaxis, np.newaxis]

    if n == -1:
        n = a.shape[0]

    _eigenvalues, _eigenvectors = np.linalg.eigh(a)

    indicies = np.argsort(_eigenvalues)[::-1]
    return _eigenvalues[indicies[0:n]], _eigenvectors[:, indicies[0:n]]


def _dream_proposals(
    currentVectors, history, dimensions, nChains, DEpairs, gamma, jitter, eps
):
    """
    generates and returns proposal vectors given the current states
    """

    sampleRange = history.ncombined_history
    currentIndex = np.arange(sampleRange - nChains, sampleRange)[:, np.newaxis]
    combined_history = history.combined_history

    # choose some chains without replacement to combine
    chains = _random_no_replace(DEpairs * 2, sampleRange - 1, nChains)

    # makes sure we have already selected the current chain so it is not replaced
    # this ensures that the the two chosen chains cannot be the same as the
    # chain for which the jump is
    chains += chains >= currentIndex

    chainDifferences = np.sum(
        combined_history[chains[:, 0:DEpairs], :], axis=1
    ) - np.sum(combined_history[chains[:, DEpairs : (DEpairs * 2)], :], axis=1)

    e = np.random.normal(0, jitter, (nChains, dimensions))

    # could replace eps with 1e-6 here
    E = np.random.normal(0, eps, (nChains, dimensions))

    proposalVectors = (
        currentVectors + (1 + e) * gamma[:, np.newaxis] * chainDifferences + E
    )
    return proposalVectors


class _GRConvergence():
    """
    Gelman Rubin convergence diagnostic calculator class. It currently only calculates the naive
    version found in the first paper. It does not check to see whether the variances have been
    stabilizing so it may be misleading sometimes.
    """

    def __init__(self):
        self._W = np.Inf
        self._R = np.Inf
        self._V = np.Inf
        self._VChange = np.Inf
        self._WChange = np.Inf

    def _get_R(self):
        return self._R

    R = property(_get_R)

    @property
    def VChange(self):
        return self._VChange

    @property
    def WChange(self):
        return self._WChange

    def update(self, history):
        """
        Updates the convergence diagnostic with the current history.
        """

        N = history.nsequence_histories

        sequences = history.sequence_histories

        variances = np.var(sequences, axis=2)

        means = np.mean(sequences, axis=2)

        withinChainVariances = np.mean(variances, axis=0)

        betweenChainVariances = np.var(means, axis=0) * N

        varEstimate = (1 - 1.0 / N) * withinChainVariances + (
            1.0 / N
        ) * betweenChainVariances

        if np.abs(withinChainVariances).all() > 0:
            self._R = np.sqrt(varEstimate / withinChainVariances)
        else:
            self._R = np.random.uniform(0.1, 0.9, withinChainVariances.__len__())

        # self._W can be a float or an array, in both cases I want to exclude 0.0 values
        try:
            if self._W != 0.0:
                if (withinChainVariances / self._W).all() <= 0.0:
                    self._WChange = np.abs(np.log(1.1) ** 0.5)
                else:
                    self._WChange = np.abs(
                        np.log(withinChainVariances / self._W) ** 0.5
                    )
            else:
                self._WChange = np.abs(np.log(1.1) ** 0.5)
        except ValueError:
            if self._W.all() != 0.0:
                if (withinChainVariances / self._W).all() <= 0.0:
                    self._WChange = np.abs(np.log(1.1) ** 0.5)
                else:
                    # log of values less then 1 gives a negative number, where I replace these values to get square working
                    tmp_WChDV = varEstimate / self._V
                    tmp_WChDV[tmp_WChDV < 1.0] = np.random.uniform(1, 5, 1)
                    self._WChange = np.abs(np.log(tmp_WChDV) ** 0.5)
            else:
                self._WChange = np.abs(np.log(1.1) ** 0.5)

        self._W = withinChainVariances

        if (varEstimate / self._V).any() <= 0.0 or np.isnan(
            varEstimate / self._V
        ).any():
            self._VChange = np.abs(np.log(1.1) ** 0.5)
        else:
            # log of values less then 1 gives a negative number, where I replace these values to get square working
            tmp_EsDV = varEstimate / self._V
            tmp_EsDV[tmp_EsDV < 1.0] = np.random.uniform(1, 5, 1)
            self._VChange = np.abs(np.log(tmp_EsDV) ** 0.5)

        self._V = varEstimate
