# spotpy
A Statistical Parameter Optimization Tool for Python

---

[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Python Versions][pypi-pyv-image]][pypi-pyv-link]
[![Build Status][travis-image]][travis-link]
[![License][license-image]][license-link]
[![Coverage Status](https://coveralls.io/repos/github/thouska/spotpy/badge.svg?branch=master)](https://coveralls.io/github/thouska/spotpy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/spotpy/badge/?version=latest)](https://readthedocs.org/projects/spotpy/badge/?version=latest)
[![DOI](https://zenodo.org/badge/47562322.svg)](https://zenodo.org/badge/latestdoi/47562322)

[pypi-v-image]: https://img.shields.io/pypi/v/spotpy.png
[pypi-v-link]: https://pypi.python.org/pypi/spotpy
[pypi-pyv-image]: https://img.shields.io/pypi/pyversions/spotpy.png
[pypi-pyv-link]: https://img.shields.io/pypi/pyversions/spotpy
[travis-image]: https://img.shields.io/travis/thouska/spotpy/master.png
[travis-link]: https://travis-ci.org/thouska/spotpy
[license-image]: https://img.shields.io/badge/license-MIT-blue.png
[license-link]: http://opensource.org/licenses/MIT



Purpose
=================

SPOTPY is a Python framework that enables the use of Computational optimization techniques for calibration, uncertainty 
and sensitivity analysis techniques of almost every (environmental-) model. The package is puplished in the open source journal PLoS One:

Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 
10(12), e0145180, doi:[10.1371/journal.pone.0145180](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180 "SPOTting Model Parameters Using a Ready-Made Python Package"), 2015
 
The simplicity and flexibility enables the use and test of different 
algorithms of almost any model, without the need of complex codes::

	sampler = spotpy.algorithms.sceua(model_setup())     # Initialize your model with a setup file
	sampler.sample(10000)                                # Run the model
	results = sampler.getdata()                          # Load the results
	spotpy.analyser.plot_parametertrace(results)         # Show the results



Features
=================

Complex algorithms bring complex tasks to link them with a model. 
We want to make this task as easy as possible. 
Some features you can use with the SPOTPY package are:

* Fitting models to evaluation data with different algorithms. 
  Available algorithms are: 
  
  * Monte Carlo (`MC`)
  * Markov-Chain Monte-Carlo (`MCMC`)
  * Maximum Likelihood Estimation (`MLE`)
  * Latin-Hypercube Sampling (`LHS`) 
  * Simulated Annealing (`SA`)
  * Shuffled Complex Evolution Algorithm (`SCE-UA`)
  * Differential Evolution Markov Chain Algorithm (`DE-MCz`)
  * Differential Evolution Adaptive Metropolis Algorithm (`DREAM`)
  * RObust Parameter Estimation (`ROPE`)
  * Fourier Amplitude Sensitivity Test (`FAST`)
  * Artificial Bee Colony (`ABC`)
  * Fitness Scaled Chaotic Artificial Bee Colony (`FSCABC`)
  * Dynamically Dimensioned Search algorithm (`DDS`)
  * Pareto Archived - Dynamicallly Dimensioned Search algorithm (`PA-DDS`)
  * Fast and Elitist Multiobjective Genetic Algorithm (`NSGA-II`)

* Wide range of objective functions (also known as loss function, fitness function or energy function) to validate the sampled results. Available functions are

  * Bias
  * PBias
  * Nash-Sutcliffe (`NSE`)
  * logarithmic Nash-Sutcliffe (`logNSE`)
  * logarithmic probability (`logp`)
  * Correlation Coefficient (`r`)
  * Coefficient of Determination (`r^2`)
  * Mean Squared Error (`MSE`)
  * Root Mean Squared Error (`RMSE`)
  * Mean Absolute Error (`MAE`)
  * Relative Root Mean Squared Error (`RRMSE`)
  * Agreement Index (`AI`)
  * Covariance, Decomposed MSE (`dMSE`)
  * Kling-Gupta Efficiency (`KGE`)
  * Non parametric Kling-Gupta Efficiency (`KGE_non_parametric`)

* Wide range of hydrological signatures functions to validate the sampled results:

  * Slope
  * Flooding/Drought events
  * Flood/Drought frequency
  * Flood/Drought duration
  * Flood/Drought variance
  * Mean flow
  * Median flow
  * Skewness
  * compare percentiles of discharge

* Prebuild parameter distribution functions: 

  * Uniform
  * Normal
  * logNormal
  * Chisquare
  * Exponential
  * Gamma
  * Wald
  * Weilbull

* Wide range to adapt algorithms to perform uncertainty-, sensitivity analysis or calibration
  of a model.

* Multi-objective support
 
* MPI support for fast parallel computing

* A progress bar monitoring the sampling loops. Enables you to plan your coffee brakes.

* Use of NumPy functions as often as possible. This makes your coffee brakes short.

* Different databases solutions: `ram` storage for fast sampling a simple , `csv` tables
  the save solution for long duration samplings.

* Automatic best run selecting and plotting

* Parameter trace plotting

* Parameter interaction plot including the Gaussian-kde function

* Regression analysis between simulation and evaluation data

* Posterior distribution plot

* Convergence diagnostics with Gelman-Rubin and the Geweke plot


Install
=================

Classical Python options exist to install SPOTPY:

From PyPi:

	pip install spotpy

From Conda-Forge:

	conda config --add channels conda-forge
	conda config --set channel_priority strict
	conda install spotpy

From [Source](https://pypi.python.org/pypi/spotpy):

	python setup.py install


Support
=================

* Documentation: https://spotpy.readthedocs.io/en/latest/

* Feel free to contact the authors of this tool for any support questions.

* Please contact the authors in case of any bug.

* If you use this package for a scientific research paper, please cite SPOTPY. It is [peer-reviewed](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180 "SPOTting Model Parameters Using a Ready-Made Python Package").

* Patches/enhancements and any other contributions to this package are very welcome!


Getting started
=================
Have a look at https://github.com/thouska/spotpy/tree/master/spotpy/examples and https://spotpy.readthedocs.io/en/latest/getting_started/


Contributing
=================
Patches/enhancements/new algorithms and any other contributions to this package are very welcome!

1. Fork it ( http://github.com/thouska/spotpy/fork )
2. Create your feature branch (``git checkout -b my-new-feature``)
3. Add your modifications
4. Add short summary of your modifications on ``CHANGELOG.rst``
5. Commit your changes (``git commit -m "Add some feature"``)
6. Push to the branch (``git push origin my-new-feature``)
7. Create new Pull Request

Papers citing SPOTPY
=====================
See [Google Scholar](https://scholar.google.de/scholar?cites=17155001516727704728&as_sdt=2005&sciodt=0,5&hl=de) for a continuously updated list.

