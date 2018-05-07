.. image:: https://img.shields.io/pypi/v/spotpy.png
  :target: https://pypi.python.org/pypi/spotpy
.. image:: https://img.shields.io/travis/thouska/spotpy/master.png
  :target: https://travis-ci.org/thouska/spotpy
.. image:: https://img.shields.io/badge/license-MIT-blue.png
  :target: http://opensource.org/licenses/MIT
.. image:: https://coveralls.io/repos/github/thouska/spotpy/badge.svg?branch=master
  :target: https://coveralls.io/github/thouska/spotpy?branch=master



Purpose
-------

SPOTPY is a Python tool that enables the use of Computational optimization techniques for calibration, uncertainty 
and sensitivity analysis techniques of almost every (environmental-) model. The package is puplished in the open source journal PLoS One

Houska, T, Kraft, P, Chamorro-Chavez, A and Breuer, L; `SPOTting Model Parameters Using a Ready-Made Python Package <http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180>`_; PLoS ONE; 2015

The simplicity and flexibility enables the use and test of different 
algorithms without the need of complex codes::

	sampler = spotpy.algorithms.sceua(model_setup())     # Initialize your model with a setup file
	sampler.sample(10000)                                # Run the model
	results = sampler.getdata()                          # Load the results
	spotpy.analyser.plot_parametertrace(results)         # Show the results


Features
--------

Complex formal Bayesian informal Bayesian and non-Bayesian algorithms bring complex tasks to link them with a given model. 
We want to make this task as easy as possible. Some features you can use with the SPOTPY package are:

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

* Wide range of objective functions (also known as loss function, fitness function or energy function) to validate the sampled results. Available functions are

  * Bias
  * Procentual Bias (`PBias`)
  * Nash-Sutcliff (`NSE`)
  * logarithmic Nash-Sutcliff (`logNSE`)
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

* Wide range of likelihood functions to validate the sampled results:

  * logLikelihood
  * Gaussian Likelihood to account for Measurement Errors
  * Gaussian Likelihood to account for Heteroscedasticity
  * Likelihood to accounr for Autocorrelation
  * Generalized Likelihood Function
  * Lapacian Likelihood
  * Skewed Student Likelihood assuming homoscedasticity
  * Skewed Student Likelihood assuming heteroscedasticity
  * Skewed Student Likelihood assuming heteroscedasticity and Autocorrelation
  * Noisy ABC Gaussian Likelihood
  * ABC Boxcar Likelihood
  * Limits Of Acceptability
  * Inverse Error Variance Shaping Factor
  * Nash Sutcliffe Efficiency Shaping Factor
  * Exponential Transform Shaping Factor
  * Sum of Absolute Error Residuals

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
  the save solution for long duration samplings and a `sql` database for larger projects.

* Automatic best run selecting and plotting

* Parameter trace plotting

* Parameter interaction plot including the Gaussian-kde function

* Regression analysis between simulation and evaluation data

* Posterior distribution plot

* Convergence diagnostics with Gelman-Rubin and the Geweke plot


Documentation
-------------

Documentation is available at `<http://fb09-pasig.umwelt.uni-giessen.de/spotpy>`__


Install
-------

Installing SPOTPY is easy. Just use:

	pip install spotpy

Or, after downloading the source code and making sure python is in your path:

	python setup.py install

Papers citing SPOTPY
-------
See `Google Scholar <https://scholar.google.de/scholar?cites=17155001516727704728&as_sdt=2005&sciodt=0,5&hl=de>`__ for a continuously updated list.


Support
-------

* Feel free to contact the authors of this tool for any support questions.

* If you use this package for a scientific research paper, please `cite <http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180>`_ SPOTPY.

* Please report any bug through mail or GitHub: https://github.com/thouska/spotpy.

* If you want to share your code with others, you are welcome to do this through GitHub: https://github.com/thouska/spotpy.


Contributing
------------
Patches/enhancements/new algorithms and any other contributions to this package are very welcome!

1. Fork it ( http://github.com/thouska/spotpy/fork )
2. Create your feature branch (``git checkout -b my-new-feature``)
3. Add your modifications
4. Add short summary of your modifications on ``CHANGELOG.rst``
5. Commit your changes (``git commit -m "Add some feature"``)
6. Push to the branch (``git push origin my-new-feature``)
7. Create new Pull Request


Getting started
---------------

Have a look at https://github.com/thouska/spotpy/tree/master/spotpy/examples and http://fb09-pasig.umwelt.uni-giessen.de/spotpy/Tutorial/2-Rosenbrock/
