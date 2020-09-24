# Calibration of HYMOD with DREAM

This chapter shows you, how to link the external hydrological model  HYMOD with SPOTPY (works only on Windows systems).

We use the hydrological model HYMOD as an example, to calibrate it with the Differential Evolution Adaptive Metropolis (DREAM) algorithm. 
For detailed information about the underlying theorie, have a look at the [Vrugt (2016)](https://doi.org/10.1016/j.envsoft.2015.08.013 "Vrugt (2016)").
The SPOTPY package comes with an example which is desgined to help you to set up your own research project. 

First we need to import some functions we will need in the following:

	from spotpy.parameter import Uniform
	from spotpy.examples.hymod_python.hymod import hymod
	import os

Now, we can setup the model within a spot setup class. The model needs some meteorological input data and five parameters to estimate discharge:
  
## Connect HYMOD with SPOTPY

Here we use class parameter, to initialize the parameter for our model.

	class spot_setup(object):
		cmax  = Uniform(low=1.0 , high=500,  optguess=412.33)
		bexp  = Uniform(low=0.1 , high=2.0,  optguess=0.1725)
		alpha = Uniform(low=0.1 , high=0.99, optguess=0.8127)
		Ks    = Uniform(low=0.001 , high=0.10, optguess=0.0404)
		Kq    = Uniform(low=0.1 , high=0.99, optguess=0.5592)
			
		def __init__(self, obj_func=None):
			#Just a way to keep this example flexible and applicable to various examples
			self.obj_func = obj_func  
			#Transform [mm/day] into [l s-1], where 1.783 is the catchment area
			self.Factor = 1.783 * 1000 * 1000 / (60 * 60 * 24) 
			self.PET,self.Precip   = [], []
			self.date,self.trueObs = [], []
			#Find Path to Hymod on users system
			self.owd = os.path.dirname(os.path.realpath(__file__))
			self.hymod_path = self.owd+os.sep+'hymod_python'
			#Load Observation data from file
			climatefile = open(self.hymod_path+os.sep+'hymod_input.csv', 'r')
			headerline = climatefile.readline()[:-1]

			#Read model forcing in working storage (this is done only ones)
			if ';' in headerline:
				self.delimiter = ';'
			else:
				self.delimiter = ','
			self.header = headerline.split(self.delimiter)
			for line in climatefile:
				values =  line.strip().split(self.delimiter)
				self.date.append(str(values[0]))
				self.Precip.append(float(values[1]))
				self.PET.append(float(values[2]))
				self.trueObs.append(float(values[3]))
			climatefile.close()

We use the simulation function to write one random parameter set into a parameter file, like it is needed for the HYMOD model, 
start the model and read the model discharge output data:

    def simulation(self,x):
        #Here the model is actualy startet with one paramter combination that it gets from spotpy for each time the model is called
        data = hymod(self.Precip, self.PET, x[0], x[1], x[2], x[3], x[4])
        sim=[]
        for val in data:
            sim.append(val*self.Factor)
        #The first year of simulation data is ignored (warm-up)
        return sim[366:]

And in a last step, we compare the observed and the simulated data. Here we set up the setup class in a way, that it can receive any
likein the SPOTPY package. Please mind that the selection of the Likelihood highly influences the results gained with this algorithm:

    def evaluation(self):
		#The first year of simulation data is ignored (warm-up)
        return self.trueObs[366:]

    def objectivefunction(self,simulation,evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation,simulation)
        return like

## Sample with DREAM

Now we can initialize the Hymod example:

	spot_setup=spot_setup()

Create the Dream sampler of spotpy, alt_objfun is set to None to force SPOTPY
to jump into the def objectivefunction in the spot_setup class (default is
spotpy.objectivefunctions.log_p). Results are saved in a DREAM_hymod.csv file:

	sampler=spotpy.algorithms.dream(spot_setup, dbname='DREAM_hymod', dbformat='csv')

Select number of maximum repetitions, the number of chains used by dream (default = 5) and set the Gelman-Rubin convergence limit (default 1.2).
We further allow 100 runs after convergence is achieved:

    #Select number of maximum repetitions
    rep=5000
    
    # Select five chains and set the Gelman-Rubin convergence limit
    nChains                = 4
    convergence_limit      = 1.2
    
    # Other possible settings to modify the DREAM algorithm, for details see Vrugt (2016)
    nCr                    = 3
    eps                    = 10e-6
    runs_after_convergence = 100
    acceptance_test_option = 6

We start the sampler and collect the gained r_hat convergence values after the sampling:

	r_hat = sampler.sample(rep, nChains, nCr, eps, convergence_limit)


## Access the results
All other results can be accessed from the SPOTPY csv-database:

	results = spotpy.analyser.load_csv_results('DREAM_hymod')

These results are structured as a numpy array. Accordingly, you can access the different columns by using simple Python code,
e.g. to access all the simulations:

	fields=[word for word in results.dtype.names if word.startswith('sim')]
	print(results[fields)

## Plot model uncertainty
For the analysis we provide some examples how to plor the data.
If you want to see the remaining posterior model uncertainty:

    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))
        q95.append(np.percentile(results[field][-100:-1],97.5))
    ax.plot(q5,color='dimgrey',linestyle='solid')
    ax.plot(q95,color='dimgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')  
    ax.plot(spot_setup.evaluation(),'r.',label='data')
    ax.set_ylim(-50,450)
    ax.set_xlim(0,729)
    ax.legend()
    fig.savefig('python_hymod.png',dpi=300)

![Posterior model uncertainty](../img/python_hymod_simulation.png)
*Figure 1: Posterior model uncertainty of HYMOD.*

## Plot convergence diagnostic
If you want to check the convergence of the DREAM algorithm:

    spotpy.analyser.plot_gelman_rubin(results, r_hat, fig_name = 'python_hymod_convergence.png')

![Convergence diagnostic](../img/python_hymod_convergence.png)

*Figure 2: Gelman-Rubin onvergence diagnostic of DREAM results.*


## Plot parameter uncertainty
Or if you want to check the posterior parameter distribution:

    parameters = spotpy.parameter.get_parameters_array(spot_setup)
    
    fig, ax = plt.subplots(nrows=5, ncols=2)
    for par_id in range(len(parameters)):
        plot_parameter_trace(ax[par_id][0], results, parameters[par_id])
        plot_posterior_parameter_histogram(ax[par_id][1], results, parameters[par_id])
    
    ax[-1][0].set_xlabel('Iterations')
    ax[-1][1].set_xlabel('Parameter range')
    
    plt.show()
    fig.savefig('hymod_parameters.png',dpi=300)

![Parameter distribution](../img/python_hymod_parameters.png)

*Figure 3: Posterior parameter distribution of HYMOD.*
