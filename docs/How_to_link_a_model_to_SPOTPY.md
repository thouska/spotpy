## Linking a model with SPOTPY

The SPOTPY package comes with several examples, which are designed to help you to set up your own research project. 

Herein we show, how to link an external model with SPOTPY. 

The hydrological model HYMOD serves as an example, which is a commonly used model in the hydrological community.
It takes meteorological data as input and produces discharge as output.

Basically any model can be linked to SPOTPY as long it is somehow start-able with the Python programming language.
Linking a model to SPOTPY is done by following five consecutive steps, which are grouped in a spotpy_setup class.
If these steps are followed, SPOTPY can work with this class and analyse the model in an automatized way, giving you
powerful tools at hand.

## Step 0: Import relevant packages and generate a spotpy setup class

	from spotpy.parameter import Uniform
	from spotpy.objectivefunctions import rmse
	from spotpy.examples.hymod_python.hymod import hymod
	import os

Generating a spotpy setup class is as easy as creating a class in Python:

	class spot_setup(object):

## Step 1: Define the parameter of the model as class parameters

Now, we can fill the setup class. In our case the model comes along with five parameters. 
Here we use Python class parameter, to initialize the parameter for our model and make them readable for SPOTPY.
Needed information is the prior distribution (herein we choose a Uniform distribution), the minimum allowed parameter
setting (low) and the maximum allow setting (high).

		cmax  = Uniform(low=1.0 , high=500)
		bexp  = Uniform(low=0.1 , high=2.0)
		alpha = Uniform(low=0.1 , high=0.99)
		Ks    = Uniform(low=0.001 , high=0.10)
		Kq    = Uniform(low=0.1 , high=0.99)

## Step 2: Write the def __init__ function, which takes care of any things which need to be done only once

In this step, we can load all the data needed to start the model and set information, which might be needed in the following,
but want change during the model assessment. In our case, we want to set the path to the model:

		def __init__(self,obj_func=None):
			#Find Path to Hymod on users system
			self.owd = os.path.dirname(os.path.realpath(__file__))
			self.hymod_path = self.owd+os.sep+'hymod_python'

Further we want to read in the forcing meteorological data and the observation data from file:

			self.PET,self.Precip   = [], []
			self.date,self.trueObs = [], []
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

Finaly, in this case a transformation factor is needed, which brings the output of the model from [mm/day] into [l s-1], where 1.783 is the catchment area [km²]

			self.Factor = 1.783 * 1000 * 1000 / (60 * 60 * 24)

The keyword `obj_func` is not necessary in most cases. Herein we use it to change the function in Step 5, which makes this example flexible and applicable to various algorithms.

        self.obj_func = obj_func 

## Step 3: Write the def simulation function, which starts your model

This function need to do three things. First it needs to be able to receive a parameter set from SPOTPY. 
This will be an list of, in our case five values. Each value represents one setting for one parameter.
The HYMOD model receives these settings through a function. However, one could also write the settings into a file, which is read by the model (most common way).
Finally we start the model with the parameter set and collect the data, which are at the end returned.

    def simulation(self,x):
        #Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        data = hymod(self.Precip, self.PET, x[0], x[1], x[2], x[3], x[4])
        sim=[]
        for val in data:
            sim.append(val*self.Factor)
        #The first year of simulation data is ignored (warm-up)
        return sim[366:]

## Step 4: Write the def evaluation function, which returns your observation data

Model assessment needs data, which tell you something about the reality that the model aims to reproduce. The HYMOD model produces discharge data, so herein we need to return observed discharge data

    def evaluation(self):
		#The first year of simulation data is ignored (warm-up)
        return self.trueObs[366:]

## Step 5: Write the def objectivefunction, which returns how good the model fits the observation data

In this last step, we compare the observed and the simulated data. So we need a function which can receive both information (simulation and evaluation).
SPOTPY will internally take care of times when this function needs to be called and will bring the corresponding data to this function. 
Important is that the first keyword is handled as simulation/model results and the second as observation/evaluation results.
The SPOTPY package gives you several functions at hand, which compare the model and observation data with each other. 
Please note that the selection of such a function highly influences the results gained with a optimization algorithm. Further some optimization algorithms
have specific needs at this function, as they wont to minimize or maximize the returned value by playing around with the parameter settings. Further some algorithms
need specific likelihood function. Herein we choose the Root Mean Squared Error (rmse) as an example. 
A value of 0 would be a perfect fit of the simulation and observation data, +inf would be the worst possible fit.

    def objectivefunction(self,simulation,evaluation, params=None):
        #SPOTPY expects to get one or multiple values back, 
        #that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation,simulation)
        else:
            #Way to ensure flexible spot setup class
            like = self.obj_func(evaluation,simulation)    
        return like
Finally, save the file and check out the next example how to use the spotpy setup class with an algorithm.
 
The shown example code is available [here](https://github.com/thouska/spotpy/blob/master/spotpy/examples/spot_setup_hymod_python.py)