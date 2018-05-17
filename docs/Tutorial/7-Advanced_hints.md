<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
# Advanced settings

This chapter will show you, how to get the full power out of SPOTPY:

## Adjust algorithms
All algorithms come along with a standard setting, to provide acceptable parameter estimations, without expert knowledge.
If you want to change the standard settings (which is absolutely recommended if you know what you are doing), try something like this:
	
	sampler = spotpy.algorithms.sceua(spotpy_setup,() dbname='RosenSCEUA', dbformat='csv')
	sampler.sample(10000,ngs=10,kstop=100,pcento=0.001,peps=0.01)

Every algorithm has it's settings, which can be user defined. Read the paper about the algorithms to know what the settings do or 
play around with this settings and see what happens. 


## MPI - Parallel Computing
If you want to use the parallel setup of the algorithms you need to have [mpi4py](http://mpi4py.scipy.org/ "mpi4py") installed on your system.
To tell SPOTPY to use MPI, just give this information to the sampler:

	sampler = spotpy.algorithms.sceua(spotpy_setup,() dbname='RosenSCEUA', dbformat='csv', parallel='mpi')
	sampler.sample(10000)
	
Now save this file and start it from a console: `mpirun -c 20 your_script.py`, where 20 is the number of cores you want to use.
This should give you and speed of neerly 20 times compared with the standard sequential sampling.

Keep in mind that `MC`, `LHS` `FAST` and `ROPE` can use as much cpu-cores as you have. `SCE-UA` will run its number of complexes 
'ngs' in parallel (see the paper [Muttil et al., 2007](http://vuir.vu.edu.au/767/) for details), which you can
change by typing:
 
	sampler = spotpy.algorithms.sceua(spotpy_setup,() dbname='RosenSCEUA', dbformat='csv')
	sampler.sample(10000,ngs=20)

DE-MCz will parallelize the selcted number of chains [terBrack and Vrugt, 2008](http://link.springer.com/article/10.1007/s11222-008-9104-9). You can adjust this number by typing:

	sampler = spotpy.algorithms.demcz(spotpy_setup,() dbname='RosenSCEUA', dbformat='csv')
	sampler.sample(10000,nChains=20)

Th algorithms `MLE`, `MCMC` and `SA` can not run in parallel.

## FAST - Sensitivity analysis
SPOTPY gives you the opportunity to start a sensitivity analysis of your model. In this case, we included a global sensitivity analysis called "Extended FAST" based on 
Saltelli et al. (1999). This is besides the Sobol´ sensitivity test the only algorithm available that is taking parameter interaction into account.

The algorithm will tell you, how sensitive your parameters are on whatever is given back by your objective function. Before you start to sample, you should know how how many
iterations you need to get an reliable information about your parameter. The number of iteration can be calculate after [Henkel et al. GLOBAL SENSITIVITY ANALYSIS OF NONLINEAR MATHEMATICAL MODELS - AN 
IMPLEMENTATION OF TWO COMPLEMENTING VARIANCE-BASED ALGORITHMS, 2012] (https://www.informs-sim.org/wsc12papers/includes/files/con308.pdf): 

$$N = (1+4M^2(1+(k-2)d))k$$ 

with N = needed parameter iterations, M= inference factor (SPOTPY default M=4) and d= frequenzy step (SPOTPY default d=2) and k as the number of parameters of your model.

You can start the simulation with

	sampler = spotpy.algorithms.fast(spotpy_setup,() dbname='Fast_sensitivity', dbformat='csv')
	
and you can analyse your results with

	results = sampler.get_data()
	analyser.plot_fast_sensitivity(results,number_of_sensitiv_pars=5)
	
This will show you a Plot with the total Sensitivity index of all your parameters and in this case the five most sensitive parameters (can be adjusted).
 
## Plotting time
If you want create plots out of your samples and you don't want to sample your results again do something like this: 

	algorithms=['MC','LHS','MLE','MCMC','SCEUA','SA','DEMCz','ROPE']
	results=[]
	for algorithm in algorithms:
		results.append(spotpy.analyser.load_csv_results('Rosen'+algorithm))

This will load your results directly from your created csv-files.
Use this code instead of the sampling code.

## Multi objective calibration
If you have more than one series of observations, or simulation or objectivefunction: You can include them into your analysis, if you use 'MC', 'LHS' or 'FAST':

Let us say you have a model producing biomass and soil moisture simulations, then you can return both in a list of your simulation function:  

	class spot_setup(object):
		def simulations(self,vector):
			biomass, soil_moisture = model(vector)
			return [list(biomass), list(soil_moisture)]

Then you consequently need also evaluation data lists for that:

		def evaluation(self):
			eval_biomass, eval_soil_moisture = load_evaluation_data()
			return [list(eval_biomass), list(eval_soil_moisture)]
			
SPOTPY will transfer those lists into the objective function. Here you can separate them again and, if you want, return also more than one objective function:

		def objectivefunction(self,eval,sim):
			obj1=spotpy.objectivefunctions.bias(eval[0],sim[0])#Biomass data
			obj2=spotpy.objectivefunctions.rmse(eval[1],sim[1])#Soil moisture data
			return [obj1,obj2]

Just note that this works not for 'MLE', 'MCMC', 'SCE-UA', 'SA', 'DE-MCz' and 'ROPE', because they need one explicit objective function during the optimization.

## Sampling from a given parameter list

SPOTPY enables you to sample directly from a given parameter list. This can be useful if you want to check specific parameter combinations or if you want to resample
calibrated parameter set, in order to test different model setups or to save further model outputs. To use this functionality you just have to rewrite your paramters function
in your spotpy setup. We will show you a example how to test parameters of the rosenbrock tutorial. Just give the values you want to test as a list into the spotpy.parameter.List function:

    def __init__(self):
        self.params = [spotpy.parameter.List('x',[1,2,3,4,6,7,8,9,0]), #Give possible x values as a List
                       spotpy.parameter.List('y',[0,1,2,5,7,8,9,0,1])  #Give possible y values as a List
                       ]
    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self,vector):
        x=np.array(vector)
        simulations= [sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)]
        return simulations
        
    def evaluation(self):
        observations=[0]
        return observations
    
    def objectivefunction(self,simulation,evaluation):
        objectivefunction=-spotpy.objectivefunctions.rmse(evaluation,simulation)      
        return objectivefunction

After that you will be able to sample the parameter combinations with the normal `MC` algorithm:

	sampler=spotpy.algorithms.mc(spot_setup(),dbname='Iterator_example',  dbformat='csv') #Parameter lists can be sampled with MC
	sampler.sample(10) #Choose equaly (or less) repetitions as you have parameters in your List
	
This will also run with MPI parallelzation.

## Create a own database

SPOTPY enables you to save results of the sampling in a own database. Users may request different sorts of databases like SQL, hdf5 files, tab separated txt files, xls timeseries.
SPOTPY does not provide all these databases yet, BUT any sort of database can be connected to SPOTPY. Therefore one just has to write his one interface. We provide a simple example how this can be done:
We use the above created example and add a selfmade txt database into a new save function:

	class spot_setup(object):
		slow = 1000
		def __init__(self):
			self.params = [spotpy.parameter.List('x',[1,2,3,4,6,7,8,9,0]), #Give possible x values as a List
						   spotpy.parameter.List('y',[0,1,2,5,7,8,9,0,1])]  #Give possible y values as a List
						   
			self.database = file('MyOwnDatabase.txt','w') #Create a file with writing rights
			
		def parameters(self):
			return spotpy.parameter.generate(self.params)
			
		def simulation(self,vector):
			x=np.array(vector)
			for i in xrange(self.slow):
				s = np.sin(i)
			simulations= [sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)]
			return simulations
			
		def evaluation(self):
			observations=[0]
			return observations
		
		def objectivefunction(self,simulation,evaluation):
			objectivefunction=-spotpy.objectivefunctions.rmse(evaluation,simulation)      
			return objectivefunction
			
		def save(self, objectivefunctions, parameter, simulations):
			line=str(objectivefunctions)+','+str(parameter).strip('[]')+','+str(simulations).strip('[]')+'\n'
			self.database.write(line)

This example can save all results gained from sampling into a txt file. To tell SPOTPY that you want to use your own database, just leave 
out the keywords 'dbname' and 'dbformat' when you initialize the algorithm:

	spot_setup=spot_setup()
	'Leave out dbformat and dbname and spotpy will return results in spot_setup.save function'
	sampler=spotpy.algorithms.mc(spot_setup) 
	sampler.sample(10) #Choose equaly or less repetitions as you have parameters in your List
	spot_setup.database.close() # Close the created txt file
	
Apart from that, some users might be interested not to save there simulation results and only the objective functions and the paramter values,
e.g. to save memory space. This is supported during the initialization of the algorithm, by setting save_sim to False (default=True):

	sampler = spotpy.algorithms.mc(spotpy_setup() dbname='no_simulations', dbformat='csv', save_sim=False)

